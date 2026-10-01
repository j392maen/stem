// マルチトラック再生（Web Audio API）。
//
// - 全 stem の AudioBuffer を同じ AudioContext の同じ時刻で start する。
// - stem の ON/OFF は GainNode を 15ms のランプで 0/1 にする（再生は止めない）。
// - シークは全 stem を止めて、同じ時刻から同時に start し直す。止める前に全体の音量（fade）を
//   8ms で 0 にし、start と同時に 8ms で戻す（全 stem 共通の GainNode なので同期は崩れない）。
// - A-B ループは AudioBufferSourceNode の loop / loopStart / loopEnd を全 stem に同じ値で設定する
//   （同じ描画単位で折り返すので stem 同士はずれない）。
// - 速度（T11）: 位置・ループ・長さはすべて「元の曲の時刻」で持つ。
//   rate = 全 stem の playbackRate（ピッチも変わる方式の倍率）。
//   bufScale = 読み込んだ音声の 1 秒が元の曲の何秒か（元の音声は 1、サーバーで r 倍に伸縮した音声は r）。
//   曲の時刻の進む速さ speed = rate × bufScale。音声データ上の時刻 = 曲の時刻 ÷ bufScale。
//   速度を変えるときは、全 stem の playbackRate を同じ AudioContext 時刻に setValueAtTime で変え、
//   位置の基準（startOffset・startCtxTime）をその時刻に置き直す（段階的に切り替え、ランプは使わない）。
// - ピッチを保つ・すぐ（T11c）: 全 stem の音量をかけて混ぜた音（bus）を、ブラウザ内の伸縮器
//   （signalsmith-stretch の AudioWorklet。1 つだけ）に通す。音源は今までどおり playbackRate = r で
//   鳴らし（同期・シーク・ループは T11 のまま）、伸縮器で音の高さを −12·log2(r) 半音戻す。
//   伸縮器を通すと音が latency 秒（約 0.12 秒）遅れて聞こえるので、再生位置（position）は
//   「latency 秒前に音源が鳴らした曲の時刻」にする。stem の ON/OFF も latency 秒遅れて聞こえる。
//   経路: stem の GainNode → bus → dry（そのまま）/ 伸縮器 → wet → fade → master。
//   位置の基準は音源の時刻で持ち、速度を変えた履歴（_past）を少し残して、聞こえている位置を数える。

export const RAMP_SEC = 0.015;
export const FADE_SEC = 0.008; // 一時停止・シークの前後のフェード（クリック音を減らす）
const START_DELAY_SEC = 0.03; // start までの余裕（全 stem の start を同じ時刻に揃えるため）
const RATE_DELAY_SEC = 0.02; // 速度を変えるまでの余裕（全 stem の playbackRate を同じ時刻に変えるため）
const HISTORY_SEC = 2; // 位置の基準の履歴を残す長さ（伸縮器の遅れより十分長く）
// 伸縮器の遅れの補正（秒）。伸縮器が報告する遅れ（latency()）に足す。
// 0.5〜2.0 倍の実測（OfflineAudioContext・ガウス形バーストの重心）で決めた（tests/test_browser_stretch.py）。
export const STRETCH_LATENCY_FIX_SEC = 0;
const STRETCH_URL = "../vendor/signalsmith-stretch/SignalsmithStretch.mjs";

/** 倍率 r で速く鳴らした音を、元の高さに戻す半音数。 */
export function semitonesFor(rate) {
  const r = Number(rate);
  return r > 0 ? -12 * Math.log2(r) : 0;
}

/**
 * 位置の基準の履歴 past（{ offset, ctxTime, speed } を ctxTime の順）と今の基準 current から、
 * AudioContext の時刻 ctxTime の曲の時刻を数える純粋関数。ctxTime が今の基準より前なら、
 * その時刻に効いていた基準（ctxTime 以前で最も新しいもの。無ければ最も古いもの）で数える。
 */
export function positionFromHistory(past, current, ctxTime, loop, duration) {
  let seg = current;
  if (ctxTime < current.ctxTime && past.length) {
    seg = past[0];
    for (const s of past) if (s.ctxTime <= ctxTime) seg = s;
  }
  return songPositionAt(seg.offset, ctxTime - seg.ctxTime, seg.speed, loop, duration);
}

/**
 * 再生位置（秒）を計算する純粋関数。
 * startOffset から elapsed 秒進んだ位置。loop（{start, end}）があれば、end に達したら start へ折り返す
 * （start より前から始めた場合も end で折り返す。AudioBufferSourceNode と同じ動き）。
 */
export function positionAt(startOffset, elapsed, loop, duration) {
  let p = startOffset + Math.max(0, elapsed);
  if (loop && loop.end > loop.start && startOffset < loop.end && p >= loop.end) {
    const len = loop.end - loop.start;
    p = loop.start + ((p - loop.end) % len);
  }
  if (Number.isFinite(duration)) p = Math.min(p, duration);
  return p;
}

/**
 * 速度つきの再生位置（曲の時刻）。基準の曲の時刻 base から、実時間で elapsedReal 秒、
 * 速さ speed（曲の秒 / 実時間の秒）で進んだ位置。ループの折り返しは曲の時刻で判定する。
 */
export function songPositionAt(base, elapsedReal, speed, loop, duration) {
  return positionAt(base, Math.max(0, elapsedReal) * speed, loop, duration);
}

/**
 * 速度を変える: 実時間 elapsedReal の時点（旧い速さ oldSpeed で進んだ位置）を新しい基準にする。
 * 戻り値 { base } を startOffset に、切り替えた時刻を startCtxTime に入れれば位置が連続する。
 */
export function rebaseAt(base, elapsedReal, oldSpeed, loop, duration) {
  return { base: songPositionAt(base, elapsedReal, oldSpeed, loop, duration) };
}

/** 位置を 0〜duration に収める。 */
export function clampTime(t, duration) {
  if (!Number.isFinite(t)) return 0;
  return Math.min(Math.max(0, t), Math.max(0, duration || 0));
}

export class Engine {
  constructor(contextFactory = () => new (window.AudioContext || window.webkitAudioContext)()) {
    this.ctx = contextFactory();
    this.master = this.ctx.createGain();
    this.master.connect(this.ctx.destination);
    this.fade = this.ctx.createGain(); // 全 stem 共通のフェード用
    this.fade.connect(this.master);
    this.bus = this.ctx.createGain(); // 全 stem を混ぜた音（伸縮器の前）
    this.dry = this.ctx.createGain(); // 伸縮器を通さない経路
    this.wet = this.ctx.createGain(); // 伸縮器を通した経路
    this.wet.gain.value = 0;
    this.bus.connect(this.dry);
    this.dry.connect(this.fade);
    this.wet.connect(this.fade);
    this.stretch = null; // 伸縮器（AudioWorkletNode）。使うときに作る
    this.stretchLatency = 0; // 伸縮器の遅れ（秒）
    this.stretchFactory = null; // テスト用: (ctx) => Promise<伸縮器>
    this._stretchLoading = null;
    this.lock = false; // ピッチを保つ・すぐ（伸縮器を通す）
    this._wantPlay = false;
    this._starting = null;
    this.tracks = new Map(); // code → { buffer, gain, source }
    this.playing = false;
    this.startCtxTime = 0;
    this.startOffset = 0; // startCtxTime の時点の曲の時刻
    this.pausedAt = 0;
    this.loop = null; // { start, end }（曲の時刻）
    this.duration = 0; // 曲の長さ（秒）
    this._past = []; // 前の位置の基準 { offset, ctxTime, speed }（切り替え時刻まではこちらで数える）
    this.rate = 1; // 全 stem の playbackRate
    this.bufScale = 1; // 音声データの 1 秒が曲の何秒か
    this.onEnded = null;
  }

  /** 曲の時刻の進む速さ（曲の秒 / 実時間の秒）。 */
  get speed() {
    return this.rate * this.bufScale;
  }

  /** 音源が鳴らしてから聞こえるまでの遅れ（秒。伸縮器を通すときだけ）。 */
  get latency() {
    return this.lock ? this.stretchLatency : 0;
  }

  /** stem（code）を足す。buffer が null の stem は音を出さない（子に分かれた親など）。 */
  addTrack(code, buffer, gainValue = 0) {
    const gain = this.ctx.createGain();
    gain.gain.value = gainValue;
    gain.connect(this.bus);
    this.tracks.set(code, { buffer, gain, source: null });
    if (buffer) this.duration = Math.max(this.duration, buffer.duration * this.bufScale);
  }

  async decode(arrayBuffer) {
    return this.ctx.decodeAudioData(arrayBuffer);
  }

  /** 音源が AudioContext の時刻 ctxTime に鳴らす曲の時刻。 */
  _positionAtCtx(ctxTime) {
    // 速度の切り替え時刻より前は、切り替える前の基準と速さで数える（位置が止まって見えないように）
    const current = { offset: this.startOffset, ctxTime: this.startCtxTime, speed: this.speed };
    return positionFromHistory(this._past, current, ctxTime, this.loop, this.duration);
  }

  /** 今の基準を履歴に残す（古いものは捨てる）。 */
  _pushHistory() {
    this._past.push({ offset: this.startOffset, ctxTime: this.startCtxTime, speed: this.speed });
    const old = this.ctx.currentTime - this.latency - HISTORY_SEC;
    while (this._past.length > 1 && this._past[1].ctxTime < old) this._past.shift();
  }

  /** 聞こえている曲の時刻（伸縮器を通すときは、その遅れの分だけ前）。 */
  get position() {
    if (!this.playing) return this.pausedAt;
    return this._positionAtCtx(this.ctx.currentTime - this.latency);
  }

  /** gains: { code: 倍率 }。15ms のランプで変える。 */
  setGains(gains, rampSec = RAMP_SEC) {
    const now = this.ctx.currentTime;
    for (const [code, value] of Object.entries(gains)) {
      const t = this.tracks.get(code);
      if (!t) continue;
      const g = t.gain.gain;
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, now);
      g.linearRampToValueAtTime(value, now + rampSec);
    }
  }

  gainValues() {
    const out = {};
    for (const [code, t] of this.tracks) out[code] = t.gain.gain.value;
    return out;
  }

  setVolume(value) {
    const now = this.ctx.currentTime;
    const g = this.master.gain;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.linearRampToValueAtTime(Math.max(0, Math.min(1, value)), now + RAMP_SEC);
  }

  _fadeOut() {
    const now = this.ctx.currentTime;
    const g = this.fade.gain;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.linearRampToValueAtTime(0, now + FADE_SEC);
    return now + FADE_SEC;
  }

  _startSources(offset, when = this.ctx.currentTime + START_DELAY_SEC) {
    // 新しい音が聞こえ始める時刻（伸縮器を通すときは遅れの分だけ後）からフェードイン
    const heard = when + this.latency;
    const g = this.fade.gain;
    g.setValueAtTime(0, heard);
    g.linearRampToValueAtTime(1, heard + FADE_SEC);
    const s = this.bufScale;
    for (const t of this.tracks.values()) {
      if (!t.buffer) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = t.buffer;
      src.playbackRate.value = this.rate;
      if (this.loop && offset < this.loop.end) {
        src.loop = true;
        src.loopStart = this.loop.start / s;
        src.loopEnd = this.loop.end / s;
      }
      src.connect(t.gain);
      src.start(when, Math.min(offset / s, t.buffer.duration));
      t.source = src;
    }
    this.startCtxTime = when;
    this.startOffset = offset;
    this._past = [];
  }

  /** 鳴っている音源を止める。at（AudioContext の時刻）を渡すとその時刻に止める。 */
  _stopSources(at = 0) {
    for (const t of this.tracks.values()) {
      const src = t.source;
      if (!src) continue;
      t.source = null;
      src.onended = () => src.disconnect();
      try { src.stop(at); } catch { src.disconnect(); }
    }
  }

  /** 再生を始める。始める途中（resume を待つ間）に呼ばれた2回目は同じ Promise を返す。 */
  play() {
    this._wantPlay = true;
    if (this.playing) return Promise.resolve();
    if (!this._starting) {
      this._starting = this._start().finally(() => { this._starting = null; });
    }
    return this._starting;
  }

  async _start() {
    if (this.ctx.state !== "running") await this.ctx.resume();
    // 待つ間に pause() された・既に鳴っているときは始めない
    if (!this._wantPlay || this.playing) return;
    let offset = this.pausedAt;
    if (offset >= this.duration - 0.01) offset = 0;
    this._startSources(offset);
    this.playing = true;
  }

  pause() {
    this._wantPlay = false;
    if (!this.playing) return;
    this.pausedAt = this.position;
    this._stopSources(this._fadeOut());
    this.playing = false;
  }

  /** 出力の遅延（秒）。再生位置の音が実際に聞こえるまでの時間（タップの補正に使う）。 */
  outputLatency() {
    const c = this.ctx;
    return Math.max(0, (Number(c.outputLatency) || 0) + (Number(c.baseLatency) || 0));
  }

  /** 鳴っている音源の数（テスト・確認用）。 */
  activeSources() {
    let n = 0;
    for (const t of this.tracks.values()) if (t.source) n++;
    return n;
  }

  /** 鳴っている音源の playbackRate（テスト・確認用）。 */
  sourceRates() {
    const out = [];
    for (const t of this.tracks.values()) if (t.source) out.push(t.source.playbackRate.value);
    return out;
  }

  seek(t) {
    const target = clampTime(t, this.duration);
    if (this.playing) {
      this._stopSources(this._fadeOut());
      this._startSources(target);
    } else {
      this.pausedAt = target;
    }
  }

  /**
   * 全 stem の playbackRate を rate にする（ピッチも変わる）。再生中は音を止めずに、
   * 全 stem を同じ AudioContext 時刻で切り替え、位置の基準をその時刻に置き直す。
   */
  setRate(rate) {
    const r = Number(rate);
    if (!(r > 0) || r === this.rate) return;
    if (!this.playing) {
      this.rate = r;
      if (this.lock) this._scheduleSemitones(r, this.ctx.currentTime);
      return;
    }
    // 音源の開始（startCtxTime）がまだ先なら、その時刻にそろえる（開始前の位置を数え違えない）
    const when = Math.max(this.ctx.currentTime + RATE_DELAY_SEC, this.startCtxTime);
    const base = this._positionAtCtx(when); // 旧い速さで when まで進んだ位置
    for (const t of this.tracks.values()) {
      if (t.source) t.source.playbackRate.setValueAtTime(r, when);
    }
    if (this.lock) this._scheduleSemitones(r, when);
    this._pushHistory();
    this.rate = r;
    this.startOffset = base;
    this.startCtxTime = when;
  }

  /**
   * 全 stem の音声を差し替える（サーバーで伸縮した音声 ⇔ 元の音声）。buffers: { code: AudioBuffer }、
   * scale: 音声の 1 秒が曲の何秒か、rate: 差し替え後の playbackRate。
   * 再生中は、少し先の時刻 when で前の音源を止めて新しい音源を始める。新しい音源は「前の音源が
   * when まで鳴って届く曲の時刻」から始めるので、同じ部分を繰り返したり飛ばしたりしない
   * （when の直前 8ms でフェードアウトし、when から 8ms でフェードイン）。GainNode はそのまま
   * なので stem の選択は保たれる。ループは曲の時刻なので音声データ上の位置に直して設定し直す。
   */
  setBuffers(buffers, scale = 1, rate = this.rate) {
    const wasPlaying = this.playing;
    const now = this.ctx.currentTime;
    const when = Math.max(now + START_DELAY_SEC, this.startCtxTime);
    const pos = wasPlaying ? this._positionAtCtx(when) : this.position;
    if (wasPlaying) {
      // 伸縮器を通すときは、when に鳴らした音が聞こえる時刻（when + 遅れ）でつなぐ
      const heard = when + this.latency;
      const g = this.fade.gain;
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, now);
      g.setValueAtTime(g.value, Math.max(now, heard - FADE_SEC));
      g.linearRampToValueAtTime(0, heard);
      this._stopSources(when);
    }
    for (const [code, t] of this.tracks) {
      if (Object.prototype.hasOwnProperty.call(buffers, code)) t.buffer = buffers[code];
    }
    const before = { offset: this.startOffset, ctxTime: this.startCtxTime, speed: this.speed };
    this.bufScale = scale;
    this.rate = rate;
    if (wasPlaying) {
      this._startSources(clampTime(pos, this.duration), when);
      // when までは前の音源が鳴っているので、前の基準で数える（位置が止まって見えないように）
      this._past = [before];
    } else {
      this.pausedAt = clampTime(pos, this.duration);
    }
  }

  /** ループ区間を設定（null で解除。曲の時刻）。再生中は音を止めずに切り替える。 */
  setLoop(loop) {
    const valid = loop && loop.end > loop.start ? { start: loop.start, end: loop.end } : null;
    const now = this.ctx.currentTime;
    // 基準を置き直す時刻: 今。ただし音源の開始・速度の切り替えが予約中ならその時刻
    const at = this.playing ? Math.max(now, this.startCtxTime) : now;
    // 区間の内外は、音源が鳴らしている位置で判断する（伸縮器の遅れの分は含めない）
    const pos = this.playing ? this._positionAtCtx(at) : this.position;
    const base = pos;
    this.loop = valid;
    if (!this.playing) {
      if (valid && (pos < valid.start || pos >= valid.end)) this.pausedAt = valid.start;
      return;
    }
    if (valid && (pos < valid.start || pos >= valid.end)) {
      this.seek(valid.start); // 区間の外にいるときは始点へ
      return;
    }
    // 区間の中にいる（または解除）: 鳴っている音源の設定だけを変え、位置の基準を置き直す
    const s = this.bufScale;
    for (const t of this.tracks.values()) {
      if (!t.source) continue;
      t.source.loop = !!valid;
      if (valid) {
        t.source.loopStart = valid.start / s;
        t.source.loopEnd = valid.end / s;
      }
    }
    this._pushHistory(); // 切り替えより前（聞こえるのが遅れている分も）は、前の基準で数える
    this.startOffset = base;
    this.startCtxTime = at;
  }

  // --- ピッチを保つ・すぐ（ブラウザ内の伸縮器。T11c） ---------------------------------------

  /** 伸縮器を作る（1 回だけ。2 回目からは同じものを返す）。失敗したら次に呼んだとき作り直す。 */
  ensureStretch() {
    if (this.stretch) return Promise.resolve(this.stretch);
    if (!this._stretchLoading) {
      const loading = (async () => {
        const create = this.stretchFactory || (await import(STRETCH_URL)).default;
        const node = await create(this.ctx);
        const lat = Number(await node.latency()) || 0;
        node.schedule({ active: false });
        node.connect(this.wet);
        this.stretchLatency = Math.max(0, lat + STRETCH_LATENCY_FIX_SEC);
        this.stretch = node;
        return node;
      })();
      this._stretchLoading = loading;
      loading.catch(() => { if (this._stretchLoading === loading) this._stretchLoading = null; });
    }
    return this._stretchLoading;
  }

  /** 伸縮器の音の高さを、音源の時刻 when から倍率 rate の分だけ戻す。 */
  _scheduleSemitones(rate, when) {
    if (!this.stretch) return;
    // 伸縮器は「出力の時刻」で予約する。音源の時刻 when に入った音が処理される頃に効くようにする
    this.stretch.schedule({ semitones: semitonesFor(rate), output: when + this.stretchLatency / 2 });
  }

  /**
   * 伸縮器を通す（ピッチを保つ・すぐ）/ 通さないを切り替える。伸縮器は ensureStretch() で先に作る
   * （まだ無ければ通さない）。戻り値は切り替え後の状態。
   * 再生中はシークと同じく、いったん音を消して、聞こえていた位置から鳴らし直す（約 0.15 秒途切れる）。
   */
  setPitchLock(on) {
    const want = !!on && !!this.stretch;
    if (want === this.lock) return want;
    const wasPlaying = this.playing;
    const pos = this.position; // 聞こえている位置
    const now = this.ctx.currentTime;
    let at = now;
    if (wasPlaying) {
      at = this._fadeOut();
      this._stopSources(at);
    }
    // 経路を切り替える（再生中は音が消えた時刻に）
    for (const [node, value] of [[this.dry, want ? 0 : 1], [this.wet, want ? 1 : 0]]) {
      node.gain.cancelScheduledValues(now);
      node.gain.setValueAtTime(node.gain.value, now);
      node.gain.setValueAtTime(value, at);
    }
    if (want) {
      this.bus.connect(this.stretch);
      this.stretch.schedule({ active: true, semitones: semitonesFor(this.rate) });
    } else {
      try { this.bus.disconnect(this.stretch); } catch { /* つないでいない */ }
      this.stretch.schedule({ active: false, output: at + this.stretchLatency });
    }
    this.lock = want;
    if (wasPlaying) this._startSources(clampTime(pos, this.duration));
    else this.pausedAt = clampTime(pos, this.duration);
    return want;
  }

  /** 毎フレーム呼ぶ。ループなしで最後まで来たら止める。 */
  tick() {
    if (this.playing && !this.loop && this.position >= this.duration - 0.005) {
      this.pause();
      this.pausedAt = 0;
      if (this.onEnded) this.onEnded();
    }
  }

  async close() {
    this._wantPlay = false;
    this._stopSources();
    this.playing = false;
    this.tracks.clear();
    if (this.stretch) {
      try { this.stretch.disconnect(); } catch { /* 外し済み */ }
    }
    try { await this.ctx.close(); } catch { /* 閉じ済み */ }
  }
}
