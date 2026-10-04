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
//   経路: stem の GainNode → bus → dryDelay → dry ─┐
//                                  └→ 伸縮器 → wet ─┴→ fade → master
//   「すぐ」の方式の間（aligned）は、伸縮器に常に音を入れておき、dryDelay を伸縮器と同じ遅れにして
//   dry と wet の時刻をそろえる。伸縮器を通す・外す（lock）は dry ⇔ wet の 15ms のクロスフェードだけで
//   切り替える（鳴らし直さないので途切れない）。aligned の間は 1.000 倍でも位置に遅れを入れる。
//   aligned の入り・切り（方式の切り替え）だけは、遅れが変わるのでシークと同じく鳴らし直す。
//   位置の基準は音源の時刻で持ち、速度を変えた履歴（_past）を少し残して、聞こえている位置を数える。
// - 音の出し方（T06b。web/js/audioroute.js）: route を付けると、再生を始めるとき（ユーザーの操作の中で、
//   await の前）に route.beforePlay() を呼ぶ（iPhone の audioSession・<audio> の再生）。<audio> を通す
//   経路（C）はその分だけ音が遅れて聞こえるので、再生位置に routeLatency を入れる。
// - あとから読み込んだ stem（T06b。スマホでは選択中の stem だけ読み込む）: addBuffer で、鳴っている
//   ほかの stem と同じ曲の時刻から鳴らし始める。開始時刻 when の曲の時刻は位置の計算（_positionAtCtx）
//   で求め、音声データ上の位置（÷ bufScale）から start(when, offset) する。速度の変更が予約中なら
//   その時刻にそろえ、新しい playbackRate で始める。

export const RAMP_SEC = 0.015;
export const FADE_SEC = 0.008; // 一時停止・シークの前後のフェード（クリック音を減らす）
const START_DELAY_SEC = 0.03; // start までの余裕（全 stem の start を同じ時刻に揃えるため）
const RATE_DELAY_SEC = 0.02; // 速度を変えるまでの余裕（全 stem の playbackRate を同じ時刻に変えるため）
const HISTORY_SEC = 2; // 位置の基準の履歴を残す長さ（伸縮器の遅れより十分長く）
export const XFADE_SEC = 0.015; // 伸縮器を通す・外すときのクロスフェード
// 伸縮器の遅れの補正（秒）。伸縮器が報告する遅れ（latency()）に足す。
// 0.5〜2.0 倍の実測（OfflineAudioContext・ガウス形バーストの重心）で決めた（tests/test_browser_stretch.py）。
export const STRETCH_LATENCY_FIX_SEC = 0;
const STRETCH_URL = "../vendor/signalsmith-stretch/SignalsmithStretch.mjs";

const RENDER_QUANTUM = 128; // Web Audio の描画単位（サンプル）

/** AudioContext の時刻 t を、次の描画単位の頭（128 サンプルの倍数）に切り上げる。 */
export function quantumCeil(t, sampleRate) {
  if (!(sampleRate > 0)) return t;
  const q = RENDER_QUANTUM / sampleRate;
  return Math.ceil(t / q - 1e-6) * q;
}

/** 倍率 r で速く鳴らした音を、元の高さに戻す半音数。 */
export function semitonesFor(rate) {
  const r = Number(rate);
  return r > 0 ? -12 * Math.log2(r) : 0;
}

/**
 * 位置の基準の履歴 past（{ offset, ctxTime, speed, loop? } を ctxTime の順）と今の基準 current から、
 * AudioContext の時刻 ctxTime の曲の時刻を数える純粋関数。ctxTime が今の基準より前なら、
 * その時刻に効いていた基準（ctxTime 以前で最も新しいもの。無ければ最も古いもの）で数える。
 * 基準が loop を持っていれば（その当時のループ）それを使い、無ければ引数の loop を使う。
 */
export function positionFromHistory(past, current, ctxTime, loop, duration) {
  let seg = current;
  if (ctxTime < current.ctxTime && past.length) {
    seg = past[0];
    for (const s of past) if (s.ctxTime <= ctxTime) seg = s;
  }
  const lp = seg.loop !== undefined ? seg.loop : loop;
  return songPositionAt(seg.offset, ctxTime - seg.ctxTime, seg.speed, lp, duration);
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
    this.dryDelay = this.ctx.createDelay(1); // aligned の間は伸縮器と同じ遅れ
    this.dryDelay.delayTime.value = 0;
    this.bus.connect(this.dryDelay);
    this.dryDelay.connect(this.dry);
    this.dry.connect(this.fade);
    this.wet.connect(this.fade);
    this.stretch = null; // 伸縮器（AudioWorkletNode）。使うときに作る
    this.stretchLatency = 0; // 伸縮器の遅れ（秒）
    this.stretchFactory = null; // テスト用: (ctx) => Promise<伸縮器>
    this._stretchLoading = null;
    this.aligned = false; // 「すぐ」の方式: 伸縮器に音を入れ、dry も同じだけ遅らせる
    this.lock = false; // 伸縮器を通した音（wet）を鳴らす（aligned の間だけ）
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
    this.route = null; // 音の出し方（AudioRoute。T06b）
    this.routeLatency = 0; // 音の出し方で増える遅れ（秒）
  }

  /** 音の出し方（web/js/audioroute.js の AudioRoute）を付ける。 */
  setRoute(route) {
    this.route = route || null;
    this.routeLatency = route ? Number(route.latency) || 0 : 0;
  }

  /** 曲の時刻の進む速さ（曲の秒 / 実時間の秒）。 */
  get speed() {
    return this.rate * this.bufScale;
  }

  /** 音源が鳴らしてから聞こえるまでの遅れ（秒。伸縮器を通すときだけ）。 */
  get latency() {
    return this.aligned ? this.stretchLatency : 0;
  }

  /** stem（code）を足す。buffer が null の stem は音を出さない（子に分かれた親など）。 */
  addTrack(code, buffer, gainValue = 0) {
    const gain = this.ctx.createGain();
    gain.gain.value = gainValue;
    gain.connect(this.bus);
    // target: 選択で決めた音量（ランプの行き先）。startInfo: 最後に鳴らし始めた時刻と音声データ上の位置
    // （テスト・確認用。再生の処理では使わない）
    this.tracks.set(code, { buffer, gain, source: null, target: gainValue, startInfo: null });
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

  /** 今の基準を履歴に残す（loop はその基準の間のループ。古いものは捨てる）。 */
  _pushHistory(loop = this.loop) {
    this._past.push({
      offset: this.startOffset, ctxTime: this.startCtxTime, speed: this.speed, loop,
    });
    const old = this.ctx.currentTime - this.latency - this.routeLatency - HISTORY_SEC;
    while (this._past.length > 1 && this._past[1].ctxTime < old) this._past.shift();
  }

  /** 聞こえている曲の時刻（伸縮器・<audio> の経路を通すときは、その遅れの分だけ前）。 */
  get position() {
    if (!this.playing) return this.pausedAt;
    return this._positionAtCtx(this.ctx.currentTime - this.latency - this.routeLatency);
  }

  /** gains: { code: 倍率 }。15ms のランプで変える。 */
  setGains(gains, rampSec = RAMP_SEC) {
    const now = this.ctx.currentTime;
    for (const [code, value] of Object.entries(gains)) {
      const t = this.tracks.get(code);
      if (!t) continue;
      t.target = value;
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
    for (const t of this.tracks.values()) {
      if (t.buffer) this._startOne(t, offset, when);
    }
    this.startCtxTime = when;
    this.startOffset = offset;
    this._past = [];
  }

  /** 1 つの stem を、AudioContext の時刻 when に曲の時刻 offset から鳴らし始める。 */
  _startOne(t, offset, when) {
    const s = this.bufScale;
    const src = this.ctx.createBufferSource();
    src.buffer = t.buffer;
    src.playbackRate.value = this.rate;
    if (this.loop && offset < this.loop.end) {
      src.loop = true;
      src.loopStart = this.loop.start / s;
      src.loopEnd = this.loop.end / s;
    }
    src.connect(t.gain);
    const bufOffset = Math.min(offset / s, t.buffer.duration);
    src.start(when, bufOffset);
    t.source = src;
    t.startInfo = { when, offset: bufOffset, rate: this.rate, scale: s };
  }

  /**
   * あとから読み込んだ stem の音声を入れる（T06b）。再生中なら、鳴っているほかの stem と同じ曲の時刻から
   * 鳴らし始める（開始時刻 when = 少し先。速度の変更が予約中ならその時刻）。その stem だけ 8ms で
   * フェードインする。同じ stem の前の音源は止める。
   */
  addBuffer(code, buffer) {
    const t = this.tracks.get(code);
    if (!t) return false;
    if (t.source) this._stopOne(t);
    t.buffer = buffer || null;
    if (!buffer) return true;
    this.duration = Math.max(this.duration, buffer.duration * this.bufScale);
    if (!this.playing) return true;
    const now = this.ctx.currentTime;
    const when = Math.max(now + START_DELAY_SEC, this.startCtxTime);
    const pos = this._positionAtCtx(when);
    const g = t.gain.gain;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.setValueAtTime(0, when);
    g.linearRampToValueAtTime(t.target, when + FADE_SEC);
    this._startOne(t, pos, when);
    return true;
  }

  /** stem の音声を捨てる（T06b。OFF にしてしばらくたった stem）。鳴っていれば止める。 */
  dropBuffer(code) {
    const t = this.tracks.get(code);
    if (!t) return;
    if (t.source) this._stopOne(t);
    t.buffer = null;
    t.startInfo = null;
  }

  /** デコード済みの音声の大きさ（バイト。32bit float × チャンネル × サンプル）。 */
  memoryBytes() {
    let n = 0;
    for (const t of this.tracks.values()) {
      if (t.buffer) n += t.buffer.length * t.buffer.numberOfChannels * 4;
    }
    return n;
  }

  _stopOne(t, at = 0) {
    const src = t.source;
    t.source = null;
    src.onended = () => src.disconnect();
    try { src.stop(at); } catch { src.disconnect(); }
  }

  /** 鳴っている音源を止める。at（AudioContext の時刻）を渡すとその時刻に止める。 */
  _stopSources(at = 0) {
    for (const t of this.tracks.values()) {
      if (t.source) this._stopOne(t, at);
    }
  }

  /** 再生を始める。始める途中（resume を待つ間）に呼ばれた2回目は同じ Promise を返す。 */
  play() {
    this._wantPlay = true;
    if (this.playing) return Promise.resolve();
    // iPhone: 音声セッション・<audio> はユーザーの操作の中（await の前）で始める
    if (this.route) this.route.beforePlay();
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
    const faded = this._fadeOut();
    this._stopSources(faded);
    this.playing = false;
    if (this.route) this.route.afterPause(faded);
  }

  /** 出力の遅延（秒）。再生位置の音が実際に聞こえるまでの時間（タップの補正に使う）。 */
  // <audio> の経路の遅れ（routeLatency）は position に入れてあるので、ここには足さない
  // （タップの補正は position − outputLatency × 速さ。足すと二重に引くことになる）。
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
      if (this.aligned) this._scheduleSemitones(r, this.ctx.currentTime);
      return;
    }
    // 音源の開始（startCtxTime）がまだ先なら、その時刻にそろえる（開始前の位置を数え違えない）
    // playbackRate は 128 サンプルごと（描画単位の頭）にしか変わらない（k-rate）ので、切り替えの時刻を
    // 描画単位の頭にそろえる（位置の計算と実際の音がずれないように。T06b で測って直した）
    const when = quantumCeil(
      Math.max(this.ctx.currentTime + RATE_DELAY_SEC, this.startCtxTime), this.ctx.sampleRate);
    const base = this._positionAtCtx(when); // 旧い速さで when まで進んだ位置
    for (const t of this.tracks.values()) {
      if (t.source) t.source.playbackRate.setValueAtTime(r, when);
    }
    if (this.aligned) this._scheduleSemitones(r, when);
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
    const prevLoop = this.loop;
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
    this._pushHistory(prevLoop); // 切り替えより前（当時のループで）（聞こえるのが遅れている分も）は、前の基準で数える
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
    // 伸縮器は「出力の時刻」で予約する。音源の時刻 when に鳴らした音が聞こえる時刻（when + 遅れ）に
    // 合わせると、playbackRate を変えた音にちょうど新しい半音数がかかる（実測で外れが無い）
    this.stretch.schedule({ semitones: semitonesFor(rate), output: when + this.stretchLatency });
  }

  /**
   * 「すぐ」の方式に入る・出る。入ると伸縮器に音を入れ続け、dry も伸縮器と同じだけ遅らせる
   * （位置にも遅れを入れる）。出ると伸縮器を外す。伸縮器は ensureStretch() で先に作る（無ければ入らない）。
   * 遅れが変わるので、再生中はシークと同じく、いったん音を消して聞こえていた位置から鳴らし直す。
   */
  setAligned(on) {
    const want = !!on && !!this.stretch;
    if (want === this.aligned) return want;
    const wasPlaying = this.playing;
    const now = this.ctx.currentTime;
    // 伸縮器の遅れの分だけ前（聞こえている位置。<audio> の経路の遅れは変わらないので含めない）
    const pos = wasPlaying ? this._positionAtCtx(now - this.latency) : this.pausedAt;
    let at = now;
    if (wasPlaying) {
      at = this._fadeOut();
      this._stopSources(at);
    }
    this.dryDelay.delayTime.cancelScheduledValues(now);
    this.dryDelay.delayTime.setValueAtTime(want ? this.stretchLatency : 0, at);
    if (want) {
      this.bus.connect(this.stretch);
      this.stretch.schedule({ active: true, semitones: semitonesFor(this.rate) });
    } else {
      this._setPath(false, at, at);
      this.lock = false;
      try { this.bus.disconnect(this.stretch); } catch { /* つないでいない */ }
      this.stretch.schedule({ active: false, output: at + this.stretchLatency });
    }
    this.aligned = want;
    if (wasPlaying) this._startSources(clampTime(pos, this.duration));
    else this.pausedAt = clampTime(pos, this.duration);
    return want;
  }

  /** dry ⇔ wet を、from から to までのクロスフェードで切り替える（wet = 伸縮器を通した音）。 */
  _setPath(wet, from, to) {
    const now = this.ctx.currentTime;
    for (const [node, value] of [[this.dry, wet ? 0 : 1], [this.wet, wet ? 1 : 0]]) {
      const g = node.gain;
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, now);
      if (to > from) {
        g.setValueAtTime(g.value, from);
        g.linearRampToValueAtTime(value, to);
      } else {
        g.setValueAtTime(value, from);
      }
    }
  }

  /**
   * 伸縮器を通した音（wet）を鳴らす / 通さない音（dry）に戻す。「すぐ」の方式（aligned）の間だけ。
   * dry と wet は時刻がそろっているので、15ms のクロスフェードだけで切り替える（鳴らし直さない）。
   * 速度の変更が予約されていれば、その音が聞こえる時刻（startCtxTime + 遅れ）を境にする:
   * 通すときはそこまでに wet へ、外すとき（1.000 倍に戻したとき）はそこから dry へ。
   */
  setPitchLock(on) {
    const want = !!on && this.aligned;
    if (want === this.lock) return want;
    const now = this.ctx.currentTime;
    const edge = this.playing ? Math.max(now, this.startCtxTime + this.latency) : now;
    if (!this.playing) this._setPath(want, now, now);
    else if (want) this._setPath(true, Math.max(now, edge - XFADE_SEC), Math.max(now + XFADE_SEC, edge));
    else this._setPath(false, edge, edge + XFADE_SEC);
    this.lock = want;
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
    if (this.route) this.route.close();
    if (this.stretch) {
      try { this.stretch.disconnect(); } catch { /* 外し済み */ }
    }
    try { await this.ctx.close(); } catch { /* 閉じ済み */ }
  }
}
