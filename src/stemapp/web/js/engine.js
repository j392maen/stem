// マルチトラック再生（Web Audio API）。
//
// - 全 stem の AudioBuffer を同じ AudioContext の同じ時刻で start する。
// - stem の ON/OFF は GainNode を 15ms のランプで 0/1 にする（再生は止めない）。
// - シークは全 stem を止めて、同じ時刻から同時に start し直す。止める前に全体の音量（fade）を
//   8ms で 0 にし、start と同時に 8ms で戻す（全 stem 共通の GainNode なので同期は崩れない）。
// - A-B ループは AudioBufferSourceNode の loop / loopStart / loopEnd を全 stem に同じ値で設定する
//   （同じ描画単位で折り返すので stem 同士はずれない）。

export const RAMP_SEC = 0.015;
export const FADE_SEC = 0.008; // 一時停止・シークの前後のフェード（クリック音を減らす）
const START_DELAY_SEC = 0.03; // start までの余裕（全 stem の start を同じ時刻に揃えるため）

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
    this._wantPlay = false;
    this._starting = null;
    this.tracks = new Map(); // code → { buffer, gain, source }
    this.playing = false;
    this.startCtxTime = 0;
    this.startOffset = 0;
    this.pausedAt = 0;
    this.loop = null; // { start, end }
    this.duration = 0;
    this.onEnded = null;
  }

  /** stem（code）を足す。buffer が null の stem は音を出さない（子に分かれた親など）。 */
  addTrack(code, buffer, gainValue = 0) {
    const gain = this.ctx.createGain();
    gain.gain.value = gainValue;
    gain.connect(this.fade);
    this.tracks.set(code, { buffer, gain, source: null });
    if (buffer) this.duration = Math.max(this.duration, buffer.duration);
  }

  async decode(arrayBuffer) {
    return this.ctx.decodeAudioData(arrayBuffer);
  }

  get position() {
    if (!this.playing) return this.pausedAt;
    return positionAt(this.startOffset, this.ctx.currentTime - this.startCtxTime, this.loop, this.duration);
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

  _startSources(offset) {
    const when = this.ctx.currentTime + START_DELAY_SEC;
    const g = this.fade.gain;
    g.setValueAtTime(0, when);
    g.linearRampToValueAtTime(1, when + FADE_SEC);
    for (const t of this.tracks.values()) {
      if (!t.buffer) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = t.buffer;
      if (this.loop && offset < this.loop.end) {
        src.loop = true;
        src.loopStart = this.loop.start;
        src.loopEnd = this.loop.end;
      }
      src.connect(t.gain);
      src.start(when, Math.min(offset, t.buffer.duration));
      t.source = src;
    }
    this.startCtxTime = when;
    this.startOffset = offset;
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

  /** 鳴っている音源の数（テスト・確認用）。 */
  activeSources() {
    let n = 0;
    for (const t of this.tracks.values()) if (t.source) n++;
    return n;
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

  /** ループ区間を設定（null で解除）。再生中は音を止めずに切り替える。 */
  setLoop(loop) {
    const valid = loop && loop.end > loop.start ? { start: loop.start, end: loop.end } : null;
    const pos = this.position;
    this.loop = valid;
    if (!this.playing) {
      if (valid && (pos < valid.start || pos >= valid.end)) this.pausedAt = valid.start;
      return;
    }
    if (valid && (pos < valid.start || pos >= valid.end)) {
      this.seek(valid.start); // 区間の外にいるときは始点へ
      return;
    }
    // 区間の中にいる（または解除）: 鳴っている音源の設定だけを変え、位置の基準を今に置き直す
    const now = this.ctx.currentTime;
    for (const t of this.tracks.values()) {
      if (!t.source) continue;
      t.source.loop = !!valid;
      if (valid) {
        t.source.loopStart = valid.start;
        t.source.loopEnd = valid.end;
      }
    }
    this.startOffset = pos;
    this.startCtxTime = now;
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
    try { await this.ctx.close(); } catch { /* 閉じ済み */ }
  }
}
