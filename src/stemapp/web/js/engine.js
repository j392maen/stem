// マルチトラック再生（Web Audio API）。
//
// - 全 stem の AudioBuffer を同じ AudioContext の同じ時刻で start する。
// - stem の ON/OFF は GainNode を 15ms のランプで 0/1 にする（再生は止めない）。
// - シークは全 stem を止めて、同じ時刻から同時に start し直す。
// - A-B ループは AudioBufferSourceNode の loop / loopStart / loopEnd を全 stem に同じ値で設定する
//   （同じ描画単位で折り返すので stem 同士はずれない）。

export const RAMP_SEC = 0.015;
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
    gain.connect(this.master);
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

  _startSources(offset) {
    const when = this.ctx.currentTime + START_DELAY_SEC;
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

  _stopSources() {
    for (const t of this.tracks.values()) {
      if (t.source) {
        try { t.source.stop(); } catch { /* 既に止まっている */ }
        t.source.disconnect();
        t.source = null;
      }
    }
  }

  async play() {
    if (this.playing) return;
    if (this.ctx.state !== "running") await this.ctx.resume();
    let offset = this.pausedAt;
    if (offset >= this.duration - 0.01) offset = 0;
    this._startSources(offset);
    this.playing = true;
  }

  pause() {
    if (!this.playing) return;
    this.pausedAt = this.position;
    this._stopSources();
    this.playing = false;
  }

  seek(t) {
    const target = clampTime(t, this.duration);
    if (this.playing) {
      this._stopSources();
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
    this._stopSources();
    this.playing = false;
    this.tracks.clear();
    try { await this.ctx.close(); } catch { /* 閉じ済み */ }
  }
}
