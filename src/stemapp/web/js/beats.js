// 拍・小節の頭と区間ごとの BPM（GET /api/tracks/{id}/beats の中身）を扱う。
// 時刻はすべて元の曲の秒。

/** arr（昇順）で t 以上の最初の位置。 */
export function lowerBound(arr, t) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** 間引きの単位: 間隔 px がこれ以上になる最小の n（1, 2, 4, 8, …）。 */
export function thinStep(pxPerItem, minPx, steps = [1, 2, 4, 8, 16, 32, 64, 128, 256]) {
  for (const n of steps) if (n * pxPerItem >= minPx) return n;
  return steps[steps.length - 1];
}

export function formatBpm(bpm) {
  return Number.isFinite(bpm) ? bpm.toFixed(1) : "—";
}

export class BeatGrid {
  constructor(data) {
    this.beats = Float64Array.from(data.beats || []);
    this.downbeats = Float64Array.from(data.downbeats || []);
    this.timeSignature = data.time_signature || 4;
    this.segments = (data.segments || []).slice().sort((a, b) => a.start_sec - b.start_sec);
    this.analyzer = data.analyzer || "";
    this.edited = !!data.edited; // ユーザーが直した結果か（T10c）
    this.canUndo = !!data.can_undo;
    // 拍の平均の間隔（描画の間引きに使う）
    const n = this.beats.length;
    this.meanBeatSec = n > 1 ? (this.beats[n - 1] - this.beats[0]) / (n - 1) : 0.5;
    const m = this.downbeats.length;
    this.meanBarSec = m > 1
      ? (this.downbeats[m - 1] - this.downbeats[0]) / (m - 1)
      : this.meanBeatSec * this.timeSignature;
  }

  get empty() {
    return this.beats.length === 0;
  }

  /** 時刻 t の区間（最初より前は最初、区間の間・最後より後は直前）。 */
  segmentAt(t) {
    const segs = this.segments;
    if (!segs.length) return null;
    let found = segs[0];
    for (const s of segs) {
      if (s.start_sec <= t) found = s;
      else break;
    }
    return found;
  }

  bpmAt(t) {
    const s = this.segmentAt(t);
    return s ? s.bpm : null;
  }

  /** 時刻 t の小節の拍数（その小節の頭から次の小節の頭までの拍の数。数えられなければ拍子）。 */
  meterAt(t) {
    const d = this.downbeats;
    const i = lowerBound(d, t + 1e-6) - 1;
    if (i < 0 || i + 1 >= d.length) return this.timeSignature;
    const n = lowerBound(this.beats, d[i + 1] - 1e-3) - lowerBound(this.beats, d[i] - 1e-3);
    return n >= 2 && n <= 12 ? n : this.timeSignature;
  }

  /** t にいちばん近い拍の番号（拍が無ければ -1）。 */
  nearestBeatIndex(t) {
    const b = this.beats;
    if (!b.length) return -1;
    const i = lowerBound(b, t);
    if (i <= 0) return 0;
    if (i >= b.length) return b.length - 1;
    return t - b[i - 1] <= b[i] - t ? i - 1 : i;
  }

  /** t にいちばん近い拍の時刻（拍が無ければ t）。 */
  snap(t) {
    const i = this.nearestBeatIndex(t);
    return i < 0 ? t : this.beats[i];
  }

  /** テンポが変わる位置（区間の境目）と前後の BPM。 */
  tempoChanges() {
    const out = [];
    for (let i = 1; i < this.segments.length; i++) {
      out.push({ t: this.segments[i].start_sec, from: this.segments[i - 1].bpm, to: this.segments[i].bpm });
    }
    return out;
  }

  /** [t0, t1) にある拍の番号の範囲 [i0, i1)。 */
  beatRange(t0, t1) {
    return [lowerBound(this.beats, t0), lowerBound(this.beats, t1)];
  }

  /** [t0, t1) にある小節の頭の番号の範囲 [i0, i1)。小節番号は i + 1。 */
  barRange(t0, t1) {
    return [lowerBound(this.downbeats, t0), lowerBound(this.downbeats, t1)];
  }
}

/** 拍の番号 i から n 拍先の時刻（拍の列を越えるときは最後の間隔で伸ばす）。 */
function beatAfter(beats, i, n) {
  const j = i + n;
  if (j < beats.length) return beats[j];
  const last = beats.length - 1;
  const step = last > 0 ? beats[last] - beats[last - 1] : 0.5;
  return beats[last] + (j - last) * step;
}

/**
 * 小節単位のループ: 再生位置 pos の小節の頭から bars 小節（1/4 小節など1未満も可）。
 * 小節の頭が無ければ近い拍から。長さは拍で数える（1小節の拍数はその位置の小節の拍数）。
 * 曲の長さ duration で終わりを切る。作れなければ null。
 */
export function barLoop(grid, pos, bars, duration = Infinity) {
  if (!grid || grid.empty || !(bars > 0)) return null;
  const d = grid.downbeats;
  let start;
  const k = lowerBound(d, pos + 1e-6) - 1;
  if (k >= 0) start = d[k];
  else if (d.length) start = d[0];
  else start = grid.beats[grid.nearestBeatIndex(pos)];
  const i = grid.nearestBeatIndex(start);
  const beatsLen = bars * grid.meterAt(start + 1e-3);
  let end;
  if (Number.isInteger(beatsLen)) end = beatAfter(grid.beats, i, beatsLen);
  else {
    const whole = Math.floor(beatsLen);
    const a = beatAfter(grid.beats, i, whole);
    const b = beatAfter(grid.beats, i, whole + 1);
    end = a + (b - a) * (beatsLen - whole);
  }
  end = Math.min(end, duration);
  if (!(end > start + 0.01)) return null;
  return { start, end, bars };
}

/**
 * タップの記録。叩いた時刻（曲の時刻）を貯め、間が空いたら（gapSec より長い）数え直す。
 * add() は { count, bpm }（bpm は2回目から。間隔の中央値から）を返す。
 */
export class TapRecorder {
  constructor(gapSec = 2.0) {
    this.gapSec = gapSec;
    this.taps = []; // 曲の時刻
    this.wall = []; // 叩いた実時刻（秒。間が空いたかの判定用）
  }

  add(songTime, wallSec) {
    const last = this.wall[this.wall.length - 1];
    if (last !== undefined && (wallSec - last > this.gapSec || songTime <= this.taps[this.taps.length - 1])) {
      this.reset();
    }
    this.taps.push(songTime);
    this.wall.push(wallSec);
    return { count: this.taps.length, bpm: this.bpm() };
  }

  bpm() {
    if (this.taps.length < 2) return null;
    const d = [];
    for (let i = 1; i < this.taps.length; i++) d.push(this.taps[i] - this.taps[i - 1]);
    d.sort((a, b) => a - b);
    const med = d[d.length >> 1];
    return med > 0 ? 60 / med : null;
  }

  reset() {
    this.taps = [];
    this.wall = [];
  }
}
