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

  /** [t0, t1) にある拍の番号の範囲 [i0, i1)。 */
  beatRange(t0, t1) {
    return [lowerBound(this.beats, t0), lowerBound(this.beats, t1)];
  }

  /** [t0, t1) にある小節の頭の番号の範囲 [i0, i1)。小節番号は i + 1。 */
  barRange(t0, t1) {
    return [lowerBound(this.downbeats, t0), lowerBound(this.downbeats, t1)];
  }
}
