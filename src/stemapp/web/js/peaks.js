// 波形 peaks（STPK 版 1、docs/PEAKS.md）の読み取りと、表示に使う解像度の選び方。

export const PEAKS_MAGIC = "STPK";
export const PEAKS_HEADER = 20;

/** ArrayBuffer を読む。形式が違えば Error。 */
export function parsePeaks(buf) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < PEAKS_HEADER) {
    throw new Error("波形データが短すぎます。");
  }
  const view = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== PEAKS_MAGIC) throw new Error("波形データの形式が違います。");
  const version = view.getUint16(4, true);
  if (version !== 1) throw new Error(`対応していない波形データの版です（${version}）。`);
  const samplesPerPx = view.getUint32(8, true);
  const sampleRate = view.getUint32(12, true);
  const points = view.getUint32(16, true);
  if (samplesPerPx <= 0 || sampleRate <= 0) throw new Error("波形データのヘッダが正しくありません。");
  if (buf.byteLength < PEAKS_HEADER + points * 2) throw new Error("波形データが途中で切れています。");
  const data = new Int8Array(buf, PEAKS_HEADER, points * 2); // data[2i]=min, data[2i+1]=max
  return { version, samplesPerPx, sampleRate, points, data };
}

/**
 * 曲全体（概観）用: 横幅 widthPx に曲全体を描くとき、1ピクセルに1点以上になる
 * いちばん細かい段階（点の数が widthPx 以下のうち samples_per_px が最小のもの）。
 * どれも多すぎるときはいちばん粗い段階。
 */
export function chooseOverviewLevel(levels, totalSamples, widthPx) {
  const sorted = [...levels].sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  for (const spp of sorted) {
    if (Math.ceil(totalSamples / spp) <= Math.max(1, widthPx)) return spp;
  }
  return sorted[sorted.length - 1];
}

/**
 * 拡大表示用: 1ピクセルあたり wantedSpp サンプルで描くとき、
 * wantedSpp 以下でいちばん粗い段階（1ピクセルに1点以上）。細かすぎる表示ならいちばん細かい段階。
 */
export function chooseZoomLevel(levels, wantedSpp) {
  const sorted = [...levels].sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  let best = sorted[0];
  for (const spp of sorted) {
    if (spp <= wantedSpp) best = spp;
  }
  return best;
}

/** 点 [i0, i1) の最小・最大（-1〜1）。範囲外は 0。 */
export function rangeMinMax(peaks, i0, i1) {
  const n = peaks.points;
  let lo = Math.max(0, Math.floor(i0));
  let hi = Math.min(n, Math.ceil(i1));
  if (hi <= lo) {
    if (lo >= n || i1 <= 0) return [0, 0];
    hi = lo + 1;
  }
  let mn = 127;
  let mx = -127;
  const d = peaks.data;
  for (let i = lo; i < hi; i++) {
    const a = d[2 * i];
    const b = d[2 * i + 1];
    if (a < mn) mn = a;
    if (b > mx) mx = b;
  }
  return [mn / 127, mx / 127];
}
