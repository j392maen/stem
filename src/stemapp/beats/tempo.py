"""拍の時刻の列から、区間ごとの BPM と拍子を求める純粋関数（numpy のみ。GPU 不要）。

流れ:
1. `clean_beats`: 拍の抜け（間隔が周りの約2倍）を等間隔の仮の拍で埋め、余分な拍（間隔が周りの
   半分程度。倍取り・誤検出）を除く。周りの間隔は前後の中央値で求めるので、1〜2拍の誤りに強い。
   長い無音（ブレイク）で拍が大きく途切れた所は埋めず、そこで列を分ける。
2. `tempo_segments`: 整えた拍の間隔（対数）を、区間ごとに一定とみなして当てはめる。
   区間の数は「区間を1つ増やす罰則」との釣り合いで決める（最適分割。動的計画法）。
   その後、BPM の差が小さい隣どうしの区間をまとめる（細かいゆれをならす）。
3. `estimate_time_signature`: 小節の頭（ダウンビート）の間にある拍の数の最頻値。

区間の BPM は「区間の拍の間隔の数 × 60 ÷ 区間の長さ」。1拍ずつの間隔には解析の時刻の刻み
（beat_this は 50 フレーム/秒 = 20ms）による揺れがあるが、区間全体の長さで割るので打ち消される。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# --- 閾値（根拠） ------------------------------------------------------------------

# 周りの間隔を求める窓（拍の間隔の数）。前後4つずつ＝2小節分。1〜2拍の抜け・倍取りが混ざっても
# 中央値は正しい間隔のまま。テンポが変わった所でも4拍ほどで新しい間隔に追いつく。
LOCAL_WINDOW = 9
# 間隔が周りの 1.5 倍を超えたら拍の抜けとみなす（1倍と2倍のちょうど中間）。
MISSING_RATIO = 1.5
# 間隔が周りの 0.6 倍未満なら余分な拍とみなす（半拍＝0.5 倍に少し余裕を持たせた値）。
# 3連符の位置（0.67 倍）に誤って置かれた拍は残るが、区間の BPM は長さで割るので影響は小さい。
EXTRA_RATIO = 0.6
# 抜けを埋めるのは最大3拍まで（間隔が4倍以下）。それより長い途切れはブレイク（無音・拍なし）
# とみなし、仮の拍で埋めずに列を分ける（ブレイクの前後でテンポが違ってもよい）。
MAX_FILL_BEATS = 3
# 区間の最小の長さ（拍の間隔の数）。4/4 で2小節。これより短いテンポの変化は無視する
# （フィルイン・ためなどの一時的な揺れを区間にしない）。
MIN_SEGMENT_BEATS = 8
# 隣の区間との BPM の差がこれ未満ならまとめる（1.5%。120 BPM で ±1.8）。
# DJ ソフトの「テンポが変わった」と感じる目安（R01 B-1）。演奏のゆれ（1% 前後）は1つの区間になる。
MERGE_RATIO = 0.015
# 区間を1つ増やす罰則の係数（BIC に近い形: 係数 × σ² × log(n)）。σ は間隔の揺れの推定値。
# 大きいほど区間が増えにくい。合成音と実曲で 120→140 のような変化を確実に分け、
# 刻みの揺れでは分けない値として 4 にした。
PENALTY_FACTOR = 4.0
# 間隔の揺れ（対数）の推定値の下限。完全に規則的な入力（合成音・Fake）で σ≈0 になり、
# 罰則が 0 になって細かく分けすぎるのを防ぐ（0.5% の揺れがあるものとして扱う）。
MIN_SIGMA = 0.005

MIN_BEATS_PER_BAR = 2
MAX_BEATS_PER_BAR = 12
DEFAULT_TIME_SIGNATURE = 4


@dataclass(frozen=True)
class TempoSegment:
    start_sec: float
    end_sec: float
    bpm: float
    beats: int  # 区間の拍の間隔の数（仮に埋めた拍を含む）

    def as_dict(self) -> dict[str, float]:
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "bpm": round(self.bpm, 2),
        }


def _as_sorted(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return np.unique(arr)  # 並べ替えと重複の除去


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - half) : i + half + 1])
    return out


def clean_beat_runs(beats: Sequence[float] | np.ndarray) -> list[np.ndarray]:
    """拍の列を整え、ブレイクで分けた列のリストを返す（各列は2拍以上。短いものは捨てる）。"""
    b = _as_sorted(beats)
    if len(b) < 3:
        return [b] if len(b) == 2 else []
    local = _rolling_median(np.diff(b), LOCAL_WINDOW)
    runs: list[list[float]] = []
    cur: list[float] = [float(b[0])]
    for i in range(1, len(b)):
        ref = float(local[i - 1])
        gap = float(b[i]) - cur[-1]
        if gap < EXTRA_RATIO * ref:
            continue  # 余分な拍
        if gap > MISSING_RATIO * ref:
            k = int(round(gap / ref))
            if k - 1 > MAX_FILL_BEATS:
                runs.append(cur)  # ブレイク
                cur = [float(b[i])]
                continue
            if k >= 2:
                last = cur[-1]
                cur.extend([last + gap * j / k for j in range(1, k)])
        cur.append(float(b[i]))
    runs.append(cur)
    return [np.asarray(r) for r in runs if len(r) >= 2]


def clean_beats(beats: Sequence[float] | np.ndarray) -> np.ndarray:
    """`clean_beat_runs` をつなげた1本の列。"""
    runs = clean_beat_runs(beats)
    return np.concatenate(runs) if runs else np.zeros(0)


def _noise_sigma(x: np.ndarray) -> float:
    """対数の間隔の揺れ（標準偏差）の頑健な推定。隣どうしの差の中央絶対偏差から求める。"""
    if len(x) < 3:
        return MIN_SIGMA
    d = np.diff(x)
    sigma = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)
    return max(sigma, MIN_SIGMA)


def _optimal_partition(x: np.ndarray, min_len: int, penalty: float) -> list[int]:
    """x を区間ごとに一定とみなしたときの、二乗誤差＋区間数×penalty が最小の区切り（開始位置）。"""
    n = len(x)
    if n < 2 * min_len:
        return [0]
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    prev = np.zeros(n + 1, dtype=np.int64)
    for end in range(min_len, n + 1):
        starts = np.arange(0, end - min_len + 1)
        # 区間の長さが min_len 未満になる区切り（start が残りの頭を作れない所）は best が inf
        length = end - starts
        seg_sum = s1[end] - s1[starts]
        cost = (s2[end] - s2[starts]) - seg_sum * seg_sum / length
        total = best[starts] + cost + penalty
        k = int(np.argmin(total))
        best[end] = total[k]
        prev[end] = starts[k]
    cuts: list[int] = []
    end = n
    while end > 0:
        start = int(prev[end])
        cuts.append(start)
        end = start
    return sorted(cuts)


@dataclass
class _Piece:
    start_sec: float
    end_sec: float
    intervals: int
    span: float  # 区間の拍の間隔の合計（ブレイクを含まない）

    @property
    def bpm(self) -> float:
        return 60.0 * self.intervals / self.span


def _merge_similar(pieces: list[_Piece]) -> list[_Piece]:
    """BPM の差が MERGE_RATIO 未満の隣どうしを、差の小さい所から順にまとめる。"""
    pieces = list(pieces)
    while len(pieces) > 1:
        diffs = [
            abs(a.bpm - b.bpm) / min(a.bpm, b.bpm) for a, b in zip(pieces, pieces[1:], strict=False)
        ]
        i = int(np.argmin(diffs))
        if diffs[i] >= MERGE_RATIO:
            break
        a, b = pieces[i], pieces[i + 1]
        pieces[i : i + 2] = [
            _Piece(a.start_sec, b.end_sec, a.intervals + b.intervals, a.span + b.span)
        ]
    return pieces


def tempo_segments(beats: Sequence[float] | np.ndarray) -> list[TempoSegment]:
    """拍の時刻の列から、テンポが一定の区間の列を求める（拍が足りなければ空）。"""
    pieces: list[_Piece] = []
    for run in clean_beat_runs(beats):
        intervals = np.diff(run)
        x = np.log(intervals)
        sigma = _noise_sigma(x)
        penalty = PENALTY_FACTOR * sigma * sigma * np.log(max(len(x), 2))
        cuts = _optimal_partition(x, MIN_SEGMENT_BEATS, penalty) + [len(x)]
        for a, b in zip(cuts, cuts[1:], strict=False):
            pieces.append(
                _Piece(float(run[a]), float(run[b]), b - a, float(run[b] - run[a]))
            )
    pieces = _merge_similar(pieces)
    return [TempoSegment(p.start_sec, p.end_sec, p.bpm, p.intervals) for p in pieces]


def segment_at(segments: Sequence[TempoSegment], t: float) -> TempoSegment | None:
    """時刻 t の区間（最初の区間より前は最初、区間の間・最後より後は直前の区間）。"""
    if not segments:
        return None
    found = segments[0]
    for seg in segments:
        if seg.start_sec <= t:
            found = seg
        else:
            break
    return found


def estimate_time_signature(
    beats: Sequence[float] | np.ndarray,
    downbeats: Sequence[float] | np.ndarray,
    default: int = DEFAULT_TIME_SIGNATURE,
) -> int:
    """小節の頭どうしの間にある拍の数の最頻値（数えられなければ default）。

    拍は `clean_beats` で整えたものを数える（拍の抜けで 3 と数えるのを防ぐ）。
    同数のときは default に近い方（4/4 を優先）。
    """
    b = clean_beats(beats)
    d = _as_sorted(downbeats)
    if len(b) < 2 or len(d) < 2:
        return default
    # 小節の頭は拍の位置とほぼ同じ所にある。刻み（20ms）の差で数え落とさないよう、
    # 小節の頭の少し手前（拍の間隔の 1/4）から数える
    tol = 0.25 * float(np.median(np.diff(b)))
    counts: Counter[int] = Counter()
    for d0, d1 in zip(d, d[1:], strict=False):
        n = int(np.count_nonzero((b >= d0 - tol) & (b < d1 - tol)))
        if MIN_BEATS_PER_BAR <= n <= MAX_BEATS_PER_BAR:
            counts[n] += 1
    if not counts:
        return default
    top = max(counts.values())
    return min((n for n, c in counts.items() if c == top), key=lambda n: (abs(n - default), n))
