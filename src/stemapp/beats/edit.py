"""拍の手動補正の計算（T10c）。DB を触らない純粋関数（numpy のみ）。

拍の格子は「拍の時刻の列」と「各拍が小節の頭かどうか（flags）」で扱う。小節の頭（downbeats）は
拍の一部（自動の結果の小節の頭は、いちばん近い拍に合わせる）。

範囲（`BeatRange`）: 補正をかける時間の範囲 [start, end)。
- 区間（segment）: 再生位置を含むテンポの区間（`tempo_segments`）。区間の始まりと終わりは
  拍の時刻で、終わりの拍は次の区間の最初の拍でもあるので、拍ごとの操作（小節の付け直し・ずらす）は
  含めず、拍の間隔の操作（×2）は終わりの拍までの間隔を含める。最初の区間は曲の頭から、
  最後の区間は曲の終わりまでを含める（区間に入らない端の数拍も直せるように）。
- 曲全体（all）、ループ区間（loop）。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from stemapp.beats.tempo import estimate_time_signature, segment_at, tempo_segments

# 自動の小節の頭を拍に合わせるときの許容（秒）。beat_this の刻みは 20ms なので十分に広い
DOWNBEAT_MATCH_SEC = 0.07
# 範囲の端と拍の時刻を同じとみなす幅（秒）。区間の端は拍の時刻を丸めた値（ms 単位）なので
EDGE_EPS = 0.002
# 拍子（1小節の拍数）の選択肢
TIME_SIGNATURES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
# 作る拍のテンポの範囲（タップ・キュー2点から）。これを外れるのは操作の誤りとみなす
MIN_BPM = 30.0
MAX_BPM = 300.0
# タップの最少回数と、間隔のばらつきの許容（中央値からの比）
MIN_TAPS = 4
TAP_TOLERANCE = 0.3
# ずらす量の上限（秒）。1回の操作でこれより大きくは動かさない
MAX_SHIFT_SEC = 1.0
# 動かした拍が隣の（範囲の外の）拍にこれより近づくなら動かさない（秒）
MIN_GAP_SEC = 0.03
# キュー2点からの小節数の上限
MAX_BARS = 1024
# 補正した結果の拍の数の上限（300 BPM で約 67 分）。これを超える操作は誤りとみなす
MAX_BEATS = 20000

RANGE_KINDS = ("segment", "all", "loop")
OPS = ("downbeat", "double", "half", "meter", "shift", "tap", "cues")


class BeatEditError(ValueError):
    """補正できない（メッセージは日本語。API では 400）。"""


@dataclass(frozen=True)
class GridState:
    beats: tuple[float, ...]
    downbeats: tuple[float, ...]
    time_signature: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "beats": list(self.beats),
            "downbeats": list(self.downbeats),
            "time_signature": self.time_signature,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GridState:
        return cls(
            tuple(float(x) for x in d.get("beats") or []),
            tuple(float(x) for x in d.get("downbeats") or []),
            int(d.get("time_signature") or 4),
        )


@dataclass(frozen=True)
class BeatRange:
    start: float  # -inf なら曲の頭から
    end: float  # inf なら曲の終わりまで
    kind: str = "all"

    @property
    def bounded_end(self) -> bool:
        return math.isfinite(self.end)


ALL = BeatRange(-math.inf, math.inf, "all")


# --- 内部の表現 ------------------------------------------------------------------------


def _to_arrays(state: GridState) -> tuple[np.ndarray, np.ndarray]:
    """拍の列と、各拍が小節の頭かどうか。"""
    beats = np.unique(np.asarray(state.beats, dtype=np.float64))
    flags = np.zeros(len(beats), dtype=bool)
    if len(beats):
        for d in state.downbeats:
            i = int(np.searchsorted(beats, d))
            cands = [j for j in (i - 1, i) if 0 <= j < len(beats)]
            j = min(cands, key=lambda k: abs(beats[k] - d))
            if abs(beats[j] - d) <= DOWNBEAT_MATCH_SEC:
                flags[j] = True
    return beats, flags


def _to_state(beats: np.ndarray, flags: np.ndarray, default_sig: int) -> GridState:
    order = np.argsort(beats, kind="stable")
    beats = beats[order]
    flags = flags[order]
    keep = np.concatenate([[True], np.diff(beats) > 1e-4]) if len(beats) else np.zeros(0, bool)
    beats = beats[keep]
    flags = flags[keep]
    if len(beats) > MAX_BEATS:
        raise BeatEditError(f"拍の数が多すぎます（{len(beats)}。上限 {MAX_BEATS}）。")
    b = tuple(round(float(x), 4) for x in beats)
    d = tuple(round(float(x), 4) for x, f in zip(beats, flags, strict=True) if f)
    sig = estimate_time_signature(b, d, default=default_sig) if len(d) >= 2 else default_sig
    return GridState(b, d, int(sig))


def _index_range(beats: np.ndarray, rng: BeatRange) -> tuple[int, int]:
    """範囲に入る拍の番号 [lo, hi)（拍ごとの操作の対象。終わりの端の拍は含めない）。"""
    lo = 0 if not math.isfinite(rng.start) else int(np.searchsorted(beats, rng.start - EDGE_EPS))
    hi = len(beats) if not rng.bounded_end else int(np.searchsorted(beats, rng.end - EDGE_EPS))
    return lo, max(lo, hi)


def _closing(beats: np.ndarray, rng: BeatRange, hi: int) -> int:
    """拍の間隔の操作で使う最後の拍の番号（範囲の終わりの端に拍があればそれ、無ければ hi-1）。"""
    if hi < len(beats) and rng.bounded_end and beats[hi] <= rng.end + EDGE_EPS:
        return hi
    return hi - 1


def _local_meter(beats: np.ndarray, flags: np.ndarray, lo: int, hi: int, default: int) -> int:
    """範囲の中の小節の拍数の最頻値（数えられなければ default）。"""
    idx = [i for i in range(lo, hi) if flags[i]]
    counts = Counter(b - a for a, b in zip(idx, idx[1:], strict=False) if 2 <= b - a <= 12)
    if not counts:
        return default
    top = max(counts.values())
    return min((n for n, c in counts.items() if c == top), key=lambda n: (abs(n - default), n))


def _rebar(flags: np.ndarray, lo: int, hi: int, anchor: int, per_bar: int) -> None:
    """[lo, hi) の小節の頭を、anchor の拍を1拍目として per_bar 拍ごとに付け直す。"""
    for i in range(lo, hi):
        flags[i] = (i - anchor) % per_bar == 0


def _first_flag(flags: np.ndarray, lo: int, hi: int) -> int:
    for i in range(lo, hi):
        if flags[i]:
            return i
    return lo


def _shared_start(beats: np.ndarray, rng: BeatRange, lo: int) -> bool:
    """範囲が区間で、その始まりの拍が前の区間と共有されている（前の区間の終わりの拍でもある）か。

    共有の拍は、ずらす・タップ・÷2 で動かさない・消さない（前の区間を変えないため）。
    """
    return (
        rng.kind == "segment"
        and math.isfinite(rng.start)
        and 0 < lo < len(beats)
        and abs(float(beats[lo]) - rng.start) <= EDGE_EPS
    )


def _need_beats(beats: np.ndarray, lo: int, hi: int, n: int = 1) -> None:
    if hi - lo < n:
        raise BeatEditError("範囲に拍がありません。")


# --- 範囲 ----------------------------------------------------------------------------


def resolve_range(
    state: GridState,
    kind: str,
    position: float = 0.0,
    loop: tuple[float, float] | None = None,
) -> BeatRange:
    """範囲の種類（segment / all / loop）と再生位置から、時間の範囲を求める。"""
    if kind == "all":
        return ALL
    if kind == "loop":
        if loop is None or not (loop[1] > loop[0]):
            raise BeatEditError("ループ区間が設定されていません。")
        return BeatRange(float(loop[0]), float(loop[1]), "loop")
    if kind != "segment":
        raise BeatEditError(f"範囲の指定が正しくありません: {kind}")
    segs = tempo_segments(state.beats)
    seg = segment_at(segs, position)
    if seg is None:
        return BeatRange(-math.inf, math.inf, "segment")
    i = segs.index(seg)
    start = -math.inf if i == 0 else seg.start_sec
    end = math.inf if i == len(segs) - 1 else seg.end_sec
    return BeatRange(start, end, "segment")


# --- 各操作 --------------------------------------------------------------------------


def set_downbeat(state: GridState, position: float, rng: BeatRange = ALL) -> GridState:
    """1小節目をここに: 再生位置にいちばん近い（範囲の中の）拍を小節の頭にし、範囲の小節を付け直す。

    拍の時刻は動かさない。1小節の拍数は範囲の中の今の小節の拍数（最頻値）。
    """
    beats, flags = _to_arrays(state)
    lo, hi = _index_range(beats, rng)
    _need_beats(beats, lo, hi)
    anchor = lo + int(np.argmin(np.abs(beats[lo:hi] - position)))
    per_bar = _local_meter(beats, flags, lo, hi, state.time_signature)
    _rebar(flags, lo, hi, anchor, per_bar)
    return _to_state(beats, flags, state.time_signature)


def scale_tempo(state: GridState, factor: int | float, rng: BeatRange = ALL) -> GridState:
    """×2（拍の間に拍を足す）／÷2（1つおきに間引く）。小節は範囲の最初の小節の頭から付け直す。

    ÷2 は小節の頭の拍を残す側で間引く。範囲が区間で、始まりの拍が前の区間と共有されている
    ときは、その拍を残し、その拍を基準に1つおきに間引く（直後に半分の間隔を残さない）。
    """
    beats, flags = _to_arrays(state)
    lo, hi = _index_range(beats, rng)
    per_bar = _local_meter(beats, flags, lo, hi, state.time_signature)
    if factor == 2:
        last = _closing(beats, rng, hi)
        _need_beats(beats, lo, last + 1, 2)
        mids = (beats[lo:last] + beats[lo + 1 : last + 1]) / 2.0
        first_db = beats[_first_flag(flags, lo, hi)]
        new_beats = np.concatenate([beats, mids])
        new_flags = np.concatenate([flags, np.zeros(len(mids), dtype=bool)])
        order = np.argsort(new_beats, kind="stable")
        new_beats, new_flags = new_beats[order], new_flags[order]
        lo2, hi2 = _index_range(new_beats, rng)
        anchor = int(np.searchsorted(new_beats, first_db))
        _rebar(new_flags, lo2, hi2, anchor, per_bar)
        return _to_state(new_beats, new_flags, state.time_signature)
    if factor == 0.5:
        _need_beats(beats, lo, hi, 2)
        base = lo if _shared_start(beats, rng, lo) else _first_flag(flags, lo, hi)
        keep = np.ones(len(beats), dtype=bool)
        for i in range(lo, hi):
            if (i - base) % 2 != 0:
                keep[i] = False
        # 小節の頭は、残した拍のうち最初の小節の頭（無ければ範囲の最初の拍）から付け直す
        kept_flags = [i for i in range(lo, hi) if keep[i] and flags[i]]
        anchor = kept_flags[0] if kept_flags else base
        new_beats, new_flags = beats[keep], flags[keep]
        lo2, hi2 = _index_range(new_beats, rng)
        anchor2 = int(np.searchsorted(new_beats, beats[anchor]))
        _rebar(new_flags, lo2, hi2, anchor2, per_bar)
        return _to_state(new_beats, new_flags, state.time_signature)
    raise BeatEditError("倍率は 2 か 0.5 にしてください。")


def set_meter(state: GridState, per_bar: int, rng: BeatRange = ALL) -> GridState:
    """拍子: 範囲の小節を per_bar 拍ごとに付け直す（範囲の最初の小節の頭から）。"""
    if per_bar not in TIME_SIGNATURES:
        raise BeatEditError("拍子は 2〜12 拍から選んでください。")
    beats, flags = _to_arrays(state)
    lo, hi = _index_range(beats, rng)
    _need_beats(beats, lo, hi)
    _rebar(flags, lo, hi, _first_flag(flags, lo, hi), per_bar)
    default = per_bar if rng.kind == "all" else state.time_signature
    result = _to_state(beats, flags, default)
    if rng.kind == "all":
        return GridState(result.beats, result.downbeats, per_bar)
    return result


def shift_beats(state: GridState, delta_sec: float, rng: BeatRange = ALL) -> GridState:
    """ずらす: 範囲の拍（と小節の頭）を delta_sec だけ前後に動かす。

    範囲の外の拍を越える（MIN_GAP_SEC より近づく）ときは動かさない。0 秒より前に出た拍は消す。
    範囲が区間のとき、前の区間と共有する始まりの拍は動かさない。
    """
    if not math.isfinite(delta_sec) or abs(delta_sec) > MAX_SHIFT_SEC:
        raise BeatEditError("ずらす量が大きすぎます。")
    beats, flags = _to_arrays(state)
    lo, hi = _index_range(beats, rng)
    if _shared_start(beats, rng, lo):
        lo += 1
    _need_beats(beats, lo, hi)
    moved = beats.copy()
    moved[lo:hi] += delta_sec
    if lo > 0 and moved[lo] - beats[lo - 1] < MIN_GAP_SEC:
        raise BeatEditError("前の拍を越えるため、これ以上ずらせません。")
    if hi < len(beats) and beats[hi] - moved[hi - 1] < MIN_GAP_SEC:
        raise BeatEditError("後ろの拍を越えるため、これ以上ずらせません。")
    keep = moved >= 0.0
    return _to_state(moved[keep], flags[keep], state.time_signature)


def _regular_replace(
    beats: np.ndarray,
    flags: np.ndarray,
    remove_from: float,
    remove_to: float,
    new_beats: np.ndarray,
    downbeat_at: float,
    per_bar: int,
    default_sig: int,
) -> GridState:
    """[remove_from, remove_to) の拍を new_beats（等間隔）で置き換え、downbeat_at にいちばん近い
    新しい拍を1拍目として per_bar 拍ごとに小節の頭を付ける。"""
    if len(new_beats) < 2:
        raise BeatEditError("範囲が短すぎて拍を作れません。")
    period = float(np.median(np.diff(new_beats)))
    outside = (beats < remove_from - EDGE_EPS) | (beats >= remove_to - EDGE_EPS)
    old_b, old_f = beats[outside], flags[outside]
    # 新しい拍に近すぎる（半拍未満の）外の拍は消す（二重の拍を作らない）
    near = np.zeros(len(old_b), dtype=bool)
    for i, t in enumerate(old_b):
        j = int(np.searchsorted(new_beats, t))
        near[i] = any(
            abs(new_beats[k] - t) < 0.5 * period for k in (j - 1, j) if 0 <= k < len(new_beats)
        )
    old_b, old_f = old_b[~near], old_f[~near]
    anchor = int(np.argmin(np.abs(new_beats - downbeat_at)))
    new_f = np.array([(k - anchor) % per_bar == 0 for k in range(len(new_beats))], dtype=bool)
    return _to_state(
        np.concatenate([old_b, new_beats]), np.concatenate([old_f, new_f]), default_sig
    )


def _check_bpm(period: float) -> None:
    bpm = 60.0 / period if period > 0 else math.inf
    if not (MIN_BPM <= bpm <= MAX_BPM):
        raise BeatEditError(f"テンポが範囲外です（{bpm:.1f} BPM。{MIN_BPM:.0f}〜{MAX_BPM:.0f}）。")


def tap_tempo(state: GridState, taps: Sequence[float], rng: BeatRange = ALL) -> GridState:
    """タップ: たたいた時刻（曲の時刻。出力の遅延は差し引き済み）から一定テンポの拍を作り、
    範囲の拍を置き換える。

    たたいた時刻に「番号 × 間隔 ＋ 位置」を最小二乗で当てはめる（1回ごとのずれをならす）。
    小節の頭は、たたき始めにいちばん近い今の小節の頭に合わせる（無ければ最初にたたいた拍）。
    1小節の拍数は曲の拍子（state.time_signature）。範囲の中の小節から推定すると、乱れた所では
    2拍などになるため。範囲が区間のとき、前の区間と共有する始まりの拍は残す。
    """
    t = np.sort(np.asarray([float(x) for x in taps], dtype=np.float64))
    if len(t) < MIN_TAPS:
        raise BeatEditError(f"{MIN_TAPS} 回以上たたいてください。")
    d = np.diff(t)
    med = float(np.median(d))
    if med <= 0 or np.any(np.abs(d / med - 1.0) > TAP_TOLERANCE):
        raise BeatEditError("たたいた間隔がそろっていません。もう一度たたいてください。")
    k = np.arange(len(t), dtype=np.float64)
    period, t0 = (float(v) for v in np.polyfit(k, t, 1))
    _check_bpm(period)
    beats, flags = _to_arrays(state)
    lo, hi = _index_range(beats, rng)
    per_bar = state.time_signature
    shared = _shared_start(beats, rng, lo)
    # 置き換える範囲（範囲が曲の端までなら、今の拍とたたいた所の広い方まで）
    first = min(float(beats[0]) if len(beats) else t[0], t[0])
    last = max(float(beats[-1]) if len(beats) else t[-1], t[-1])
    start = max(rng.start, 0.0) if math.isfinite(rng.start) else max(0.0, first)
    end = rng.end if rng.bounded_end else last + 0.5 * period
    k0 = math.ceil((start - EDGE_EPS - t0) / period)
    k1 = math.floor((end - EDGE_EPS - t0) / period)
    new_beats = t0 + period * np.arange(k0, k1 + 1, dtype=np.float64)
    new_beats = new_beats[new_beats >= 0.0]
    remove_from = start
    if shared:
        # 共有の拍は残し、それに半拍より近い拍は作らない
        remove_from = float(beats[lo]) + 2 * EDGE_EPS
        new_beats = new_beats[new_beats > float(beats[lo]) + 0.5 * period]
    if rng.bounded_end:
        # 範囲の後ろの最初の拍（区間なら次の区間の最初の拍）は残す。それに半拍より近い拍は作らない
        after = beats[beats >= end - EDGE_EPS]
        if len(after):
            new_beats = new_beats[new_beats < after[0] - 0.5 * period]
    db_idx = [i for i in range(lo, hi) if flags[i]]
    downbeat_at = float(t[0])
    if db_idx:
        downbeat_at = float(min((beats[i] for i in db_idx), key=lambda x: abs(x - t[0])))
    return _regular_replace(
        beats, flags, remove_from, end, new_beats, downbeat_at, per_bar, state.time_signature
    )


def grid_from_cues(
    state: GridState, start: float, end: float, bars: int, per_bar: int | None = None
) -> GridState:
    """キュー2点から: start と end の間を bars 小節の一定テンポの拍で置き換える。

    start と end は小節の頭になる。1小節の拍数は per_bar（省略時は曲の拍子 state.time_signature）。
    その間の今の小節から推定しないのは、直したい所は小節の頭も乱れていて、2拍などと数えるため
    （実データ「水槽」で 67.65 BPM になった）。
    """
    if not (end > start >= 0.0):
        raise BeatEditError("2つ目のキューは1つ目より後ろにしてください。")
    if not (1 <= bars <= MAX_BARS):
        raise BeatEditError(f"小節数は 1〜{MAX_BARS} にしてください。")
    beats, flags = _to_arrays(state)
    if per_bar is None:
        per_bar = state.time_signature
    if per_bar not in TIME_SIGNATURES:
        raise BeatEditError("拍子は 2〜12 拍から選んでください。")
    n = bars * per_bar
    period = (end - start) / n
    _check_bpm(period)
    new_beats = start + period * np.arange(n + 1, dtype=np.float64)
    new_beats[-1] = end
    return _regular_replace(
        beats, flags, start - 0.5 * period, end + 0.5 * period, new_beats, start, per_bar,
        state.time_signature,
    )


# --- まとめ（API から呼ぶ） --------------------------------------------------------------


def apply_edit(state: GridState, op: str, params: Mapping[str, Any]) -> GridState:
    """操作の種類と引数から、補正後の拍を求める。

    params: position（再生位置）, range（segment/all/loop）, loop_start/loop_end,
    操作ごとの引数（factor, beats_per_bar, delta_sec, taps, cue_start/cue_end/bars）。
    """
    if op not in OPS:
        raise BeatEditError(f"操作の種類が正しくありません: {op}")
    if op == "cues":
        return grid_from_cues(
            state, float(params["cue_start"]), float(params["cue_end"]), int(params["bars"]),
            int(params["beats_per_bar"]) if params.get("beats_per_bar") else None,
        )
    loop = None
    if params.get("loop_start") is not None and params.get("loop_end") is not None:
        loop = (float(params["loop_start"]), float(params["loop_end"]))
    position = float(params.get("position") or 0.0)
    rng = resolve_range(state, str(params.get("range") or "segment"), position, loop)
    if op == "downbeat":
        return set_downbeat(state, position, rng)
    if op == "double":
        return scale_tempo(state, 2, rng)
    if op == "half":
        return scale_tempo(state, 0.5, rng)
    if op == "meter":
        return set_meter(state, int(params["beats_per_bar"]), rng)
    if op == "shift":
        return shift_beats(state, float(params["delta_sec"]), rng)
    return tap_tempo(state, [float(x) for x in params.get("taps") or []], rng)
