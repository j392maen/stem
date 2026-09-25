"""拍の手動補正の純粋関数（T10c、`stemapp.beats.edit`）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stemapp.beats.edit import (
    ALL,
    BeatEditError,
    BeatRange,
    GridState,
    apply_edit,
    grid_from_cues,
    resolve_range,
    scale_tempo,
    set_downbeat,
    set_meter,
    shift_beats,
    tap_tempo,
)
from stemapp.beats.tempo import tempo_segments


def grid(sections: list[tuple[float, float]], until: float, per_bar: int = 4,
         offset: float = 0.0) -> GridState:
    """[(開始秒, BPM), ...] の規則的な拍（FakeBeatAnalyzer と同じ作り方）。"""
    beats: list[float] = []
    downs: list[float] = []
    t = offset
    while t < until:
        bpm = [b for s, b in sections if s <= t + 1e-9][-1]
        if len(beats) % per_bar == 0:
            downs.append(round(t, 4))
        beats.append(round(t, 4))
        t += 60.0 / bpm
    return GridState(tuple(beats), tuple(downs), per_bar)


def bpms(state: GridState) -> list[float]:
    return [round(s.bpm, 1) for s in tempo_segments(state.beats)]


def bar_lengths(state: GridState) -> list[int]:
    """小節の頭どうしの間の拍の数。"""
    b = np.asarray(state.beats)
    idx = [int(np.argmin(np.abs(b - d))) for d in state.downbeats]
    return [j - i for i, j in zip(idx, idx[1:], strict=False)]


# --- 範囲 -----------------------------------------------------------------------------


def test_resolve_range_segment_all_loop() -> None:
    g = grid([(0, 120), (16, 150)], 32)
    first = resolve_range(g, "segment", 5.0)
    assert first.start == -math.inf and first.end == pytest.approx(16.0)
    last = resolve_range(g, "segment", 20.0)
    assert last.start == pytest.approx(16.0) and last.end == math.inf
    # 区間の境目ちょうどは後ろの区間
    assert resolve_range(g, "segment", 16.0).start == pytest.approx(16.0)
    assert resolve_range(g, "all", 5.0) == ALL
    loop = resolve_range(g, "loop", 0, (2.0, 6.0))
    assert (loop.start, loop.end, loop.kind) == (2.0, 6.0, "loop")
    with pytest.raises(BeatEditError, match="ループ区間"):
        resolve_range(g, "loop", 0, None)
    with pytest.raises(BeatEditError):
        resolve_range(g, "xyz", 0)
    # 拍が少なく区間が無いときは曲全体
    few = GridState((0.0, 0.5, 1.0), (0.0,), 4)
    r = resolve_range(few, "segment", 0.7)
    assert r.start == -math.inf and r.end == math.inf


# --- 1小節目をここに ------------------------------------------------------------------------


def test_set_downbeat_moves_bar_start_without_moving_beats() -> None:
    g = grid([(0, 120)], 16)
    out = set_downbeat(g, 1.04)  # 3拍目（1.0 秒）がいちばん近い
    assert out.beats == g.beats
    assert out.downbeats[:3] == (1.0, 3.0, 5.0)
    assert out.time_signature == 4
    # 1.0 秒より前も同じ拍子で区切る（0.5 秒前の拍は前の小節の4拍目）
    assert 1.0 - 2.0 not in out.downbeats and out.downbeats[0] == 1.0


def test_set_downbeat_only_in_segment() -> None:
    g = grid([(0, 120), (16, 150)], 32)
    out = set_downbeat(g, 17.2, resolve_range(g, "segment", 17.2))
    # 前の区間の小節の頭は変わらない
    before = [d for d in g.downbeats if d < 16]
    assert [d for d in out.downbeats if d < 16 - 0.01] == before
    b = np.asarray(g.beats)
    near = float(b[np.argmin(np.abs(b - 17.2))])
    assert near in out.downbeats
    assert out.beats == g.beats


def test_set_downbeat_keeps_three_four() -> None:
    g = grid([(0, 120)], 12, per_bar=3)
    out = set_downbeat(g, 0.5)
    assert set(bar_lengths(out)) == {3}
    assert out.downbeats[0] == 0.5 and out.time_signature == 3


def test_set_downbeat_empty_range() -> None:
    g = grid([(0, 120)], 4)
    with pytest.raises(BeatEditError, match="範囲に拍がありません"):
        set_downbeat(g, 10.0, BeatRange(8.0, 9.0, "loop"))


# --- ×2 / ÷2 ---------------------------------------------------------------------------


def test_double_and_half_whole_song() -> None:
    half_tempo = grid([(0, 64)], 30)
    doubled = scale_tempo(half_tempo, 2)
    assert bpms(doubled) == [128.0]
    assert len(doubled.beats) == 2 * len(half_tempo.beats) - 1
    assert set(bar_lengths(doubled)) == {4}
    assert doubled.downbeats[0] == 0.0

    double_tempo = grid([(0, 256)], 30)
    halved = scale_tempo(double_tempo, 0.5)
    assert bpms(halved) == [128.0]
    assert set(bar_lengths(halved)) == {4}
    assert halved.downbeats[0] == 0.0


def test_half_keeps_downbeat_side() -> None:
    # 小節の頭が奇数番目の拍（0.25 秒）にある: その拍の側を残す
    g = grid([(0, 240)], 16, offset=0.0)
    g = set_downbeat(g, 0.25)
    out = scale_tempo(g, 0.5)
    assert 0.25 in out.beats and 0.0 not in out.beats
    assert out.downbeats[0] == 0.25


def test_double_in_segment_only() -> None:
    g = grid([(0, 120), (16, 75)], 40)  # 後ろの区間を半分に取り違えた（本当は 150）
    rng = resolve_range(g, "segment", 30.0)
    out = scale_tempo(g, 2, rng)
    assert bpms(out) == [120.0, 150.0]
    # 前の区間の拍は変わらない
    assert [b for b in out.beats if b < 16] == [b for b in g.beats if b < 16]
    # ÷2 で戻る
    back = scale_tempo(out, 0.5, resolve_range(out, "segment", 30.0))
    assert bpms(back) == [120.0, 75.0]


def test_double_in_middle_segment_uses_closing_beat() -> None:
    g = grid([(0, 120), (10, 60), (30, 120)], 40)
    rng = resolve_range(g, "segment", 15.0)
    assert rng.start == pytest.approx(10.0) and rng.end == pytest.approx(30.0)
    out = scale_tempo(g, 2, rng)
    assert bpms(out) == [120.0]  # 3つの区間が1つにまとまる
    b = np.asarray(out.beats)
    assert np.allclose(np.diff(b), 0.5, atol=1e-3)


def test_scale_errors() -> None:
    g = GridState((1.0,), (1.0,), 4)
    with pytest.raises(BeatEditError):
        scale_tempo(g, 2)
    with pytest.raises(BeatEditError):
        scale_tempo(grid([(0, 120)], 4), 3)


# --- 拍子 --------------------------------------------------------------------------------


@pytest.mark.parametrize("per_bar", [3, 4, 5, 6, 7])
def test_set_meter_whole(per_bar: int) -> None:
    g = grid([(0, 120)], 30)
    out = set_meter(g, per_bar)
    assert out.time_signature == per_bar
    assert set(bar_lengths(out)) == {per_bar}
    assert out.downbeats[0] == 0.0 and out.beats == g.beats


def test_set_meter_in_segment() -> None:
    g = grid([(0, 120), (30, 150)], 40)
    out = set_meter(g, 3, resolve_range(g, "segment", 35.0))
    assert out.time_signature == 4  # 曲全体では 4 拍子のまま（前の区間の方が長い）
    b = np.asarray(out.beats)
    later = [d for d in out.downbeats if d >= 30]
    idx = [int(np.argmin(np.abs(b - d))) for d in later]
    assert set(np.diff(idx)) == {3}
    assert [d for d in out.downbeats if d < 30] == [d for d in g.downbeats if d < 30]
    with pytest.raises(BeatEditError):
        set_meter(g, 13)


# --- ずらす -----------------------------------------------------------------------------


def test_shift_whole_and_segment() -> None:
    g = grid([(0.0, 120)], 10, offset=0.1)
    out = shift_beats(g, 0.01)
    assert out.beats[0] == pytest.approx(0.11) and out.downbeats[0] == pytest.approx(0.11)
    assert len(out.beats) == len(g.beats)
    out = shift_beats(g, -0.001)
    assert out.beats[1] == pytest.approx(0.599)

    g2 = grid([(0, 120), (16, 150)], 32)
    rng = resolve_range(g2, "segment", 5.0)
    out2 = shift_beats(g2, 0.01, rng)
    assert out2.beats[0] == pytest.approx(0.01)
    # 境目の拍（次の区間の最初の拍）は動かさない
    assert 16.0 in out2.beats


def test_shift_drops_negative_and_refuses_crossing() -> None:
    g = grid([(0, 120)], 4)
    out = shift_beats(g, -0.01)
    assert out.beats[0] == pytest.approx(0.49)  # 0 秒の拍は 0 より前に出たので消える
    loop = BeatRange(1.0, 2.0, "loop")
    with pytest.raises(BeatEditError, match="後ろの拍"):
        shift_beats(g, 0.48, loop)
    with pytest.raises(BeatEditError, match="前の拍"):
        shift_beats(g, -0.48, loop)
    with pytest.raises(BeatEditError):
        shift_beats(g, 5.0)


# --- タップ ------------------------------------------------------------------------------


def test_tap_replaces_range_with_constant_tempo() -> None:
    g = grid([(0, 100)], 30)  # 自動の結果が 100 BPM（本当は 128）
    period = 60 / 128
    rng = np.random.default_rng(1)
    taps = [4.0 + k * period + rng.normal(0, 0.008) for k in range(8)]
    out = tap_tempo(g, taps)
    assert bpms(out) == [pytest.approx(128.0, abs=0.6)]
    b = np.asarray(out.beats)
    assert b[0] >= 0 and b[-1] <= 30.0
    # たたいた位置に拍がある
    assert np.min(np.abs(b - 4.0)) < 0.02
    assert set(bar_lengths(out)) == {4}


def test_tap_in_segment_keeps_other_segments() -> None:
    g = grid([(0, 120), (16, 100)], 40)
    period = 60 / 150
    taps = [20.0 + k * period for k in range(6)]
    out = tap_tempo(g, taps, resolve_range(g, "segment", 21.0))
    assert bpms(out) == [120.0, pytest.approx(150.0, abs=0.3)]
    assert [b for b in out.beats if b < 16] == [b for b in g.beats if b < 16]


def test_tap_in_bounded_segment_keeps_boundary_beat() -> None:
    g = grid([(0, 120), (10, 90), (20, 120)], 30)
    rng = resolve_range(g, "segment", 12.0)
    taps = [11.0 + k * 0.5 for k in range(5)]
    out = tap_tempo(g, taps, rng)
    assert 20.0 in out.beats  # 範囲の終わりの拍（次の区間の最初の拍）は残る
    assert bpms(out) == [120.0]
    d = np.diff(np.asarray(out.beats))
    assert d.min() > 0.2  # 二重の拍が無い


def test_tap_errors() -> None:
    g = grid([(0, 120)], 10)
    with pytest.raises(BeatEditError, match="4 回以上"):
        tap_tempo(g, [1.0, 1.5, 2.0])
    with pytest.raises(BeatEditError, match="そろっていません"):
        tap_tempo(g, [1.0, 1.5, 2.0, 3.5])
    with pytest.raises(BeatEditError, match="範囲外"):
        tap_tempo(g, [1.0, 1.1, 1.2, 1.3])


# --- キュー2点から ---------------------------------------------------------------------------


def test_grid_from_cues() -> None:
    g = grid([(0, 100)], 40)
    out = grid_from_cues(g, 8.0, 8.0 + 8 * 4 * 60 / 128, 8)
    b = np.asarray(out.beats)
    inside = b[(b >= 8.0 - 1e-6) & (b <= 23.0 + 1e-6)]
    assert len(inside) == 33
    assert np.allclose(np.diff(inside), 60 / 128, atol=1e-3)
    assert 8.0 in out.downbeats and 23.0 in out.downbeats
    assert np.diff(b).min() > 0.2
    # 外の拍は残る
    assert [x for x in out.beats if x < 7.7] == [x for x in g.beats if x < 7.7]


def test_grid_from_cues_errors() -> None:
    g = grid([(0, 120)], 10)
    with pytest.raises(BeatEditError, match="後ろ"):
        grid_from_cues(g, 5.0, 4.0, 2)
    with pytest.raises(BeatEditError, match="範囲外"):
        grid_from_cues(g, 0.0, 8.0, 100)
    with pytest.raises(BeatEditError, match="小節数"):
        grid_from_cues(g, 0.0, 8.0, 0)


# --- まとめ ---------------------------------------------------------------------------------


def test_apply_edit_dispatch() -> None:
    g = grid([(0, 64)], 20)
    assert bpms(apply_edit(g, "double", {"range": "all"})) == [128.0]
    assert bpms(apply_edit(g, "half", {"range": "segment", "position": 3})) == [32.0]
    assert apply_edit(g, "meter", {"range": "all", "beats_per_bar": 3}).time_signature == 3
    assert apply_edit(g, "downbeat", {"position": 0.95}).downbeats[0] == pytest.approx(0.9375)
    assert apply_edit(g, "shift", {"range": "all", "delta_sec": 0.01}).beats[0] == 0.01
    out = apply_edit(g, "double", {"range": "loop", "loop_start": 0.0, "loop_end": 5.0})
    assert len(out.beats) > len(g.beats)
    taps = [1.0 + k * 0.5 for k in range(4)]
    assert bpms(apply_edit(g, "tap", {"range": "all", "taps": taps})) == [120.0]
    out = apply_edit(g, "cues", {"cue_start": 0.0, "cue_end": 8.0, "bars": 4})
    assert bpms(out)[0] == 120.0
    with pytest.raises(BeatEditError):
        apply_edit(g, "nope", {})


# 実データ「水槽」（T10 の報告）の 110〜140 秒の拍。123〜134 秒で beat_this が 3/4 の間隔
# （約 100 BPM）で拍を取り、区間の吸収で 165.6 BPM の区間が出た
SUISOU_110_140 = [
    110.3, 110.74, 111.2, 111.64, 112.08, 112.52, 112.96, 113.4, 113.86, 114.3, 114.72, 115.18,
    115.64, 116.08, 116.52, 116.96, 117.42, 117.86, 118.3, 118.74, 119.2, 119.62, 120.08,
    120.32, 120.52, 120.98, 121.42, 121.86, 122.28, 122.74, 123.06, 123.2, 123.34, 123.94,
    124.1, 124.52, 125.12, 125.7, 126.32, 126.9, 127.48, 128.08, 128.68, 129.26, 129.86,
    130.46, 130.76, 131.04, 131.64, 131.94, 132.54, 132.82, 133.12, 133.42, 133.72, 134.32,
    134.9, 135.5, 135.64, 136.04, 136.5, 136.96, 137.4, 137.82, 138.26, 138.72, 139.16, 139.6,
]


def _suisou_like() -> GridState:
    period = 60 / 134.8
    head = [round(110.3 - k * period, 4) for k in range(40, 0, -1)]
    tail = [round(139.6 + k * period, 4) for k in range(1, 40)]
    beats = tuple(head + SUISOU_110_140 + tail)
    return GridState(beats, beats[::4], 4)


def test_suisou_wrong_segment_is_fixed_by_cues() -> None:
    g = _suisou_like()
    wrong = [s for s in tempo_segments(g.beats) if abs(s.bpm - 134.8) > 3]
    assert wrong, "乱れた区間が再現できていません"
    # 乱れた所をはさむ小節の頭（120.08 秒と 137.82 秒）にキューを打ち、間を 10 小節とする
    out = grid_from_cues(g, 120.08, 137.82, 10)
    assert [round(s.bpm) for s in tempo_segments(out.beats)] == [135]


def test_suisou_wrong_segment_is_fixed_by_tap_in_loop() -> None:
    g = _suisou_like()
    period = 60 / 134.8
    taps = [120.98 + k * period for k in range(8)]
    out = tap_tempo(g, taps, BeatRange(119.9, 138.0, "loop"))
    # 165.6 BPM の区間は消える。たたいた拍と後ろの自動の拍の位相の差（数十 ms）で、ループの
    # 終わりに短い区間が残ることはあるが、BPM の差は数 % 以内（ずらす・キュー2点からで詰められる）
    segs = tempo_segments(out.beats)
    assert all(abs(s.bpm / 134.8 - 1) < 0.03 for s in segs), segs
    assert max(s.end_sec - s.start_sec for s in segs if abs(s.bpm - 134.8) > 1) < 5
