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




# 実データ「水槽」（track 4、T10 の報告）の 100〜160 秒の自動の拍と小節の頭（DB から読み取りだけで
# 取り出した値）。123〜134 秒で beat_this が 3/4 の間隔（約 100 BPM）で拍を取り、区間の吸収で
# 165.6 BPM の区間が出た。小節の頭もこの辺りで乱れている（2拍・1拍の小節がある）
SUISOU_BEATS_100_160 = [
    100.08, 100.52, 100.98, 101.42, 101.86, 102.3, 102.76, 103.2, 103.64, 104.08, 104.52, 104.96,
    105.4, 105.84, 106.28, 106.74, 107.2, 107.64, 108.08, 108.52, 108.96, 109.42, 109.86, 110.3,
    110.74, 111.2, 111.64, 112.08, 112.52, 112.96, 113.4, 113.86, 114.3, 114.72, 115.18, 115.64,
    116.08, 116.52, 116.96, 117.42, 117.86, 118.3, 118.74, 119.2, 119.62, 120.08, 120.32, 120.52,
    120.98, 121.42, 121.86, 122.28, 122.74, 123.06, 123.2, 123.34, 123.94, 124.1, 124.52, 125.12,
    125.7, 126.32, 126.9, 127.48, 128.08, 128.68, 129.26, 129.86, 130.46, 130.76, 131.04, 131.64,
    131.94, 132.54, 132.82, 133.12, 133.42, 133.72, 134.32, 134.9, 135.5, 135.64, 136.04, 136.5,
    136.96, 137.4, 137.82, 138.26, 138.72, 139.16, 139.6, 140.02, 140.48, 140.92, 141.38, 141.82,
    142.28, 142.74, 143.18, 143.64, 144.08, 144.52, 144.96, 145.42, 145.86, 146.3, 146.74, 147.2,
    147.64, 148.1, 148.54, 148.96, 149.42, 149.86, 150.3, 150.76, 151.18, 151.64, 152.08, 152.52,
    152.96, 153.42, 153.86, 154.3, 154.74, 155.2, 155.64, 156.08, 156.52, 156.98, 157.42, 157.86,
    158.3, 158.74, 159.2, 159.64,
]
SUISOU_DOWNBEATS_100_160 = [
    101.42, 103.2, 104.96, 106.74, 108.52, 110.3, 112.08, 113.86, 115.64, 117.42, 119.2, 120.98,
    121.86, 122.74, 124.52, 126.32, 127.48, 128.08, 129.26, 129.86, 130.76, 131.64, 131.94,
    133.42, 134.32, 136.96, 138.72, 140.48, 142.28, 144.08, 145.86, 147.64, 149.42, 151.18,
    152.96, 154.74, 156.52, 158.3,
]


def _suisou() -> GridState:
    """実データの 100〜160 秒に、前後 30 秒ほどの規則的な拍（134.8 BPM・4/4）を足したもの。"""
    period = 60 / 134.8
    head = [round(100.08 - k * period, 4) for k in range(68, 0, -1)]
    tail = [round(159.64 + k * period, 4) for k in range(1, 68)]
    beats = tuple(head + SUISOU_BEATS_100_160 + tail)
    downs = tuple(head[1::4] + SUISOU_DOWNBEATS_100_160 + tail[3::4])
    return GridState(beats, downs, 4)


def test_suisou_wrong_segment_is_fixed_by_cues() -> None:
    g = _suisou()
    wrong = [s for s in tempo_segments(g.beats) if abs(s.bpm - 134.8) > 3]
    assert any(abs(s.bpm - 165.6) < 0.5 for s in wrong), "乱れた区間が再現できていません"
    # 乱れた所の小節の頭の数え方（1小節の拍数）は 2 拍が多い。そこから推定すると 67.65 BPM に
    # なった（レビューの指摘）。省略時は曲の拍子（4）を使う
    out = grid_from_cues(g, 119.2, 138.72, 11)
    b = np.asarray(out.beats)
    inside = b[(b >= 119.2 - 1e-6) & (b <= 138.72 + 1e-6)]
    assert len(inside) == 11 * 4 + 1
    assert 60 / float(np.median(np.diff(inside))) == pytest.approx(135.2, abs=0.1)
    # 165.6 BPM の区間は消える（キューの外の実データの揺れで 1〜2% 違う短い区間は残りうる）
    segs = tempo_segments(out.beats)
    assert all(abs(s.bpm / 134.8 - 1) < 0.02 for s in segs), segs
    # 拍子を明示しても同じ。3 拍子を指定すれば 3 拍子で作る
    assert grid_from_cues(g, 119.2, 138.72, 11, 4) == out
    three = grid_from_cues(g, 119.2, 138.72, 11, 3)
    b3 = np.asarray(three.beats)
    assert len(b3[(b3 >= 119.2 - 1e-6) & (b3 <= 138.72 + 1e-6)]) == 11 * 3 + 1


def test_suisou_wrong_segment_is_fixed_by_tap_in_loop() -> None:
    g = _suisou()
    period = 60 / 134.8
    taps = [120.98 + k * period for k in range(8)]
    out = tap_tempo(g, taps, BeatRange(119.9, 138.0, "loop"))
    # 165.6 BPM の区間は消える。たたいた拍と後ろの自動の拍の位相の差（数十 ms）で、ループの
    # 終わりに短い区間が残ることはあるが、BPM の差は数 % 以内（ずらす・キュー2点からで詰められる）
    segs = tempo_segments(out.beats)
    assert all(abs(s.bpm / 134.8 - 1) < 0.03 for s in segs), segs
    # 小節は曲の拍子（4 拍）で付く（乱れた所の 2 拍ではない）
    b = np.asarray(out.beats)
    idx = [int(np.argmin(np.abs(b - d))) for d in out.downbeats if 121 <= d <= 137]
    assert set(np.diff(idx)) == {4}


# --- 区間の境目の拍（前の区間と共有する拍）------------------------------------------------


def _three_sections() -> GridState:
    # 0〜10 秒 120 BPM、10〜30 秒 150 BPM を倍に取り違えた 300 BPM、30〜40 秒 120 BPM
    return grid([(0, 120), (10, 300), (30, 120)], 40)


def test_half_in_segment_keeps_shared_start_and_parity() -> None:
    g = _three_sections()
    # 小節の頭を奇数番目の拍にずらしておく（共有の拍と偶奇が合わない）
    rng = resolve_range(g, "segment", 20.0)
    g = set_downbeat(g, 10.2, rng)
    rng = resolve_range(g, "segment", 20.0)
    assert rng.start == pytest.approx(10.0)
    out = scale_tempo(g, 0.5, rng)
    b = np.asarray(out.beats)
    mid = b[(b >= 10.0 - 1e-6) & (b <= 30.0 + 1e-6)]
    # 共有の拍（10.0）は残り、直後に半分の間隔（0.2 秒）が残らない
    assert mid[0] == pytest.approx(10.0)
    assert np.allclose(np.diff(mid), 0.4, atol=1e-3)
    assert bpms(out) == [120.0, 150.0, 120.0]


def test_shift_in_segment_keeps_shared_start() -> None:
    g = grid([(0, 120), (10, 150), (30, 120)], 40)
    rng = resolve_range(g, "segment", 20.0)
    out = shift_beats(g, 0.01, rng)
    assert 10.0 in out.beats and 10.01 not in out.beats
    assert 10.41 in out.beats and 30.0 in out.beats  # 終わりの境目も動かない
    # 動かすと前の拍（共有の拍）を越えるときは断る
    with pytest.raises(BeatEditError, match="前の拍"):
        shift_beats(g, -0.39, rng)


def test_tap_in_segment_keeps_shared_start() -> None:
    g = grid([(0, 120), (10, 100), (30, 120)], 40)
    rng = resolve_range(g, "segment", 20.0)
    taps = [12.03 + k * 0.4 for k in range(6)]  # 150 BPM、位相は 10.0 秒と少しずれている
    out = tap_tempo(g, taps, rng)
    b = np.asarray(out.beats)
    assert 10.0 in out.beats and rng.end in out.beats  # 両端の境目の拍は残る
    after = b[b > 10.0][0]
    assert after - 10.0 >= 0.2  # 共有の拍の直後に半拍より短い間隔を作らない
    assert [x for x in out.beats if x <= 10.0] == [x for x in g.beats if x <= 10.0]


def test_too_many_beats(monkeypatch: pytest.MonkeyPatch) -> None:
    from stemapp.beats import edit

    monkeypatch.setattr(edit, "MAX_BEATS", 50)
    g = grid([(0, 120)], 20)  # 40 拍
    with pytest.raises(BeatEditError, match="多すぎ"):
        scale_tempo(g, 2)
