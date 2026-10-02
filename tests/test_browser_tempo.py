"""速度の変更の画面テスト（T11。PC の Edge を Playwright で）。

`-m browser` で実行する。拍は FakeBeatAnalyzer（6 秒で 120 → 150 BPM）、ピッチを保つ方式の伸縮は
FakeStretcher（長さだけを変えた WAV。ffmpeg を使わない）。
スクリーンショットは data/cache/screens/ に保存する（コミットしない）。
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from browser_helpers import LiveServer, run_server
from stemapp.beats import FakeBeatAnalyzer
from stemapp.config import Settings
from stemapp.tempo.stretch import FakeStretcher
from test_browser import (  # noqa: F401  fixture を使う
    PHONE,
    _shot,
    _wait_gains,
    browser,
    page,
)
from test_browser_beats import TEMPO, _done_track, _wait_bpm

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

VIEW = "window.__stemapp.view"


@pytest.fixture
def server(tmp_path: Path) -> Iterator[LiveServer]:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    with run_server(
        settings, fake_delay=0.2, beat_analyzer=FakeBeatAnalyzer(TEMPO),
        tempo_stretcher=FakeStretcher(delay_sec=2.0, steps=20),
    ) as srv:
        yield srv


def _open(pg: Any, server: LiveServer, track_id: int) -> None:
    pg.goto(f"{server.base_url}/#/track/{track_id}")
    pg.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)


def _engine(pg: Any, expr: str) -> Any:
    return pg.evaluate(f"() => {{ const e = {VIEW}.engine; return {expr}; }}")


SRC_LOOPS = (
    "[...e.tracks.values()].filter((t) => t.source)"
    ".map((t) => [t.source.loopStart, t.source.loopEnd])"
)
BUF_LENGTHS = "[...e.tracks.values()].filter((t) => t.buffer).map((t) => t.buffer.duration)"


def _positions(pg: Any, n: int) -> list[float]:
    """50ms おきに n 回、再生位置を読む。"""
    return list(pg.evaluate(
        """async (n) => {
        const e = window.__stemapp.view.engine;
        const out = [];
        for (let i = 0; i < n; i++) {
          out.push(e.position);
          await new Promise((r) => setTimeout(r, 50));
        }
        return out;
    }""",
        n,
    ))


def _hide_toast(pg: Any) -> None:
    pg.evaluate("() => { document.getElementById('toast').hidden = true; }")


def _speed_over(pg: Any, ms: int = 600) -> float:
    """実時間 ms の間に曲の時刻がどれだけ進んだか（曲の秒 / 実時間の秒）。"""
    return float(pg.evaluate(
        """async (ms) => {
        const e = window.__stemapp.view.engine;
        const c0 = e.ctx.currentTime, p0 = e.position;
        await new Promise((r) => setTimeout(r, ms));
        const c1 = e.ctx.currentTime, p1 = e.position;
        let d = p1 - p0;
        if (d < 0 && e.loop) d += e.loop.end - e.loop.start;  // ループで折り返した
        return d / (c1 - c0);
    }""",
        ms,
    ))


def test_tempo_pure_functions(page: Any, server: LiveServer) -> None:  # noqa: F811
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    res = page.evaluate(
        """() => {
        const { engine: E, tempo: T, beatedit: B } = window.__stemapp.modules;
        const loop = { start: 10, end: 14 };
        return {
          // 速度つきの位置: 基準 5 秒から実時間 2 秒、1.5 倍 → 8 秒
          plain: E.songPositionAt(5, 2, 1.5, null, 100),
          slow: E.songPositionAt(5, 2, 0.5, null, 100),
          // ループの折り返しは曲の時刻で: 12 秒から実時間 2 秒・1.5 倍 → 15 → 11
          looped: E.songPositionAt(12, 2, 1.5, loop, 100),
          // 速度を変える: 実時間 1 秒たった所（1.25 倍で 11.25）を新しい基準にする
          rebased: E.rebaseAt(10, 1, 1.25, null, 100).base,
          rebasedLoop: E.rebaseAt(13, 2, 1.0, loop, 100).base,  // 15 → 11
          // 曲の長さで止まる
          end: E.songPositionAt(99, 10, 2, null, 100),
          neg: E.songPositionAt(3, -1, 2, null, 100),
          clamp: [T.clampRatio(0.1), T.clampRatio(3), T.clampRatio(1.23456), T.clampRatio("x")],
          fromBpm: [T.ratioFromBpm(132, 120), T.ratioFromBpm(300, 120), T.ratioFromBpm(120, null)],
          pct: [1, 1.035, 0.92, 1.0004].map((r) => T.formatPercent(r)),
          slider: [
            T.sliderToRatio(35), T.sliderToRatio(-80),
            T.ratioToSlider(1.2, 8), T.ratioToSlider(0.95, 16),
          ],
          key: T.ratioKey(1.1),
          tap: [B.tapSongTime(10, 0.05, 1), B.tapSongTime(10, 0.05, 1.2)],
          // 幅に入らない倍率なら幅を広げる（2.0 倍は ±50 のまま端に寄せる）
          widen: [T.rangeFor(1.05, 8), T.rangeFor(1.1, 8), T.rangeFor(0.7, 16), T.rangeFor(2.0, 8),
                  T.rangeFor(1.03, 50)],
          lowmem: [T.defaultLowMemory(true, 100), T.defaultLowMemory(false, 100),
                   T.defaultLowMemory(false, 600)],
        };
    }"""
    )
    assert res["plain"] == pytest.approx(8.0)
    assert res["slow"] == pytest.approx(6.0)
    assert res["looped"] == pytest.approx(11.0)
    assert res["rebased"] == pytest.approx(11.25)
    assert res["rebasedLoop"] == pytest.approx(11.0)
    assert res["end"] == 100
    assert res["neg"] == 3
    assert res["clamp"] == [0.5, 2.0, 1.235, 1]
    assert res["fromBpm"] == [1.1, 2.0, None]
    assert res["pct"] == ["±0.0%", "+3.5%", "−8.0%", "±0.0%"]
    assert res["slider"] == [1.035, 0.92, 80, -50]
    assert res["key"] == "1.100"
    # タップの出力遅延は「遅延 × 速さ」を曲の時刻から引く
    assert res["tap"] == [pytest.approx(9.95), pytest.approx(9.94)]
    assert res["widen"] == [8, 16, 50, 50, 50]
    assert res["lowmem"] == [True, False, True]


def test_pitch_mode_rate_keeps_sync(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    track_id, _ = _done_track(server, tmp_path, seconds=24.0)
    _open(page, server, track_id)
    assert page.inner_text("#tp-readout") == "±0.0%"
    # PC の既定は「ピッチを保つ・すぐ」（T11c）。このテストはピッチも変わる方式を選ぶ
    assert page.get_attribute("#tp-mode .seg-btn[data-value='instant']", "aria-pressed") == "true"
    page.click("#tp-mode .seg-btn[data-value='pitch']")
    assert page.get_attribute("#tp-mode .seg-btn[data-value='pitch']", "aria-pressed") == "true"
    # 一部の stem だけ鳴らす（1 キー = 最初の stem を OFF）
    page.evaluate(f"() => {VIEW}.seek(1.0)")
    page.click("#play-btn")
    page.wait_for_function(f"() => {VIEW}.engine.playing && {VIEW}.engine.position > 1.3")
    page.wait_for_timeout(150)  # GainNode のランプ（15ms）が終わるのを待つ
    all_on = _engine(page, "e.gainValues()")
    page.locator("body").focus()
    page.keyboard.press("1")
    page.wait_for_timeout(150)
    gains = _engine(page, "e.gainValues()")
    assert sorted(set(gains.values())) == [0.0, 1.0]
    n_sources = _engine(page, "e.activeSources()")
    assert n_sources > 1
    before = _engine(page, "e.position")

    # 速度を +10%（目標 BPM 132 = 区間の BPM 120 × 1.1）
    page.fill("#tp-bpm", "132")
    page.press("#tp-bpm", "Enter")
    page.wait_for_function("() => document.querySelector('#tp-readout').textContent === '+10.0%'")
    assert _engine(page, "e.rate") == pytest.approx(1.1)
    # 全 stem が同じ playbackRate で、音源は作り直していない（再生は止まらない）
    page.wait_for_timeout(100)
    rates = _engine(page, "e.sourceRates()")
    assert len(rates) == n_sources and all(r == pytest.approx(1.1) for r in rates), rates
    assert _engine(page, "e.playing") is True
    after = _engine(page, "e.position")
    assert after >= before  # 位置は戻らない
    assert _speed_over(page) == pytest.approx(1.1, abs=0.05)
    # GainNode（stem の選択）はそのまま、stem の切り替えも効く
    assert _engine(page, "e.gainValues()") == gains
    page.locator("body").focus()
    page.keyboard.press("1")
    _wait_gains(page, all_on)
    # BPM 表示 = 区間の BPM × 速さ、元の BPM も出す
    _wait_bpm(page, "132.0")
    assert page.inner_text("#bpm-orig") == "元 120.0"

    # キー操作: . で +0.1%、Shift+, で −1%、R で元に戻す
    page.keyboard.press(".")
    assert page.inner_text("#tp-readout") == "+10.1%"
    page.keyboard.press("Shift+Comma")
    assert page.inner_text("#tp-readout") == "+9.1%"
    assert _engine(page, "e.rate") == pytest.approx(1.091)

    # シーク: 曲の時刻で動き、その後も 1.091 倍で進む
    page.evaluate(f"() => {VIEW}.seek(8.0)")
    page.wait_for_function(f"() => Math.abs({VIEW}.engine.position - 8.0) < 0.2")
    page.wait_for_timeout(100)  # 鳴らし直す（start の予約の 30ms）のを待つ
    assert _speed_over(page) == pytest.approx(1.091, abs=0.05)
    assert all(r == pytest.approx(1.091) for r in _engine(page, "e.sourceRates()"))

    # 小節ループ（曲の時刻）: 再生位置の小節（150 BPM で 1.6 秒）。速度を変えても曲の時刻で折り返す
    page.evaluate(f"() => {VIEW}.setBarLoop(1)")
    loop = page.evaluate(f"() => {VIEW}.engine.loop")
    assert loop["end"] - loop["start"] == pytest.approx(1.6, abs=0.01)
    assert 6.0 <= loop["start"] <= 8.3
    page.keyboard.press(".")  # ループ中に速度を変える
    seen = _positions(page, 30)
    assert all(loop["start"] - 0.01 <= p <= loop["end"] + 0.01 for p in seen), seen
    assert any(b < a for a, b in zip(seen, seen[1:], strict=False)), "折り返していない"
    src_loop = _engine(page, SRC_LOOPS)
    assert all(a == pytest.approx(loop["start"]) and b == pytest.approx(loop["end"])
               for a, b in src_loop)
    page.evaluate(f"() => {{ {VIEW}.loopOn = false; {VIEW}.applyLoop(); }}")

    # 一時停止 → 再生でも速度はそのまま
    page.click("#play-btn")
    page.wait_for_function(f"() => !{VIEW}.engine.playing")
    page.click("#play-btn")
    page.wait_for_function(f"() => {VIEW}.engine.playing")
    assert all(r == pytest.approx(1.092) for r in _engine(page, "e.sourceRates()"))
    _hide_toast(page)
    _shot(page, "tempo_pitch_desktop.png")

    # 曲ごとに保存され、開き直すと戻る
    page.reload()
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    assert page.inner_text("#tp-readout") == "+9.2%"
    assert _engine(page, "e.rate") == pytest.approx(1.092)
    # 元の速度に戻す
    page.click("#tp-reset")
    assert page.inner_text("#tp-readout") == "±0.0%"
    assert _engine(page, "e.rate") == 1
    assert page.locator("#tp-reset").is_disabled()
    assert not page.errors  # type: ignore[attr-defined]


def _wait_status(pg: Any, text: str, timeout_ms: int = 20_000) -> None:
    pg.wait_for_function(
        "(t) => document.querySelector('#tp-status').textContent.startsWith(t)", arg=text,
        timeout=timeout_ms,
    )


def test_keep_mode_render_and_swap(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    track_id, job_id = _done_track(server, tmp_path, seconds=24.0)
    _open(page, server, track_id)
    page.evaluate(f"() => {VIEW}.seek(2.0)")
    page.click("#play-btn")
    page.wait_for_function(f"() => {VIEW}.engine.position > 2.3")
    page.wait_for_timeout(150)  # GainNode のランプ（15ms）が終わるのを待つ
    all_on = _engine(page, "e.gainValues()")
    page.locator("body").focus()
    page.keyboard.press("2")  # 2 番目の stem を OFF
    page.wait_for_timeout(150)
    gains = _engine(page, "e.gainValues()")
    assert gains != all_on
    # 小節ループ（2〜4 秒の 1 小節 ×2 = 2〜6 秒）を作っておく
    page.evaluate(f"() => {VIEW}.setBarLoop(2)")
    loop = page.evaluate(f"() => {VIEW}.engine.loop")
    assert loop["start"] == pytest.approx(2.0) and loop["end"] == pytest.approx(6.0)

    # ピッチを保つ方式で +20%（作成中は既定の「今の音のまま」）
    page.click("#tp-mode .seg-btn[data-value='keep']")
    assert page.is_visible("#tp-keep")
    assert page.input_value("#tp-pending") == "original"
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.2)")
    _wait_status(page, "作成")
    assert _engine(page, "e.speed") == pytest.approx(1.0)  # まだ元の速度のまま
    page.wait_for_function(
        "() => /作成中 [1-9]/.test(document.querySelector('#tp-status').textContent)",
        timeout=20_000,
    )
    _hide_toast(page)
    _shot(page, "tempo_keep_rendering_desktop.png")
    _wait_status(page, "ピッチを保って再生中（×1.200）")
    # 差し替え後: 音声の 1 秒 = 曲の 1.2 秒、playbackRate は 1、再生は続いている
    assert _engine(page, "e.bufScale") == pytest.approx(1.2)
    assert _engine(page, "e.rate") == 1
    assert _engine(page, "e.playing") is True
    assert all(r == 1 for r in _engine(page, "e.sourceRates()"))
    assert _engine(page, "e.gainValues()") == gains  # stem の選択は保たれる
    loop_now = page.evaluate(f"() => {VIEW}.engine.loop")
    assert loop_now == loop  # ループ（曲の時刻）も保たれる
    # 音声データ上のループ位置 = 曲の時刻 ÷ 1.2
    src_loop = _engine(page, SRC_LOOPS)
    assert all(a == pytest.approx(2 / 1.2) and b == pytest.approx(6 / 1.2) for a, b in src_loop)
    # 読み込んだ音声は伸縮済み（長さ = 元の長さ ÷ 1.2）
    lengths = _engine(page, BUF_LENGTHS)
    assert all(d == pytest.approx(24.0 / 1.2, abs=0.01) for d in lengths)
    assert _speed_over(page) == pytest.approx(1.2, abs=0.05)
    seen = _positions(page, 40)
    assert all(1.99 <= p <= 6.01 for p in seen)
    _wait_bpm(page, "144.0")  # 120 × 1.2
    assert page.inner_text("#bpm-orig") == "元 120.0"
    # stem の即時切り替えも効く
    page.locator("body").focus()
    page.keyboard.press("2")
    _wait_gains(page, all_on)

    # 作成中に「ピッチを変えて先に速度を変える」: すぐ 0.9 倍の速さになり、できたら差し替える
    page.select_option("#tp-pending", "pitch")
    page.evaluate(f"() => {VIEW}.tempo.setRatio(0.9)")
    page.wait_for_function(f"() => Math.abs({VIEW}.engine.speed - 0.9) < 1e-6")
    assert _engine(page, "e.bufScale") == pytest.approx(1.2)  # まだ前の音声（playbackRate 0.75）
    page.wait_for_timeout(100)  # playbackRate は 20ms 先の時刻で切り替わる
    assert all(r == pytest.approx(0.75) for r in _engine(page, "e.sourceRates()"))
    _wait_status(page, "ピッチを保って再生中（×0.900）")
    assert _engine(page, "e.bufScale") == pytest.approx(0.9) and _engine(page, "e.rate") == 1

    # 作成済みの倍率（1.2）を選ぶとすぐ切り替わる（作成しない）
    started = time.monotonic()
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.2)")
    _wait_status(page, "ピッチを保って再生中（×1.200）", timeout_ms=5000)
    assert time.monotonic() - started < 1.9  # 作成（FakeStretcher で 2 秒）を待っていない

    # 作成をやめる
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.3)")
    page.wait_for_selector("#tp-cancel:not([hidden])", timeout=10_000)
    page.click("#tp-cancel")
    _wait_status(page, "作成をやめました")
    assert page.is_visible("#tp-retry")
    assert _engine(page, "e.bufScale") == pytest.approx(1.2)  # 前の音声のまま鳴っている

    # ピッチも変わる方式に戻すと、元の音声を読み直して playbackRate で鳴らす
    page.click("#tp-mode .seg-btn[data-value='pitch']")
    page.wait_for_function(f"() => {VIEW}.engine.bufScale === 1 && {VIEW}.engine.rate === 1.3",
                           timeout=10_000)
    assert page.is_hidden("#tp-keep")
    # サーバーには作成済みの倍率が残っている（1曲 3 つまで）
    with httpx.Client(base_url=server.base_url) as c:
        renders = c.get(f"/api/jobs/{job_id}/tempo").json()["renders"]
    done = sorted(r["ratio_key"] for r in renders if r["status"] == "done")
    assert done == ["0.900", "1.200"]
    assert not page.errors  # type: ignore[attr-defined]


def test_cue_loop_independent_of_speed(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    """キューのループ（engine.setLoop、曲の時刻）は速度を変えても同じ区間で折り返す。"""
    track_id, _ = _done_track(server, tmp_path, seconds=24.0)
    with httpx.Client(base_url=server.base_url) as c:
        res = c.post(f"/api/tracks/{track_id}/cues",
                     json={"position_sec": 3.0, "loop_end_sec": 4.5, "label": "A"})
        assert res.status_code == 201, res.text
    _open(page, server, track_id)
    page.click("#play-btn")
    page.wait_for_function(f"() => {VIEW}.engine.playing")
    page.click("#cues li .name")  # キューへ（ループ付きのキューがループの対象になる）
    page.click("#loop-btn")
    assert page.evaluate(f"() => {VIEW}.engine.loop") == {"start": 3.0, "end": 4.5}
    for ratio in (1.25, 0.8):
        page.evaluate(f"(r) => {VIEW}.tempo.setRatio(r)", ratio)
        page.wait_for_timeout(100)
        assert page.evaluate(f"() => {VIEW}.engine.loop") == {"start": 3.0, "end": 4.5}
        assert all(r == pytest.approx(ratio) for r in _engine(page, "e.sourceRates()"))
        seen = _positions(page, 50)  # 2.5 秒: 1.5 秒の区間を必ず1回以上折り返す
        assert all(2.99 <= p <= 4.51 for p in seen), seen
        assert any(b < a for a, b in zip(seen, seen[1:], strict=False))
    assert not page.errors  # type: ignore[attr-defined]


def test_tempo_screens_phone(browser: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    track_id, _ = _done_track(server, tmp_path)
    ctx = browser.new_context(viewport=PHONE, locale="ja-JP", is_mobile=True, has_touch=True)
    pg = ctx.new_page()
    try:
        pg.goto(f"{server.base_url}/#/track/{track_id}")
        pg.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
        # スマホの既定はサーバーで作る方式（T11c）。まずピッチも変わる方式の画面
        assert pg.evaluate(f"() => {VIEW}.tempo.mode") == "keep"
        pg.click("#tp-mode .seg-btn[data-value='pitch']")
        pg.evaluate(f"() => {VIEW}.tempo.setRatio(0.955)")
        pg.wait_for_function("() => document.querySelector('#tp-readout').textContent === '−4.5%'")
        # はみ出さない（横スクロールが出ない）
        assert pg.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
        _hide_toast(pg)
        _shot(pg, "tempo_pitch_phone.png")
        pg.click("#tp-mode .seg-btn[data-value='keep']")
        _wait_status(pg, "作成")
        assert pg.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
        _hide_toast(pg)
        _shot(pg, "tempo_keep_phone.png")
        _wait_status(pg, "ピッチを保って再生中")
    finally:
        ctx.close()


def test_swap_continuity_range_blur_lowmem(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    track_id, _ = _done_track(server, tmp_path, seconds=24.0)
    _open(page, server, track_id)
    page.click("#play-btn")
    page.wait_for_function(f"() => {VIEW}.engine.position > 0.5")

    # 音声の差し替え（setBuffers）の前後で位置が戻らず飛ばない（繰り返し・欠けが無い）
    trace = page.evaluate(
        """async () => {
        const e = window.__stemapp.view.engine;
        const bufs = Object.fromEntries([...e.tracks].map(([c, t]) => [c, t.buffer]));
        const out = [];
        for (let i = 0; i < 40; i++) {
          if (i === 20) e.setBuffers(bufs, 1, 1);
          out.push([e.ctx.currentTime, e.position]);
          await new Promise((r) => setTimeout(r, 10));
        }
        return out;
    }"""
    )
    steps = [(p1 - p0) - (c1 - c0) for (c0, p0), (c1, p1) in zip(trace, trace[1:], strict=False)]
    assert all(abs(d) < 0.005 for d in steps), steps  # 実時間どおりに進む（5ms 未満の誤差）

    # スライダーの幅を超える倍率なら幅が自動で広がる
    assert page.get_attribute("#tp-range .seg-btn[data-value='8']", "aria-pressed") == "true"
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.3)")
    assert page.get_attribute("#tp-range .seg-btn[data-value='50']", "aria-pressed") == "true"
    assert page.input_value("#tp-slider") == "300"

    # ボタン・スライダーを操作した後はフォーカスが外れ、キー操作がプレイヤーに戻る
    page.click("#tp-plus")
    assert page.evaluate("() => document.activeElement === document.body")
    page.keyboard.press("Space")
    page.wait_for_function(f"() => !{VIEW}.engine.playing")
    page.locator("#tp-slider").focus()
    page.keyboard.press("ArrowLeft")  # 動かし終わり（change）で外れる
    assert page.evaluate("() => document.activeElement !== document.querySelector('#tp-slider')")
    page.keyboard.press("Space")
    page.wait_for_function(f"() => {VIEW}.engine.playing")

    # 省メモリ: 前の組を捨ててから読み込み、終わったら同じ位置から続ける
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1)")
    page.click("#tp-mode .seg-btn[data-value='keep']")
    page.check("#tp-lowmem")
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.2)")
    _wait_status(page, "ピッチを保って再生中（×1.200）")
    page.wait_for_function(f"() => {VIEW}.engine.playing")
    assert _engine(page, "e.bufScale") == pytest.approx(1.2)
    lengths = _engine(page, BUF_LENGTHS)
    assert lengths and all(d == pytest.approx(24.0 / 1.2, abs=0.01) for d in lengths)
    assert _speed_over(page) == pytest.approx(1.2, abs=0.05)
    # 読み込み中に別の組に変えると、前の読み込みは止まる（ピッチも変わる方式 = 元の音声へ）
    page.evaluate(f"() => {{ {VIEW}.tempo.setMode('pitch'); {VIEW}.tempo.setMode('keep'); }}")
    page.wait_for_function(
        f"() => {VIEW}.engine.bufScale === 1.2 && {VIEW}.tempo.activeKey === '1.200'"
        f" && {VIEW}.tempo.loadingKey === null && {VIEW}.engine.playing",
        timeout=10_000,
    )
    # 省メモリで前の組を捨てた後に読み込めなかったら、無音のままにせず元の音声（1.0 倍）で鳴らす
    page.route("**/api/files/tempo/**", lambda route: route.abort())
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.25)")
    page.wait_for_function(
        f"() => {VIEW}.tempo.activeKey === 'orig' && {VIEW}.engine.bufScale === 1"
        f" && {VIEW}.engine.rate === 1 && {VIEW}.engine.playing"
        f" && {VIEW}.tempo.loadingKey === null",
        timeout=20_000,
    )
    _wait_status(page, "読み込めませんでした（元の速度で再生中）")
    lengths = _engine(page, BUF_LENGTHS)
    assert lengths and all(d == pytest.approx(24.0, abs=0.01) for d in lengths)
    page.wait_for_timeout(500)  # 読み直しを繰り返さない（落ち着いている）
    assert page.evaluate(f"() => {VIEW}.tempo.activeKey") == "orig"
    page.unroute("**/api/files/tempo/**")
    page.evaluate("() => { document.getElementById('toast').hidden = true; }")
    assert not page.errors  # type: ignore[attr-defined]
