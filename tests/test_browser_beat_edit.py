"""拍の補正パネル・小節ループ・キューのスナップの画面テスト（T10c。PC の Edge を Playwright で）。

`-m browser` で実行する。拍は FakeBeatAnalyzer（6 秒で 120 → 150 BPM、4/4）。
スクリーンショットは data/cache/screens/ に保存する（コミットしない）。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from browser_helpers import LiveServer
from test_browser import DESKTOP, PHONE, _shot, browser, page  # noqa: F401  fixture を使う
from test_browser_beats import _done_track, _wait_bpm, server  # noqa: F401

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

VIEW = "window.__stemapp.view"


def _open_player(page: Any, server: LiveServer, track_id: int) -> None:  # noqa: F811
    page.goto(f"{server.base_url}/#/track/{track_id}")
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    page.evaluate(f"() => {VIEW}.engine.pause()")


def _seek(pg: Any, t: float) -> None:
    pg.evaluate(f"(t) => {VIEW}.seek(t)", t)
    pg.wait_for_function(f"(t) => Math.abs({VIEW}.engine.position - t) < 1e-6", arg=t)


def _bars_after_draw(pg: Any, time_text: str) -> int:
    """再生位置の表示が time_text になった後の、拡大波形の小節線の数。"""
    pg.wait_for_function(
        "(t) => document.querySelector('#time-now').textContent === t", arg=time_text
    )
    pg.wait_for_timeout(50)  # 次の描画（同じフレームで小節の数も書く）
    return int(pg.get_attribute("#wave-zoom", "data-bars") or 0)


def _bpm_text(pg: Any) -> str:
    return pg.inner_text("#bpm-value")


def _hide_toast(pg: Any) -> None:
    """スクリーンショットの前に通知を消す（普段の画面を撮る）。"""
    pg.evaluate("() => { document.getElementById('toast').hidden = true; }")


def _wait_state(pg: Any, text: str) -> None:
    pg.wait_for_function(
        "(t) => document.querySelector('#be-state').textContent === t", arg=text
    )


def test_edit_panel_double_tap_undo_reset(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    track_id, _ = _done_track(server, tmp_path)
    _open_player(page, server, track_id)
    panel = page.locator("#beat-edit")
    # 普段は閉じている（見出しの1行だけ）
    assert panel.is_visible() and panel.get_attribute("open") is None
    assert page.inner_text("#be-state") == "自動"
    assert not page.locator("#be-double").is_visible()
    _shot(page, "beat_edit_closed.png")

    page.click("#beat-edit > summary")
    page.wait_for_selector("#be-double", state="visible")
    assert page.locator("#be-undo").is_disabled() and page.locator("#be-reset").is_disabled()
    assert page.input_value("#be-range") == "segment"

    # ×2（範囲は再生位置の区間 = 0〜6 秒）: 小節線と BPM 表示がすぐ変わる
    _seek(page, 2.0)
    _wait_bpm(page, "120.0")
    bars_before = _bars_after_draw(page, "0:02.0")
    page.click("#be-double")
    _wait_bpm(page, "240.0")
    _wait_state(page, "補正済み")
    page.wait_for_function(
        "(n) => Number(document.querySelector('#wave-zoom').dataset.bars) > n", arg=bars_before
    )
    assert page.inner_text("#meter") == "4/4"
    _seek(page, 9.0)
    _wait_bpm(page, "150.0")  # 後ろの区間はそのまま
    assert not page.locator("#be-undo").is_disabled()
    _seek(page, 2.0)
    _hide_toast(page)
    _shot(page, "beat_edit_double.png")

    # Ctrl+Z は補正パネルを開いているときだけ（閉じていると何もしない）
    page.click("#beat-edit > summary")
    page.wait_for_selector("#be-double", state="hidden")
    page.locator("body").focus()
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)
    assert page.inner_text("#be-state") == "補正済み" and _bpm_text(page) == "240.0"
    page.click("#beat-edit > summary")
    page.wait_for_selector("#be-double", state="visible")
    # 元に戻す（Ctrl+Z）
    page.locator("body").focus()
    page.keyboard.press("Control+z")
    _wait_bpm(page, "120.0")
    _wait_state(page, "自動")
    assert _bars_after_draw(page, "0:02.0") == bars_before
    page.wait_for_function("() => document.querySelector('#be-undo').disabled")

    # タップ（テスト用に曲の時刻を直接渡す）: 150 BPM の間隔で 5 回 → 範囲（曲全体）を置き換える
    page.select_option("#be-range", "all")
    page.evaluate(
        f"""() => {{
        const p = {VIEW}.beatEdit;
        for (let k = 0; k < 5; k++) p.tap(1.0 + k * 0.4, 100 + k * 0.4);
    }}"""
    )
    assert page.inner_text("#be-tap-info").startswith("5 回・150.0 BPM")
    # 最後にたたいてから少したつと拍を作る
    _wait_bpm(page, "150.0", timeout_ms=5000)
    page.wait_for_function(
        "() => document.querySelector('#be-tap-info').textContent === 'タップから拍を作りました'"
    )
    _seek(page, 9.0)
    _wait_bpm(page, "150.0")
    # 少ないタップは使わない
    page.evaluate(f"() => {{ const p = {VIEW}.beatEdit; p.tap(1, 1); p.tap(1.5, 1.5); }}")
    assert page.evaluate(f"() => {VIEW}.beatEdit.commitTaps()") is False
    assert "4 回以上" in page.inner_text("#be-tap-info")

    # 拍子（曲全体を 3 拍子に）
    page.select_option("#be-meter", "3")
    page.wait_for_function("() => document.querySelector('#meter').textContent === '3/4'")

    # 自動に戻す（確認つき。やめると何もしない）
    page.click("#be-reset")
    page.click(".modal button:has-text('やめる')")
    assert page.inner_text("#be-state") == "補正済み"
    page.click("#be-reset")
    page.click(".modal button:has-text('自動に戻す')")
    _wait_state(page, "自動")
    _seek(page, 2.0)
    _wait_bpm(page, "120.0")
    assert page.inner_text("#meter") == "4/4"
    # 自動に戻すのも元に戻せる
    page.click("#be-undo")
    _wait_state(page, "補正済み")
    page.wait_for_function("() => document.querySelector('#meter').textContent === '3/4'")

    # 画面を開き直しても（保存は API で即時）直した結果が出る
    page.reload()
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    assert page.inner_text("#be-state") == "補正済み"
    assert "手動で補正済み" in (page.get_attribute("#tempo", "title") or "")
    assert not page.errors  # type: ignore[attr-defined]


def test_snap_bar_loop_cues_and_tempo_marks(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    # 24 秒（0〜6 秒 120 BPM、6〜24 秒 150 BPM）
    track_id, _ = _done_track(server, tmp_path, seconds=24.0)
    _open_player(page, server, track_id)

    # テンポが変わる位置の印（6 秒）。ポインタを合わせると前後の BPM
    page.wait_for_function(
        "() => document.querySelector('#wave-overview').dataset.tempoMarks === '1'"
    )
    box = page.locator("#wave-overview").bounding_box()
    duration = page.evaluate(f"() => {VIEW}.wave.duration")
    page.mouse.move(box["x"] + box["width"] * 6.0 / duration, box["y"] + box["height"] - 3)
    page.wait_for_function(
        "() => document.querySelector('#wave-overview').title.includes('120.0 → 150.0')"
    )
    page.mouse.move(box["x"] + 3, box["y"] + 3)
    page.wait_for_function("() => document.querySelector('#wave-overview').title === ''")

    # キューのスナップ（既定 ON）: 8.07 秒で打つと 8.0 秒の拍に合う
    assert page.is_checked("#snap-toggle")
    _seek(page, 8.07)
    page.click("#add-cue-btn")
    page.wait_for_function(f"() => {VIEW}.cues.length === 1")
    assert page.evaluate(f"() => {VIEW}.cues[0].position_sec") == 8.0
    page.uncheck("#snap-toggle")
    _seek(page, 12.83)
    page.click("#add-cue-btn")
    page.wait_for_function(f"() => {VIEW}.cues.length === 2")
    assert page.evaluate(f"() => {VIEW}.cues[1].position_sec") == 12.83
    page.check("#snap-toggle")

    # 小節ループ: 2.3 秒で「2」→ 小節の頭 2.0 秒から 2 小節（6.0 秒まで）
    _seek(page, 2.3)
    page.click("#bar-loops button[data-bars='2']")
    loop = page.evaluate(f"() => {VIEW}.engine.loop")
    assert loop == {"start": 2.0, "end": 6.0}
    assert page.get_attribute("#loop-btn", "aria-pressed") == "true"
    assert page.get_attribute("#bar-loops button[data-bars='2']", "aria-pressed") == "true"
    # ×2 → 4 小節（6 秒からは 150 BPM の拍で数える）、½ で戻る
    page.click("#loop-double")
    assert page.evaluate(f"() => {VIEW}.engine.loop") == {"start": 2.0, "end": 9.2}
    assert page.get_attribute("#bar-loops button[data-bars='4']", "aria-pressed") == "true"
    page.locator("body").focus()
    page.keyboard.press("[")
    assert page.evaluate(f"() => {VIEW}.engine.loop") == {"start": 2.0, "end": 6.0}
    page.keyboard.press("[")
    page.keyboard.press("[")
    page.keyboard.press("[")  # 1/4 小節（下限）
    assert page.evaluate(f"() => {VIEW}.engine.loop") == {"start": 2.0, "end": 2.5}
    assert "1/4 小節" in page.inner_text("#bar-loop-state")
    # L で止める・B で最後に選んだ長さ（2 小節）を作り直す
    page.keyboard.press("l")
    assert page.evaluate(f"() => {VIEW}.engine.loop") is None
    _seek(page, 7.0)
    page.keyboard.press("b")
    assert page.evaluate(f"() => {VIEW}.engine.loop") == {"start": 6.0, "end": 9.2}
    # 再生してもループの中で折り返す
    page.click("#play-btn")
    page.wait_for_timeout(600)
    pos = page.evaluate(f"() => {VIEW}.engine.position")
    assert 6.0 <= pos < 9.2
    page.click("#play-btn")

    # 範囲「ループ区間」で +10ms → ループの中（6.0〜9.2 秒）の拍だけ動く
    page.click("#beat-edit > summary")
    page.select_option("#be-range", "loop")
    _seek(page, 7.0)
    page.click("#be-shift-plus")
    _wait_state(page, "補正済み")
    beats = page.evaluate(f"() => Array.from({VIEW}.beatGrid.beats)")
    assert 6.01 in beats and 8.81 in beats and 5.5 in beats and 9.2 in beats
    assert 6.0 not in beats
    page.click("#be-undo")
    _wait_state(page, "自動")

    # キュー2点から: 8.0 秒と 12.83 秒の間（今の 150 BPM・4/4 なら約 3 小節）を 5 小節に。
    # 1小節の拍数は選択欄（既定は曲の拍子 4）で、必ずサーバーに送る
    page.wait_for_selector("#be-cues", state="visible")
    assert page.input_value("#be-cue-meter") == "4"
    assert page.input_value("#be-bars") == "3"
    page.select_option("#be-cue-meter", "3")
    assert page.input_value("#be-bars") == "4"  # 3 拍子なら約 4 小節
    page.select_option("#be-cue-meter", "4")
    page.fill("#be-bars", "5")
    sent: list[Any] = []
    page.on("request", lambda r: sent.append(r.post_data_json)
            if r.url.endswith("/beats/edit") else None)
    page.click("#be-cues")
    _wait_state(page, "補正済み")
    _seek(page, 10.0)
    page.wait_for_function(
        "() => document.querySelector('#bpm-value').textContent === (20 * 60 / 4.83).toFixed(1)"
    )
    downbeats = page.evaluate(f"() => Array.from({VIEW}.beatGrid.downbeats)")
    assert 8.0 in downbeats and 12.83 in downbeats
    assert sent and sent[-1]["op"] == "cues" and sent[-1]["beats_per_bar"] == 4

    # 拍子の選択欄は、開いている間は再生位置の小節の拍数に合わせて変わる
    page.select_option("#be-range", "segment")
    _seek(page, 20.0)
    page.select_option("#be-meter", "3")
    page.wait_for_function("() => document.querySelector('#meter').textContent === '3/4'")
    page.locator("body").focus()
    _seek(page, 3.0)
    page.wait_for_function("() => document.querySelector('#be-meter').value === '4'")
    _seek(page, 20.0)
    page.wait_for_function("() => document.querySelector('#be-meter').value === '3'")
    page.click("#be-undo")
    _wait_state(page, "補正済み")
    # キューのループに ½・×2 をかけると保存した終点を書き換える（title と通知で知らせる）
    assert "書き換え" in (page.get_attribute("#loop-double", "title") or "")

    # スクリーンショット（パネルを開いた PC・スマホ幅。通知が写らない普段の状態）
    _seek(page, 10.0)
    page.evaluate(f"() => {VIEW}.setBarLoop(4)")
    _hide_toast(page)
    page.wait_for_timeout(200)
    _shot(page, "beat_edit_open.png")
    page.set_viewport_size(PHONE)
    page.wait_for_timeout(300)
    _hide_toast(page)
    _shot(page, "beat_edit_phone.png")
    page.set_viewport_size(DESKTOP)
    assert not page.errors  # type: ignore[attr-defined]


def test_tap_uses_output_latency(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    """タップの時刻は、再生位置から出力の遅延（outputLatency + baseLatency）を引いたもの。"""
    track_id, _ = _done_track(server, tmp_path, seconds=8.0)
    _open_player(page, server, track_id)
    page.click("#beat-edit > summary")
    # 再生していないときは使えない
    page.click("#be-tap")
    page.wait_for_function(
        "() => document.querySelector('#toast').textContent.includes('再生しながら')"
    )
    # タップのボタンにフォーカスがあるときの Space はタップだけ（再生/停止にはならない）
    page.evaluate("() => { document.getElementById('toast').textContent = ''; }")
    page.focus("#be-tap")
    page.keyboard.press("Space")
    page.wait_for_function(
        "() => document.querySelector('#toast').textContent.includes('再生しながら')"
    )
    page.wait_for_timeout(200)
    assert page.evaluate(f"() => {VIEW}.engine.playing") is False
    got = page.evaluate(
        f"""() => {{
        const v = {VIEW};
        const p = v.beatEdit;
        v.engine.outputLatency = () => 0.1;
        Object.defineProperty(v.engine, "playing", {{ value: true, configurable: true }});
        Object.defineProperty(v.engine, "position", {{ value: 3.0, configurable: true }});
        p.tapNow();
        const t = p.taps.taps[0];
        delete v.engine.playing; delete v.engine.position;
        v.engine.playing = false;
        p.taps.reset();
        clearTimeout(p.tapTimer);
        return t;
    }}"""
    )
    assert got == pytest.approx(2.9)
    # T キーは補正パネルを開いているときだけ
    assert page.evaluate(f"() => {VIEW}.beatEdit.open") is True
    assert not page.errors  # type: ignore[attr-defined]
