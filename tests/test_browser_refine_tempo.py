"""「もっと分ける」と速度変更（T11）を一緒に使う画面テスト（T07。PC の Edge を Playwright で）。

`-m browser` で実行する。分割・詳細分割は FakeSeparator、ピッチを保つ方式の伸縮は FakeStretcher。
- 速度を変えて再生している最中に詳細分割が終わり、画面を読み直しても、速度・再生位置・選択が
  保たれる。
- ピッチを保つ方式では、古い伸縮の音声（分ける前の stem の組）を使わず、子を含む組を作り直す。
"""

from __future__ import annotations

import shutil
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
    _shot,
    browser,
    page,
)
from test_browser_beats import TEMPO, _done_track

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

VIEW = "window.__stemapp.view"
DRUM_KIDS = ["kick", "snare", "toms", "hihat", "ride", "crash", "drums_rest"]


@pytest.fixture
def server(tmp_path: Path) -> Iterator[LiveServer]:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    with run_server(
        settings, fake_delay=0.2, beat_analyzer=FakeBeatAnalyzer(TEMPO),
        tempo_stretcher=FakeStretcher(delay_sec=1.0, steps=5),
    ) as srv:
        yield srv


def _open(pg: Any, server: LiveServer, track_id: int) -> None:
    pg.goto(f"{server.base_url}/#/track/{track_id}")
    pg.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)


def _engine(pg: Any, expr: str) -> Any:
    return pg.evaluate(f"() => {{ const e = {VIEW}.engine; return {expr}; }}")


def _wait_status(pg: Any, text: str, timeout_ms: int = 30_000) -> None:
    pg.wait_for_function(
        "(t) => (document.querySelector('#tp-status')?.textContent || '').includes(t)",
        arg=text, timeout=timeout_ms,
    )


def _refine(pg: Any, code: str, model_prefix: str) -> None:
    pg.click(f".stem-cell[data-code='{code}'] .refine-act.split")
    pg.wait_for_selector(".refine-modal")
    pg.click(f".refine-option[data-model^='{model_prefix}']")
    pg.wait_for_selector(".refine-modal", state="detached")


def _wait_children(pg: Any, code: str) -> None:
    pg.wait_for_selector(f".stem-btn[data-code='{code}']", timeout=60_000)
    pg.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)


def test_refine_keeps_speed_in_pitch_mode(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    """ピッチも変わる方式（playbackRate）で 1.1 倍にして再生中に、その他を分ける。"""
    track_id, _job_id = _done_track(server, tmp_path, seconds=60.0)
    _open(page, server, track_id)
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.1)")
    page.click(".stem-btn[data-code='bass']")  # ベースを OFF
    page.evaluate(f"() => {VIEW}.seek(5.0)")
    page.click("#play-btn")
    page.wait_for_function(f"() => {VIEW}.engine.position > 5.2")
    assert _engine(page, "e.speed") == pytest.approx(1.1)
    _refine(page, "other", "hpss")
    _wait_children(page, "sustained")
    page.wait_for_function(f"() => {VIEW}.engine.playing", timeout=10_000)
    # 速度・方式・再生位置（分ける前より先）・選択が保たれる
    assert _engine(page, "e.speed") == pytest.approx(1.1)
    assert _engine(page, "e.rate") == pytest.approx(1.1)
    assert page.evaluate(f"() => {VIEW}.tempo.mode") == "pitch"
    assert _engine(page, "e.position") > 5.2
    gains = _engine(page, "e.gainValues()")
    assert gains["bass"] == 0
    assert gains["sustained"] == 1 and gains["transient"] == 1 and gains["other"] == 0
    assert not page.errors  # type: ignore[attr-defined]


def test_refine_rebuilds_keep_mode_render(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    """ピッチを保つ方式で 1.2 倍にして再生中に、ドラムを分ける。読み直した後は古い伸縮の音声
    （ドラムのまま）を使わず、子を含む組を作り直して差し替える。"""
    track_id, job_id = _done_track(server, tmp_path, seconds=60.0)
    _open(page, server, track_id)
    page.click(".stem-btn[data-code='bass']")  # ベースを OFF
    page.evaluate(f"() => {VIEW}.seek(5.0)")
    page.click("#play-btn")
    page.click("#tp-mode .seg-btn[data-value='keep']")
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.2)")
    _wait_status(page, "ピッチを保って再生中（×1.200）")
    old_files = page.evaluate(f"() => Object.keys({VIEW}.tempo.render.files)")
    assert "drums" in old_files and "kick" not in old_files
    pos_before = _engine(page, "e.position")

    _refine(page, "drums", "MDX23C")
    _wait_children(page, "kick")
    # 読み直した後: 古い組（ドラムのまま）は使わず、子を含む組を新しく作って差し替える
    # （SQLite は消した行の番号を使い直すことがあるので、render_id ではなく中身で確かめる）
    _wait_status(page, "ピッチを保って再生中（×1.200）")
    render = page.evaluate(f"() => {VIEW}.tempo.render")
    assert set(DRUM_KIDS) <= set(render["files"]) and "drums" not in render["files"]
    assert _engine(page, "e.bufScale") == pytest.approx(1.2)
    loaded = _engine(page, "[...e.tracks].filter(([, t]) => t.buffer).map(([c]) => c)")
    assert set(DRUM_KIDS) <= set(loaded)
    page.wait_for_function(f"() => {VIEW}.engine.playing", timeout=10_000)
    assert page.evaluate(f"() => {VIEW}.tempo.mode") == "keep"
    assert _engine(page, "e.position") >= pos_before
    gains = _engine(page, "e.gainValues()")
    assert gains["bass"] == 0 and gains["drums"] == 0
    assert all(gains[k] == 1 for k in DRUM_KIDS)
    # サーバーにも古い組（分ける前の stem）は残っていない
    with httpx.Client(base_url=server.base_url) as c:
        renders = c.get(f"/api/jobs/{job_id}/tempo").json()["renders"]
    assert renders and all("drums" not in r["files"] for r in renders)
    _shot(page, "refine_with_tempo_keep_pc.png")
    assert not page.errors  # type: ignore[attr-defined]
