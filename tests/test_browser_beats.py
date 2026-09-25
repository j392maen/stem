"""拍・小節線と BPM 表示の画面テスト（T10。PC の Microsoft Edge を Playwright で動かす）。

`uv run pytest -m browser` で実行する。拍は FakeBeatAnalyzer（6 秒で 120 → 150 BPM）。
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

from audio_helpers import synth_drums, write_source
from browser_helpers import LiveServer, run_server
from stemapp.beats import FakeBeatAnalyzer
from stemapp.config import Settings
from stemapp.models import BeatGrid
from test_browser import _shot, browser, page  # noqa: F401  fixture を使う

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

TEMPO = [(0.0, 120.0), (6.0, 150.0)]


@pytest.fixture
def server(tmp_path: Path) -> Iterator[LiveServer]:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    with run_server(settings, fake_delay=0.2, beat_analyzer=FakeBeatAnalyzer(TEMPO)) as srv:
        yield srv


def _done_track(server: LiveServer, tmp_path: Path, seconds: float = 12.0) -> tuple[int, int]:
    src = write_source(tmp_path / "テンポが変わる曲.wav", synth_drums(TEMPO, seconds), "PCM_16")
    with httpx.Client(base_url=server.base_url, timeout=30) as c:
        res = c.post(
            "/api/imports",
            files={"file": (src.name, src.read_bytes(), "audio/wav")},
            data={"separate": "true", "preset": "fast"},
        )
        assert res.status_code == 202, res.text
        source_id = res.json()["source_id"]
        deadline = time.monotonic() + 90
        while True:
            imp = c.get(f"/api/imports/{source_id}").json()
            if imp["status"] == "done" and imp["job_id"]:
                job = c.get(f"/api/jobs/{imp['job_id']}").json()
                if job["status"] == "done":
                    return imp["track_id"], job["job_id"]
            assert imp["status"] != "failed", imp
            assert time.monotonic() < deadline, "分割が終わりません"
            time.sleep(0.2)


def _bpm(pg: Any) -> str:
    return pg.inner_text("#bpm-value")


def _wait_bpm(pg: Any, want: str, timeout_ms: int = 5000) -> None:
    pg.wait_for_function(
        "(w) => document.querySelector('#bpm-value').textContent === w", arg=want,
        timeout=timeout_ms,
    )


def test_beat_grid_pure_functions(page: Any, server: LiveServer) -> None:  # noqa: F811
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    res = page.evaluate(
        """() => {
        const B = window.__stemapp.modules.beats;
        const g = new B.BeatGrid({
          beats: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.4, 4.8],
          downbeats: [0, 2, 4],
          time_signature: 4,
          segments: [
            { start_sec: 4, end_sec: 4.8, bpm: 150 }, { start_sec: 0, end_sec: 4, bpm: 120 },
          ],
        });
        return {
          bpm: [g.bpmAt(-1), g.bpmAt(1), g.bpmAt(4), g.bpmAt(99)],
          beats: g.beatRange(1, 3), bars: g.barRange(1, 4.1),
          lb: [B.lowerBound([1, 2, 3], 0), B.lowerBound([1, 2, 3], 2), B.lowerBound([1, 2, 3], 9)],
          thin: [B.thinStep(40, 30), B.thinStep(10, 30), B.thinStep(1, 30, [8, 16, 32, 64])],
          fmt: [B.formatBpm(128), B.formatBpm(127.96), B.formatBpm(null)],
          empty: new B.BeatGrid({}).empty,
        };
    }"""
    )
    assert res["bpm"] == [120, 120, 150, 150]
    assert res["beats"] == [2, 6] and res["bars"] == [1, 3]
    assert res["lb"] == [0, 1, 3]
    assert res["thin"] == [1, 4, 32]
    assert res["fmt"] == ["128.0", "128.0", "—"]
    assert res["empty"] is True


def test_bar_lines_and_bpm_follow_position(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    track_id, _ = _done_track(server, tmp_path)
    with httpx.Client(base_url=server.base_url, timeout=30) as c:
        beats = c.get(f"/api/tracks/{track_id}/beats").json()
    assert [s["bpm"] for s in beats["segments"]] == [120.0, 150.0]

    page.goto(f"{server.base_url}/#/track/{track_id}")
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    view = "window.__stemapp.view"
    page.evaluate(f"() => {{ {view}.engine.pause(); {view}.engine.seek(2.0); }}")
    _wait_bpm(page, "120.0")
    assert page.inner_text("#meter") == "4/4"
    assert page.get_attribute("#tempo", "class") == "tempo"
    # 拡大波形は拍の線・小節線（8 秒表示で 120 BPM → 4 小節前後）
    page.wait_for_function("() => document.querySelector('#wave-zoom').dataset.grid === 'beats'")
    bars = int(page.get_attribute("#wave-zoom", "data-bars") or 0)
    assert 3 <= bars <= 6, bars
    _shot(page, "beats_120.png")

    page.evaluate(f"() => {view}.engine.seek(9.0)")
    _wait_bpm(page, "150.0")
    _shot(page, "beats_150.png")

    # 再生して区間の境目（6 秒）を越えると表示が変わる
    page.evaluate(f"() => {view}.engine.seek(5.3)")
    _wait_bpm(page, "120.0")
    page.click("#play-btn")
    _wait_bpm(page, "150.0", timeout_ms=5000)
    assert page.evaluate(f"() => {view}.engine.position") > 6.0
    page.click("#play-btn")

    # 拡大・縮小しても小節線は描ける（32 秒表示では曲全体の小節が見える）
    for _ in range(3):
        page.click(".wave-tools button[aria-label='縮小']")
    page.wait_for_timeout(200)
    total = len(beats["downbeats"])
    assert int(page.get_attribute("#wave-zoom", "data-bars") or 0) == total
    assert not page.errors  # type: ignore[attr-defined]


def test_track_without_beats_keeps_second_grid(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    track_id, _ = _done_track(server, tmp_path, seconds=6.0)
    with server.session_factory() as s:
        s.delete(s.get(BeatGrid, track_id))
        s.commit()
    page.goto(f"{server.base_url}/#/track/{track_id}")
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    page.wait_for_function("() => document.querySelector('#wave-zoom').dataset.grid === 'seconds'")
    assert _bpm(page) == "—"
    assert "none" in (page.get_attribute("#tempo", "class") or "")
    assert page.locator("#meter[hidden]").count() == 1
    assert "まだ解析されていません" in (page.get_attribute("#tempo", "title") or "")
    assert not page.errors  # type: ignore[attr-defined]
