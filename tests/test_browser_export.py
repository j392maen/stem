"""書き出しメニューの画面テスト（T08。PC の Microsoft Edge を Playwright で動かす）。

`uv run pytest -m browser` で実行する。分割は FakeSeparator、書き出し（MP3）は本物の ffmpeg。
ダウンロードは Playwright の download イベントで発生を確かめる。
スクリーンショットは data/cache/screens/ に保存する（コミットしない）。
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from browser_helpers import LiveServer
from test_browser import PHONE, _done_track, _shot, browser, page, server  # noqa: F401

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

IPHONE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) CriOS/129.0 Mobile/15E148 Safari/604.1"
)


def _open_player(pg: Any, srv: LiveServer, track_id: int) -> None:
    pg.goto(f"{srv.base_url}/#/track/{track_id}")
    pg.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)


def _choose(pg: Any, group: str, value: str) -> None:
    pg.click(f"#export-{group} .seg-btn[data-value='{value}']")
    pg.wait_for_selector(f"#export-{group} .seg-btn.on[data-value='{value}']")


def _create_and_wait(pg: Any, filename: str) -> None:
    pg.click("#export-start")
    pg.wait_for_function(
        """(name) => {
            const f = document.querySelector('.export-filename');
            return !!document.querySelector('#export-download') && f && f.textContent === name;
        }""",
        arg=filename, timeout=60_000,
    )


def test_export_zip_and_mix(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    track_id, _job_id = _done_track(server, tmp_path)
    _open_player(page, server, track_id)

    # メニューを開く（開いている間はキー操作が再生に効かない）
    page.click("#export-btn")
    page.wait_for_selector(".export-modal")
    assert page.locator("#export-type .seg-btn").count() == 3
    assert page.locator("#export-format .seg-btn").count() == 3
    assert page.locator("#export-stem option").count() == 8
    _shot(page, "export_menu_pc.png")

    # 全部を ZIP（FLAC）
    _choose(page, "type", "all")
    _choose(page, "format", "flac")
    assert page.locator("#export-level .seg-btn").count() == 2
    _create_and_wait(page, "用意した曲 - stems.zip")
    _shot(page, "export_done_pc.png")
    with page.expect_download() as info:
        page.click("#export-download")
    dl = info.value
    assert dl.suggested_filename == "用意した曲 - stems.zip"
    saved = tmp_path / "dl" / dl.suggested_filename
    dl.save_as(str(saved))
    names = zipfile.ZipFile(saved).namelist()
    assert len(names) == 7 and all(n.endswith(".flac") for n in names)
    assert "用意した曲 - ドラム.flac" in names

    # 閉じて、stem を切り替えてから「今の組み合わせ」を MP3 でミックス
    page.click(".export-close")
    page.wait_for_selector(".export-modal", state="detached")
    page.click(".stem-btn[data-code='drums']")
    page.click(".stem-btn[data-code='bass']")
    page.click("#export-btn")
    _choose(page, "type", "mix")
    assert page.inner_text("#export-mix") == "ボーカル＋ギター＋ピアノ＋その他"
    _choose(page, "format", "mp3")
    expect = "用意した曲 - ボーカル＋ギター＋ピアノ＋その他.mp3"
    _create_and_wait(page, expect)
    with page.expect_download() as info:
        page.click("#export-download")
    assert info.value.suggested_filename == expect
    mp3 = tmp_path / "dl" / "mix.mp3"
    info.value.save_as(str(mp3))
    head = mp3.read_bytes()[:3]
    assert head == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xfa")

    # Esc で閉じる
    page.keyboard.press("Escape")
    page.wait_for_selector(".export-modal", state="detached")
    assert page.errors == []  # type: ignore[attr-defined]


def test_export_phone(browser: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    track_id, _job_id = _done_track(server, tmp_path)
    ctx = browser.new_context(
        viewport=PHONE, locale="ja-JP", user_agent=IPHONE_UA, has_touch=True, is_mobile=True,
        accept_downloads=True,
    )
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    try:
        _open_player(pg, server, track_id)
        pg.click("#export-btn")
        pg.wait_for_selector(".export-modal")
        # 組み合わせプリセットを選んでいると、その名前で書き出す
        _shot(pg, "export_menu_phone.png")
        pg.click(".export-close")
        pg.click("#presets li .name >> text=カラオケ（伴奏）")
        pg.click("#export-btn")
        _choose(pg, "type", "mix")
        assert pg.inner_text("#export-mix") == "組み合わせ「カラオケ（伴奏）」"
        _create_and_wait(pg, "用意した曲 - カラオケ（伴奏）.wav")
        # iPhone のときは保存の案内を出す。リンクは <a download>
        assert pg.locator("#export-ios-hint").is_visible()
        link = pg.locator("#export-download")
        assert link.get_attribute("download") == "用意した曲 - カラオケ（伴奏）.wav"
        _shot(pg, "export_done_phone.png")
        with pg.expect_download() as info:
            link.click()
        assert info.value.suggested_filename == "用意した曲 - カラオケ（伴奏）.wav"
        assert errors == []
    finally:
        ctx.close()
