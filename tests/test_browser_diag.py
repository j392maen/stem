"""端末の診断ページ・PWA・パスコードの注意の画面テスト（PC の Edge）。

`uv run pytest -m browser` で実行する。スクリーンショットは data/cache/screens/ に保存する。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from browser_helpers import EDGE_ARGS, SCREENS_DIR, LiveServer, run_server
from stemapp.config import Settings

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

playwright_api = pytest.importorskip("playwright.sync_api")

DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}


@pytest.fixture
def server(tmp_path: Path) -> Iterator[LiveServer]:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    with run_server(settings, with_worker=False) as srv:
        yield srv


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge", headless=True, args=EDGE_ARGS)
        except Exception as e:  # Edge が無い環境
            pytest.skip(f"Microsoft Edge を起動できません: {e}")
        yield b
        b.close()


def _page(browser: Any, viewport: dict[str, int]) -> Any:
    ctx = browser.new_context(viewport=viewport, locale="ja-JP")
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.errors = errors  # type: ignore[attr-defined]
    return pg


def _shot(page: Any, name: str) -> Path:
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENS_DIR / name
    page.screenshot(path=str(path), full_page=True)
    return path


def _saved(server: LiveServer) -> list[dict[str, Any]]:
    with httpx.Client(base_url=server.base_url, timeout=10) as c:
        return c.get("/api/diag").json()["results"]


def test_diag_page_desktop(browser: Any, server: LiveServer) -> None:
    page = _page(browser, DESKTOP)
    try:
        page.goto(server.base_url + "/#/library")
        link = page.locator("#diag-link")
        link.wait_for()
        assert link.inner_text() == "端末の診断"
        link.click()
        page.wait_for_selector("#diag-run")
        assert page.title() == "stemapp - 端末の診断"
        page.fill("#diag-device", "PC Edge test")
        page.click("#diag-run")
        page.wait_for_selector("#diag-status[data-saved]", timeout=30000)
        name = page.get_attribute("#diag-status", "data-saved")
        assert name and name.endswith("-PC-Edge-test.json")

        results = _saved(server)
        assert [r["name"] for r in results] == [name]
        result = results[0]["result"]
        assert result["device"] == "PC Edge test"
        assert set(result["decode"]) == {"webm", "m4a", "mp3", "flac", "wav"}
        for code, d in result["decode"].items():
            assert d["ok"] is True, (code, d)  # Edge はどれもデコードできる
            assert abs(d["duration"] - 1.0) < 0.1
        assert result["audio_context"]["sample_rate"] > 0
        assert 'audio/webm; codecs="opus"' in result["can_play_type"]
        assert result["features"]["service_worker"] is True
        assert result["features"]["indexed_db"]["open_ok"] is True
        assert result["standalone"] is False
        assert result["secure_context"] is True  # 127.0.0.1 は安全な接続扱い
        assert results[0]["server"]["local_client"] is True
        assert page.locator(".diag-ok").count() >= 5
        _shot(page, "diag-desktop.png")

        # 画面ロックの試験（ここではロックせずに答えるだけ）
        page.click("#lock-webaudio")
        page.wait_for_selector("[data-answer=continued]")
        page.wait_for_timeout(300)
        page.click("[data-answer=continued]")
        page.wait_for_function(
            "() => document.querySelector('#diag-status').dataset.saved !== " + json.dumps(name)
        )
        # 保存した名前で引く（一覧の並びに頼らない）。保存の応答の後に名前が付くので、
        # ファイルは既にある
        name2 = page.get_attribute("#diag-status", "data-saved")
        saved = {r["name"]: r for r in _saved(server)}
        assert set(saved) == {name, name2}
        assert _saved(server)[0]["name"] == name2  # 新しいものが先頭
        latest = saved[name2]["result"]
        assert len(latest["lock_tests"]) == 1
        lock = latest["lock_tests"][0]
        assert lock["mode"] == "webaudio" and lock["answer"] == "continued"
        assert lock["events"][0]["type"] == "start"
        assert page.locator(".diag-lock-list li").count() == 1
        assert "null" not in page.locator("#diag-lock").inner_text()

        # <audio> 要素でも鳴らせる
        page.click("#lock-element")
        page.wait_for_selector("[data-answer=unknown]")
        page.click("[data-answer=unknown]")
        page.wait_for_function("() => document.querySelectorAll('.diag-lock-list li').length === 2")
        assert page.errors == []  # type: ignore[attr-defined]
    finally:
        page.context.close()


def test_diag_page_phone_and_pwa(browser: Any, server: LiveServer) -> None:
    page = _page(browser, PHONE)
    try:
        page.goto(server.base_url + "/#/diag")
        page.click("#diag-run")
        page.wait_for_selector("#diag-status[data-saved]", timeout=30000)
        # 横にはみ出さない
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 0
        _shot(page, "diag-phone.png")

        # PWA: manifest・アイコンのリンクと Service Worker
        assert page.get_attribute("link[rel=manifest]", "href") == "manifest.webmanifest"
        assert page.get_attribute("link[rel=apple-touch-icon]", "href")
        page.wait_for_function(
            "() => navigator.serviceWorker.getRegistration().then((r) => !!r && !!r.active)",
            timeout=10000,
        )
        # Service Worker があっても API（音声を含む）はそのまま届く
        page.reload()
        page.wait_for_selector("#diag-run")
        assert page.evaluate("navigator.serviceWorker.controller !== null")
        assert page.evaluate("fetch('/api/health').then((r) => r.status)") == 200
        assert page.evaluate("fetch('/api/diag/samples/wav').then((r) => r.status)") == 200
        # 音声を <audio> でも読み込む（Range 付きの要求）
        page.evaluate(
            """() => new Promise((ok) => {
                const a = new Audio('/api/diag/samples/m4a');
                a.addEventListener('loadedmetadata', () => ok(true), { once: true });
                a.addEventListener('error', () => ok(false), { once: true });
                a.load();
            })"""
        )
        # Cache Storage の中身: 画面ファイルはあり、/api/ と音声は1つも無い
        cached = page.evaluate(
            """async () => {
                const out = [];
                for (const key of await caches.keys()) {
                    const cache = await caches.open(key);
                    for (const req of await cache.keys()) out.push(req.url);
                }
                return out;
            }"""
        )
        paths = [u.removeprefix(server.base_url) for u in cached]
        assert any(p.startswith("/js/") or p.startswith("js/") for p in paths), paths
        assert not [p for p in paths if "/api/" in p or p.startswith("api/")], paths
        audio_ext = (".webm", ".m4a", ".mp3", ".flac", ".wav", ".opus")
        assert not [p for p in paths if p.split("?")[0].endswith(audio_ext)], paths
        assert page.errors == []  # type: ignore[attr-defined]
    finally:
        page.context.close()


def test_passcode_notice_via_proxy(browser: Any, server: LiveServer) -> None:
    page = _page(browser, PHONE)
    try:
        def proxied_me(route: Any) -> None:
            route.fulfill(json={
                "authenticated": True, "passcode_required": False, "passcode_recommended": True,
                "local_client": False, "can_open_folder": False,
            })

        page.route("**/api/me", proxied_me)
        page.goto(server.base_url + "/#/library")
        notice = page.locator("#passcode-notice")
        notice.wait_for()
        assert "STEMAPP_PASSCODE" in notice.inner_text()
        page.wait_for_selector("#tracks")  # 使えなくはしない
        _shot(page, "notice-phone.png")
        page.click("#notice button")
        assert page.locator("#notice").is_hidden()
        page.reload()
        page.wait_for_selector("#tracks")
        assert page.locator("#notice").is_hidden()  # 閉じたらこのタブでは出さない
    finally:
        page.context.close()


def test_worker_down_notice(browser: Any, server: LiveServer) -> None:
    """ワーカーが止まっていると、画面の上に「分割の処理が止まっています」を出す。"""
    page = _page(browser, PHONE)
    try:
        def down(route: Any) -> None:
            route.fulfill(json={"status": "ok", "worker": {
                "state": "down", "running": False, "message": "分割の処理が止まっています。",
                "managed": True, "restart_in_sec": 42.0,
            }})

        page.route("**/api/health", down)
        page.goto(server.base_url + "/#/library")
        box = page.locator("#worker-notice")
        box.wait_for()
        text = box.inner_text()
        assert "分割の処理が止まっています" in text and "42 秒" in text
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 0
        _shot(page, "worker-down-phone.png")
    finally:
        page.context.close()


def test_no_notice_on_local(browser: Any, server: LiveServer) -> None:
    page = _page(browser, DESKTOP)
    try:
        page.goto(server.base_url + "/#/library")
        page.wait_for_selector("#tracks")
        assert page.locator("#notice").is_hidden()
        # ワーカーを起動していない構成（状態は unknown）では止まっている表示は出さない
        page.wait_for_function(
            "() => fetch('/api/health').then((r) => r.json()).then((b) => b.worker.state)"
            " .then((s) => s === 'unknown')"
        )
        assert page.locator("#worker-notice").is_hidden()
        # アイコンは赤い丸ではなく波形の棒の画像
        bg = page.eval_on_selector(".brand-mark", "e => getComputedStyle(e).backgroundImage")
        assert "icon.svg" in bg
    finally:
        page.context.close()
