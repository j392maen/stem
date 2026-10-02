"""外出先（Tailscale Serve）から使うための安全策と PWA の配信。

- Host ヘッダーの許可リスト（DNS リバインディング対策）
- HTTPS（X-Forwarded-Proto: https）で来たときの Secure Cookie
- 中継経由でパスコードが無いときの「パスコードの設定を勧める」
- manifest・Service Worker・アイコンの配信
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from stemapp.config import Settings
from stemapp.hosts import HostCheckMiddleware, normalize_host
from test_api import _app

TS_NAME = "unagi.tail8b25a2.ts.net"
PROXY = {"X-Forwarded-For": "100.100.95.89", "X-Forwarded-Proto": "https",
         "Tailscale-User-Login": "someone@example.com"}


def _client(settings: Settings, passcode: str | None = None, **update: object) -> TestClient:
    app = _app(settings.model_copy(update=update), passcode=passcode)
    return TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with _client(settings) as c:
        yield c


# --- Host の許可リスト --------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1:8000", "127.0.0.1"),
        ("LOCALHOST", "localhost"),
        ("[::1]:8000", "::1"),
        ("[::1]", "::1"),
        ("::1", "::1"),
        (f"{TS_NAME}.", TS_NAME),
        (f"{TS_NAME}:443", TS_NAME),
        ("", ""),
        ("[::1", ""),
        ("[::1]x", ""),
        ("host:abc", ""),
        # 全角数字や上付き数字は isdigit() が True でもポートではない
        ("127.0.0.1:８０００", ""),
        ("localhost:²", ""),
        ("[::1]:８０", ""),
    ],
)
def test_normalize_host(value: str, expected: str) -> None:
    assert normalize_host(value) == expected


def _run_asgi(scope: dict[str, Any]) -> list[dict[str, Any]]:
    """HostCheckMiddleware を直接呼ぶ（TestClient は Host を必ず付けるため）。"""
    sent: list[dict[str, Any]] = []
    reached: list[str] = []

    async def inner(_scope: Any, _receive: Any, _send: Any) -> None:
        reached.append(_scope["type"])

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    mw = HostCheckMiddleware(inner, allowed=frozenset({"127.0.0.1", "localhost", "::1"}))
    asyncio.run(mw(scope, receive, send))
    if reached:
        sent.append({"type": "reached"})
    return sent


def test_missing_host_header_is_400() -> None:
    sent = _run_asgi({"type": "http", "method": "GET", "path": "/api/health", "headers": []})
    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 400
    body = json.loads(sent[1]["body"].decode("utf-8"))
    assert "STEMAPP_ALLOWED_HOSTS" in body["detail"]


@pytest.mark.parametrize("headers", [[], [(b"host", b"evil.example")]])
def test_websocket_with_bad_host_is_closed_1008(headers: list[tuple[bytes, bytes]]) -> None:
    sent = _run_asgi({"type": "websocket", "path": "/ws", "headers": headers})
    assert sent == [{"type": "websocket.close", "code": 1008}]


def test_websocket_with_good_host_passes() -> None:
    sent = _run_asgi(
        {"type": "websocket", "path": "/ws", "headers": [(b"host", b"127.0.0.1:8000")]}
    )
    assert sent == [{"type": "reached"}]


def test_missing_host_via_real_server(settings: Settings) -> None:
    """本物のサーバー（uvicorn）に Host なしの HTTP/1.0 を送ると 400。"""
    import socket
    import threading
    import time

    import uvicorn

    from stemapp.app import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
            c.sendall(b"GET /api/health HTTP/1.0\r\n\r\n")
            data = b""
            while chunk := c.recv(4096):
                data += chunk
        assert data.startswith(b"HTTP/1.1 400"), data[:100]
    finally:
        server.should_exit = True
        t.join(timeout=15)


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000", "127.0.0.1"])
def test_local_hosts_are_always_allowed(settings: Settings, host: str) -> None:
    with _client(settings, allowed_hosts="") as c:
        res = c.get("/api/health", headers={"Host": host})
        assert res.status_code == 200
        assert c.get("/", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "host", ["evil.example", "evil.example:8000", TS_NAME, "127.0.0.1.evil.example", "0.0.0.0"]
)
def test_unknown_host_is_400(settings: Settings, host: str) -> None:
    with _client(settings, allowed_hosts="") as c:
        for path in ("/api/health", "/", "/api/tracks", "/manifest.webmanifest"):
            res = c.get(path, headers={"Host": host})
            assert res.status_code == 400, (path, host)
            assert "STEMAPP_ALLOWED_HOSTS" in res.json()["detail"]
        assert c.post("/api/login", json={}, headers={"Host": host}).status_code == 400


def test_allowed_hosts_setting(settings: Settings) -> None:
    with _client(settings, allowed_hosts=f" {TS_NAME.upper()} , other.example ,") as c:
        assert c.get("/api/health", headers={"Host": TS_NAME}).status_code == 200
        assert c.get("/api/health", headers={"Host": f"{TS_NAME}:443"}).status_code == 200
        assert c.get("/api/health", headers={"Host": "other.example"}).status_code == 200
        assert c.get("/api/health", headers={"Host": "evil.example"}).status_code == 400


def test_allowed_hosts_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEMAPP_ALLOWED_HOSTS", f"{TS_NAME},[::2]")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.allowed_host_names == {"127.0.0.1", "localhost", "::1", TS_NAME, "::2"}
    assert Settings(_env_file=None, allowed_hosts="").allowed_host_names == {  # type: ignore[call-arg]
        "127.0.0.1", "localhost", "::1",
    }


# --- Cookie ------------------------------------------------------------------------


def _login_cookie(c: TestClient, headers: dict[str, str]) -> str:
    res = c.post("/api/login", json={"passcode": "pw"}, headers=headers)
    assert res.status_code == 200, res.text
    return res.headers["set-cookie"]


def test_cookie_secure_when_https(settings: Settings) -> None:
    with _client(settings, passcode="pw") as c:
        cookie = _login_cookie(c, {"X-Forwarded-Proto": "https"})
        assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
        # テストの接続は http なので Secure の Cookie は自動では送られない。手で付ける
        token = cookie.split(";")[0]
        res = c.post("/api/logout", headers={"X-Forwarded-Proto": "https", "Cookie": token})
        assert res.status_code == 200
        assert "Secure" in res.headers["set-cookie"]


@pytest.mark.parametrize("headers", [{}, {"X-Forwarded-Proto": "http"}])
def test_cookie_not_secure_over_http(settings: Settings, headers: dict[str, str]) -> None:
    with _client(settings, passcode="pw") as c:
        cookie = _login_cookie(c, headers)
        assert "Secure" not in cookie
        assert "HttpOnly" in cookie


# --- パスコードを勧める -----------------------------------------------------------------


def test_passcode_recommended_via_proxy_without_passcode(client: TestClient) -> None:
    assert client.get("/api/me").json()["passcode_recommended"] is False
    me = client.get("/api/me", headers=PROXY).json()
    assert me["passcode_recommended"] is True
    assert me["local_client"] is False
    for key in ("X-Real-IP", "Via", "X-Forwarded-Proto"):
        assert client.get("/api/me", headers={key: "x"}).json()["passcode_recommended"] is True


def test_passcode_not_recommended_when_set(settings: Settings) -> None:
    with _client(settings, passcode="pw") as c:
        _login_cookie(c, {})
        assert c.get("/api/me", headers=PROXY).json()["passcode_recommended"] is False


# --- PWA ---------------------------------------------------------------------------


def test_manifest_is_served(client: TestClient) -> None:
    res = client.get("/manifest.webmanifest")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/manifest+json")
    data = json.loads(res.content.decode("utf-8"))
    assert data["display"] == "standalone"
    assert data["name"] and data["short_name"]
    assert data["start_url"] and data["scope"]
    assert data["background_color"].lower() == "#0b0b0e"
    sizes = {icon["sizes"] for icon in data["icons"]}
    assert {"192x192", "512x512"} <= sizes
    # purpose は "any" と "maskable" を別の項目に分けて書く（"any maskable" は非推奨）
    purposes = {(i["sizes"], i.get("purpose", "any")) for i in data["icons"]}
    for size in ("192x192", "512x512"):
        assert {(size, "any"), (size, "maskable")} <= purposes
    assert all(" " not in i.get("purpose", "any") for i in data["icons"])
    for icon in data["icons"]:
        got = client.get("/" + icon["src"].lstrip("/"))
        assert got.status_code == 200, icon
        assert got.headers["content-type"] == icon["type"]


@pytest.mark.parametrize(
    ("path", "ctype"),
    [
        ("/icons/apple-touch-icon.png", "image/png"),
        ("/icons/icon-192.png", "image/png"),
        ("/icons/icon-512.png", "image/png"),
        ("/icons/icon.svg", "image/svg+xml"),
    ],
)
def test_icons_are_served(client: TestClient, path: str, ctype: str) -> None:
    res = client.get(path)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith(ctype)
    if ctype == "image/png":
        assert res.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_apple_touch_icon_is_180(client: TestClient) -> None:
    data = client.get("/icons/apple-touch-icon.png").content
    # PNG の IHDR（幅・高さ）
    assert int.from_bytes(data[16:20], "big") == 180
    assert int.from_bytes(data[20:24], "big") == 180


def test_service_worker_is_served(client: TestClient) -> None:
    res = client.get("/sw.js")
    assert res.status_code == 200
    assert "javascript" in res.headers["content-type"]
    assert res.headers["cache-control"] == "no-cache"
    text = res.text
    assert "/api/" in text  # API（音声を含む）はキャッシュしない
    assert "VERSION" in text


def test_index_links_pwa(client: TestClient) -> None:
    html = client.get("/").text
    assert 'rel="manifest"' in html
    assert 'rel="apple-touch-icon"' in html
    assert "viewport-fit=cover" in html
    assert "apple-mobile-web-app-capable" in html
