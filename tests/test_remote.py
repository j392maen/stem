"""外出先（Tailscale Serve）から使うための安全策と PWA の配信。

- Host ヘッダーの許可リスト（DNS リバインディング対策）
- HTTPS（X-Forwarded-Proto: https）で来たときの Secure Cookie
- 中継経由でパスコードが無いときの「パスコードの設定を勧める」
- manifest・Service Worker・アイコンの配信
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from stemapp.config import Settings
from stemapp.hosts import normalize_host
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
    ],
)
def test_normalize_host(value: str, expected: str) -> None:
    assert normalize_host(value) == expected


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
