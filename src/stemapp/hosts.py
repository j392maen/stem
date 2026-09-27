"""Host ヘッダーの許可リスト（DNS リバインディング対策）。

悪意のあるサイトが自分のドメイン名を 127.0.0.1 に向け直すと、ブラウザはそのサイトのページから
この PC の stemapp を「同じサイト」として呼べてしまう（DNS リバインディング）。
そのとき Host ヘッダーは相手のドメイン名になるので、知らない名前は 400 で断る。

常に許可: 127.0.0.1・localhost・[::1]。それ以外（Tailscale の名前など）は設定
`STEMAPP_ALLOWED_HOSTS`（カンマ区切り）で足す。
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Receive, Scope, Send

ALWAYS_ALLOWED: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

MSG_BAD_HOST = (
    "この名前（Host）からの接続は受け付けていません。"
    "Tailscale などの名前で開くときは、.env の STEMAPP_ALLOWED_HOSTS に名前を足してください。"
)


def normalize_host(value: str) -> str:
    """Host ヘッダーや設定の値を「小文字・ポートなし・IPv6 は角かっこなし」にする。

    形が正しくなければ空文字。
    """
    value = value.strip().lower()
    if not value:
        return ""
    if value.startswith("["):
        end = value.find("]")
        if end < 0:
            return ""
        name, rest = value[1:end], value[end + 1 :]
        if rest and not (rest.startswith(":") and rest[1:].isdigit()):
            return ""
        return name
    if value.count(":") == 1:
        name, port = value.split(":")
        if not port.isdigit():
            return ""
        value = name
    elif value.count(":") > 1:
        return value  # 角かっこの無い IPv6（設定に書かれた "::1" など）
    return value.rstrip(".")


class HostCheckMiddleware:
    """許可リストに無い Host（無い場合も）のリクエストを 400 にする ASGI ミドルウェア。"""

    def __init__(self, app: ASGIApp, allowed: frozenset[str]) -> None:
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        host = ""
        for key, val in scope.get("headers", []):
            if key == b"host":
                host = normalize_host(val.decode("latin-1"))
                break
        if host and host in self.allowed:
            await self.app(scope, receive, send)
            return
        body = json.dumps({"detail": MSG_BAD_HOST}, ensure_ascii=False).encode("utf-8")
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await send({
            "type": "http.response.start",
            "status": 400,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
