"""簡易パスコード認証。

- 設定 `passcode` が空なら認証なし。
- 設定されていれば `POST /api/login` で署名付き Cookie（HttpOnly、SameSite=Lax、30日）を発行し、
  `/api/health` と `/api/login` 以外の `/api/*` は有効な Cookie が無いと 401。
- Cookie の中身は「有効期限.乱数.署名」。署名は HMAC-SHA256（鍵は `data/secret.key`）で、
  パスコードのハッシュも混ぜるので、パスコードを変えると発行済みの Cookie は無効になる。
- ログインの失敗は、同じ接続元で1分に5回まで（超えると 429）。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from stemapp.api.common import is_local_request
from stemapp.api.folders import supports_open_folder
from stemapp.config import Settings

COOKIE_NAME = "stemapp_session"
SESSION_MAX_AGE_SEC = 30 * 24 * 60 * 60
PUBLIC_PATHS: frozenset[str] = frozenset({"/api/health", "/api/login"})
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SEC = 60.0

MSG_LOGIN_REQUIRED = "ログインしてください。"
MSG_WRONG_PASSCODE = "パスコードが違います。"
MSG_TOO_MANY = (
    "ログインの失敗が続いたため、しばらく受け付けません。1分ほど待ってから再度お試しください。"
)


def passcode_of(settings: Settings) -> str | None:
    code = settings.passcode
    return code if code else None


def load_or_create_secret(path: Path) -> bytes:
    """署名用の秘密鍵。無ければランダムに作って保存する。"""
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if len(text) >= 32:
            return bytes.fromhex(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    path.write_text(key + "\n", encoding="utf-8")
    return bytes.fromhex(key)


def _signature(secret: bytes, passcode: str, payload: str) -> str:
    pass_hash = hashlib.sha256(passcode.encode("utf-8")).hexdigest()
    msg = f"{payload}|{pass_hash}".encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def make_token(secret: bytes, passcode: str, now: float | None = None) -> str:
    expires = int((now if now is not None else time.time()) + SESSION_MAX_AGE_SEC)
    payload = f"{expires}.{secrets.token_hex(8)}"
    return f"{payload}.{_signature(secret, passcode, payload)}"


def verify_token(secret: bytes, passcode: str, token: str | None, now: float | None = None) -> bool:
    if not token or not token.isascii():  # compare_digest は非 ASCII の str を受け付けない
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    expires_s, nonce, sig = parts
    payload = f"{expires_s}.{nonce}"
    if not hmac.compare_digest(sig, _signature(secret, passcode, payload)):
        return False
    try:
        expires = int(expires_s)
    except ValueError:
        return False
    return expires > (now if now is not None else time.time())


class LoginLimiter:
    """接続元ごとのログイン失敗回数（直近 window 秒）を数える。"""

    def __init__(
        self, max_failures: int = LOGIN_MAX_FAILURES, window_sec: float = LOGIN_WINDOW_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_failures = max_failures
        self.window_sec = window_sec
        self.clock = clock
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str) -> deque[float]:
        q = self._failures.setdefault(key, deque())
        limit = self.clock() - self.window_sec
        while q and q[0] <= limit:
            q.popleft()
        return q

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key)) >= self.max_failures

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._recent(key).append(self.clock())

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


def is_authenticated(request: Request) -> bool:
    settings: Settings = request.app.state.settings
    passcode = passcode_of(settings)
    if passcode is None:
        return True
    return verify_token(request.app.state.secret_key, passcode, request.cookies.get(COOKIE_NAME))


async def auth_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    path = request.url.path
    if path.startswith("/api/") and path not in PUBLIC_PATHS and not is_authenticated(request):
        return JSONResponse({"detail": MSG_LOGIN_REQUIRED}, status_code=401)
    return await call_next(request)


class LoginRequest(BaseModel):
    passcode: str = ""


router = APIRouter(prefix="/api", tags=["auth"])


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict[str, bool]:
    settings: Settings = request.app.state.settings
    passcode = passcode_of(settings)
    if passcode is None:
        return {"authenticated": True, "passcode_required": False}
    limiter: LoginLimiter = request.app.state.login_limiter
    key = _client_key(request)
    if limiter.is_blocked(key):
        raise HTTPException(status_code=429, detail=MSG_TOO_MANY)
    if not hmac.compare_digest(body.passcode.encode("utf-8"), passcode.encode("utf-8")):
        limiter.record_failure(key)
        raise HTTPException(status_code=401, detail=MSG_WRONG_PASSCODE)
    limiter.reset(key)
    response.set_cookie(
        COOKIE_NAME,
        make_token(request.app.state.secret_key, passcode),
        max_age=SESSION_MAX_AGE_SEC,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return {"authenticated": True, "passcode_required": True}


@router.post("/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax")
    return {"authenticated": False}


@router.get("/me")
def me(request: Request) -> dict[str, bool]:
    # ここに来た時点で認証済み（未ログインならミドルウェアが 401 を返す）
    local = is_local_request(request)
    return {
        "authenticated": True,
        "passcode_required": passcode_of(request.app.state.settings) is not None,
        # サーバーと同じ PC のブラウザか（保存フォルダを開くボタンを出すかどうか）
        "local_client": local,
        "can_open_folder": local and supports_open_folder(),
    }
