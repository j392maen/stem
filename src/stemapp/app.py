"""FastAPI アプリ。

分割ワーカーは別プロセス（`stemapp serve` が一緒に起動する）。このアプリは REST API と、
取り込みを実行するスレッド（ImportManager）を持つ。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from stemapp import __version__
from stemapp.api import auth, cues, files, imports, master, tracks
from stemapp.api.imports import ImportDeps, ImportManager
from stemapp.config import Settings, get_settings
from stemapp.db import init_db, make_engine, make_session_factory
from stemapp.ingest.service import recover_interrupted_imports
from stemapp.seed import seed

WEB_DIR: Path = Path(__file__).resolve().parent / "web"

# Starlette の既定の英語メッセージを日本語にする
_DEFAULT_DETAILS: dict[int, str] = {
    404: "見つかりません。",
    405: "この操作には対応していません。",
}


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        if not isinstance(detail, str) or detail.isascii():
            detail = _DEFAULT_DETAILS.get(exc.status_code, str(detail))
        return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        reasons = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
            parts.append(loc or "本文")
            # 自前の検証（model_validator）の日本語メッセージはそのまま添える
            msg = str(err.get("msg", "")).removeprefix("Value error, ")
            if msg and not msg.isascii() and msg not in reasons:
                reasons.append(msg)
        where = "、".join(parts) if parts else "本文"
        detail = f"リクエストの内容が正しくありません（{where}）。" + "".join(reasons)
        return JSONResponse({"detail": detail}, status_code=422)


class WebFiles(StaticFiles):
    """画面（src/stemapp/web）の配信。/api 以下は扱わない。更新がすぐ効くよう毎回確認させる。"""

    async def get_response(self, path: str, scope: Scope) -> Response:
        if path == "api" or path.startswith("api/"):
            raise StarletteHTTPException(status_code=404)
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(
    settings: Settings | None = None, *, import_deps: ImportDeps | None = None
) -> FastAPI:
    """アプリを作る。

    起動時に DB 作成・列の追加・初期データ投入と、中断された取り込みの片付けを行う。
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.db_path)
        init_db(engine)
        session_factory = make_session_factory(engine)
        with session_factory() as session:
            seed(session)
            recover_interrupted_imports(session)
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.secret_key = auth.load_or_create_secret(settings.data_root / "secret.key")
        manager = ImportManager(settings, session_factory, import_deps or ImportDeps())
        manager.start()
        app.state.import_manager = manager
        try:
            yield
        finally:
            manager.stop()
            engine.dispose()

    # パスコードを設定しているときは API の説明ページ（/docs 等）を出さない
    docs_enabled = auth.passcode_of(settings) is None
    app = FastAPI(
        title="stemapp",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.settings = settings
    app.state.login_limiter = auth.LoginLimiter()
    app.middleware("http")(auth.auth_middleware)
    _install_error_handlers(app)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    for module in (auth, imports, tracks, cues, files, master):
        app.include_router(module.router)
    # 画面。API のルートより後に登録する（/api/* はここまで来ない）
    app.mount("/", WebFiles(directory=WEB_DIR, html=True), name="web")
    return app
