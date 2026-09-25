"""FastAPI アプリ。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from stemapp import __version__
from stemapp.config import Settings, get_settings
from stemapp.db import init_db, make_engine, make_session_factory
from stemapp.seed import seed


def create_app(settings: Settings | None = None) -> FastAPI:
    """アプリを作る。起動時に DB 作成と初期データ投入を行う。"""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.db_path)
        init_db(engine)
        session_factory = make_session_factory(engine)
        with session_factory() as session:
            seed(session)
        app.state.engine = engine
        app.state.session_factory = session_factory
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="stemapp", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app
