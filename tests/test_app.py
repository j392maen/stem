from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from stemapp import __version__
from stemapp.app import create_app
from stemapp.config import Settings
from stemapp.models import StemType


def test_health(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "version": __version__}


def test_startup_creates_db_and_seeds(settings: Settings) -> None:
    app = create_app(settings)
    with TestClient(app):
        assert settings.db_path.is_file()
        with app.state.session_factory() as s:
            assert s.scalar(select(func.count()).select_from(StemType))
