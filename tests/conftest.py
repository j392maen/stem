from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from stemapp.config import Settings
from stemapp.db import init_db, make_engine, make_session_factory


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """PC の STEMAPP_* 環境変数がテストに混ざらないようにする。"""
    for key in list(os.environ):
        if key.upper().startswith("STEMAPP_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """一時ディレクトリを使う設定（.env は読まない）。"""
    return Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]


@pytest.fixture
def engine(settings: Settings) -> Iterator[Engine]:
    eng = make_engine(settings.db_path)
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with make_session_factory(engine)() as s:
        yield s
