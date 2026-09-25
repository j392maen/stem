"""DB 接続（SQLite + SQLAlchemy 2.0）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """全モデルの基底クラス。"""


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(db_path: Path | str) -> Engine:
    """SQLite エンジンを作る。接続ごとに外部キー制約を有効にする。

    db_path に ":memory:" を渡すとメモリ DB。
    """
    if str(db_path) == ":memory:":
        url = "sqlite://"
    else:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path.resolve().as_posix()}"
    engine = create_engine(url)
    event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """全テーブルを作成する（既にあれば何もしない）。"""
    from stemapp import models  # noqa: F401  モデルを Base.metadata に登録する

    Base.metadata.create_all(engine)
