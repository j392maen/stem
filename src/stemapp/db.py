"""DB 接続（SQLite + SQLAlchemy 2.0）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.schema import Column

# 他のプロセス（Web サーバー・ワーカー・分割の子プロセス）が書き込み中のとき待つ秒数
BUSY_TIMEOUT_SEC = 30.0


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
    engine = create_engine(url, connect_args={"timeout": BUSY_TIMEOUT_SEC})
    event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def _column_ddl(engine: Engine, column: Column[object]) -> str:
    """`ALTER TABLE ... ADD COLUMN` に続ける列の定義。"""
    col_type = column.type.compile(dialect=engine.dialect)
    ddl = f'"{column.name}" {col_type}'
    default = column.server_default
    if default is not None:
        arg = getattr(default, "arg", None)
        value = getattr(arg, "text", arg)
        ddl += f" DEFAULT {value}"
    if not column.nullable:
        if default is None:
            raise RuntimeError(
                f"{column.table.name}.{column.name} は NOT NULL で既定値が無いため、"
                "既存の DB に列を追加できません。"
            )
        ddl += " NOT NULL"
    return ddl


def migrate_db(engine: Engine) -> list[str]:
    """既存のテーブルに無い列を `ALTER TABLE ... ADD COLUMN` で足す（簡易的な移行）。

    列の追加だけを扱う（型の変更・削除はしない）。足した列を「テーブル.列」で返す。
    """
    added: list[str] = []
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for column in table.columns:
                if column.name in have:
                    continue
                conn.exec_driver_sql(
                    f'ALTER TABLE "{table.name}" ADD COLUMN {_column_ddl(engine, column)}'
                )
                added.append(f"{table.name}.{column.name}")
    return added


def init_db(engine: Engine) -> None:
    """全テーブルを作成し（既にあれば何もしない）、足りない列を追加する。"""
    from stemapp import models  # noqa: F401  モデルを Base.metadata に登録する

    Base.metadata.create_all(engine)
    migrate_db(engine)
