"""DB 接続（SQLite + SQLAlchemy 2.0）。"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, Table, create_engine, event, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.schema import Column, CreateIndex, CreateTable

log = logging.getLogger(__name__)

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

    列の追加と、無い索引（index）の作成だけを扱う（型の変更・削除はしない）。
    足した列を「テーブル.列」で返す。
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
            for index in table.indexes:
                index.create(conn, checkfirst=True)
    return added


# --- 番号の使い回しを防ぐ（AUTOINCREMENT）への移行 ---------------------------------------------


class DbMigrationError(RuntimeError):
    """DB の移行に失敗した（元の状態に戻してある）。メッセージは日本語で、そのまま画面に出す。"""


def _db_file(engine: Engine) -> Path | None:
    """ファイルの DB ならそのパス（メモリ DB は None）。"""
    name = engine.url.database
    if not name or name == ":memory:":
        return None
    return Path(name)


def backup_dir_of(db_path: Path) -> Path:
    """DB のバックアップの置き場所（データフォルダの backup）。"""
    return db_path.parent / "backup"


def _backup(conn: sqlite3.Connection, db_path: Path) -> Path:
    """DB を `backup/stemapp-<日時>.db` に複製する（SQLite の backup API）。"""
    folder = backup_dir_of(db_path)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = folder / f"stemapp-{stamp}.db"
    n = 2
    while dst.exists():
        dst = folder / f"stemapp-{stamp}-{n}.db"
        n += 1
    try:
        _copy_db(conn, dst)
    except Exception as e:
        # 書きかけのバックアップは残さない（不完全なものを正しいバックアップと思わないように）
        for p in (dst, dst.with_name(dst.name + "-journal")):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise DbMigrationError(
            "DB の更新の前に作るバックアップを作れなかったため、DB には何もしていません"
            f"（{dst}）。\n原因: {e}\n"
            "データフォルダの空き容量と書き込み権限を確かめてから、もう一度起動してください。"
        ) from e
    return dst


def _copy_db(conn: sqlite3.Connection, dst: Path) -> None:
    """SQLite の backup API で dst に複製する。"""
    target = sqlite3.connect(dst)
    try:
        conn.backup(target)
    finally:
        target.close()


def _foreign_keys_on(conn: sqlite3.Connection) -> int:
    """今の接続の PRAGMA foreign_keys（0 = 止まっている）。"""
    return int(conn.execute("PRAGMA foreign_keys").fetchone()[0])


def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """全表の行数（作り直しの前後で比べる）。"""
    names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {n: int(conn.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0]) for n in names}


def _has_autoincrement(conn: sqlite3.Connection, table: str) -> bool | None:
    """表の定義に AUTOINCREMENT があるか。表が無ければ None。"""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    if row is None:
        return None
    return "AUTOINCREMENT" in str(row[0]).upper()


def autoincrement_tables() -> list[Table]:
    """番号を使い回さない（sqlite_autoincrement=True の）表。"""
    from stemapp import models  # noqa: F401  モデルを Base.metadata に登録する

    return [
        t for t in Base.metadata.sorted_tables if t.dialect_options["sqlite"].get("autoincrement")
    ]


def _fk_problems(conn: sqlite3.Connection) -> set[tuple[Any, ...]]:
    return {tuple(r) for r in conn.execute("PRAGMA foreign_key_check").fetchall()}


def _rebuild_table(conn: sqlite3.Connection, engine: Engine, table: Table) -> None:
    """表を新しい定義（AUTOINCREMENT つき）で作り直し、行・索引を移す。

    トランザクションの中で呼ぶ。SQLite は ALTER で AUTOINCREMENT を付けられないため、公式の手順
    （新しい表を作る → 行を移す → 古い表を消す → 名前を変える → 索引を作り直す）で行う。
    番号はそのまま移すので、外部キーは保たれる。モデルに無い列（古い版の列）も消さずに移す。

    注意: トリガー（TRIGGER）とビュー（VIEW）は作り直さない（DROP TABLE で表のトリガーは
    消える）。stemapp はどちらも使っていないため。使うようになったら、ここで退避して戻すこと。
    """
    name = table.name
    tmp = f"_new_{name}"
    ddl = str(CreateTable(table).compile(dialect=engine.dialect)).strip()
    ddl, n = re.subn(rf'^CREATE TABLE "?{re.escape(name)}"? \(', f'CREATE TABLE "{tmp}" (', ddl)
    if n != 1:
        raise RuntimeError(f"{name} の定義を作れませんでした。")
    old_indexes = [
        sql
        for (sql,) in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND tbl_name = ? "
            "AND sql IS NOT NULL",
            (name,),
        )
    ]
    old_cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
    conn.execute(f'DROP TABLE IF EXISTS "{tmp}"')
    conn.execute(ddl)
    new_names = {r[1] for r in conn.execute(f'PRAGMA table_info("{tmp}")')}
    for _cid, col, col_type, _notnull, default, _pk in old_cols:
        if col in new_names:
            continue
        # モデルに無い列: 型と既定値だけ付けて残す（NOT NULL は付けない）
        extra = f'"{col}" {col_type or ""}'.rstrip()
        if default is not None:
            extra += f" DEFAULT {default}"
        conn.execute(f'ALTER TABLE "{tmp}" ADD COLUMN {extra}')
    cols = ", ".join(f'"{r[1]}"' for r in old_cols)
    conn.execute(f'INSERT INTO "{tmp}" ({cols}) SELECT {cols} FROM "{name}"')
    conn.execute(f'DROP TABLE "{name}"')
    conn.execute(f'ALTER TABLE "{tmp}" RENAME TO "{name}"')
    for sql in old_indexes:
        conn.execute(sql)
    for index in table.indexes:
        conn.execute(str(CreateIndex(index, if_not_exists=True).compile(dialect=engine.dialect)))


# テスト用: 表を1つ作り直すたびに表の名前を渡して呼ぶ（途中で失敗させ、元に戻ることを確かめる）
after_rebuild_hook: Callable[[str], None] | None = None


def migrate_autoincrement(engine: Engine) -> list[str]:
    """番号を使い回さない表のうち、古い定義（AUTOINCREMENT なし）のものを作り直す。

    作り直す表があれば、先に DB を `backup/stemapp-<日時>.db` に複製する。すべての表を1つの
    トランザクションで作り直し、外部キーの確認（PRAGMA foreign_key_check）で新しい問題が出たら、
    または途中で失敗したら、元に戻して DbMigrationError を出す。作り直した表の名前を返す
    （何度実行しても同じ結果。2回目からは何もしない）。
    """
    tables = autoincrement_tables()
    raw = engine.raw_connection()
    try:
        conn = raw.driver_connection
        assert isinstance(conn, sqlite3.Connection)
        old_isolation = conn.isolation_level
        conn.isolation_level = None  # BEGIN / COMMIT を自分で出す
        try:
            todo = [t for t in tables if _has_autoincrement(conn, t.name) is False]
            if not todo:
                return []
            db_path = _db_file(engine)
            backup = _backup(conn, db_path) if db_path is not None else None
            log.warning(
                "DB を更新します（番号の使い回しを防ぐ: %s）。バックアップ: %s",
                ", ".join(t.name for t in todo), backup,
            )
            # 表を消して作り直す間は外部キーを止める（止めないと DROP TABLE が CASCADE で
            # 子の行を消す）。PRAGMA foreign_keys はトランザクションの外でしか変えられない
            conn.execute("PRAGMA foreign_keys=OFF")
            try:
                if _foreign_keys_on(conn) != 0:
                    # 外部キーが止まらない（トランザクションの中など）と、DROP TABLE が子の行を
                    # 消してしまう。何もせずに止める
                    raise DbMigrationError(
                        "DB の更新（番号の使い回しを防ぐ AUTOINCREMENT への作り直し）の準備で、"
                        "外部キーの確認を止められなかったため、DB には何もしていません"
                        f"（バックアップ: {backup}）。\n"
                        "ほかに stemapp が動いていないかを確かめてから、もう一度起動してください。"
                    )
                conn.execute("BEGIN IMMEDIATE")
                try:
                    # 別のプロセスが先に済ませていないか、ロックを取ってから確かめ直す
                    todo = [t for t in todo if _has_autoincrement(conn, t.name) is False]
                    before = _fk_problems(conn)
                    counts_before = _row_counts(conn)
                    for table in todo:
                        _rebuild_table(conn, engine, table)
                        if after_rebuild_hook is not None:
                            after_rebuild_hook(table.name)
                    new_problems = _fk_problems(conn) - before
                    if new_problems:
                        raise RuntimeError(f"外部キーの確認で問題が見つかりました: {new_problems}")
                    counts_after = _row_counts(conn)
                    if counts_after != counts_before:
                        diff = {
                            n: (counts_before.get(n), counts_after.get(n))
                            for n in set(counts_before) | set(counts_after)
                            if counts_before.get(n) != counts_after.get(n)
                        }
                        raise RuntimeError(f"作り直しの前後で行の数が違います: {diff}")
                    conn.execute("COMMIT")
                except BaseException as e:
                    conn.execute("ROLLBACK")
                    where = f"（バックアップ: {backup}）" if backup is not None else ""
                    raise DbMigrationError(
                        "DB の更新（番号の使い回しを防ぐ AUTOINCREMENT への作り直し）に"
                        f"失敗したため、元の状態に戻しました{where}。\n原因: {e}\n"
                        "データフォルダの空き容量と、ほかに stemapp が動いていないかを"
                        "確かめてから、もう一度起動してください。"
                        "直らないときはこの表示を開発者に伝えてください。"
                    ) from e
            finally:
                conn.execute("PRAGMA foreign_keys=ON")
            done = [t.name for t in todo]
            if done:
                log.warning("DB を更新しました（%s）。", ", ".join(done))
            return done
        finally:
            conn.isolation_level = old_isolation
    finally:
        raw.close()


def init_db(engine: Engine) -> None:
    """全テーブルを作成し（既にあれば何もしない）、足りない列を追加し、
    番号を使い回さない表（AUTOINCREMENT）への作り直しを行う。

    作り直しに失敗したら元に戻して DbMigrationError（起動を止め、メッセージを表示する）。
    """
    from stemapp import models  # noqa: F401  モデルを Base.metadata に登録する

    Base.metadata.create_all(engine)
    migrate_db(engine)
    migrate_autoincrement(engine)
