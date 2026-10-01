"""番号の使い回しを防ぐ（AUTOINCREMENT）への移行（`stemapp.db.migrate_autoincrement`）。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, MetaData

from stemapp import db as dbmod
from stemapp.db import (
    Base,
    DbMigrationError,
    autoincrement_tables,
    init_db,
    make_engine,
    make_session_factory,
    migrate_autoincrement,
)
from stemapp.models import SeparationJob

TARGETS = {
    "track", "input_source", "separation_job", "stem", "export", "tempo_render", "beat_edit",
}
T0 = "2026-01-01 00:00:00"


def _old_schema(engine: Engine) -> None:
    """AUTOINCREMENT の無い（T14 より前の）定義で全テーブルを作る。"""
    from stemapp import models  # noqa: F401

    md = MetaData()
    for t in Base.metadata.sorted_tables:
        t.to_metadata(md)
    for t in md.tables.values():
        t.dialect_kwargs["sqlite_autoincrement"] = False
    md.create_all(engine)


def _fill(engine: Engine) -> None:
    rows = [
        "INSERT INTO track (track_id, title, audio_hash, created_at) VALUES (1, '曲A', 'h1', '%s')",
        "INSERT INTO track (track_id, title, audio_hash, created_at) VALUES (2, '曲B', 'h2', '%s')",
        "INSERT INTO input_source (source_id, track_id, source_type) VALUES (5, 1, 'file')",
        "INSERT INTO stem_type (stem_type_id, code, display_name, tier, experimental, color, "
        "display_order) VALUES (1, 'vocals', 'ボーカル', 'base', 0, '#ff0000', 0)",
        "INSERT INTO stem_type (stem_type_id, code, display_name, tier, experimental, color, "
        "display_order) VALUES (2, 'lead', 'リード', 'detail', 0, '#ff0000', 1)",
        "INSERT INTO separation_job (job_id, track_id, job_kind, status, run_on, progress, "
        "created_at) VALUES (3, 1, 'full', 'done', 'gpu', 1.0, '%s')",
        "INSERT INTO separation_job (job_id, track_id, job_kind, status, run_on, progress, "
        "created_at) VALUES (9, 2, 'full', 'done', 'gpu', 1.0, '%s')",
        "INSERT INTO stem (stem_id, job_id, stem_type_id, is_residual, is_silent) "
        "VALUES (10, 3, 1, 0, 0)",
        "INSERT INTO stem (stem_id, job_id, stem_type_id, parent_stem_id, is_residual, is_silent)"
        " VALUES (11, 3, 2, 10, 0, 0)",
        "INSERT INTO separation_job (job_id, track_id, job_kind, input_stem_id, status, run_on, "
        "progress, created_at) VALUES (12, 1, 'refine', 10, 'done', 'gpu', 1.0, '%s')",
        "INSERT INTO stem_rendition (rendition_id, stem_id, purpose, codec, file_path) "
        "VALUES (1, 10, 'master', 'flac', 'x.flac')",
        "INSERT INTO export (export_id, job_id, export_type, format, created_at) "
        "VALUES (4, 3, 'single', 'wav', '%s')",
        "INSERT INTO export_item (export_id, stem_id, gain_db) VALUES (4, 10, 0.0)",
        "INSERT INTO tempo_render (render_id, job_id, ratio, pitch_mode, status, progress, "
        "cancel_requested, created_at, last_used_at) "
        "VALUES (6, 3, 1.1, 'keep', 'done', 1.0, 0, '%s', '%s')",
        "INSERT INTO tempo_rendition (render_id, stem_id, codec, file_path) "
        "VALUES (6, 11, 'opus', 'y.webm')",
        "INSERT INTO beat_edit (edit_id, track_id, op, created_at) VALUES (8, 1, 'reset', '%s')",
    ]
    with engine.begin() as conn:
        for sql in rows:
            conn.exec_driver_sql(sql.replace("'%s'", f"'{T0}'"))


def _snapshot(path: Path) -> dict[str, Any]:
    """表ごとの行・外部キー・索引（比べる用）。"""
    con = sqlite3.connect(path)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        snap: dict[str, Any] = {}
        for t in tables:
            snap[t] = {
                "rows": sorted(con.execute(f'SELECT * FROM "{t}"').fetchall(), key=repr),
                # (参照先の表, 列, 参照先の列, ON UPDATE, ON DELETE, MATCH)。番号の振り方は除く
                "fks": sorted(
                    r[2:] for r in con.execute(f'PRAGMA foreign_key_list("{t}")').fetchall()
                ),
                "indexes": sorted(
                    (r[1], r[2]) for r in con.execute(f'PRAGMA index_list("{t}")').fetchall()
                ),
                "cols": [r[1:] for r in con.execute(f'PRAGMA table_info("{t}")').fetchall()],
            }
        return snap
    finally:
        con.close()


def _schema_sql(path: Path, table: str) -> str:
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def old_db(tmp_path: Path) -> Iterator[tuple[Path, Engine]]:
    path = tmp_path / "data" / "stemapp.db"
    engine = make_engine(path)
    _old_schema(engine)
    _fill(engine)
    yield path, engine
    engine.dispose()


def test_targets_are_marked() -> None:
    assert {t.name for t in autoincrement_tables()} == TARGETS


def test_new_db_has_autoincrement_and_does_not_reuse_ids(tmp_path: Path) -> None:
    path = tmp_path / "new.db"
    engine = make_engine(path)
    try:
        init_db(engine)
        for t in TARGETS:
            assert "AUTOINCREMENT" in _schema_sql(path, t).upper(), t
        assert migrate_autoincrement(engine) == []
        assert not (tmp_path / "backup").exists()  # 新しい DB はバックアップを作らない
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "INSERT INTO track (title, audio_hash, created_at) VALUES ('a', 'h', '2026-01-01')"
            )
            conn.exec_driver_sql("DELETE FROM track")
            conn.exec_driver_sql(
                "INSERT INTO track (title, audio_hash, created_at) VALUES ('b', 'h', '2026-01-01')"
            )
            assert conn.exec_driver_sql("SELECT track_id FROM track").scalar() == 2
    finally:
        engine.dispose()


def test_old_db_is_rebuilt_keeping_rows_fks_and_indexes(old_db: tuple[Path, Engine]) -> None:
    path, engine = old_db
    for t in TARGETS:
        assert "AUTOINCREMENT" not in _schema_sql(path, t).upper()
    before = _snapshot(path)

    init_db(engine)

    after = _snapshot(path)
    for t in TARGETS:
        assert "AUTOINCREMENT" in _schema_sql(path, t).upper(), t
    assert set(after) == set(before)
    for t in before:
        assert after[t]["rows"] == before[t]["rows"], t
        assert after[t]["fks"] == before[t]["fks"], t
        assert after[t]["indexes"] == before[t]["indexes"], t
        assert after[t]["cols"] == before[t]["cols"], t
    assert not any(t.startswith("_new_") for t in after)
    with engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert conn.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar() == "ok"

    # バックアップ（移行前の DB）ができている
    backups = list((path.parent / "backup").glob("stemapp-*.db"))
    assert len(backups) == 1
    assert "AUTOINCREMENT" not in _schema_sql(backups[0], "track").upper()
    assert _snapshot(backups[0])["stem"]["rows"] == before["stem"]["rows"]

    # 2回目は何もしない（バックアップも増えない）
    assert migrate_autoincrement(engine) == []
    init_db(engine)
    assert len(list((path.parent / "backup").glob("stemapp-*.db"))) == 1
    assert _snapshot(path) == after


def test_after_migration_max_id_is_not_reused_and_cascade_works(
    old_db: tuple[Path, Engine],
) -> None:
    path, engine = old_db
    init_db(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        # いちばん大きい番号のジョブ（12）を消しても、次は 13
        s.delete(s.get(SeparationJob, 12))
        s.commit()
        job = SeparationJob(track_id=2, job_kind="full", status="queued")
        s.add(job)
        s.commit()
        assert job.job_id == 13
    with engine.begin() as conn:
        # 外部キーの CASCADE は作り直した後も効く（曲を消すとジョブ・stem・書き出しも消える）
        conn.exec_driver_sql("DELETE FROM track WHERE track_id = 1")
        assert conn.exec_driver_sql("SELECT count(*) FROM separation_job").scalar() == 2
        assert conn.exec_driver_sql("SELECT count(*) FROM stem").scalar() == 0
        assert conn.exec_driver_sql("SELECT count(*) FROM export_item").scalar() == 0
        assert conn.exec_driver_sql("SELECT count(*) FROM tempo_rendition").scalar() == 0
        assert conn.exec_driver_sql("SELECT count(*) FROM beat_edit").scalar() == 0
        # 消した曲の番号（1）も使い直さない（曲は 2 が最大だった）
        conn.exec_driver_sql(
            "INSERT INTO track (title, audio_hash, created_at) VALUES ('c', 'h3', '2026-01-01')"
        )
        assert conn.exec_driver_sql("SELECT max(track_id) FROM track").scalar() == 3


def test_failure_rolls_back_everything(
    old_db: tuple[Path, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, engine = old_db
    before = _snapshot(path)

    def boom(name: str) -> None:
        if name == "stem":
            raise OSError("ディスクがいっぱいです（テスト）")

    monkeypatch.setattr(dbmod, "after_rebuild_hook", boom)
    with pytest.raises(DbMigrationError) as ei:
        init_db(engine)
    msg = str(ei.value)
    assert "元の状態に戻しました" in msg and "ディスクがいっぱい" in msg and "backup" in msg
    # どの表も作り直されていない（途中まで作り直した表も戻っている）
    assert _snapshot(path) == before
    for t in TARGETS:
        assert "AUTOINCREMENT" not in _schema_sql(path, t).upper()
    with engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1

    # 原因を取り除けば、次の起動で移行できる
    monkeypatch.setattr(dbmod, "after_rebuild_hook", None)
    init_db(engine)
    for t in TARGETS:
        assert "AUTOINCREMENT" in _schema_sql(path, t).upper()


def test_new_fk_problem_rolls_back(
    old_db: tuple[Path, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, engine = old_db
    before = _snapshot(path)

    # 外部キーの確認で新しい問題が出たら元に戻す: 確認の関数を差し替えて問題を作る
    calls = {"n": 0}
    orig = dbmod._fk_problems

    def fake_fk(conn: sqlite3.Connection) -> set[tuple[Any, ...]]:
        calls["n"] += 1
        found = orig(conn)
        return found if calls["n"] == 1 else found | {("stem", 99, "separation_job", 0)}

    monkeypatch.setattr(dbmod, "_fk_problems", fake_fk)
    with pytest.raises(DbMigrationError, match="外部キー"):
        init_db(engine)
    assert _snapshot(path) == before


def test_extra_old_column_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    engine = make_engine(path)
    try:
        _old_schema(engine)
        _fill(engine)
        with engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE track ADD COLUMN legacy_note TEXT DEFAULT 'x'")
            conn.exec_driver_sql("UPDATE track SET legacy_note = '残す' WHERE track_id = 2")
        init_db(engine)
        with engine.connect() as conn:
            assert (
                conn.exec_driver_sql("SELECT legacy_note FROM track WHERE track_id = 2").scalar()
                == "残す"
            )
        assert "AUTOINCREMENT" in _schema_sql(path, "track").upper()
    finally:
        engine.dispose()


def test_cli_stops_with_japanese_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from stemapp import cli

    def fail(_engine: Engine) -> None:
        raise DbMigrationError("DB の更新に失敗したため、元の状態に戻しました。")

    monkeypatch.setenv("STEMAPP_DATA_DIR", str(tmp_path / "data"))
    cli.get_settings.cache_clear()
    monkeypatch.setattr(dbmod, "init_db", fail)
    monkeypatch.setattr("sys.argv", ["stemapp", "init-db"])
    try:
        with pytest.raises(SystemExit) as ei:
            cli.main()
    finally:
        cli.get_settings.cache_clear()
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "起動できません" in err and "元の状態に戻しました" in err
