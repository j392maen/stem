"""init_db の簡易移行（足りない列を ALTER TABLE で追加する）。"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, text

from stemapp.db import init_db, make_engine, migrate_db


def test_old_db_gets_new_columns_and_keeps_rows(tmp_path: Path) -> None:
    engine = make_engine(tmp_path / "old.db")
    try:
        init_db(engine)
        with engine.begin() as conn:
            # T03 までの DB を再現する（T04 で足した2列が無い）
            conn.exec_driver_sql("ALTER TABLE separation_job DROP COLUMN cancel_requested")
            conn.exec_driver_sql("ALTER TABLE separation_job DROP COLUMN output_gain_db")
            conn.exec_driver_sql("ALTER TABLE separation_job DROP COLUMN postprocess_status")
            conn.exec_driver_sql("DROP INDEX ux_listen_preset_seed_code")
            conn.exec_driver_sql("ALTER TABLE listen_preset DROP COLUMN seed_code")
            conn.exec_driver_sql("ALTER TABLE listen_preset DROP COLUMN hidden")
            conn.exec_driver_sql(
                "INSERT INTO track (track_id, title, audio_hash, created_at) "
                "VALUES (1, '古い曲', 'h1', '2026-01-01 00:00:00')"
            )
            conn.exec_driver_sql(
                "INSERT INTO separation_job (job_id, track_id, job_kind, status, run_on, "
                "progress, created_at) VALUES (7, 1, 'full', 'done', 'gpu', 1.0, "
                "'2026-01-01 00:00:00')"
            )
        cols = {c["name"] for c in inspect(engine).get_columns("separation_job")}
        assert "cancel_requested" not in cols and "output_gain_db" not in cols

        init_db(engine)

        cols = {c["name"] for c in inspect(engine).get_columns("separation_job")}
        assert {"cancel_requested", "output_gain_db", "postprocess_status"} <= cols
        lp_cols = {c["name"] for c in inspect(engine).get_columns("listen_preset")}
        assert {"seed_code", "hidden"} <= lp_cols
        indexes = {i["name"]: i for i in inspect(engine).get_indexes("listen_preset")}
        assert indexes["ux_listen_preset_seed_code"]["unique"]
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT job_id, status, cancel_requested, output_gain_db, postprocess_status "
                    "FROM separation_job"
                )
            ).one()
            assert tuple(row) == (7, "done", 0, 0.0, None)
            assert conn.execute(text("SELECT title FROM track")).scalar() == "古い曲"
        # 2回目は何もしない
        assert migrate_db(engine) == []
    finally:
        engine.dispose()
