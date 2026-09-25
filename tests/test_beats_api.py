"""拍の API・CLI・DB の移行（T10）。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from job_helpers import make_track, sync_launcher
from stemapp import cli
from stemapp.beats import FakeBeatAnalyzer, analyze_job_beats
from stemapp.config import Settings
from stemapp.db import init_db, make_engine
from stemapp.delivery import fake_encoder
from stemapp.jobs.worker import Worker
from stemapp.models import BeatGrid, SeparationJob
from test_api import _app


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(_app(settings)) as c:
        yield c


def _factory(client: TestClient) -> sessionmaker[Session]:
    return client.app.state.session_factory  # type: ignore[attr-defined]


def _worker(client: TestClient, analyzer: FakeBeatAnalyzer | None = None) -> Worker:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    factory = _factory(client)

    def runner(job_id: int, _stop) -> None:
        with factory() as s:
            analyze_job_beats(s, settings, job_id, FakeBeatAnalyzer([(0, 120), (4, 150)]))

    return Worker(
        settings, factory, sync_launcher(settings, beat_analyzer=analyzer),
        postprocess_encoder=fake_encoder, beat_runner=runner,
    )


def _track(client: TestClient, tmp_path: Path, seconds: float = 8.0) -> int:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _factory(client)() as s:
        return make_track(s, settings, tmp_path, seconds=seconds)


def _separate(client: TestClient, track_id: int, analyzer: FakeBeatAnalyzer) -> int:
    res = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"})
    job_id = res.json()["job"]["job_id"]
    assert _worker(client, analyzer).run_one() == job_id
    return job_id


def test_get_beats(client: TestClient, tmp_path: Path) -> None:
    track_id = _track(client, tmp_path)
    res = client.get(f"/api/tracks/{track_id}/beats")
    assert res.status_code == 404 and "まだ解析されていません" in res.json()["detail"]
    assert client.get("/api/tracks/9999/beats").status_code == 404

    _separate(client, track_id, FakeBeatAnalyzer([(0, 120), (4, 150)], beats_per_bar=3))
    res = client.get(f"/api/tracks/{track_id}/beats")
    assert res.status_code == 200
    body = res.json()
    assert body["time_signature"] == 3 and body["analyzer"] == "fake 1"
    assert body["beats"][:3] == [0.0, 0.5, 1.0] and body["downbeats"][:2] == [0.0, 1.5]
    assert body["segments"] == [
        {"start_sec": 0.0, "end_sec": 4.0, "bpm": 120.0},
        {"start_sec": 4.0, "end_sec": 7.6, "bpm": 150.0},
    ]


def test_beat_warning_in_job_and_stems(client: TestClient, tmp_path: Path) -> None:
    track_id = _track(client, tmp_path)
    job_id = _separate(client, track_id, FakeBeatAnalyzer(fail=True))
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done" and "拍を解析できませんでした" in job["beat_warning"]
    stems = client.get(f"/api/jobs/{job_id}/stems").json()
    assert stems["delivery_ready"] is True and stems["beat_warning"] == job["beat_warning"]
    assert client.get(f"/api/tracks/{track_id}/beats").status_code == 404

    # 作り直し（postprocess）で拍が無ければ拍を作る
    res = client.post(f"/api/jobs/{job_id}/postprocess")
    assert res.status_code == 202 and res.json()["missing"] == ["beats"]
    assert _worker(client).run_postprocess_one() == job_id
    assert client.get(f"/api/tracks/{track_id}/beats").status_code == 200
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["beat_warning"] is None and job["postprocess_status"] == "done"
    res = client.post(f"/api/jobs/{job_id}/postprocess")
    assert res.status_code == 200 and res.json()["reason"] == "ready"


def test_reanalyze_beats(client: TestClient, tmp_path: Path) -> None:
    track_id = _track(client, tmp_path)
    assert client.post(f"/api/tracks/{track_id}/beats").status_code == 409  # 未分割
    assert client.post("/api/tracks/9999/beats").status_code == 404
    job_id = _separate(client, track_id, FakeBeatAnalyzer(100))
    assert client.get(f"/api/tracks/{track_id}/beats").json()["segments"][0]["bpm"] == 100.0

    res = client.post(f"/api/tracks/{track_id}/beats")
    assert res.status_code == 202 and res.json()["created"] is True
    assert res.json()["job"]["postprocess_status"] == "queued"
    again = client.post(f"/api/tracks/{track_id}/beats")
    assert again.status_code == 200 and again.json()["reason"] == "active"
    assert client.get(f"/api/tracks/{track_id}/beats").status_code == 404  # 解析待ち
    assert _worker(client).run_postprocess_one() == job_id
    segs = client.get(f"/api/tracks/{track_id}/beats").json()["segments"]
    assert [s["bpm"] for s in segs] == [120.0, 150.0]


def test_reanalyze_while_postprocess_active_drops_beats(
    client: TestClient, tmp_path: Path
) -> None:
    """作り直しが作成待ちのときに再解析を頼むと、拍を消してから active を返す（依頼が生きる）。"""
    track_id = _track(client, tmp_path)
    job_id = _separate(client, track_id, FakeBeatAnalyzer(100))
    with _factory(client)() as s:
        s.get(SeparationJob, job_id).postprocess_status = "queued"
        s.commit()
    res = client.post(f"/api/tracks/{track_id}/beats")
    assert res.status_code == 200 and res.json()["reason"] == "active"
    assert client.get(f"/api/tracks/{track_id}/beats").status_code == 404
    assert _worker(client).run_postprocess_one() == job_id
    segs = client.get(f"/api/tracks/{track_id}/beats").json()["segments"]
    assert [s["bpm"] for s in segs] == [120.0, 150.0]


def test_track_delete_removes_beats(client: TestClient, tmp_path: Path) -> None:
    track_id = _track(client, tmp_path)
    _separate(client, track_id, FakeBeatAnalyzer())
    with _factory(client)() as s:
        assert s.get(BeatGrid, track_id) is not None
    assert client.delete(f"/api/tracks/{track_id}").status_code == 200
    with _factory(client)() as s:
        assert s.get(BeatGrid, track_id) is None


# --- CLI -------------------------------------------------------------------------------

runner = CliRunner()


def test_beats_command(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, session: Session, tmp_path: Path
) -> None:
    track_id = make_track(session, settings, tmp_path, seconds=8.0)
    fake = FakeBeatAnalyzer([(0, 120), (4, 150)])
    seen: list[bool] = []

    def make(_s: Settings, cpu: bool = False) -> FakeBeatAnalyzer:
        seen.append(cpu)
        return fake

    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)
    monkeypatch.setattr(cli, "make_beat_analyzer", make)

    res = runner.invoke(cli.app, ["beats", str(track_id)])
    assert res.exit_code == 0, res.output
    assert "120.00" in res.output and "150.00" in res.output and "4/4" in res.output
    assert "装置: cpu" in res.output
    res = runner.invoke(cli.app, ["beats", str(track_id)])
    assert res.exit_code == 0 and "解析済みです" in res.output and len(fake.calls) == 1
    res = runner.invoke(cli.app, ["beats", str(track_id), "--force", "--cpu"])
    assert res.exit_code == 0 and len(fake.calls) == 2 and seen == [False, False, True]
    res = runner.invoke(cli.app, ["beats", "9999"])
    assert res.exit_code == 1 and "曲が見つかりません" in res.output


# --- 移行 -------------------------------------------------------------------------------


def test_old_db_gets_beat_tables_and_column(tmp_path: Path) -> None:
    engine = make_engine(tmp_path / "old.db")
    try:
        init_db(engine)
        with engine.begin() as conn:
            # T05b までの DB を再現する
            conn.exec_driver_sql("DROP TABLE beat_anchor")
            conn.exec_driver_sql("DROP TABLE beat_grid")
            conn.exec_driver_sql("ALTER TABLE separation_job DROP COLUMN beat_warning")
            conn.exec_driver_sql(
                "INSERT INTO track (track_id, title, audio_hash, created_at) "
                "VALUES (1, '古い曲', 'h1', '2026-01-01 00:00:00')"
            )
            conn.exec_driver_sql(
                "INSERT INTO separation_job (job_id, track_id, job_kind, status, run_on, "
                "progress, created_at) VALUES (7, 1, 'full', 'done', 'gpu', 1.0, "
                "'2026-01-01 00:00:00')"
            )
        init_db(engine)
        insp = inspect(engine)
        assert {"beat_grid", "beat_anchor"} <= set(insp.get_table_names())
        assert "beat_warning" in {c["name"] for c in insp.get_columns("separation_job")}
        anchor_cols = {c["name"] for c in insp.get_columns("beat_anchor")}
        assert anchor_cols == {
            "anchor_id", "track_id", "position_sec", "kind", "bar_number", "bpm", "created_at",
        }
        with engine.connect() as conn:
            row = conn.execute(text("SELECT job_id, beat_warning FROM separation_job")).one()
            assert tuple(row) == (7, None)
        with Session(engine) as s:
            assert s.get(SeparationJob, 7).status == "done"
    finally:
        engine.dispose()
