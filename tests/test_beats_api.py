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


def test_reanalyze_with_job_id_puts_warning_on_that_job(
    client: TestClient, tmp_path: Path
) -> None:
    """job_id を渡すと、そのジョブ（表示中の分け方）で解析し、失敗の警告もそこに付く。"""
    track_id = _track(client, tmp_path)
    first = _separate(client, track_id, FakeBeatAnalyzer(100))
    res = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "standard"})
    newest = res.json()["job"]["job_id"]
    assert _worker(client).run_one() == newest
    assert newest > first

    # 指定の誤り
    assert client.post(f"/api/tracks/{track_id}/beats", json={"job_id": 9999}).status_code == 404
    other = _track(client, tmp_path / "o", seconds=4.0)
    other_job = _separate(client, other, FakeBeatAnalyzer(90))
    assert client.post(
        f"/api/tracks/{track_id}/beats", json={"job_id": other_job}
    ).status_code == 404

    res = client.post(f"/api/tracks/{track_id}/beats", json={"job_id": first})
    assert res.status_code == 202 and res.json()["job"]["job_id"] == first
    settings = client.app.state.settings  # type: ignore[attr-defined]

    def failing(_job_id: int, _stop: object) -> None:
        raise RuntimeError("解析に失敗（テスト）")

    worker = Worker(settings, _factory(client), sync_launcher(settings),
                    postprocess_encoder=fake_encoder, beat_runner=failing)
    assert worker.run_postprocess_one() == first
    assert client.get(f"/api/jobs/{first}").json()["beat_warning"]
    assert client.get(f"/api/jobs/{newest}").json()["beat_warning"] is None
    # 省略時は最新の分割
    res = client.post(f"/api/tracks/{track_id}/beats")
    assert res.json()["job"]["job_id"] == newest


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


# --- 補正（T10c） --------------------------------------------------------------------------


def _edit(client: TestClient, track_id: int, **body: object):
    return client.post(f"/api/tracks/{track_id}/beats/edit", json=body)


def test_edit_undo_reset(client: TestClient, tmp_path: Path) -> None:
    track_id = _track(client, tmp_path, seconds=12.0)
    _separate(client, track_id, FakeBeatAnalyzer(60))
    body = client.get(f"/api/tracks/{track_id}/beats").json()
    assert body["edited"] is False and body["can_undo"] is False
    auto_beats = body["beats"]

    res = _edit(client, track_id, op="double", range="all")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["edited"] is True and body["can_undo"] is True
    assert [s["bpm"] for s in body["segments"]] == [120.0]
    assert len(body["beats"]) == 2 * len(auto_beats) - 1
    # GET も有効な拍（直した結果）を返す
    assert client.get(f"/api/tracks/{track_id}/beats").json()["beats"] == body["beats"]
    # 自動の結果は書き換えない
    with _factory(client)() as s:
        grid = s.get(BeatGrid, track_id)
        assert grid.beats_json == auto_beats and grid.edited_beats_json == body["beats"]

    body = _edit(client, track_id, op="meter", range="all", beats_per_bar=3).json()
    assert body["time_signature"] == 3 and body["auto_time_signature"] == 4

    # 元に戻す（複数回）
    body = client.post(f"/api/tracks/{track_id}/beats/undo").json()
    assert body["time_signature"] == 4 and body["edited"] is True and body["can_undo"] is True
    body = client.post(f"/api/tracks/{track_id}/beats/undo").json()
    assert body["edited"] is False and body["can_undo"] is False and body["beats"] == auto_beats
    res = client.post(f"/api/tracks/{track_id}/beats/undo")
    assert res.status_code == 409 and "元に戻せる操作" in res.json()["detail"]

    # 自動に戻す（元に戻すで取り消せる）
    _edit(client, track_id, op="double", range="all")
    _edit(client, track_id, op="shift", range="all", delta_sec=0.01)
    body = client.post(f"/api/tracks/{track_id}/beats/reset").json()
    assert body["edited"] is False and body["beats"] == auto_beats and body["can_undo"] is True
    body = client.post(f"/api/tracks/{track_id}/beats/undo").json()
    assert body["edited"] is True and body["beats"][0] == 0.01
    # 直していないときの reset は何もしない（履歴も増やさない）
    client.post(f"/api/tracks/{track_id}/beats/undo")
    client.post(f"/api/tracks/{track_id}/beats/undo")
    body = client.post(f"/api/tracks/{track_id}/beats/reset").json()
    assert body["edited"] is False and body["can_undo"] is False


def test_edit_errors(client: TestClient, tmp_path: Path) -> None:
    track_id = _track(client, tmp_path)
    assert _edit(client, track_id, op="double").status_code == 404  # 未解析
    assert _edit(client, 9999, op="double").status_code == 404
    _separate(client, track_id, FakeBeatAnalyzer(120))
    assert _edit(client, track_id, op="nope").status_code == 422
    res = _edit(client, track_id, op="meter")
    assert res.status_code == 422 and "beats_per_bar" in res.text
    res = _edit(client, track_id, op="tap", taps=[1.0, 1.5])
    assert res.status_code == 400 and "4 回以上" in res.json()["detail"]
    res = _edit(client, track_id, op="double", range="loop")
    assert res.status_code == 400 and "ループ区間" in res.json()["detail"]
    res = _edit(client, track_id, op="cues", cue_start=4.0, cue_end=2.0, bars=1)
    assert res.status_code == 400
    # 失敗した操作は履歴に残らない
    assert client.get(f"/api/tracks/{track_id}/beats").json()["can_undo"] is False


def test_edit_ops_via_api(client: TestClient, tmp_path: Path) -> None:
    track_id = _track(client, tmp_path, seconds=12.0)
    _separate(client, track_id, FakeBeatAnalyzer([(0, 120), (4, 150)]))
    # 区間（再生位置 6 秒 → 150 BPM の区間）だけ ÷2
    body = _edit(client, track_id, op="half", position=6.0).json()
    assert [s["bpm"] for s in body["segments"]] == [120.0, 75.0]
    body = _edit(client, track_id, op="downbeat", position=5.3, range="segment").json()
    assert any(abs(d - 5.6) < 0.01 for d in body["downbeats"])  # ÷2 の後の拍は 4.0, 4.8, 5.6…
    taps = [2.0 + k * 0.4 for k in range(6)]
    body = _edit(client, track_id, op="tap", taps=taps, range="all").json()
    assert [s["bpm"] for s in body["segments"]] == [150.0]
    body = _edit(client, track_id, op="cues", cue_start=0.0, cue_end=8.0, bars=4).json()
    assert body["segments"][0]["bpm"] == 120.0
    body = _edit(
        client, track_id, op="shift", range="loop", loop_start=0.0, loop_end=4.0, delta_sec=-0.01
    ).json()
    assert body["beats"][0] == 0.49


def test_reanalyze_moves_edits_to_history(client: TestClient, tmp_path: Path) -> None:
    """再解析すると新しい自動の結果を使い、直した結果は「元に戻す」で戻せる。"""
    track_id = _track(client, tmp_path)
    job_id = _separate(client, track_id, FakeBeatAnalyzer(100))
    edited = _edit(client, track_id, op="double", range="all").json()
    assert client.post(f"/api/tracks/{track_id}/beats").status_code == 202
    assert client.post(f"/api/tracks/{track_id}/beats/undo").status_code == 404  # 解析待ち
    assert _worker(client).run_postprocess_one() == job_id
    body = client.get(f"/api/tracks/{track_id}/beats").json()
    assert body["edited"] is False and body["can_undo"] is True
    assert [s["bpm"] for s in body["segments"]] == [120.0, 150.0]
    body = client.post(f"/api/tracks/{track_id}/beats/undo").json()
    assert body["edited"] is True and body["beats"] == edited["beats"]


def test_cli_force_reanalyze_moves_edits_to_history(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    from stemapp.beats.service import analyze_track, beats_payload, can_undo, edit_grid

    track_id = make_track(session, settings, tmp_path, seconds=8.0)
    analyze_track(session, settings, track_id, FakeBeatAnalyzer(100))
    grid = session.get(BeatGrid, track_id)
    edit_grid(session, grid, "double", {"range": "all"})
    session.commit()
    analyze_track(session, settings, track_id, FakeBeatAnalyzer(90), force=True)
    session.commit()
    body = beats_payload(grid, undo=can_undo(session, track_id))
    assert body["edited"] is False and body["segments"][0]["bpm"] == 90.0 and body["can_undo"]


def test_history_is_capped(settings: Settings, session: Session, tmp_path: Path) -> None:
    from stemapp.beats import service
    from stemapp.models import BeatEdit

    track_id = make_track(session, settings, tmp_path, seconds=8.0)
    service.analyze_track(session, settings, track_id, FakeBeatAnalyzer(100))
    grid = session.get(BeatGrid, track_id)
    for i in range(service.MAX_HISTORY + 5):
        service.edit_grid(
            session, grid, "shift", {"range": "all", "delta_sec": 0.001 if i % 2 else -0.001}
        )
    session.commit()
    assert session.query(BeatEdit).count() == service.MAX_HISTORY


def test_old_db_gets_edit_columns_and_table(tmp_path: Path) -> None:
    from stemapp.beats.service import beats_payload

    engine = make_engine(tmp_path / "old.db")
    try:
        init_db(engine)
        with engine.begin() as conn:
            # T10 までの DB を再現する
            conn.exec_driver_sql("DROP TABLE beat_edit")
            for col in ("edited_beats_json", "edited_downbeats_json", "edited_time_signature"):
                conn.exec_driver_sql(f"ALTER TABLE beat_grid DROP COLUMN {col}")
            conn.exec_driver_sql(
                "INSERT INTO track (track_id, title, audio_hash, created_at) "
                "VALUES (1, '古い曲', 'h1', '2026-01-01 00:00:00')"
            )
            conn.exec_driver_sql(
                "INSERT INTO beat_grid (track_id, analyzer, beats_json, downbeats_json, "
                "time_signature, created_at) VALUES (1, 'fake 1', '[0.0, 0.5, 1.0]', '[0.0]', 4, "
                "'2026-01-01 00:00:00')"
            )
        init_db(engine)
        insp = inspect(engine)
        assert "beat_edit" in set(insp.get_table_names())
        cols = {c["name"] for c in insp.get_columns("beat_grid")}
        assert {"edited_beats_json", "edited_downbeats_json", "edited_time_signature"} <= cols
        with Session(engine) as s:
            grid = s.get(BeatGrid, 1)
            assert grid.edited_beats_json is None
            body = beats_payload(grid)
            assert body["edited"] is False and body["beats"] == [0.0, 0.5, 1.0]
    finally:
        engine.dispose()


def test_edit_rejects_times_outside_track(client: TestClient, tmp_path: Path) -> None:
    track_id = _track(client, tmp_path, seconds=8.0)
    _separate(client, track_id, FakeBeatAnalyzer(120))
    taps = [1.0, 1.5, 2.0, 2.5]
    res = _edit(client, track_id, op="tap", range="all", taps=[-1.0, *taps])
    assert res.status_code == 422  # 0 秒より前
    res = _edit(client, track_id, op="tap", range="all", taps=[*taps, 500.0])
    assert res.status_code == 400 and "曲の長さ" in res.json()["detail"]
    res = _edit(client, track_id, op="cues", cue_start=1.0, cue_end=60.0, bars=4)
    assert res.status_code == 400
    res = _edit(client, track_id, op="double", range="loop", loop_start=1.0, loop_end=99.0)
    assert res.status_code == 400
    res = _edit(client, track_id, op="tap", range="all", taps=[1.0 + 0.01 * k for k in range(65)])
    assert res.status_code == 422  # タップの回数の上限
    assert _edit(client, track_id, op="tap", range="all", taps=taps).status_code == 200
