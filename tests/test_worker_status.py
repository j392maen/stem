"""ワーカーの状態（生きている合図・/api/health）と、serve によるワーカーの再起動。"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from job_helpers import sync_launcher
from stemapp.config import Settings
from stemapp.db import make_session_factory
from stemapp.jobs.supervisor import (
    EXIT_WORKER_LOCKED,
    MSG_DOWN,
    MSG_OTHER,
    Heartbeat,
    WorkerSupervisor,
    read_status,
    status_file,
    worker_health,
    write_status,
)
from stemapp.jobs.worker import Worker
from test_api import _app


class FakeProc:
    def __init__(self) -> None:
        self.rc: int | None = None

    def poll(self) -> int | None:
        return self.rc


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _supervisor(clock: Clock) -> tuple[WorkerSupervisor, list[FakeProc], list[FakeProc]]:
    started: list[FakeProc] = []
    stopped: list[FakeProc] = []

    def start() -> FakeProc:
        p = FakeProc()
        started.append(p)
        return p

    sup = WorkerSupervisor(
        start, stopped.append, quick_restart_sec=2, burst_window_sec=60, burst_count=3,
        backoff_sec=60, clock=clock, check_sec=3600,
    )
    return sup, started, stopped


# --- 見張りと再起動 ---------------------------------------------------------------------------


def test_supervisor_restarts_crashed_worker_and_backs_off() -> None:
    clock = Clock()
    sup, started, stopped = _supervisor(clock)
    sup.start()
    try:
        assert len(started) == 1 and sup.snapshot()["alive"]
        sup.check_once()
        assert len(started) == 1  # 動いている間は何もしない

        # 1回目に落ちた: 2 秒後に起動し直す（それまでは down）
        started[-1].rc = 3
        sup.check_once()
        snap = sup.snapshot()
        assert not snap["alive"] and snap["last_exit_code"] == 3 and snap["restart_in_sec"] == 2
        clock.t += 1
        sup.check_once()
        assert len(started) == 1
        clock.t += 1
        sup.check_once()
        assert len(started) == 2 and sup.snapshot()["alive"] and sup.restarts == 1

        # 短い間に3回落ちたら、間隔を空ける（60 秒）
        for _ in range(2):
            started[-1].rc = 1
            sup.check_once()
            clock.t += 2
            sup.check_once()
        assert len(started) == 3  # 2回目までは 2 秒で起動し直した
        snap = sup.snapshot()
        # 3回目に落ちてから 2 秒たった: 起動し直すのは 60 秒後（あと 58 秒）
        assert not snap["alive"] and snap["restart_in_sec"] == 58
        clock.t += 57
        sup.check_once()
        assert len(started) == 3
        clock.t += 1
        sup.check_once()
        assert len(started) == 4 and sup.snapshot()["alive"]

        # しばらく落ちなければ、また 2 秒で起動し直す
        clock.t += 120
        started[-1].rc = 1
        sup.check_once()
        assert sup.snapshot()["restart_in_sec"] == 2
    finally:
        sup.stop()
    # 止めるときは今のワーカーを止め、起動し直さない
    assert stopped == [started[-1]]
    n = len(started)
    clock.t += 1000
    sup.check_once()
    assert len(started) == n


def test_supervisor_does_not_restart_when_other_worker_runs(tmp_path: Path) -> None:
    """ワーカーがロックを取れずに専用の終了コードで終わったら、起動し直さない。"""
    clock = Clock()
    sup, started, _stopped = _supervisor(clock)
    sup.start()
    try:
        started[0].rc = EXIT_WORKER_LOCKED
        sup.check_once()
        for _ in range(5):
            clock.t += 120
            sup.check_once()
        assert len(started) == 1
        snap = sup.snapshot()
        assert snap["locked_out"] and not snap["alive"] and snap["restart_in_sec"] is None
        # 別のワーカーが合図を書いている
        write_status(status_file(tmp_path), {
            "pid": 99, "alive_at": datetime.now(UTC).isoformat(), "job_id": 4,
        })
        h = worker_health(tmp_path, sup)
        assert h["state"] == "other" and h["message"] == MSG_OTHER
        assert "別のワーカーが動いています" in h["message"]
        assert h["running"] is True and h["job_id"] == 4 and h["locked_out"] is True
    finally:
        sup.stop()


def test_cli_worker_exit_code_when_locked(settings: Settings, monkeypatch: Any) -> None:
    from typer.testing import CliRunner

    from stemapp import cli
    from stemapp.jobs.worker import WorkerLock

    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)
    held = WorkerLock(settings.data_root / "worker.lock")
    held.acquire()
    try:
        res = CliRunner().invoke(cli.app, ["worker"])
    finally:
        held.release()
    assert res.exit_code == EXIT_WORKER_LOCKED
    assert "別のワーカー" in res.output


def test_supervisor_thread_restarts_real_loop() -> None:
    started: list[FakeProc] = []

    def start() -> FakeProc:
        p = FakeProc()
        started.append(p)
        return p

    sup = WorkerSupervisor(start, lambda _p: None, check_sec=0.02, quick_restart_sec=0.05)
    sup.start()
    try:
        started[0].rc = 1
        deadline = time.monotonic() + 5
        while len(started) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(started) == 2
    finally:
        sup.stop()


# --- 生きている合図 ---------------------------------------------------------------------------


def test_heartbeat_writes_and_removes(tmp_path: Path) -> None:
    path = tmp_path / "run" / "worker-status.json"
    state = {"job_id": 7, "postprocess_job_id": None, "tempo_render_id": 3}
    hb = Heartbeat(path, lambda: state, interval=0.02)
    hb.start()
    try:
        data = read_status(path)
        assert data is not None and data["job_id"] == 7 and data["tempo_render_id"] == 3
        state["job_id"] = None
        deadline = time.monotonic() + 5
        while (read_status(path) or {}).get("job_id") is not None:
            assert time.monotonic() < deadline
            time.sleep(0.02)
    finally:
        hb.stop()
    assert not path.exists()  # きちんと止まったら消す


def test_worker_run_forever_writes_heartbeat(settings: Settings, engine: Any) -> None:
    path = status_file(settings.data_root)
    worker = Worker(
        settings, make_session_factory(engine), sync_launcher(settings), poll_interval=0.02,
        heartbeat_file=path,
    )
    t = threading.Thread(target=worker.run_forever)
    t.start()
    try:
        deadline = time.monotonic() + 10
        while read_status(path) is None:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert worker_health(settings.data_root, None)["state"] == "running"
    finally:
        worker.stop()
        t.join(timeout=30)
    assert not path.exists()
    assert worker_health(settings.data_root, None)["state"] == "unknown"


def test_worker_health_without_supervisor(tmp_path: Path) -> None:
    root = tmp_path
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    assert worker_health(root, None, now=now)["state"] == "unknown"
    write_status(status_file(root), {
        "pid": 1, "alive_at": (now - timedelta(seconds=3)).isoformat(),
        "job_id": 5, "postprocess_job_id": None, "tempo_render_id": 9,
    })
    h = worker_health(root, None, now=now)
    assert h["state"] == "running" and h["running"] and h["message"] is None
    assert h["job_id"] == 5 and h["tempo_render_id"] == 9 and h["managed"] is False
    # 合図が古い（ワーカーが落ちた）
    h = worker_health(root, None, now=now + timedelta(minutes=5))
    assert h["state"] == "down" and h["message"] == MSG_DOWN
    assert h["job_id"] is None and h["last_seen_at"] == (now - timedelta(seconds=3)).isoformat()
    # ログインしていないときは状態だけ
    brief = worker_health(root, None, now=now, detail=False)
    assert set(brief) == {"state", "running", "message"}
    # 壊れたファイルは無いのと同じ
    status_file(root).write_text("{", encoding="utf-8")
    assert worker_health(root, None, now=now)["state"] == "unknown"


def test_worker_health_with_supervisor(tmp_path: Path) -> None:
    clock = Clock()
    sup, started, _stopped = _supervisor(clock)
    sup.start()
    try:
        h = worker_health(tmp_path, sup)
        assert h["state"] == "running" and h["managed"] is True
        started[0].rc = 9
        sup.check_once()
        h = worker_health(tmp_path, sup)
        assert h["state"] == "down" and h["message"] == MSG_DOWN
        assert h["last_exit_code"] == 9 and h["restart_in_sec"] == 2
    finally:
        sup.stop()


def test_health_api_reports_worker(settings: Settings) -> None:
    app = _app(settings)
    with TestClient(app) as c:
        body = c.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["worker"]["state"] == "unknown"
        # serve が見張っていて、ワーカーが落ちて起動し直し待ち
        clock = Clock()
        sup, started, _ = _supervisor(clock)
        sup.start()
        app.state.worker_supervisor = sup
        started[0].rc = 1
        sup.check_once()
        w = c.get("/api/health").json()["worker"]
        assert w["state"] == "down" and w["message"] == MSG_DOWN and w["managed"] is True
        sup.stop()


def test_health_api_hides_details_without_login(settings: Settings) -> None:
    write_status(status_file(settings.data_root), {
        "pid": 1, "alive_at": datetime.now(UTC).isoformat(), "job_id": 5,
    })
    with TestClient(_app(settings, passcode="secret-pass")) as c:
        w = c.get("/api/health").json()["worker"]
        assert w == {"state": "running", "running": True, "message": None}
        assert c.post("/api/login", json={"passcode": "secret-pass"}).status_code == 200
        w = c.get("/api/health").json()["worker"]
        assert w["job_id"] == 5


def test_status_file_is_json(tmp_path: Path) -> None:
    path = status_file(tmp_path)
    assert write_status(path, {"a": "日本語"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": "日本語"}
