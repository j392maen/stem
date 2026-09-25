from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from job_helpers import FinishedHandle, make_track, sync_launcher
from stemapp.config import Settings
from stemapp.db import make_session_factory
from stemapp.jobs import (
    INTERRUPTED_MESSAGE,
    JobConflict,
    JobNotFound,
    enqueue_full_job,
    request_cancel,
)
from stemapp.jobs.worker import (
    ChildHandle,
    Worker,
    WorkerLock,
    WorkerLockError,
    stop_on_stdin_eof,
    subprocess_launcher,
)
from stemapp.models import SeparationJob, Stem, StemRendition, Waveform
from stemapp.peaks import DEFAULT_LEVELS
from stemapp.seed import SW, seed
from stemapp.separation import FakeSeparator
from stemapp.separation.pipeline import SeparationError


@pytest.fixture
def factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)


@pytest.fixture
def seeded(session: Session) -> Session:
    seed(session)
    return session


@pytest.fixture
def track_id(seeded: Session, settings: Settings, tmp_path: Path) -> int:
    return make_track(seeded, settings, tmp_path)


def _job(factory: sessionmaker[Session], job_id: int) -> SeparationJob:
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None
        return job


def _wait_for(cond: Callable[[], bool], timeout: float = 30.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("時間内に条件を満たしませんでした")


# --- 登録 --------------------------------------------------------------------------


def test_enqueue_creates_queued_job(seeded: Session, track_id: int) -> None:
    res = enqueue_full_job(seeded, track_id, "fast")
    assert res.created is True
    job = res.job
    assert (job.status, job.job_kind, job.progress) == ("queued", "full", 0.0)
    assert job.cancel_requested is False and job.output_gain_db == 0.0

    # 分割待ちがあれば新しく作らない（force でも）
    again = enqueue_full_job(seeded, track_id, "fast", force=True)
    assert again.created is False and again.reason == "active"
    assert again.job.job_id == job.job_id


def test_enqueue_errors(seeded: Session, track_id: int) -> None:
    with pytest.raises(JobNotFound):
        enqueue_full_job(seeded, 9999, "fast")
    with pytest.raises(SeparationError):
        enqueue_full_job(seeded, track_id, "no-such-preset")


# --- 実行 ---------------------------------------------------------------------------


def test_worker_runs_job_to_done(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    worker = Worker(settings, factory, sync_launcher(settings))
    assert worker.run_one() == job_id
    assert worker.run_one() is None  # もう無い

    job = _job(factory, job_id)
    assert job.status == "done" and job.progress == 1.0 and job.stage == "完了"
    assert job.started_at is not None and job.finished_at is not None
    with factory() as s:
        stems = s.scalars(select(Stem).where(Stem.job_id == job_id)).all()
        assert len(stems) == 8  # 上位6（vocals を含む）+ lead / backing
        for st in stems:
            rends = s.scalars(select(StemRendition).where(StemRendition.stem_id == st.stem_id))
            by_purpose = {r.purpose: r for r in rends}
            assert set(by_purpose) == {"master", "stream"}
            stream = by_purpose["stream"]
            assert (stream.codec, stream.bitrate_kbps) == ("opus", 128)
            assert stream.file_path.startswith(f"stems/{job_id}/stream/")
            assert stream.bytes and stream.bytes > 0
            levels = s.scalars(
                select(Waveform.samples_per_px).where(Waveform.stem_id == st.stem_id)
            ).all()
            assert sorted(levels) == list(DEFAULT_LEVELS)

    # 分割済みなので、force なしでは登録されない
    res = enqueue_full_job(seeded, track_id, "fast")
    assert res.created is False and res.reason == "done" and res.job.job_id == job_id
    assert enqueue_full_job(seeded, track_id, "fast", force=True).created is True


class ProgressProbe(FakeSeparator):
    """分離の呼び出しごとに、DB 上のジョブの進捗を記録する。"""

    def __init__(self, factory: sessionmaker[Session], job_id: int) -> None:
        super().__init__()
        self.factory = factory
        self.job_id = job_id
        self.seen: list[tuple[str, float, str | None]] = []

    def separate(self, *args: object, **kw: object):  # type: ignore[override]
        job = _job(self.factory, self.job_id)
        self.seen.append((job.status, job.progress, job.stage))
        return super().separate(*args, **kw)  # type: ignore[arg-type]


def test_progress_is_written_to_job(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    probe = ProgressProbe(factory, job_id)
    Worker(settings, factory, sync_launcher(settings, lambda: probe)).run_one()
    assert [s for s, _, _ in probe.seen] == ["running", "running"]
    progresses = [p for _, p, _ in probe.seen]
    assert 0.0 < progresses[0] < progresses[1] < 1.0
    assert all(stage and stage.startswith("分離中") for _, _, stage in probe.seen)
    assert _job(factory, job_id).status == "done"


def test_failure_marks_failed(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    launcher = sync_launcher(settings, lambda: FakeSeparator(fail_models={SW}))
    Worker(settings, factory, launcher).run_one()
    job = _job(factory, job_id)
    assert job.status == "failed"
    assert job.error_message and "fake failure" in job.error_message
    assert not (settings.stems_dir / str(job_id)).exists()
    with factory() as s:
        assert s.scalars(select(Stem).where(Stem.job_id == job_id)).all() == []


def test_postprocess_failure_marks_failed(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    """配信用データの作成に失敗したらジョブは failed、保存済みの stem も消える。"""
    from stemapp.jobs.child import run_job

    def broken_encoder(_args: object) -> None:
        raise RuntimeError("encoder broken")

    def launch(job_id: int) -> ChildHandle:
        return FinishedHandle(run_job(settings, job_id, FakeSeparator(), encoder=broken_encoder))

    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    Worker(settings, factory, launch).run_one()
    job = _job(factory, job_id)
    assert job.status == "failed"
    assert job.error_message and "配信用データを作成中" in job.error_message
    with factory() as s:
        assert s.scalars(select(Stem).where(Stem.job_id == job_id)).all() == []
    assert not (settings.stems_dir / str(job_id)).exists()


def test_child_crash_marks_failed(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    (settings.stems_dir / str(job_id)).mkdir(parents=True)
    Worker(settings, factory, lambda _jid: FinishedHandle(3)).run_one()
    job = _job(factory, job_id)
    assert job.status == "failed"
    assert job.error_message and "異常終了" in job.error_message and "3" in job.error_message
    assert not (settings.stems_dir / str(job_id)).exists()


def test_jobs_run_one_at_a_time_in_order(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], tmp_path: Path
) -> None:
    ids = []
    for i in range(3):
        tid = make_track(seeded, settings, tmp_path, name=f"s{i}", seed_offset=0.1 * i)
        ids.append(enqueue_full_job(seeded, tid, "fast").job.job_id)

    launched: list[int] = []
    active = {"now": 0, "max": 0}
    inner = sync_launcher(settings, launched=launched)

    def launch(job_id: int) -> ChildHandle:
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        try:
            return inner(job_id)
        finally:
            active["now"] -= 1

    worker = Worker(settings, factory, launch, poll_interval=0.05)
    t = threading.Thread(target=worker.run_forever)
    t.start()
    try:
        _wait_for(lambda: all(_job(factory, j).status == "done" for j in ids))
    finally:
        worker.stop()
        t.join(10)
    assert launched == ids
    assert active["max"] == 1
    jobs = [_job(factory, j) for j in ids]
    for a, b in zip(jobs, jobs[1:], strict=False):
        assert a.finished_at is not None and b.started_at is not None
        assert a.finished_at <= b.started_at


# --- キャンセル ------------------------------------------------------------------------


def test_cancel_queued_is_immediate(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    job = request_cancel(seeded, job_id)
    assert job.status == "canceled" and job.finished_at is not None
    # ワーカーは取り出さない
    launched: list[int] = []
    assert Worker(settings, factory, sync_launcher(settings, launched=launched)).run_one() is None
    assert launched == []
    # 終わったジョブはキャンセルできない
    with pytest.raises(JobConflict):
        request_cancel(seeded, job_id)
    with pytest.raises(JobNotFound):
        request_cancel(seeded, 9999)


def test_cancel_running_kills_child_process(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    """実際に子プロセスを起動し、実行中にキャンセルすると子が止まって canceled になる。"""
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    launched: list[ChildHandle] = []
    real = subprocess_launcher(settings, ["--fake", "--fake-delay", "60"])

    def launch(jid: int) -> ChildHandle:
        child = real(jid)
        launched.append(child)
        return child

    worker = Worker(settings, factory, launch, cancel_check_interval=0.2)
    t = threading.Thread(target=worker.run_one)
    t.start()
    try:
        # 子プロセスが分離を始める（stage が「分離中」になる）まで待つ
        _wait_for(lambda: (_job(factory, job_id).stage or "").startswith("分離中"), 60)
        partial = settings.stems_dir / str(job_id)
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "partial.flac").write_bytes(b"x")
        with factory() as s:
            assert request_cancel(s, job_id).status == "running"
        t.join(30)
        assert not t.is_alive()
    finally:
        for c in launched:
            if c.poll() is None:
                c.kill()
    assert launched and launched[0].poll() is not None  # 子プロセスは終了している
    job = _job(factory, job_id)
    assert job.status == "canceled" and job.cancel_requested is True
    assert job.finished_at is not None
    assert not (settings.stems_dir / str(job_id)).exists()


def test_stop_event_interrupts_running_job(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id

    class Running:
        killed = False

        def poll(self) -> int | None:
            return -9 if self.killed else None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            return -9

    child = Running()
    worker = Worker(settings, factory, lambda _j: child, cancel_check_interval=0.05)
    worker.stop()
    worker.run_one()
    assert child.killed
    job = _job(factory, job_id)
    assert job.status == "failed" and job.error_message == INTERRUPTED_MESSAGE


def test_keyboard_interrupt_cleans_up(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    (settings.stems_dir / str(job_id)).mkdir(parents=True)

    class Interrupting:
        killed = False
        polled = 0

        def poll(self) -> int | None:
            self.polled += 1
            if self.polled == 1:
                raise KeyboardInterrupt  # 監視中に Ctrl+C
            return 1 if self.killed else None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            return 1

    child = Interrupting()
    with pytest.raises(KeyboardInterrupt):
        Worker(settings, factory, lambda _j: child).run_one()
    assert child.killed
    job = _job(factory, job_id)
    assert job.status == "failed" and job.error_message == INTERRUPTED_MESSAGE
    assert not (settings.stems_dir / str(job_id)).exists()


# --- 起動時の片付け ----------------------------------------------------------------------


def test_recover_marks_running_as_failed(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    # 前回 running のまま終わったジョブ（stem もファイルも途中まで作られている）
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    Worker(settings, factory, sync_launcher(settings)).run_one()
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None
        job.status = "running"
        job.finished_at = None
        s.commit()
    partial = settings.stems_dir / str(job_id)
    assert partial.is_dir()

    queued = enqueue_full_job(seeded, track_id, "fast")
    assert queued.created is False  # running があるので登録されない
    worker = Worker(settings, factory, sync_launcher(settings))
    assert worker.recover() == [job_id]
    job = _job(factory, job_id)
    assert job.status == "failed" and job.error_message == INTERRUPTED_MESSAGE
    assert job.finished_at is not None
    assert not partial.exists()
    with factory() as s:
        assert s.scalars(select(Stem).where(Stem.job_id == job_id)).all() == []
    assert worker.recover() == []


# --- ロック・停止の合図 ---------------------------------------------------------------------


def test_worker_lock_is_exclusive(tmp_path: Path) -> None:
    a = WorkerLock(tmp_path / "worker.lock")
    b = WorkerLock(tmp_path / "worker.lock")
    a.acquire()
    try:
        with pytest.raises(WorkerLockError):
            b.acquire()
    finally:
        a.release()
    b.acquire()
    b.release()


def test_stop_on_stdin_eof() -> None:
    import io

    ev = threading.Event()
    t = stop_on_stdin_eof(ev, io.BytesIO(b"abc"))
    t.join(5)
    assert ev.is_set()


def test_worker_process_stops_when_stdin_closes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stemapp serve` と同じ方法でワーカーを起動し、標準入力を閉じると止まる。"""
    from stemapp import cli

    monkeypatch.setenv("STEMAPP_DATA_DIR", str(settings.data_dir))
    proc = cli._start_worker_process()
    try:
        _wait_for(lambda: (settings.data_dir / "worker.lock").exists(), 60)
        time.sleep(1.0)
        assert proc.poll() is None  # 動き続けている
        cli._stop_worker_process(proc)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()


# --- 残った処理が結果を書き込まない・子プロセスの後始末 -------------------------------------


class Interfere(FakeSeparator):
    """最初の分離の呼び出しで、DB 上のジョブを他のプロセスが変えたことにする。"""

    def __init__(self, factory: sessionmaker[Session], job_id: int, **values: object) -> None:
        super().__init__()
        self.factory = factory
        self.job_id = job_id
        self.values = values

    def separate(self, *args: object, **kw: object):  # type: ignore[override]
        if len(self.calls) == 0:
            with self.factory() as s:
                job = s.get(SeparationJob, self.job_id)
                assert job is not None
                for k, v in self.values.items():
                    setattr(job, k, v)
                s.commit()
        return super().separate(*args, **kw)  # type: ignore[arg-type]


def test_abandoned_job_does_not_write_done(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    """実行中にジョブが failed にされた（ワーカーの再起動で片付けられた等）ら、結果を書かない。"""
    from stemapp.jobs.child import EXIT_FAILED, run_job

    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    with factory() as s:
        from stemapp.jobs.queue import claim_next_job

        assert claim_next_job(s) == job_id
    sep = Interfere(factory, job_id, status="failed", error_message=INTERRUPTED_MESSAGE)
    assert run_job(settings, job_id, sep) == EXIT_FAILED
    job = _job(factory, job_id)
    assert job.status == "failed" and job.error_message == INTERRUPTED_MESSAGE
    with factory() as s:
        assert s.scalars(select(Stem).where(Stem.job_id == job_id)).all() == []
    assert not (settings.stems_dir / str(job_id)).exists()
    assert not (settings.cache_dir / "tmp" / f"job-{job_id}").exists()


def test_cancel_without_worker_is_honored_by_child(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    """ワーカーがいなくても、子はキャンセル依頼に気づいて canceled にしてやめる。"""
    from stemapp.jobs.child import run_job

    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    with factory() as s:
        from stemapp.jobs.queue import claim_next_job

        claim_next_job(s)
    run_job(settings, job_id, Interfere(factory, job_id, cancel_requested=True))
    job = _job(factory, job_id)
    assert job.status == "canceled"
    with factory() as s:
        assert s.scalars(select(Stem).where(Stem.job_id == job_id)).all() == []


def test_tmp_dirs_are_cleaned(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int
) -> None:
    from stemapp.jobs.queue import finish_job

    tmp = settings.cache_dir / "tmp"
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    (tmp / f"job-{job_id}").mkdir(parents=True)
    (tmp / f"job-{job_id}" / "vocals.wav").write_bytes(b"x")
    with factory() as s:
        finish_job(s, settings, job_id, "canceled")
    assert not (tmp / f"job-{job_id}").exists()

    # 起動時: running でないジョブの一時フォルダは消す（取り込み用の一時フォルダは残す）
    (tmp / "job-999").mkdir()
    (tmp / "abcdef").mkdir()
    Worker(settings, factory, sync_launcher(settings)).recover()
    assert not (tmp / "job-999").exists()
    assert (tmp / "abcdef").exists()


def _pid_alive(pid: int) -> bool:
    import subprocess
    import sys

    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout
        return f'"{pid}"' in out
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


WORKER_SCRIPT = """
import sys
from pathlib import Path
from stemapp.config import Settings
from stemapp.db import make_engine, make_session_factory
from stemapp.jobs.worker import Worker, subprocess_launcher

settings = Settings(_env_file=None, data_dir=Path(sys.argv[1]))
pid_file = Path(sys.argv[2])
real = subprocess_launcher(settings, ["--fake", "--fake-delay", "60"])

def launch(job_id):
    child = real(job_id)
    pid_file.write_text(str(child.pid), encoding="utf-8")
    return child

factory = make_session_factory(make_engine(settings.db_path))
Worker(settings, factory, launch, poll_interval=0.1).run_forever()
"""


def test_killing_worker_stops_child(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], track_id: int,
    tmp_path: Path,
) -> None:
    """ワーカーだけを強制終了しても、分割の子プロセスは残らない。"""
    import subprocess
    import sys

    from stemapp.jobs.worker import child_env, new_group_kwargs

    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    pid_file = tmp_path / "child.pid"
    worker = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", WORKER_SCRIPT, str(settings.data_dir), str(pid_file)],
        env=child_env(), stdin=subprocess.DEVNULL, **new_group_kwargs(),  # type: ignore[call-overload]
    )
    child_pid = None
    try:
        _wait_for(lambda: (_job(factory, job_id).stage or "").startswith("分離中"), 60)
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        assert _pid_alive(child_pid)
        worker.kill()  # ワーカーだけを強制終了（後始末の機会なし）
        worker.wait(10)
        _wait_for(lambda: not _pid_alive(child_pid), 15)
    finally:
        if worker.poll() is None:
            worker.kill()
        if child_pid is not None and _pid_alive(child_pid):
            subprocess.run(["taskkill", "/F", "/PID", str(child_pid)], check=False)
    # ジョブは running のまま残るが、次のワーカー起動時に片付けられる
    assert _job(factory, job_id).status == "running"
    assert Worker(settings, factory, sync_launcher(settings)).recover() == [job_id]
    assert _job(factory, job_id).status == "failed"
