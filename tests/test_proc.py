"""子プロセスの起動方法（Job Object / 親の見張り）に関するテスト。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from job_helpers import make_track
from stemapp.config import Settings
from stemapp.db import make_session_factory
from stemapp.jobs import enqueue_full_job
from stemapp.jobs.worker import Worker, subprocess_launcher
from stemapp.models import SeparationJob
from stemapp.seed import seed


def _wait_for(cond: Callable[[], bool], timeout: float) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError("時間内に条件を満たしませんでした")


def pid_alive(pid: int) -> bool:
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


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="scipy がありません")
def test_child_can_import_scipy_signal(session: Session, settings: Settings,
                                       engine: Engine, tmp_path: Path) -> None:
    """回帰テスト: 実際の起動方法（worker の launcher）で起動した子が、起動直後に
    `import scipy.signal` をしても固まらずに完了する。

    （stdin を別スレッドで読んでいたときは、Windows で DLL の読み込みが止まって固まった）
    """
    seed(session)
    track_id = make_track(session, settings, tmp_path)
    job_id = enqueue_full_job(session, track_id, "fast").job.job_id
    factory: sessionmaker[Session] = make_session_factory(engine)
    launcher = subprocess_launcher(settings, ["--fake", "--preimport", "scipy.signal"])
    t0 = time.monotonic()
    assert Worker(settings, factory, launcher, cancel_check_interval=0.2).run_one() == job_id
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.status == "done", job.error_message
    assert time.monotonic() - t0 < 120


INTERMEDIATE = """
import subprocess, sys, time
from pathlib import Path
from stemapp.proc import start_bound_process

out = Path(sys.argv[1])
# 子（Job に入る）が、さらに孫（ffmpeg の代わり。run_ffmpeg と同じ popen_bound）を起動する
code = (
    "import sys, time; from stemapp.proc import popen_bound; "
    "g = popen_bound([sys.executable, '-c', 'import time; time.sleep(120)']); "
    "open(sys.argv[1], 'w').write(str(g.pid)); time.sleep(120)"
)
child = start_bound_process([sys.executable, "-c", code, str(out) + ".grand"])
out.write_text(str(child.pid))
time.sleep(120)
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Job Object は Windows のみ")
def test_job_object_kills_child_and_grandchild(tmp_path: Path) -> None:
    """親が強制終了されると、Job に入れた子も、子が起動した孫（ffmpeg など）も終了する。"""
    out = tmp_path / "pids"
    parent = subprocess.Popen([sys.executable, "-c", INTERMEDIATE, str(out)])  # noqa: S603
    pids: list[int] = []
    try:
        grand_file = Path(str(out) + ".grand")
        _wait_for(lambda: out.exists() and grand_file.exists()
                  and out.read_text() != "" and grand_file.read_text() != "", 30)
        pids = [int(out.read_text()), int(grand_file.read_text())]
        assert all(pid_alive(p) for p in pids)
        parent.kill()
        parent.wait(10)
        _wait_for(lambda: not any(pid_alive(p) for p in pids), 15)
    finally:
        if parent.poll() is None:
            parent.kill()
        for p in pids:
            if pid_alive(p):
                subprocess.run(["taskkill", "/F", "/PID", str(p)], check=False)
