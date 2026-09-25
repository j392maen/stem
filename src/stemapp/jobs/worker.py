"""ワーカー（`stemapp worker`）。Web サーバーとは別のプロセスで動く。

- DB の queued ジョブを古い順に1件ずつ取り出し、1ジョブごとに子プロセス
  （`stemapp.jobs.child`）を起動して分割させる（GPU は同時に1ジョブ）。
- 実行中は 0.5 秒ごとに cancel_requested を調べ、立っていれば子プロセスを終了させて
  canceled にし、stems フォルダを消す。
- 起動時、running のまま残ったジョブを failed（中断されました）にする。
- 停止（stop_event、Ctrl+C）のときは実行中の子プロセスを終了させ、同じ後始末をする。
- 同じデータフォルダで2つのワーカーが動かないよう、`data/worker.lock` をロックする。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Protocol

from sqlalchemy.orm import Session, sessionmaker

from stemapp.config import Settings
from stemapp.jobs.queue import (
    CANCELED,
    FAILED,
    INTERRUPTED_MESSAGE,
    RUNNING,
    STAGE_CANCELED,
    claim_next_job,
    finish_job,
    is_cancel_requested,
    recover_interrupted_jobs,
)
from stemapp.models import SeparationJob

log = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 1.0  # queued ジョブを探す間隔
CANCEL_CHECK_INTERVAL_SEC = 0.5  # 実行中にキャンセルを調べる間隔（1秒以内）
KILL_WAIT_SEC = 10.0


class ChildHandle(Protocol):
    """子プロセス（subprocess.Popen と同じ一部のメソッド）。"""

    def poll(self) -> int | None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


ChildLauncher = Callable[[int], ChildHandle]


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def new_group_kwargs() -> dict[str, object]:
    """コンソールの Ctrl+C が子に届かないようにする（後始末は親が行う）。"""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def subprocess_launcher(settings: Settings, extra_args: Sequence[str] = ()) -> ChildLauncher:
    """`python -m stemapp.jobs.child <job_id>` を起動する launcher。"""

    def launch(job_id: int) -> ChildHandle:
        cmd = [
            sys.executable, "-m", "stemapp.jobs.child", str(job_id),
            "--data-dir", str(settings.data_dir),
            *extra_args,
        ]
        log.info("子プロセスを起動します（job %d）。", job_id)
        return subprocess.Popen(  # noqa: S603
            cmd, env=child_env(), stdin=subprocess.DEVNULL, **new_group_kwargs()  # type: ignore[call-overload]
        )

    return launch


class WorkerLockError(RuntimeError):
    pass


class WorkerLock:
    """`data/worker.lock` の排他ロック（プロセスが終われば OS が外す）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: IO[bytes] | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")  # noqa: SIM115
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            fh.close()
            raise WorkerLockError(
                "別のワーカーが同じデータフォルダで動いています（stemapp serve / worker を"
                "二重に起動していないか確認してください）。"
            ) from e
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._fh.close()
        self._fh = None


class Worker:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        launcher: ChildLauncher,
        *,
        poll_interval: float = POLL_INTERVAL_SEC,
        cancel_check_interval: float = CANCEL_CHECK_INTERVAL_SEC,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.launcher = launcher
        self.poll_interval = poll_interval
        self.cancel_check_interval = cancel_check_interval
        self.stop_event = stop_event or threading.Event()
        self.current_job_id: int | None = None
        self.current_child: ChildHandle | None = None

    # --- 起動時・停止時 ---------------------------------------------------------------

    def recover(self) -> list[int]:
        with self.session_factory() as session:
            return recover_interrupted_jobs(session, self.settings)

    def stop(self) -> None:
        self.stop_event.set()

    def _kill_child(self, child: ChildHandle) -> None:
        if child.poll() is None:
            child.kill()
        try:
            child.wait(timeout=KILL_WAIT_SEC)
        except subprocess.TimeoutExpired:
            log.error("子プロセスが終了しません。")

    def _finish(self, job_id: int, status: str, message: str | None, stage: str | None) -> None:
        with self.session_factory() as session:
            finish_job(session, self.settings, job_id, status, message=message, stage=stage)

    # --- 1ジョブの実行 ----------------------------------------------------------------

    def run_one(self) -> int | None:
        """queued のジョブを1件実行する（終わるまで戻らない）。無ければ None。"""
        with self.session_factory() as session:
            job_id = claim_next_job(session)
        if job_id is None:
            return None
        log.info("ジョブを開始します（job %d）。", job_id)
        try:
            child = self.launcher(job_id)
        except Exception as e:
            log.exception("子プロセスを起動できませんでした（job %d）", job_id)
            self._finish(job_id, FAILED, f"分割処理を起動できませんでした: {e}", None)
            return job_id
        self.current_job_id, self.current_child = job_id, child
        try:
            self._watch(job_id, child)
        except BaseException:
            # Ctrl+C（KeyboardInterrupt）など: 子を止めて「中断」にする
            self._kill_child(child)
            self._finish(job_id, FAILED, INTERRUPTED_MESSAGE, None)
            raise
        finally:
            self.current_job_id, self.current_child = None, None
        return job_id

    def _watch(self, job_id: int, child: ChildHandle) -> None:
        while True:
            rc = child.poll()
            if rc is not None:
                self._after_exit(job_id, rc)
                return
            if self.stop_event.is_set():
                log.warning("停止の指示で job %d を中断します。", job_id)
                self._kill_child(child)
                self._finish(job_id, FAILED, INTERRUPTED_MESSAGE, None)
                return
            with self.session_factory() as session:
                cancel = is_cancel_requested(session, job_id)
            if cancel:
                log.info("job %d をキャンセルします（子プロセスを終了）。", job_id)
                self._kill_child(child)
                self._finish(job_id, CANCELED, None, STAGE_CANCELED)
                return
            self.stop_event.wait(self.cancel_check_interval)

    def _after_exit(self, job_id: int, rc: int) -> None:
        with self.session_factory() as session:
            job = session.get(SeparationJob, job_id)
            status = job.status if job is not None else None
            cancel = bool(job.cancel_requested) if job is not None else False
        if status == RUNNING:
            # 子プロセスが JOB を更新せずに終わった（異常終了）
            if cancel:
                self._finish(job_id, CANCELED, None, STAGE_CANCELED)
            else:
                self._finish(
                    job_id, FAILED, f"分割処理が異常終了しました（終了コード {rc}）。", None
                )
        log.info("ジョブが終わりました（job %d, %s, 終了コード %d）。", job_id, status, rc)

    # --- ループ ---------------------------------------------------------------------

    def run_forever(self) -> None:
        """stop_event が立つまでジョブを実行し続ける。"""
        recovered = self.recover()
        if recovered:
            log.warning("中断されたジョブを片付けました: %s", recovered)
        log.info("ワーカーを開始しました。")
        while not self.stop_event.is_set():
            job_id = self.run_one()
            if job_id is None:
                self.stop_event.wait(self.poll_interval)
        log.info("ワーカーを停止しました。")


def stop_on_stdin_eof(event: threading.Event, stream: IO[bytes] | None = None) -> threading.Thread:
    """標準入力が閉じられたら event を立てるスレッド（`stemapp serve` からの停止の合図）。"""
    src = stream if stream is not None else sys.stdin.buffer

    def reader() -> None:
        try:
            while src.read(4096):
                pass
        except (OSError, ValueError):
            pass
        event.set()

    t = threading.Thread(target=reader, name="stdin-eof", daemon=True)
    t.start()
    return t
