"""子プロセスの寿命を親に結びつける（親がどう終わっても子が残らないようにする）。

- Windows: プロセスごとに1つ Job Object（JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE）を作り、
  起動した子をそこに入れる。親が終了すると（強制終了でも）Job のハンドルが閉じ、
  OS が Job 内の全プロセス（子が起動した ffmpeg なども含む）を終了させる。
  入れ子の Job（serve → ワーカー → 分割の子）は Windows 8 以降で使える。
- 子のさらに子（ffmpeg・ffprobe・yt-dlp）も `run_bound` で起動して Job に入れる。
  venv の python.exe（ランチャー）は自分用の Job を「子が抜け出せる」設定で作るため、
  ふつうに起動した孫は親の Job から抜けてしまう。明示的に Job に入れればそれを防げる。
- Linux など: 子は環境変数 STEMAPP_PARENT_PID を受け取り、`watch_parent()` のスレッドが
  1秒ごとに親が変わっていないか（os.getppid）を調べ、親がいなくなったら終了する。

標準入力は読まない（Windows で stdin パイプを別スレッドで同期読み取りしていると、
その間の DLL 読み込み（例: import scipy.signal）が止まってしまうため）。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

log = logging.getLogger(__name__)

PARENT_PID_ENV = "STEMAPP_PARENT_PID"
EXIT_ORPHANED = 3
PARENT_CHECK_SEC = 1.0

_job_handle: int | None = None
_job_lock = threading.Lock()


def _win_job() -> int | None:
    """このプロセス用の Job Object（KILL_ON_JOB_CLOSE）。作れなければ None。"""
    global _job_handle
    with _job_lock:
        if _job_handle is not None:
            return _job_handle
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):  # noqa: N801
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class BASIC_LIMIT(ctypes.Structure):  # noqa: N801
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):  # noqa: N801
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            log.warning("Job Object を作れませんでした（%d）。", ctypes.get_last_error())
            return None
        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)  # 9 = ExtendedLimitInformation
        )
        if not ok:
            log.warning("Job Object を設定できませんでした（%d）。", ctypes.get_last_error())
            kernel32.CloseHandle(job)
            return None
        # ハンドルは閉じずに持ち続ける（このプロセスが終わると OS が閉じ、子も終了する）
        _job_handle = int(job)
        return _job_handle


def _assign_to_job(proc: subprocess.Popen[Any]) -> bool:
    import ctypes
    from ctypes import wintypes

    job = _win_job()
    if job is None:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    handle = int(proc._handle)  # type: ignore[attr-defined]
    if not kernel32.AssignProcessToJobObject(job, handle):
        log.warning(
            "子プロセス %d を Job Object に入れられませんでした（%d）。",
            proc.pid, ctypes.get_last_error(),
        )
        return False
    return True


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env[PARENT_PID_ENV] = str(os.getpid())
    return env


def new_group_kwargs() -> dict[str, Any]:
    """コンソールの Ctrl+C が子に届かないようにする（後始末は親が行う）。"""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def popen_bound(cmd: Sequence[str], **kwargs: Any) -> subprocess.Popen[Any]:
    """subprocess.Popen と同じ。Windows では起動した子をこのプロセスの Job に入れる。"""
    proc: subprocess.Popen[Any] = subprocess.Popen(list(cmd), **kwargs)  # noqa: S603
    if os.name == "nt":
        _assign_to_job(proc)
    return proc


def run_bound(
    cmd: Sequence[str], *, timeout: float | None = None, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """subprocess.run(capture_output=True) と同じ。子はこのプロセスの Job に入る。"""
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    with popen_bound(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs) as proc:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        except BaseException:
            proc.kill()
            raise
    return subprocess.CompletedProcess(list(cmd), proc.returncode, out, err)


def start_bound_process(cmd: Sequence[str]) -> subprocess.Popen[bytes]:
    """親（このプロセス）が終わると一緒に終わる子プロセスを起動する。標準入力は使わない。"""
    return popen_bound(cmd, env=child_env(), stdin=subprocess.DEVNULL, **new_group_kwargs())


def watch_parent() -> threading.Thread | None:
    """（Windows 以外）親がいなくなったら終了するスレッドを始める。Windows では何もしない。

    Windows では親の Job Object が子を終了させるので不要。
    """
    if os.name == "nt":
        return None
    expected = os.environ.get(PARENT_PID_ENV)
    if not expected or not expected.isdigit():
        return None
    parent = int(expected)

    def check() -> None:
        while True:
            if os.getppid() != parent:
                sys.stderr.write("親プロセスがいなくなったため終了します。\n")
                os._exit(EXIT_ORPHANED)
            time.sleep(PARENT_CHECK_SEC)

    t = threading.Thread(target=check, name="parent-watch", daemon=True)
    t.start()
    return t
