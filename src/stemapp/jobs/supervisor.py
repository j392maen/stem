"""ワーカーの状態（生きているかの合図）と、`stemapp serve` によるワーカーの再起動。

- ワーカーは `data/run/worker-status.json` に、数秒ごとに「生きている時刻」と実行中のもの
  （分割のジョブ・配信用データの作り直し・速度変更の作成）を書く（`Heartbeat`）。止まるときは消す。
- `stemapp serve` は `WorkerSupervisor` でワーカーのプロセスを見張り、落ちたら起動し直す。
  短い間に何度も落ちるときは間隔を空ける（その間は画面に「分割の処理が止まっています」を出す）。
- `/api/health` は `worker_health` でこの2つをまとめて返す。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

HEARTBEAT_SEC = 2.0  # 合図を書く間隔
STALE_SEC = 15.0  # これより古い合図は「止まっている」とみなす
STATUS_FILE_NAME = "worker-status.json"

STATE_RUNNING = "running"
STATE_DOWN = "down"
STATE_UNKNOWN = "unknown"  # 合図も見張りも無い（ワーカーを起動していない構成）
MSG_DOWN = "分割の処理が止まっています。"


def status_file(data_root: Path) -> Path:
    return data_root / "run" / STATUS_FILE_NAME


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_status(path: Path, data: dict[str, Any]) -> bool:
    """合図を書く（一時ファイルに書いて置き換える）。読み手と重なって失敗したら False。"""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        # Windows では読み手が開いている間は置き換えられない。次の合図で書く
        tmp.unlink(missing_ok=True)
        return False


def read_status(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class Heartbeat:
    """ワーカーの中で動くスレッド。snapshot() の内容に時刻を足して数秒ごとに書く。"""

    def __init__(
        self, path: Path, snapshot: Callable[[], dict[str, Any]], interval: float = HEARTBEAT_SEC
    ) -> None:
        self.path = path
        self.snapshot = snapshot
        self.interval = interval
        self.started_at = _now_iso()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> None:
        data = {"pid": os.getpid(), "started_at": self.started_at, "alive_at": _now_iso()}
        try:
            data.update(self.snapshot())
        except Exception:  # noqa: BLE001 - 合図は止めない
            log.exception("ワーカーの状態を読めませんでした。")
        write_status(self.path, data)

    def start(self) -> None:
        self.beat()
        self._thread = threading.Thread(target=self._loop, name="worker-heartbeat", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.beat()

    def stop(self) -> None:
        """止める（きちんと止まったので合図のファイルを消す）。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        for _ in range(5):
            try:
                self.path.unlink(missing_ok=True)
                return
            except OSError:
                time.sleep(0.1)


# --- serve による見張りと再起動 ------------------------------------------------------------


class Proc(Protocol):
    """subprocess.Popen の一部。"""

    def poll(self) -> int | None: ...


QUICK_RESTART_SEC = 2.0  # 落ちてから起動し直すまで
BURST_WINDOW_SEC = 60.0  # この時間の中で
BURST_COUNT = 3  # この回数落ちたら
BACKOFF_SEC = 60.0  # 間隔を空ける
CHECK_SEC = 1.0


class WorkerSupervisor:
    """ワーカーのプロセスを見張り、落ちたら起動し直す（`stemapp serve` のスレッド）。

    start_proc はワーカーを起動して Proc を返す関数、stop_proc はきちんと止める関数。
    """

    def __init__(
        self,
        start_proc: Callable[[], Proc],
        stop_proc: Callable[[Proc], None],
        *,
        check_sec: float = CHECK_SEC,
        quick_restart_sec: float = QUICK_RESTART_SEC,
        burst_window_sec: float = BURST_WINDOW_SEC,
        burst_count: int = BURST_COUNT,
        backoff_sec: float = BACKOFF_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._start_proc = start_proc
        self._stop_proc = stop_proc
        self.check_sec = check_sec
        self.quick_restart_sec = quick_restart_sec
        self.burst_window_sec = burst_window_sec
        self.burst_count = burst_count
        self.backoff_sec = backoff_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self.proc: Proc | None = None
        self.restarts = 0
        self.last_exit_code: int | None = None
        self.last_exit_at: str | None = None
        self._crashes: deque[float] = deque()
        self._restart_at: float | None = None  # 起動し直す予定（clock の値）。None は動いている

    def start(self) -> None:
        with self._lock:
            self.proc = self._start_proc()
        self._thread = threading.Thread(target=self._loop, name="worker-supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """見張りをやめ（起動し直さない）、ワーカーを止める。"""
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        with self._lock:
            proc = self.proc
        if proc is not None:
            self._stop_proc(proc)

    def check_once(self) -> None:
        """1回分の見張り（テスト用にも使う）。落ちていれば記録し、時刻が来たら起動し直す。"""
        now = self._clock()
        with self._lock:
            if self._stopping.is_set():
                return
            if self._restart_at is None:
                proc = self.proc
                rc = proc.poll() if proc is not None else None
                if proc is None or rc is None:
                    return
                self.last_exit_code = rc
                self.last_exit_at = _now_iso()
                self._crashes.append(now)
                while self._crashes and now - self._crashes[0] > self.burst_window_sec:
                    self._crashes.popleft()
                delay = (
                    self.backoff_sec
                    if len(self._crashes) >= self.burst_count
                    else self.quick_restart_sec
                )
                self._restart_at = now + delay
                log.error(
                    "ワーカーが終了しました（終了コード %s）。%.0f 秒後に起動し直します。",
                    rc, delay,
                )
            if now < self._restart_at:
                return
            try:
                self.proc = self._start_proc()
            except Exception:
                log.exception("ワーカーを起動し直せませんでした。")
                self._crashes.append(now)
                self._restart_at = now + self.backoff_sec
                return
            self._restart_at = None
            self.restarts += 1
            log.warning("ワーカーを起動し直しました（%d 回目）。", self.restarts)

    def _loop(self) -> None:
        while not self._stopping.wait(self.check_sec):
            try:
                self.check_once()
            except Exception:  # noqa: BLE001 - 見張りは止めない
                log.exception("ワーカーの見張りで想定外のエラー")

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            proc = self.proc
            alive = (
                self._restart_at is None and proc is not None and proc.poll() is None
            )
            restart_in = (
                max(0.0, self._restart_at - now) if self._restart_at is not None else None
            )
            return {
                "alive": alive,
                "restarts": self.restarts,
                "last_exit_code": self.last_exit_code,
                "last_exit_at": self.last_exit_at,
                "restart_in_sec": round(restart_in, 1) if restart_in is not None else None,
            }


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def worker_health(
    data_root: Path,
    supervisor: WorkerSupervisor | None,
    *,
    now: datetime | None = None,
    detail: bool = True,
) -> dict[str, Any]:
    """`/api/health` の worker。detail=False（ログインしていない）なら状態だけ。

    - serve が見張っているとき: プロセスが動いていれば running、落ちて起動し直し待ちなら down。
    - 見張りが無いとき（`stemapp worker` を別に起動した等）: 合図が新しければ running、
      古ければ down（落ちた）、合図のファイルが無ければ unknown。
    """
    now = now or datetime.now(UTC)
    status = read_status(status_file(data_root))
    alive_at = _parse_time(status.get("alive_at")) if status else None
    fresh = alive_at is not None and (now - alive_at).total_seconds() <= STALE_SEC
    sup = supervisor.snapshot() if supervisor is not None else None
    if sup is not None:
        state = STATE_RUNNING if sup["alive"] else STATE_DOWN
    elif status is None:
        state = STATE_UNKNOWN
    else:
        state = STATE_RUNNING if fresh else STATE_DOWN
    out: dict[str, Any] = {
        "state": state,
        "running": state == STATE_RUNNING,
        "message": MSG_DOWN if state == STATE_DOWN else None,
    }
    if not detail:
        return out
    current = status if (status is not None and fresh) else {}
    out.update(
        {
            "managed": sup is not None,
            "last_seen_at": alive_at.isoformat() if alive_at is not None else None,
            "pid": status.get("pid") if status else None,
            # 実行中のもの（合図が古いときは分からないので None）
            "job_id": current.get("job_id"),
            "postprocess_job_id": current.get("postprocess_job_id"),
            "tempo_render_id": current.get("tempo_render_id"),
            "restarts": sup["restarts"] if sup else 0,
            "last_exit_code": sup["last_exit_code"] if sup else None,
            "last_exit_at": sup["last_exit_at"] if sup else None,
            "restart_in_sec": sup["restart_in_sec"] if sup else None,
        }
    )
    return out
