"""書き出しをサーバー内のスレッドで1件ずつ実行する（GPU は使わない）。

分割ワーカー（別プロセス）は GPU の長い処理で埋まるため、書き出しはそこに入れず、
Web サーバーのプロセスのスレッドで行う（取り込みの ImportManager と同じ形）。
一定時間ごと・書き出しの前後に片付け（期限・容量）も行う。
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field

from sqlalchemy.orm import Session, sessionmaker

from stemapp.audio import FfmpegRunner
from stemapp.config import Settings
from stemapp.exports.service import cleanup_exports, recover_interrupted_exports, run_export

log = logging.getLogger(__name__)

CLEANUP_INTERVAL_SEC = 600.0


@dataclass
class ExportManager:
    settings: Settings
    session_factory: sessionmaker[Session]
    # ffmpeg の代わり（テスト用）。None なら PATH の ffmpeg
    runner: FfmpegRunner | None = None
    cleanup_interval_sec: float = CLEANUP_INTERVAL_SEC
    _queue: queue.Queue[int | None] = field(default_factory=queue.Queue, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        """前回中断した書き出しを failed にし、片付けてからスレッドを始める。"""
        with self.session_factory() as s:
            recover_interrupted_exports(s, self.settings)
        self.cleanup()
        # daemon: サーバー終了時に書き出し中のものを待たない（次回起動時に failed にする）
        self._thread = threading.Thread(target=self._loop, name="export", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)

    def join(self) -> None:
        """登録済みの書き出しがすべて終わるまで待つ（テスト用）。"""
        self._queue.join()

    def submit(self, export_id: int) -> None:
        self._queue.put(export_id)

    def cleanup(self) -> list[int]:
        try:
            with self.session_factory() as s:
                return cleanup_exports(s, self.settings)
        except Exception:
            log.exception("書き出しの片付けに失敗しました。")
            return []

    def _loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self.cleanup_interval_sec)
            except queue.Empty:
                self.cleanup()
                continue
            try:
                if item is None:
                    return
                run_export(self.settings, self.session_factory, item, self.runner)
                self.cleanup()
            except Exception:
                log.exception("書き出しで想定外のエラー（export %s）", item)
            finally:
                self._queue.task_done()
