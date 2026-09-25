"""ブラウザのテスト用: 一時データフォルダで動くサーバー（Fake の分離器＋本物の ffmpeg）。

- Web サーバー（uvicorn）と、ワーカー（分割は FakeSeparator、配信用データは本物の ffmpeg で
  Opus/WebM を作る）を、このプロセスのスレッドで動かす。
- ブラウザは PC の Microsoft Edge（Playwright の channel="msedge"）。ブラウザ本体は
  ダウンロードしない（`playwright install` は不要）。
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from sqlalchemy.orm import Session, sessionmaker

from job_helpers import FinishedHandle
from stemapp.api.imports import ImportDeps
from stemapp.app import create_app
from stemapp.beats.base import BeatAnalyzer
from stemapp.beats.service import analyze_job_beats
from stemapp.config import REPO_ROOT, Settings
from stemapp.db import init_db, make_engine, make_session_factory
from stemapp.jobs.child import run_job
from stemapp.jobs.worker import ChildHandle, Worker
from stemapp.seed import seed
from stemapp.separation import FakeSeparator

SCREENS_DIR = REPO_ROOT / "data" / "cache" / "screens"
EDGE_ARGS = ["--autoplay-policy=no-user-gesture-required"]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class LiveServer:
    base_url: str
    settings: Settings
    session_factory: sessionmaker[Session]
    # 「保存フォルダを開く」で開こうとしたフォルダ（実際にはエクスプローラーを開かない）
    opened_folders: list[Path] = field(default_factory=list)


@contextmanager
def run_server(
    settings: Settings,
    *,
    fake_delay: float = 0.2,
    with_worker: bool = True,
    beat_analyzer: BeatAnalyzer | None = None,
) -> Iterator[LiveServer]:
    """beat_analyzer を渡すと、分割の後処理と作り直しで拍を作る（省略時は拍を作らない）。"""
    engine = make_engine(settings.db_path)
    init_db(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        seed(s)

    app = create_app(settings, import_deps=ImportDeps(threads=1))
    app.state.sse_poll_sec = 0.1
    opened: list[Path] = []
    app.state.folder_opener = opened.append
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    )
    server_thread = threading.Thread(target=server.run, name="test-uvicorn", daemon=True)
    server_thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline or not server_thread.is_alive():
            raise RuntimeError("テスト用サーバーが起動しません。")
        time.sleep(0.05)

    worker: Worker | None = None
    worker_thread: threading.Thread | None = None
    if with_worker:

        def launch(job_id: int) -> ChildHandle:
            # 子プロセスの代わりにその場で実行する（配信用データは本物の ffmpeg で作る）
            rc = run_job(
                settings, job_id, FakeSeparator(delay_sec=fake_delay), encoder=None,
                beat_analyzer=beat_analyzer,
            )
            return FinishedHandle(rc)

        def beat_runner(job_id: int, _stop: object) -> None:
            assert beat_analyzer is not None
            with factory() as s:
                analyze_job_beats(s, settings, job_id, beat_analyzer)

        worker = Worker(
            settings, factory, launch, poll_interval=0.2, cancel_check_interval=0.1,
            beat_runner=beat_runner if beat_analyzer is not None else None,
        )
        worker_thread = threading.Thread(target=worker.run_forever, name="test-worker", daemon=True)
        worker_thread.start()
    try:
        yield LiveServer(f"http://127.0.0.1:{port}", settings, factory, opened)
    finally:
        if worker is not None and worker_thread is not None:
            worker.stop()
            worker_thread.join(timeout=30)
        server.should_exit = True
        server_thread.join(timeout=15)
        engine.dispose()
