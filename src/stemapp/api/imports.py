"""取り込み API と、取り込みをバックグラウンドで実行する ImportManager。

取り込み（URL の取得・正規化）は時間がかかるので、サーバー内のスレッドで実行する。
INPUT_SOURCE を先に作り（fetch_status=queued）、スレッドが fetching → done / failed と更新する。
取り込みそのものは T03 の `import_file` / `fetch_url` を使う。
separate=true なら、取り込み後に分割ジョブを登録する（分割済みなら force のときだけ）。
"""

from __future__ import annotations

import logging
import queue
import re
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker
from starlette.datastructures import UploadFile

from stemapp.api.common import SessionDep, iso, not_found
from stemapp.audio import FfmpegRunner
from stemapp.config import Settings
from stemapp.ingest.service import (
    FETCH_DONE,
    FETCH_FAILED,
    FETCH_FETCHING,
    FETCH_QUEUED,
    SOURCE_FILE,
    SOURCE_URL,
    ImportResult,
    TagReader,
    import_file,
)
from stemapp.ingest.url import (
    ERROR_MESSAGES,
    UNKNOWN,
    UrlImportError,
    YtDlpRunner,
    choose_runner,
    fetch_url,
    is_url,
)
from stemapp.jobs import enqueue_full_job
from stemapp.models import InputSource
from stemapp.separation.pipeline import SeparationError, load_plan

log = logging.getLogger(__name__)

INVALID_AUDIO = "invalid_audio"  # ファイル取り込みで音声として読めなかった
IMPORT_MESSAGES: dict[str, str] = {
    **ERROR_MESSAGES,
    INVALID_AUDIO: "音声ファイルとして読み込めませんでした。ファイルの形式を確認してください。",
}
STATUS_MESSAGES: dict[str, str] = {
    FETCH_QUEUED: "取り込みの順番を待っています。",
    FETCH_FETCHING: "取得・取り込み中です。",
    FETCH_DONE: "取り込みました。",
}


@dataclass
class ImportDeps:
    """取り込みで使う外部処理（テストで差し替える）。"""

    ffmpeg_runner: FfmpegRunner | None = None
    tag_reader: TagReader | None = None
    ytdlp_runner: Callable[[Settings], YtDlpRunner] = choose_runner
    threads: int = 2


@dataclass(frozen=True)
class ImportTask:
    source_id: int
    source_type: str
    file_path: Path | None = None  # 受け取ったファイル（取り込み後に消す）
    original_name: str | None = None
    url: str | None = None
    separate: bool = False
    preset: str | None = None
    force: bool = False


@dataclass
class ImportOutcome:
    job_id: int | None = None
    job_created: bool = False


@dataclass
class ImportManager:
    settings: Settings
    session_factory: sessionmaker[Session]
    deps: ImportDeps = field(default_factory=ImportDeps)

    def __post_init__(self) -> None:
        self._queue: queue.Queue[ImportTask | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._active: set[int] = set()
        self._outcomes: dict[int, ImportOutcome] = {}
        self._lock = threading.Lock()

    # --- 起動・停止 -------------------------------------------------------------------

    def start(self) -> None:
        for i in range(max(1, self.deps.threads)):
            # daemon: サーバー終了時に取得中の yt-dlp を待たない（残った行は次回起動時に failed）
            t = threading.Thread(target=self._loop, name=f"import-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        for _ in self._threads:
            self._queue.put(None)

    def join(self) -> None:
        """登録済みの取り込みがすべて終わるまで待つ（テスト用）。"""
        self._queue.join()

    # --- 登録 -------------------------------------------------------------------------

    def _create_source(self, source_type: str, original_name: str | None, url: str | None) -> int:
        with self.session_factory() as session:
            src = InputSource(
                track_id=None,
                source_type=source_type,
                original_name=original_name,
                url=url,
                fetch_status=FETCH_QUEUED,
            )
            session.add(src)
            session.commit()
            return src.source_id

    def submit(self, task_without_id: dict[str, Any]) -> int:
        source_id = self._create_source(
            task_without_id["source_type"],
            task_without_id.get("original_name"),
            task_without_id.get("url"),
        )
        task = ImportTask(source_id=source_id, **task_without_id)
        with self._lock:
            self._active.add(source_id)
        self._queue.put(task)
        return source_id

    def is_active(self, source_id: int) -> bool:
        with self._lock:
            return source_id in self._active

    def outcome(self, source_id: int) -> ImportOutcome | None:
        with self._lock:
            return self._outcomes.get(source_id)

    # --- 実行 -------------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                self._run(task)
            except Exception:
                log.exception("取り込みで想定外のエラー")
            finally:
                if task is not None:
                    with self._lock:
                        self._active.discard(task.source_id)
                self._queue.task_done()

    def _set_status(self, session: Session, source_id: int, status: str) -> None:
        src = session.get(InputSource, source_id)
        if src is not None:
            src.fetch_status = status
            session.commit()

    def _fail(self, session: Session, source_id: int, code: str, detail: str) -> None:
        session.rollback()
        src = session.get(InputSource, source_id)
        if src is None:
            return
        src.fetch_status = FETCH_FAILED
        src.error_code = code
        src.error_detail = detail[-2000:]
        src.fetched_at = datetime.now(UTC)
        session.commit()

    def _run(self, task: ImportTask) -> None:
        with self.session_factory() as session:
            self._set_status(session, task.source_id, FETCH_FETCHING)
            result: ImportResult | None = None
            try:
                if task.source_type == SOURCE_URL:
                    assert task.url is not None
                    result = fetch_url(
                        session,
                        self.settings,
                        task.url,
                        self.deps.ytdlp_runner(self.settings),
                        ffmpeg_runner=self.deps.ffmpeg_runner,
                        tag_reader=self.deps.tag_reader,
                        source_id=task.source_id,
                    )
                else:
                    assert task.file_path is not None
                    result = import_file(
                        session,
                        self.settings,
                        task.file_path,
                        original_name=task.original_name,
                        ffmpeg_runner=self.deps.ffmpeg_runner,
                        tag_reader=self.deps.tag_reader,
                        source_id=task.source_id,
                    )
            except UrlImportError as e:
                log.warning("URL の取り込みに失敗しました（source %d）: %s", task.source_id, e)
            except Exception as e:
                code = INVALID_AUDIO if task.source_type == SOURCE_FILE else UNKNOWN
                self._fail(session, task.source_id, code, f"{type(e).__name__}: {e}")
                log.warning("取り込みに失敗しました（source %d）: %s", task.source_id, e)
            finally:
                if task.file_path is not None:
                    shutil.rmtree(task.file_path.parent, ignore_errors=True)
            if result is None or not task.separate:
                return
            try:
                res = enqueue_full_job(session, result.track_id, task.preset, force=task.force)
            except Exception:
                log.exception("分割ジョブを登録できませんでした（track %d）", result.track_id)
                return
            with self._lock:
                self._outcomes[task.source_id] = ImportOutcome(res.job.job_id, res.created)


# --- API ----------------------------------------------------------------------------

router = APIRouter(prefix="/api", tags=["imports"])

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name: str | None) -> str:
    """受け取ったファイル名から、保存に使える名前（フォルダ部分を除く）を作る。"""
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = _UNSAFE_CHARS.sub("_", base).strip(" .")
    return base[:150] or "upload"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _manager(request: Request) -> ImportManager:
    return request.app.state.import_manager


def _check_preset(session_factory: sessionmaker[Session], preset: str | None) -> None:
    with session_factory() as session:
        load_plan(session, preset)


@router.post("/imports", status_code=202)
async def create_import(request: Request) -> JSONResponse:
    """ファイル（multipart の file）か URL（JSON の url）を取り込む。

    任意で separate（取り込み後に分割）、preset、force（分割済みでも分割）。
    """
    settings: Settings = request.app.state.settings
    ctype = request.headers.get("content-type", "").lower()
    task: dict[str, Any]
    dest_dir: Path | None = None
    if ctype.startswith("multipart/form-data"):
        limit = settings.max_upload_mb * 1024 * 1024
        too_large = HTTPException(
            status_code=413,
            detail=f"ファイルが大きすぎます（上限 {settings.max_upload_mb} MB）。",
        )
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > limit:
            raise too_large
        # 受け取ったファイル（一時ファイル）は with を抜けると閉じられる
        async with request.form() as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile) or not upload.filename:
                raise HTTPException(
                    status_code=400, detail="ファイルが指定されていません（file）。"
                )
            options: dict[str, Any] = {k: form.get(k) for k in ("separate", "preset", "force")}
            dest_dir = settings.cache_dir / "uploads" / uuid.uuid4().hex
            dest = dest_dir / safe_filename(upload.filename)

            def save() -> bool:
                """保存する。上限を超えたら途中で消して False。"""
                dest_dir.mkdir(parents=True, exist_ok=True)
                total = 0
                with open(dest, "wb") as fh:
                    while chunk := upload.file.read(1024 * 1024):
                        total += len(chunk)
                        if total > limit:
                            break
                        fh.write(chunk)
                if total > limit:
                    shutil.rmtree(dest_dir, ignore_errors=True)
                    return False
                return True

            if not await run_in_threadpool(save):
                raise too_large
            task = {
                "source_type": SOURCE_FILE,
                "file_path": dest,
                "original_name": upload.filename.replace("\\", "/").split("/")[-1],
            }
    elif ctype.startswith("application/json"):
        try:
            body = await request.json()
        except ValueError as e:
            raise HTTPException(status_code=400, detail="JSON を読めません。") from e
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON はオブジェクトで送ってください。")
        url = str(body.get("url") or "").strip()
        if not is_url(url):
            raise HTTPException(
                status_code=400,
                detail="URL（http:// か https:// で始まるもの）を指定してください。",
            )
        options = {k: body.get(k) for k in ("separate", "preset", "force")}
        task = {"source_type": SOURCE_URL, "url": url}
    else:
        raise HTTPException(
            status_code=415,
            detail="multipart/form-data（file）か application/json（url）で送ってください。",
        )

    preset = str(options["preset"]).strip() if options.get("preset") else None
    separate = _as_bool(options.get("separate"))
    try:
        if separate:
            await run_in_threadpool(_check_preset, request.app.state.session_factory, preset)
    except SeparationError as e:
        if dest_dir is not None:
            shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    task.update(separate=separate, preset=preset, force=_as_bool(options.get("force")))
    source_id = await run_in_threadpool(_manager(request).submit, task)
    return JSONResponse(
        {"source_id": source_id, "status": FETCH_QUEUED, "message": STATUS_MESSAGES[FETCH_QUEUED]},
        status_code=202,
    )


@router.get("/imports/{source_id}")
def get_import(
    source_id: int, request: Request, session: SessionDep
) -> dict[str, Any]:
    src = session.get(InputSource, source_id)
    if src is None:
        raise not_found("取り込み")
    manager = _manager(request)
    status = src.fetch_status or FETCH_DONE
    if status == FETCH_DONE and manager.is_active(source_id):
        status = FETCH_FETCHING  # 取り込み後の分割ジョブ登録がまだ
    if status == FETCH_FAILED:
        message = IMPORT_MESSAGES.get(src.error_code or UNKNOWN, IMPORT_MESSAGES[UNKNOWN])
    else:
        message = STATUS_MESSAGES.get(status, "")
    outcome = manager.outcome(source_id)
    return {
        "source_id": src.source_id,
        "source_type": src.source_type,
        "status": status,
        "error_code": src.error_code if status == FETCH_FAILED else None,
        "message": message,
        "error_detail": src.error_detail if status == FETCH_FAILED else None,
        "track_id": src.track_id if status == FETCH_DONE else None,
        "original_name": src.original_name,
        "url": src.url,
        "fetched_at": iso(src.fetched_at),
        "job_id": outcome.job_id if outcome else None,
        "job_created": outcome.job_created if outcome else False,
    }
