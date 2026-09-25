"""サーバーの PC でフォルダを開く（stem の保存場所をエクスプローラーで表示する）。

同じ PC のブラウザからだけ使える（iPhone など外からは 403）。Windows 以外では 501。
開く処理は `app.state.folder_opener`（テストでは差し替える）。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.api.common import SessionDep, is_local_request, not_found
from stemapp.config import Settings
from stemapp.library import resolve_data_path
from stemapp.models import SeparationJob, Stem, StemRendition
from stemapp.proc import open_in_explorer

router = APIRouter(prefix="/api", tags=["folders"])

FolderOpener = Callable[[Path], None]

MSG_NOT_LOCAL = "フォルダを開けるのは、サーバーと同じ PC のブラウザからだけです。"
MSG_OPEN_FAILED = "エクスプローラーを起動できませんでした。"


def supports_open_folder() -> bool:
    return os.name == "nt"


def _same_origin(request: Request) -> bool:
    """Origin ヘッダーがあれば、自分のページからか確かめる（他サイトからの POST 対策）。"""
    origin = request.headers.get("origin")
    if origin is None:
        return True
    return urlsplit(origin).netloc == request.headers.get("host", "")


def master_folder(session: Session, settings: Settings, job_id: int) -> Path:
    """ジョブの stem（master FLAC）が入っているフォルダ。無ければ 404。"""
    if session.get(SeparationJob, job_id) is None:
        raise not_found("ジョブ")
    stored = session.scalar(
        select(StemRendition.file_path)
        .join(Stem, Stem.stem_id == StemRendition.stem_id)
        .where(Stem.job_id == job_id, StemRendition.purpose == "master")
        .order_by(StemRendition.rendition_id)
        .limit(1)
    )
    if stored is None:
        raise not_found("stem の保存フォルダ")
    root = settings.data_dir.resolve()
    folder = resolve_data_path(settings, stored).resolve().parent
    if not folder.is_relative_to(root) or not folder.is_dir():
        raise not_found("stem の保存フォルダ")
    return folder


@router.post("/jobs/{job_id}/open-folder")
def open_folder(job_id: int, request: Request, session: SessionDep) -> dict[str, str]:
    if not is_local_request(request) or not _same_origin(request):
        raise HTTPException(status_code=403, detail=MSG_NOT_LOCAL)
    if not supports_open_folder():
        raise HTTPException(status_code=501, detail="フォルダを開く機能は Windows でだけ使えます。")
    folder = master_folder(session, request.app.state.settings, job_id)
    opener: FolderOpener = getattr(request.app.state, "folder_opener", open_in_explorer)
    try:
        opener(folder)
    except OSError as e:
        raise HTTPException(status_code=500, detail=MSG_OPEN_FAILED) from e
    return {"folder": str(folder)}
