"""書き出しの API（作成・状態・ダウンロード）。

- POST /api/jobs/{job_id}/exports: 種類・形式・対象を受け取り、書き出しを登録する（202）。
- GET /api/exports/{export_id}: 状態・進捗・ファイル名・大きさ。
- GET /api/exports/{export_id}/download: ファイル（Range 対応、日本語のファイル名は RFC 5987）。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from stemapp.api.common import SessionDep, iso, not_found, safe_data_file
from stemapp.config import Settings
from stemapp.exports import (
    ExportConflict,
    ExportInvalid,
    ExportManager,
    ExportNotFound,
    ExportRequest,
    content_disposition,
    create_export,
    plan_export,
)
from stemapp.exports.render import FORMAT_LABELS
from stemapp.exports.service import DONE, FAILED
from stemapp.models import Export, SeparationJob

router = APIRouter(prefix="/api", tags=["exports"])

DOWNLOAD_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".zip": "application/zip",
}


class MixStemIn(BaseModel):
    code: str = Field(min_length=1, max_length=50)
    gain_db: float = Field(default=0.0, ge=-60.0, le=12.0)


class ExportIn(BaseModel):
    export_type: Literal["single", "all", "mix"]
    format: Literal["wav", "flac", "mp3"]
    # single: 書き出す stem の code
    stem_code: str | None = Field(default=None, max_length=50)
    # all: true なら子に分かれていても親（いちばん上の stem）だけ
    parents_only: bool = False
    # mix: 組み合わせプリセット（指定すると stems は使わない）
    listen_preset_id: int | None = None
    # mix: 選択中の stem と音量
    stems: list[MixStemIn] = Field(default_factory=list, max_length=200)


def export_to_dict(exp: Export, settings: Settings, track_id: int | None) -> dict[str, Any]:
    ready = exp.status == DONE and exp.output_path is not None
    expires = exp.created_at + timedelta(hours=settings.export_ttl_hours)
    return {
        "export_id": exp.export_id,
        "job_id": exp.job_id,
        "track_id": track_id,
        "export_type": exp.export_type,
        "format": exp.format,
        "format_label": FORMAT_LABELS.get(exp.format, exp.format),
        "zip": exp.export_type == "all",
        "listen_preset_id": exp.listen_preset_id,
        "status": exp.status,
        "progress": exp.progress,
        "stage": exp.stage,
        "error_message": exp.error_message,
        "filename": exp.filename,
        "bytes": exp.bytes,
        "mix_gain_db": exp.mix_gain_db,
        "created_at": iso(exp.created_at),
        "finished_at": iso(exp.finished_at),
        "expires_at": iso(expires),
        "download_url": f"/api/exports/{exp.export_id}/download" if ready else None,
    }


def _manager(request: Request) -> ExportManager:
    return request.app.state.export_manager


def _track_id(session: SessionDep, job_id: int) -> int | None:
    job = session.get(SeparationJob, job_id)
    return job.track_id if job is not None else None


@router.post("/jobs/{job_id}/exports", status_code=202)
def create_job_export(
    job_id: int, body: ExportIn, request: Request, session: SessionDep
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    req = ExportRequest(
        export_type=body.export_type,
        format=body.format,
        stem_code=body.stem_code,
        parents_only=body.parents_only,
        listen_preset_id=body.listen_preset_id,
        stems=[(s.code, s.gain_db) for s in body.stems],
    )
    try:
        plan = plan_export(session, job_id, req)
    except ExportNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ExportConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ExportInvalid as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    manager = _manager(request)
    manager.cleanup()
    exp = create_export(session, plan)
    manager.submit(exp.export_id)
    return {"export": export_to_dict(exp, settings, plan.track_id)}


def _get_export(session: SessionDep, export_id: int) -> Export:
    exp = session.get(Export, export_id)
    if exp is None:
        raise not_found("書き出し（期限切れで消えた可能性があります）")
    return exp


@router.get("/exports/{export_id}")
def get_export(export_id: int, request: Request, session: SessionDep) -> dict[str, Any]:
    exp = _get_export(session, export_id)
    settings: Settings = request.app.state.settings
    return {"export": export_to_dict(exp, settings, _track_id(session, exp.job_id))}


@router.get("/exports/{export_id}/download")
def download_export(export_id: int, request: Request, session: SessionDep) -> FileResponse:
    exp = _get_export(session, export_id)
    if exp.status == FAILED:
        raise HTTPException(status_code=409, detail="この書き出しは失敗しました。")
    if exp.status != DONE or not exp.output_path:
        raise HTTPException(status_code=409, detail="まだ書き出し中です。")
    path = safe_data_file(request.app.state.settings, exp.output_path)
    filename = exp.filename or path.name
    media_type = DOWNLOAD_TYPES.get(path.suffix.lower(), "application/octet-stream")
    # FileResponse は Range（部分取得）に対応している（206 と Content-Range を返す）
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "Content-Disposition": content_disposition(filename),
            "Cache-Control": "private, no-store",
        },
    )
