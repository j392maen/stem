"""ファイル配信（音声は HTTP Range に対応）。データフォルダの外は返さない。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from stemapp.api.common import SessionDep, not_found, safe_data_file
from stemapp.delivery import MEDIA_TYPES
from stemapp.models import StemRendition, Waveform

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/renditions/{rendition_id}")
def rendition_file(
    rendition_id: int, request: Request, session: SessionDep
) -> FileResponse:
    rend = session.get(StemRendition, rendition_id)
    if rend is None:
        raise not_found("音声")
    path = safe_data_file(request.app.state.settings, rend.file_path)
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    # FileResponse は Range（部分取得）に対応している（206 と Content-Range を返す）
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private"})


@router.get("/peaks/{stem_id}/{samples_per_px}")
def peaks_file(
    stem_id: int, samples_per_px: int, request: Request, session: SessionDep
) -> FileResponse:
    wave = session.get(Waveform, (stem_id, samples_per_px))
    if wave is None:
        raise not_found("波形データ")
    path = safe_data_file(request.app.state.settings, wave.peaks_path)
    return FileResponse(
        path, media_type="application/octet-stream", headers={"Cache-Control": "private"}
    )
