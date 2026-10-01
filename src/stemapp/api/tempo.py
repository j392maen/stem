"""速度変更（ピッチを保つ方式）の API: 伸縮済み音声の作成の登録・状態・キャンセル・進み具合（SSE）。

ピッチも変わる方式（playbackRate）はブラウザの中だけで完結するので API は無い。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from stemapp.api.common import SessionDep, not_found, safe_data_file, sse_response
from stemapp.delivery import MEDIA_TYPES
from stemapp.models import SeparationJob, TempoRender, TempoRendition
from stemapp.tempo.service import (
    FINISHED_STATUSES,
    MAX_RATIO,
    MIN_RATIO,
    TempoConflict,
    TempoInvalid,
    TempoNotFound,
    cancel_render,
    get_render,
    render_to_dict,
    request_render,
)

router = APIRouter(prefix="/api", tags=["tempo"])


class TempoRequest(BaseModel):
    ratio: float = Field(ge=MIN_RATIO, le=MAX_RATIO, allow_inf_nan=False)


@router.get("/jobs/{job_id}/tempo")
def list_renders(job_id: int, session: SessionDep) -> dict[str, Any]:
    """ジョブの速度を変えた音声（作成済み・作成待ち・作成中・失敗）の一覧（倍率の順）。"""
    if session.get(SeparationJob, job_id) is None:
        raise not_found("ジョブ")
    renders = session.scalars(
        select(TempoRender).where(TempoRender.job_id == job_id).order_by(TempoRender.ratio)
    ).all()
    return {"job_id": job_id, "renders": [render_to_dict(session, r) for r in renders]}


@router.post("/jobs/{job_id}/tempo")
def create_render(
    job_id: int, body: TempoRequest, response: Response, session: SessionDep
) -> dict[str, Any]:
    """ratio 倍（ピッチを保つ）の音声の作成を登録する。登録したら 202。

    作成済み（最後に使った時刻を更新）・作成待ち・作成中なら 200 でそれを返す。
    分割が終わっていないジョブは 409、1.000 倍は 400。
    """
    try:
        res = request_render(session, job_id, body.ratio)
    except TempoNotFound as e:
        raise not_found("ジョブ") from e
    except TempoConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except TempoInvalid as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    response.status_code = 202 if res.created else 200
    return {"created": res.created, "render": render_to_dict(session, res.render)}


def _render_or_404(session: SessionDep, render_id: int) -> TempoRender:
    try:
        return get_render(session, render_id)
    except TempoNotFound as e:
        raise not_found("速度を変えた音声") from e


@router.get("/tempo/{render_id}")
def get_tempo(render_id: int, session: SessionDep) -> dict[str, Any]:
    return render_to_dict(session, _render_or_404(session, render_id))


@router.post("/tempo/{render_id}/cancel")
def cancel_tempo(render_id: int, request: Request, session: SessionDep) -> dict[str, Any]:
    """作成をキャンセルする。終わっていれば 409。"""
    _render_or_404(session, render_id)
    try:
        render = cancel_render(session, request.app.state.settings, render_id)
    except TempoConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return render_to_dict(session, render)


@router.get("/tempo/{render_id}/events")
def tempo_events(render_id: int, request: Request) -> StreamingResponse:
    """Server-Sent Events。progress・stage・status が変わるたびに `event: tempo` を送る。

    終わった状態（done / failed / canceled）を送ったら閉じる（ジョブの SSE と同じ形）。
    """
    factory = request.app.state.session_factory
    with factory() as session:
        _render_or_404(session, render_id)

    def load() -> dict[str, Any] | None:
        with factory() as s:
            render = s.get(TempoRender, render_id)
            return render_to_dict(s, render) if render is not None else None

    return sse_response(
        request, load, event="tempo", gone_message="速度を変えた音声が削除されました。",
        finished=FINISHED_STATUSES,
    )


@router.get("/files/tempo/{render_id}/{stem_id}")
def tempo_file(
    render_id: int, stem_id: int, request: Request, session: SessionDep
) -> FileResponse:
    """伸縮済みの stem の音声（HTTP Range に対応）。データフォルダの外は返さない。"""
    rend = session.get(TempoRendition, (render_id, stem_id))
    if rend is None:
        raise not_found("音声")
    path = safe_data_file(request.app.state.settings, rend.file_path)
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private"})
