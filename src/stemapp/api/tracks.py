"""曲・ジョブ・stem の API。"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.api.common import SessionDep, iso, job_to_dict, not_found, preset_codes
from stemapp.config import Settings
from stemapp.delivery import missing_delivery
from stemapp.jobs import (
    ACTIVE_STATUSES,
    FINISHED_STATUSES,
    JobConflict,
    JobNotFound,
    delete_job,
    enqueue_full_job,
    request_cancel,
    request_postprocess,
)
from stemapp.models import (
    InputSource,
    SeparationJob,
    Stem,
    StemRendition,
    StemType,
    Track,
    Waveform,
)
from stemapp.separation.pipeline import SeparationError

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tracks"])

SSE_POLL_SEC = 0.5
SSE_KEEPALIVE_SEC = 15.0


# --- 曲 ---------------------------------------------------------------------------


def _jobs_by_track(session: Session) -> dict[int, list[SeparationJob]]:
    """曲ごとのジョブ（新しい順）。"""
    out: dict[int, list[SeparationJob]] = {}
    for job in session.scalars(select(SeparationJob).order_by(SeparationJob.job_id.desc())):
        out.setdefault(job.track_id, []).append(job)
    return out


def _track_summary(
    track: Track, jobs: list[SeparationJob], presets: dict[int, str]
) -> dict[str, Any]:
    """jobs はその曲のジョブ（新しい順）。"""
    playable = next((j for j in jobs if j.job_kind == "full" and j.status == "done"), None)
    return {
        "track_id": track.track_id,
        "title": track.title,
        "artist": track.artist,
        "duration_sec": track.duration_sec,
        "created_at": iso(track.created_at),
        "latest_job": job_to_dict(jobs[0], presets) if jobs else None,
        # 再生に使うジョブ（完了した full ジョブのうち新しいもの）
        "playable_job_id": playable.job_id if playable is not None else None,
    }


@router.get("/tracks")
def list_tracks(session: SessionDep) -> dict[str, Any]:
    presets = preset_codes(session)
    jobs = _jobs_by_track(session)
    tracks = session.scalars(select(Track).order_by(Track.track_id.desc())).all()
    return {"tracks": [_track_summary(t, jobs.get(t.track_id, []), presets) for t in tracks]}


@router.get("/tracks/{track_id}")
def get_track(track_id: int, session: SessionDep) -> dict[str, Any]:
    track = session.get(Track, track_id)
    if track is None:
        raise not_found("曲")
    presets = preset_codes(session)
    jobs = session.scalars(
        select(SeparationJob)
        .where(SeparationJob.track_id == track_id)
        .order_by(SeparationJob.job_id.desc())
    ).all()
    sources = session.scalars(
        select(InputSource)
        .where(InputSource.track_id == track_id)
        .order_by(InputSource.source_id)
    ).all()
    out = _track_summary(track, list(jobs), presets)
    levels = _stem_levels(session, [j.job_id for j in jobs if j.status == "done"])
    out["jobs"] = [
        {**job_to_dict(j, presets), "stem_rms_db": levels.get(j.job_id, {})} for j in jobs
    ]
    out["sources"] = [
        {
            "source_id": s.source_id,
            "source_type": s.source_type,
            "original_name": s.original_name,
            "url": s.url,
            "fetched_at": iso(s.fetched_at),
        }
        for s in sources
    ]
    return out


def _stem_levels(session: Session, job_ids: list[int]) -> dict[int, dict[str, float | None]]:
    """ジョブごとの stem の RMS（dBFS）。{job_id: {stem の code: rms_db}}（表示順）。"""
    out: dict[int, dict[str, float | None]] = {}
    if not job_ids:
        return out
    rows = session.execute(
        select(Stem.job_id, StemType.code, Stem.rms_db)
        .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
        .where(Stem.job_id.in_(job_ids))
        .order_by(Stem.job_id, StemType.display_order)
    ).all()
    for job_id, code, level in rows:
        out.setdefault(job_id, {})[code] = round(level, 2) if level is not None else None
    return out


@router.delete("/tracks/{track_id}")
def delete_track(
    track_id: int, request: Request, session: SessionDep
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    track = session.get(Track, track_id)
    if track is None:
        raise not_found("曲")
    jobs = session.scalars(select(SeparationJob).where(SeparationJob.track_id == track_id)).all()
    if any(j.status in ACTIVE_STATUSES for j in jobs):
        raise HTTPException(
            status_code=409,
            detail="分割待ち・分割中のジョブがあるため削除できません。キャンセルしてから削除してください。",
        )
    job_ids = [j.job_id for j in jobs]
    session.delete(track)  # JOB・STEM・INPUT_SOURCE などは外部キーの CASCADE で消える
    session.commit()
    shutil.rmtree(settings.tracks_dir / str(track_id), ignore_errors=True)
    for job_id in job_ids:
        shutil.rmtree(settings.stems_dir / str(job_id), ignore_errors=True)
    log.info("曲を削除しました（track %d, job %s）。", track_id, job_ids)
    return {"deleted": True, "track_id": track_id, "job_ids": job_ids}


# --- ジョブ -------------------------------------------------------------------------


class JobRequest(BaseModel):
    preset: str | None = None
    force: bool = False


ENQUEUE_MESSAGES = {
    None: "分割ジョブを登録しました。",
    "active": "この曲はこの分け方で分割待ち・分割中です。",
    "done": "この曲はこの分け方で分割済みです（分割し直すには force を指定してください）。",
}


@router.post("/tracks/{track_id}/jobs")
def create_job(
    track_id: int,
    response: Response,
    session: SessionDep,
    body: JobRequest | None = None,
) -> dict[str, Any]:
    body = body or JobRequest()
    try:
        res = enqueue_full_job(session, track_id, body.preset, force=body.force)
    except JobNotFound as e:
        raise not_found("曲") from e
    except SeparationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    response.status_code = 201 if res.created else 200
    return {
        "created": res.created,
        "reason": res.reason,
        "message": ENQUEUE_MESSAGES.get(res.reason, ""),
        "job": job_to_dict(res.job, preset_codes(session)),
    }


def _get_job(session: Session, job_id: int) -> SeparationJob:
    job = session.get(SeparationJob, job_id)
    if job is None:
        raise not_found("ジョブ")
    return job


@router.get("/jobs/{job_id}")
def get_job(job_id: int, session: SessionDep) -> dict[str, Any]:
    return job_to_dict(_get_job(session, job_id), preset_codes(session))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, session: SessionDep) -> dict[str, Any]:
    try:
        job = request_cancel(session, job_id)
    except JobNotFound as e:
        raise not_found("ジョブ") from e
    except JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return job_to_dict(job, preset_codes(session))


@router.delete("/jobs/{job_id}")
def delete_job_api(job_id: int, request: Request, session: SessionDep) -> dict[str, Any]:
    """ジョブ（1つの分け方）を消す。stem のファイルも消える。曲とほかのジョブは残る。

    分割待ち・分割中（配信用データの作成中を含む）は 409。
    """
    settings: Settings = request.app.state.settings
    try:
        job = delete_job(session, settings, job_id)
    except JobNotFound as e:
        raise not_found("ジョブ") from e
    except JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"deleted": True, "job_id": job_id, "track_id": job.track_id}


POSTPROCESS_MESSAGES = {
    None: "配信用データの作成を登録しました。",
    "active": "配信用データを作成待ち・作成中です。",
    "ready": "配信用データはそろっています。",
}


@router.post("/jobs/{job_id}/postprocess")
def postprocess_job(job_id: int, response: Response, session: SessionDep) -> dict[str, Any]:
    """配信用データ（stream rendition・peaks）が欠けているとき、ワーカーで作り直す。

    登録したら 202、欠けていない・作成待ちなら 200。done 以外のジョブは 409。
    """
    job = _get_job(session, job_id)
    missing = missing_delivery(session, job_id) if job.status == "done" else []
    try:
        res = request_postprocess(session, job_id, missing)
    except JobNotFound as e:
        raise not_found("ジョブ") from e
    except JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    response.status_code = 202 if res.created else 200
    return {
        "created": res.created,
        "reason": res.reason,
        "message": POSTPROCESS_MESSAGES.get(res.reason, ""),
        "missing": missing,
        "job": job_to_dict(res.job, preset_codes(session)),
    }


@router.get("/jobs/{job_id}/events")
def job_events(job_id: int, request: Request) -> StreamingResponse:
    """Server-Sent Events。progress・stage・status が変わるたびに `event: job` を送る。

    終わった状態（done / failed / canceled）を送ったら閉じる。
    """
    factory = request.app.state.session_factory
    with factory() as session:
        _get_job(session, job_id)
    poll_sec: float = getattr(request.app.state, "sse_poll_sec", SSE_POLL_SEC)

    def load() -> dict[str, Any] | None:
        with factory() as s:
            job = s.get(SeparationJob, job_id)
            return job_to_dict(job, preset_codes(s)) if job is not None else None

    async def stream() -> AsyncIterator[str]:
        last: tuple[object, ...] | None = None
        last_sent = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            data = await run_in_threadpool(load)
            if data is None:
                payload = json.dumps({"detail": "ジョブが削除されました。"}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n"
                return
            key = (data["status"], data["progress"], data["stage"])
            if key != last:
                yield f"event: job\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                last = key
                last_sent = time.monotonic()
            elif time.monotonic() - last_sent >= SSE_KEEPALIVE_SEC:
                yield ": keepalive\n\n"
                last_sent = time.monotonic()
            if data["status"] in FINISHED_STATUSES:
                return
            await asyncio.sleep(poll_sec)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- stem ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/stems")
def job_stems(job_id: int, session: SessionDep) -> dict[str, Any]:
    job = _get_job(session, job_id)
    rows = session.execute(
        select(Stem, StemType)
        .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
        .where(Stem.job_id == job_id)
        .order_by(StemType.display_order)
    ).all()
    code_of = {s.stem_id: t.code for s, t in rows}
    stem_ids = list(code_of)
    renditions: dict[int, list[StemRendition]] = {}
    for r in session.scalars(
        select(StemRendition)
        .where(StemRendition.stem_id.in_(stem_ids))
        .order_by(StemRendition.rendition_id)
    ):
        renditions.setdefault(r.stem_id, []).append(r)
    waves: dict[int, list[Waveform]] = {}
    for w in session.scalars(
        select(Waveform).where(Waveform.stem_id.in_(stem_ids)).order_by(Waveform.samples_per_px)
    ):
        waves.setdefault(w.stem_id, []).append(w)

    stems = []
    for s, t in rows:
        stems.append(
            {
                "stem_id": s.stem_id,
                "stem_type_id": t.stem_type_id,
                "code": t.code,
                "display_name": t.display_name,
                "color": t.color,
                "parent_stem_id": s.parent_stem_id,
                "parent_code": code_of.get(s.parent_stem_id) if s.parent_stem_id else None,
                "is_residual": s.is_residual,
                "rms_db": s.rms_db,
                "is_silent": s.is_silent,
                "renditions": [
                    {
                        "rendition_id": r.rendition_id,
                        "purpose": r.purpose,
                        "codec": r.codec,
                        "bitrate_kbps": r.bitrate_kbps,
                        "bytes": r.bytes,
                        "url": f"/api/files/renditions/{r.rendition_id}",
                    }
                    for r in renditions.get(s.stem_id, [])
                ],
                "peaks": [
                    {
                        "samples_per_px": w.samples_per_px,
                        "url": f"/api/files/peaks/{s.stem_id}/{w.samples_per_px}",
                    }
                    for w in waves.get(s.stem_id, [])
                ],
            }
        )
    missing = missing_delivery(session, job_id) if job.status == "done" else []
    return {
        "job_id": job.job_id,
        "track_id": job.track_id,
        "status": job.status,
        "output_gain_db": job.output_gain_db,
        "postprocess_status": job.postprocess_status,
        # 配信用データ（stream と全解像度の peaks）がそろっているか
        "delivery_ready": job.status == "done" and not missing,
        "delivery_missing": missing,
        "stems": stems,
    }
