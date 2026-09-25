"""キューポイント（曲の中の目印。loop_end_sec があれば A-B ループの区間）。"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, StringConstraints, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.api.common import SessionDep, not_found
from stemapp.models import CuePoint, Track

router = APIRouter(prefix="/api", tags=["cues"])

Color = Annotated[str, StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$")]
Label = Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]
Seconds = Annotated[float, Field(ge=0.0, le=24 * 60 * 60)]


class CueCreate(BaseModel):
    position_sec: Seconds
    loop_end_sec: Seconds | None = None
    label: Label | None = None
    color: Color | None = None

    @model_validator(mode="after")
    def _loop_after_start(self) -> CueCreate:
        if self.loop_end_sec is not None and self.loop_end_sec <= self.position_sec:
            raise ValueError("ループの終点は始点より後にしてください。")
        return self


class CueUpdate(BaseModel):
    """指定した項目だけ変える。loop_end_sec に null を送るとループを外す。"""

    position_sec: Seconds | None = None
    loop_end_sec: Seconds | None = None
    label: Label | None = None
    color: Color | None = None


def cue_to_dict(c: CuePoint) -> dict[str, Any]:
    return {
        "cue_id": c.cue_id,
        "track_id": c.track_id,
        "position_sec": c.position_sec,
        "loop_end_sec": c.loop_end_sec,
        "label": c.label,
        "color": c.color,
    }


def _check_range(track: Track, position: float, loop_end: float | None) -> None:
    if loop_end is not None and loop_end <= position:
        raise HTTPException(status_code=400, detail="ループの終点は始点より後にしてください。")
    limit = track.duration_sec
    if limit is not None and (position > limit + 0.01 or (loop_end or 0.0) > limit + 0.01):
        raise HTTPException(status_code=400, detail="曲の長さを超える位置は指定できません。")


def _get_track(session: Session, track_id: int) -> Track:
    track = session.get(Track, track_id)
    if track is None:
        raise not_found("曲")
    return track


def _get_cue(session: Session, cue_id: int) -> CuePoint:
    cue = session.get(CuePoint, cue_id)
    if cue is None:
        raise not_found("キュー")
    return cue


@router.get("/tracks/{track_id}/cues")
def list_cues(track_id: int, session: SessionDep) -> dict[str, Any]:
    _get_track(session, track_id)
    cues = session.scalars(
        select(CuePoint)
        .where(CuePoint.track_id == track_id)
        .order_by(CuePoint.position_sec, CuePoint.cue_id)
    ).all()
    return {"cues": [cue_to_dict(c) for c in cues]}


@router.post("/tracks/{track_id}/cues", status_code=201)
def create_cue(track_id: int, body: CueCreate, session: SessionDep) -> dict[str, Any]:
    track = _get_track(session, track_id)
    _check_range(track, body.position_sec, body.loop_end_sec)
    cue = CuePoint(
        track_id=track_id,
        position_sec=body.position_sec,
        loop_end_sec=body.loop_end_sec,
        label=body.label or None,
        color=body.color.upper() if body.color else None,
    )
    session.add(cue)
    session.commit()
    return cue_to_dict(cue)


@router.put("/cues/{cue_id}")
def update_cue(cue_id: int, body: CueUpdate, session: SessionDep) -> dict[str, Any]:
    cue = _get_cue(session, cue_id)
    fields = body.model_fields_set
    position = body.position_sec if body.position_sec is not None else cue.position_sec
    loop_end = body.loop_end_sec if "loop_end_sec" in fields else cue.loop_end_sec
    track = _get_track(session, cue.track_id)
    _check_range(track, position, loop_end)
    cue.position_sec = position
    cue.loop_end_sec = loop_end
    if "label" in fields:
        cue.label = body.label or None
    if "color" in fields:
        cue.color = body.color.upper() if body.color else None
    session.commit()
    return cue_to_dict(cue)


@router.delete("/cues/{cue_id}", status_code=204)
def delete_cue(cue_id: int, session: SessionDep) -> Response:
    cue = _get_cue(session, cue_id)
    session.delete(cue)
    session.commit()
    return Response(status_code=204)
