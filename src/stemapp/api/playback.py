"""端末（DEVICE）と、曲ごと・端末ごとの再生の状態（PLAYBACK_STATE。続きから再生。T06b）。

- 端末はブラウザが作るランダムな ID（device_key。localStorage に保存）で見分ける。
  `POST /api/devices` で登録（既にあれば最後に見た時刻を更新）し、device_id を受け取る。
- 再生の状態は `PUT /api/tracks/{track_id}/playback/{device_id}` で上書き保存する（一定間隔・
  一時停止・画面を離れるとき）。`GET /api/tracks/{track_id}/playback` は全端末の状態を新しい順に返す
  （自分の端末の状態で続きから再生し、ほかの端末の状態は「PC で 1:23 まで聴いた」に使う）。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.api.common import SessionDep, iso, not_found
from stemapp.models import (
    Device,
    ListenPreset,
    PlaybackState,
    SeparationJob,
    Track,
    utcnow,
)

router = APIRouter(prefix="/api", tags=["playback"])

DeviceKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{8,64}$")]
DeviceName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
Kind = Literal["pc", "iphone", "ipad", "other"]
Code = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
TempoMode = Literal["pitch", "instant", "keep"]

# 名前を付けずに登録したときの名前
DEFAULT_NAMES: dict[str, str] = {"pc": "PC", "iphone": "iPhone", "ipad": "iPad", "other": "スマホ"}
MAX_STEMS = 200


class DeviceRegister(BaseModel):
    device_key: DeviceKey
    kind: Kind = "other"
    # 省略すると、新しい端末は種類から（「iPhone」「PC」など）。既にある端末は今の名前のまま
    name: DeviceName | None = None


class DeviceUpdate(BaseModel):
    name: DeviceName | None = None
    kind: Kind | None = None


class PlaybackSave(BaseModel):
    # 送ってきた端末の ID（device_id の端末と照合する。違えば 403）
    device_key: DeviceKey
    position_sec: float = Field(ge=0.0, le=24 * 60 * 60)
    job_id: int | None = None
    selected: list[Code] | None = Field(default=None, max_length=MAX_STEMS)
    # stem（code）→ 音量 dB
    gains_db: dict[Code, Annotated[float, Field(ge=-60.0, le=24.0)]] | None = Field(
        default=None, max_length=MAX_STEMS
    )
    listen_preset_id: int | None = None
    tempo_ratio: float | None = Field(default=None, ge=0.5, le=2.0)
    tempo_mode: TempoMode | None = None


def device_to_dict(d: Device) -> dict[str, Any]:
    return {
        "device_id": d.device_id,
        "name": d.name,
        "kind": d.kind,
        "last_seen_at": iso(d.last_seen_at),
    }


def state_to_dict(st: PlaybackState, device: Device, session: Session) -> dict[str, Any]:
    # 消えたジョブ・組み合わせは null にして返す（古い DB では外部キーが無いことがあるため）
    job = session.get(SeparationJob, st.job_id) if st.job_id is not None else None
    preset = (
        session.get(ListenPreset, st.listen_preset_id) if st.listen_preset_id is not None else None
    )
    return {
        "device_id": st.device_id,
        "device_name": device.name,
        "device_kind": device.kind,
        "track_id": st.track_id,
        "position_sec": st.position_sec,
        "job_id": job.job_id if job is not None and job.track_id == st.track_id else None,
        "selected": st.selected_json if isinstance(st.selected_json, list) else None,
        "gains_db": st.channel_gains_json if isinstance(st.channel_gains_json, dict) else None,
        "listen_preset_id": preset.listen_preset_id if preset is not None else None,
        "listen_preset_name": preset.name if preset is not None else None,
        "tempo_ratio": st.tempo_ratio,
        "tempo_mode": st.tempo_mode,
        "updated_at": iso(st.updated_at),
    }


def _get_device(session: Session, device_id: int) -> Device:
    device = session.get(Device, device_id)
    if device is None:
        raise not_found("端末")
    return device


def _get_track(session: Session, track_id: int) -> Track:
    track = session.get(Track, track_id)
    if track is None:
        raise not_found("曲")
    return track


@router.post("/devices")
def register_device(body: DeviceRegister, session: SessionDep) -> dict[str, Any]:
    """端末を登録する（同じ device_key なら同じ行。最後に見た時刻を更新する）。"""
    device = session.scalars(select(Device).where(Device.device_key == body.device_key)).first()
    if device is None:
        device = Device(
            device_key=body.device_key,
            kind=body.kind,
            name=body.name or DEFAULT_NAMES[body.kind],
        )
        session.add(device)
    elif body.name:
        device.name = body.name
    device.last_seen_at = utcnow()
    session.commit()
    return device_to_dict(device)


@router.get("/devices")
def list_devices(session: SessionDep) -> dict[str, Any]:
    rows = session.scalars(select(Device).order_by(Device.device_id)).all()
    return {"devices": [device_to_dict(d) for d in rows]}


@router.put("/devices/{device_id}")
def update_device(device_id: int, body: DeviceUpdate, session: SessionDep) -> dict[str, Any]:
    device = _get_device(session, device_id)
    if body.name is not None:
        device.name = body.name
    if body.kind is not None:
        device.kind = body.kind
    session.commit()
    return device_to_dict(device)


@router.get("/tracks/{track_id}/playback")
def list_playback(track_id: int, session: SessionDep) -> dict[str, Any]:
    """この曲の、端末ごとの再生の状態（新しい順）。"""
    _get_track(session, track_id)
    rows = session.execute(
        select(PlaybackState, Device)
        .join(Device, Device.device_id == PlaybackState.device_id)
        .where(PlaybackState.track_id == track_id)
        .order_by(PlaybackState.updated_at.desc(), PlaybackState.device_id)
    ).all()
    return {"states": [state_to_dict(st, d, session) for st, d in rows]}


@router.put("/tracks/{track_id}/playback/{device_id}")
def save_playback(
    track_id: int, device_id: int, body: PlaybackSave, session: SessionDep
) -> dict[str, Any]:
    """この端末の、この曲の再生の状態を上書き保存する。"""
    track = _get_track(session, track_id)
    device = _get_device(session, device_id)
    if device.device_key is None or device.device_key != body.device_key:
        raise HTTPException(status_code=403, detail="この端末の再生の状態ではありません。")
    job_id = body.job_id
    if job_id is not None:
        job = session.get(SeparationJob, job_id)
        if job is None:
            job_id = None  # 消えた分け方は覚えない（エラーにしない）
        elif job.track_id != track_id:
            raise HTTPException(status_code=400, detail="この曲の分け方ではありません。")
    preset_id = body.listen_preset_id
    if preset_id is not None and session.get(ListenPreset, preset_id) is None:
        preset_id = None  # 消えた組み合わせは覚えない
    position = body.position_sec
    if track.duration_sec is not None:
        position = min(position, track.duration_sec)
    st = session.get(PlaybackState, (device_id, track_id))
    if st is None:
        st = PlaybackState(device_id=device_id, track_id=track_id)
        session.add(st)
    st.position_sec = position
    st.job_id = job_id
    st.selected_json = list(body.selected) if body.selected is not None else None
    gains = {k: v for k, v in (body.gains_db or {}).items() if v != 0}
    st.channel_gains_json = gains or None
    st.listen_preset_id = preset_id
    st.tempo_ratio = body.tempo_ratio
    st.tempo_mode = body.tempo_mode
    st.updated_at = utcnow()
    device.last_seen_at = utcnow()
    session.commit()
    return state_to_dict(st, device, session)
