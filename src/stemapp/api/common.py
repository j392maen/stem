"""API の共通部品（DB セッション、JSON への変換、安全なファイルパス）。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.config import Settings
from stemapp.library import resolve_data_path
from stemapp.models import SeparationJob, SeparationPreset


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:  # SQLite から読むとタイムゾーンが落ちる（保存は UTC）
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def preset_codes(session: Session) -> dict[int, str]:
    return {p.preset_id: p.code for p in session.scalars(select(SeparationPreset))}


def job_to_dict(job: SeparationJob, presets: dict[int, str]) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "track_id": job.track_id,
        "job_kind": job.job_kind,
        "preset": presets.get(job.preset_id) if job.preset_id is not None else None,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "run_on": job.run_on,
        "cancel_requested": bool(job.cancel_requested),
        "output_gain_db": job.output_gain_db,
        "error_message": job.error_message,
        "created_at": iso(job.created_at),
        "started_at": iso(job.started_at),
        "finished_at": iso(job.finished_at),
    }


def not_found(what: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{what}が見つかりません。")


def safe_data_file(settings: Settings, stored: str) -> Path:
    """DB に保存したパスを実際のファイルにする。データフォルダの外や存在しないものは 404。"""
    root = settings.data_dir.resolve()
    try:
        path = resolve_data_path(settings, stored).resolve()
    except (OSError, ValueError) as e:
        raise not_found("ファイル") from e
    if not path.is_relative_to(root) or not path.is_file():
        raise not_found("ファイル")
    return path
