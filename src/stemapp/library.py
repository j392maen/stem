"""曲ライブラリの共通処理（データフォルダ内のパス、完了済みジョブの検索）。

取り込み（ingest）と分離（separation）の両方から使うので、どちらにも依存しない場所に置く。
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.config import Settings
from stemapp.models import SeparationJob


def data_relative(settings: Settings, path: Path) -> str:
    """DB に保存するパス（データフォルダからの相対、/ 区切り）。"""
    return path.resolve().relative_to(settings.data_dir.resolve()).as_posix()


def resolve_data_path(settings: Settings, stored: str) -> Path:
    """DB に保存したパスを実際のパスにする（相対ならデータフォルダ基準）。"""
    p = Path(stored)
    return p if p.is_absolute() else settings.data_dir / p


def find_done_job(
    session: Session, track_id: int, preset_id: int | None = None
) -> SeparationJob | None:
    """その曲の完了済み full ジョブ（新しいもの）。無ければ None。

    preset_id を渡すと、そのプリセットで分割したものだけを探す。
    """
    stmt = select(SeparationJob).where(
        SeparationJob.track_id == track_id,
        SeparationJob.job_kind == "full",
        SeparationJob.status == "done",
    )
    if preset_id is not None:
        stmt = stmt.where(SeparationJob.preset_id == preset_id)
    return session.scalars(stmt.order_by(SeparationJob.job_id.desc())).first()
