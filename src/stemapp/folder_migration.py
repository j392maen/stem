"""T13 より前の保存フォルダ `data/stems/<job_id>/` を `data/stems/<曲>/<分け方>/` へ移す。

`stemapp migrate-folders`（`--dry-run` で予定だけ）から呼ぶ。`init_db` では自動で行わない。

- 対象: SEPARATION_JOB.output_dir が NULL で、`data/stems/<job_id>/` があるジョブ。
- 1ジョブずつ、フォルダを移し（同じドライブ内の名前の変更）、DB のパス（STEM_RENDITION.file_path、
  WAVEFORM.peaks_path、EXPORT.output_path のうち `stems/<job_id>/` で始まるもの）を書き換え、
  output_dir を入れて commit する。
- 移した後のファイル数・合計サイズが移す前と違う、DB が指すファイルが無い、などの失敗は、
  そのジョブの分だけ元に戻して（DB は rollback、フォルダは元の名前へ）次のジョブへ進む。
- 移し終えたジョブは output_dir が入るので、何度実行しても同じ結果になる。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.config import Settings
from stemapp.library import resolve_data_path
from stemapp.models import Export, SeparationJob, Stem, StemRendition, Waveform
from stemapp.stem_folders import STEMS_DIRNAME, FolderPlanner, remove_if_empty

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")

PLANNED = "planned"  # dry-run の予定
MOVED = "moved"
SKIPPED = "skipped"
FAILED = "failed"


@dataclass
class MigrationItem:
    job_id: int
    track_id: int
    old_dir: str  # データフォルダからの相対パス
    new_dir: str | None
    files: int = 0
    bytes: int = 0
    after_files: int | None = None
    after_bytes: int | None = None
    db_paths: int = 0  # 書き換える（書き換えた）DB のパスの数
    status: str = PLANNED
    message: str = ""


@dataclass
class MigrationReport:
    dry_run: bool
    items: list[MigrationItem] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for i in self.items if i.status == status)

    @property
    def before(self) -> tuple[int, int]:
        """対象（予定・移動済み・失敗）の移行前のファイル数と合計バイト数。"""
        targets = [i for i in self.items if i.status != SKIPPED]
        return sum(i.files for i in targets), sum(i.bytes for i in targets)

    @property
    def after(self) -> tuple[int, int]:
        """移した後（移動済みは新しい場所、失敗は元に戻した場所）のファイル数と合計バイト数。"""
        done = [i for i in self.items if i.status in (MOVED, FAILED)]
        return (
            sum(i.after_files or 0 for i in done),
            sum(i.after_bytes or 0 for i in done),
        )


def folder_stats(path: Path) -> tuple[int, int]:
    """フォルダの中の（下の階層も含む）ファイル数と合計バイト数。"""
    files = 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            total += p.stat().st_size
    return files, total


def _path_rows(session: Session, job_id: int) -> list[tuple[object, str]]:
    """ジョブのパスを持つ行と属性名（STEM_RENDITION・WAVEFORM・EXPORT）。"""
    stem_ids = select(Stem.stem_id).where(Stem.job_id == job_id)
    rows: list[tuple[object, str]] = []
    rows += [
        (r, "file_path")
        for r in session.scalars(select(StemRendition).where(StemRendition.stem_id.in_(stem_ids)))
    ]
    rows += [
        (w, "peaks_path")
        for w in session.scalars(select(Waveform).where(Waveform.stem_id.in_(stem_ids)))
    ]
    rows += [
        (e, "output_path")
        for e in session.scalars(select(Export).where(Export.job_id == job_id))
    ]
    return rows


def _matching(rows: list[tuple[object, str]], prefix: str) -> list[tuple[object, str]]:
    return [
        (row, attr)
        for row, attr in rows
        if isinstance(getattr(row, attr), str) and getattr(row, attr).startswith(prefix)
    ]


Rename = Callable[[Path, Path], None]


def migrate_folders(
    session: Session,
    settings: Settings,
    *,
    dry_run: bool = False,
    rename: Rename = os.rename,
) -> MigrationReport:
    """古い保存フォルダを新しい名前へ移す（dry_run なら予定だけ）。rename はテストで差し替える。"""
    report = MigrationReport(dry_run=dry_run)
    planner = FolderPlanner(session, settings)
    jobs = session.scalars(
        select(SeparationJob)
        .where(SeparationJob.output_dir.is_(None))
        .order_by(SeparationJob.job_id)
    ).all()
    for job in jobs:
        old_rel = f"{STEMS_DIRNAME}/{job.job_id}"
        old = settings.data_dir / STEMS_DIRNAME / str(job.job_id)
        prefix = old_rel + "/"
        refs = _matching(_path_rows(session, job.job_id), prefix)
        if not old.is_dir():
            if refs:
                report.items.append(
                    MigrationItem(
                        job.job_id, job.track_id, old_rel, None, db_paths=len(refs),
                        status=SKIPPED,
                        message=f"フォルダがありません（DB には {len(refs)} 件のパスがあります）",
                    )
                )
            continue
        files, total = folder_stats(old)
        if job.status in ACTIVE or job.postprocess_status in ACTIVE:
            report.items.append(
                MigrationItem(
                    job.job_id, job.track_id, old_rel, None, files, total, db_paths=len(refs),
                    status=SKIPPED,
                    message="分割中・配信用データの作成中のため移しません",
                )
            )
            continue
        new_rel = planner.job_folder(job)
        item = MigrationItem(
            job.job_id, job.track_id, old_rel, new_rel, files, total, db_paths=len(refs)
        )
        report.items.append(item)
        if dry_run:
            continue
        _move_one(session, settings, job, item, rename)
    return report


def _move_one(
    session: Session,
    settings: Settings,
    job: SeparationJob,
    item: MigrationItem,
    rename: Rename,
) -> None:
    assert item.new_dir is not None
    job_id = job.job_id
    old = settings.data_dir / STEMS_DIRNAME / str(job_id)
    new = settings.data_dir / Path(*item.new_dir.split("/"))
    old_prefix = item.old_dir + "/"
    new_prefix = item.new_dir + "/"
    moved = False
    try:
        if new.exists():
            raise RuntimeError(f"移し先が既にあります: {new}")
        new.parent.mkdir(parents=True, exist_ok=True)
        rename(old, new)
        moved = True
        refs = _matching(_path_rows(session, job_id), old_prefix)
        for row, attr in refs:
            setattr(row, attr, new_prefix + getattr(row, attr)[len(old_prefix):])
        job.output_dir = item.new_dir
        session.flush()
        item.after_files, item.after_bytes = folder_stats(new)
        if (item.after_files, item.after_bytes) != (item.files, item.bytes):
            raise RuntimeError(
                f"移した後のファイル数・サイズが違います（前 {item.files} 個 {item.bytes} バイト、"
                f"後 {item.after_files} 個 {item.after_bytes} バイト）"
            )
        missing = [
            getattr(row, attr)
            for row, attr in refs
            if not resolve_data_path(settings, getattr(row, attr)).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"DB が指すファイルがありません: {missing[0]} など {len(missing)} 件"
            )
        session.commit()
        item.status = MOVED
        item.db_paths = len(refs)
        log.info("job %d: %s → %s", job_id, item.old_dir, item.new_dir)
    except Exception as e:
        session.rollback()
        item.status = FAILED
        item.message = f"{type(e).__name__}: {e}"
        if moved:
            try:
                rename(new, old)
            except OSError as back:
                item.message += f"／元に戻せませんでした: {back}"
        remove_if_empty(settings, new.parent)
        if old.is_dir():
            item.after_files, item.after_bytes = folder_stats(old)
        log.error("job %d の移行に失敗しました（元に戻しました）: %s", job_id, item.message)
