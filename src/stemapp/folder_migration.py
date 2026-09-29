"""T13 より前の保存フォルダ `data/stems/<job_id>/` を `data/stems/<曲>/<分け方>/` へ移す。

`stemapp migrate-folders`（`--dry-run` で予定だけ）から呼ぶ。`init_db` では自動で行わない。

- 対象: SEPARATION_JOB.output_dir が NULL で、`data/stems/<job_id>/` があるジョブ。
- 1ジョブずつ、フォルダを移し（同じドライブ内の名前の変更）、DB のパス（STEM_RENDITION.file_path、
  WAVEFORM.peaks_path、EXPORT.output_path のうち `stems/<job_id>/` で始まるもの）を書き換え、
  output_dir を入れて commit する。
- 移した後のファイル数・合計サイズが移す前と違う、DB が指すファイルが無い、などの失敗は、
  そのジョブの分だけ元に戻して（DB は rollback、フォルダは元の名前へ）次のジョブへ進む。
- 移し終えたジョブは output_dir が入るので、何度実行しても同じ結果になる。
- `stems/<job_id>` が別の曲の新しい形のフォルダ（ほかのジョブの output_dir）だったり、中身が
  旧形式でなかったりするときは移さない。
- 移す前に予定（job_id, 元, 先）を `data/stems/.migrate-journal.json` に書き、commit 後に消す。
  Ctrl+C などでは元に戻してから止まる。プロセスが強制終了して記録が残ったときは、次の実行の
  最初に照合して直す（`recover_journal`）。
"""

from __future__ import annotations

import json
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
from stemapp.stem_folders import (
    STEMS_DIRNAME,
    FolderPlanner,
    is_legacy_job_dir,
    legacy_dir,
    remove_if_empty,
)

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")

PLANNED = "planned"  # dry-run の予定
MOVED = "moved"
SKIPPED = "skipped"
FAILED = "failed"
NEEDS_CHECK = "check"  # 元に戻せなかった（要確認。次の実行で直す）


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
    notes: list[str] = field(default_factory=list)  # 前回の中断を直した内容など

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
        done = [i for i in self.items if i.status in (MOVED, FAILED, NEEDS_CHECK)]
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

JOURNAL_NAME = ".migrate-journal.json"


class MigrationBlocked(RuntimeError):
    """前回の中断の跡を自動では直せない（人の確認が要る）。"""


def journal_path(settings: Settings) -> Path:
    return settings.data_dir / STEMS_DIRNAME / JOURNAL_NAME


def _write_journal(settings: Settings, job_id: int, old_rel: str, new_rel: str) -> None:
    """これから移すもの（job_id, 元, 先）を書いておく（中断したら次の実行で直すため）。"""
    path = journal_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"job_id": job_id, "old": old_rel, "new": new_rel}, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _clear_journal(settings: Settings) -> None:
    journal_path(settings).unlink(missing_ok=True)


def _data_path(settings: Settings, rel: str) -> Path:
    return settings.data_dir / Path(*rel.split("/"))


def recover_journal(
    session: Session, settings: Settings, *, dry_run: bool = False, rename: Rename = os.rename
) -> str | None:
    """`_recover_journal` の直した内容（無ければ None）だけを返す。"""
    return _recover_journal(session, settings, dry_run=dry_run, rename=rename)[0]


def _recover_journal(
    session: Session, settings: Settings, *, dry_run: bool = False, rename: Rename = os.rename
) -> tuple[str | None, tuple[int, str] | None]:
    """前回の移行が途中で止まっていたら（予定の記録が残っていたら）直す。

    - DB が新しい場所を指している（commit 済み）→ 記録を消すだけ。
    - DB は古いままで、フォルダだけ新しい場所にある → フォルダを元に戻す（この後の移行で移し直す）。
    - フォルダがまだ元の場所にある → 何も移っていないので記録を消す。
    - 元にも先にもある・どちらにも無い → 自動では決められないので MigrationBlocked。
    dry_run なら何も変えず、する予定を返す。
    (直した内容（無ければ None）, dry_run でフォルダを戻す予定のとき (job_id, 新しい場所)) を返す。
    """
    path = journal_path(settings)
    if not path.is_file():
        return None, None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        job_id = int(entry["job_id"])
        old_rel = str(entry["old"])
        new_rel = str(entry["new"])
    except (ValueError, KeyError, TypeError, OSError) as e:
        raise MigrationBlocked(f"前回の移行の記録 {path} を読めません: {e}") from e
    old = _data_path(settings, old_rel)
    new = _data_path(settings, new_rel)
    output_dir = session.scalar(
        select(SeparationJob.output_dir).where(SeparationJob.job_id == job_id)
    )
    if output_dir == new_rel:
        note = f"前回の移行（job {job_id}）は完了していました。記録を消します。"
        if not dry_run:
            _clear_journal(settings)
        return note, None
    if new.is_dir() and not old.exists():
        note = (
            f"前回中断した移行（job {job_id}）のフォルダを元に戻します: {new_rel} → {old_rel}"
        )
        if dry_run:
            return note, (job_id, new_rel)
        rename(new, old)
        remove_if_empty(settings, new.parent)
        _clear_journal(settings)
        return note, None
    if old.is_dir() and not new.exists():
        note = f"前回の移行（job {job_id}）はフォルダを移す前に止まっていました。記録を消します。"
        if not dry_run:
            _clear_journal(settings)
        return note, None
    raise MigrationBlocked(
        f"前回中断した移行（job {job_id}）を自動では直せません。{old} と {new} を確かめ、"
        f"直してから {path} を消してください。"
    )


def migrate_folders(
    session: Session,
    settings: Settings,
    *,
    dry_run: bool = False,
    rename: Rename = os.rename,
) -> MigrationReport:
    """古い保存フォルダを新しい名前へ移す（dry_run なら予定だけ）。rename はテストで差し替える。

    前回の中断の跡を直せないときは MigrationBlocked。
    """
    report = MigrationReport(dry_run=dry_run)
    note, pending = _recover_journal(session, settings, dry_run=dry_run, rename=rename)
    if note:
        report.notes.append(note)
    planner = FolderPlanner(session, settings)
    if pending is not None:
        # dry-run: 前回中断したジョブは、元に戻してから同じ場所へ移す予定として扱う
        pending_job = session.get(SeparationJob, pending[0])
        if pending_job is not None:
            planner.reserve(pending_job.track_id, pending[1])
    jobs = session.scalars(
        select(SeparationJob)
        .where(SeparationJob.output_dir.is_(None))
        .order_by(SeparationJob.job_id)
    ).all()
    for job in jobs:
        old_rel = f"{STEMS_DIRNAME}/{job.job_id}"
        old = legacy_dir(settings, job.job_id)
        prefix = old_rel + "/"
        refs = _matching(_path_rows(session, job.job_id), prefix)
        if pending is not None and pending[0] == job.job_id:
            files, total = folder_stats(_data_path(settings, pending[1]))
            report.items.append(
                MigrationItem(
                    job.job_id, job.track_id, old_rel, pending[1], files, total,
                    db_paths=len(refs), status=PLANNED,
                    message=(
                        f"前回中断した移行を元に戻してから移す予定（今は {pending[1]} にあります）"
                    ),
                )
            )
            continue
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
        if not is_legacy_job_dir(session, settings, job.job_id):
            # 新しい形の曲フォルダ（別の曲）などは、このジョブのものとして扱わない
            if refs:
                report.items.append(
                    MigrationItem(
                        job.job_id, job.track_id, old_rel, None, db_paths=len(refs),
                        status=SKIPPED,
                        message=f"{old_rel} は古い形の保存フォルダではないため移しません",
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
        if item.status == NEEDS_CHECK:
            break  # 記録を残したまま止める（次の実行で直す）
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
    old = legacy_dir(settings, job_id)
    new = _data_path(settings, item.new_dir)
    old_prefix = item.old_dir + "/"
    new_prefix = item.new_dir + "/"
    moved = False
    committed = False
    try:
        if new.exists():
            raise RuntimeError(f"移し先が既にあります: {new}")
        new.parent.mkdir(parents=True, exist_ok=True)
        _write_journal(settings, job_id, item.old_dir, item.new_dir)
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
        # ここから先は移行済み（DB もフォルダも新しい場所）。何が起きても戻さない
        committed = True
        item.status = MOVED
        item.db_paths = len(refs)
        log.info("job %d: %s → %s", job_id, item.old_dir, item.new_dir)
        _clear_journal(settings)
    except BaseException as e:
        if committed:
            # 記録の削除に失敗した・Ctrl+C など。移行は済んでいるので MOVED のまま。
            # 残った記録は次の実行で recover_journal が（output_dir == 新しい場所として）消す
            item.message = f"記録を消せませんでした（{type(e).__name__}: {e}）。次の実行で消します"
            log.warning("job %d: %s", job_id, item.message)
            if not isinstance(e, Exception):
                raise
            return
        # Ctrl+C などでも、このジョブの分を元に戻してから止める
        session.rollback()
        item.status = FAILED
        item.message = f"{type(e).__name__}: {e}"
        restored = True
        # rename の途中・直後に止められたときも、実際の状態を見て戻す
        if moved or (new.is_dir() and not old.exists()):
            try:
                rename(new, old)
            except OSError as back:
                restored = False
                item.status = NEEDS_CHECK
                item.message += (
                    f"／元に戻せませんでした（{back}）。フォルダは {new} にあり、DB は "
                    f"{item.old_dir} を指しています。次の実行で直します"
                )
        if restored:
            try:
                _clear_journal(settings)
            except OSError as clear_err:  # 残っても次の実行で照合して消す
                item.message += f"／記録を消せませんでした（{clear_err}）"
            remove_if_empty(settings, new.parent)
        if old.is_dir():
            item.after_files, item.after_bytes = folder_stats(old)
        elif new.is_dir():
            item.after_files, item.after_bytes = folder_stats(new)
        log.error("job %d の移行に失敗しました: %s", job_id, item.message)
        if not isinstance(e, Exception):
            raise
