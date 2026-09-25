"""stem の保存フォルダ（`data/stems/<元のファイル名>/<分け方>/`）の名前を決める。

- 曲のフォルダ名は INPUT_SOURCE.original_name（最初に取り込みに成功したもの。ファイルなら
  拡張子を除く）を `safe_folder_name` で整形したもの。別の曲が同じ名前を使っていたら ` (2)`,
  ` (3)` … を付ける。一度決めたら変えない（その曲のジョブの SEPARATION_JOB.output_dir から読む）。
- ジョブのフォルダ名はプリセットの code。同じ曲の同じ code が使われていたら ` (2)` … を付ける。
- 決めた場所は SEPARATION_JOB.output_dir（データフォルダからの相対パス、/ 区切り）に保存する。
  NULL は T13 より前の `data/stems/<job_id>/`（`stemapp migrate-folders` で移す）。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.config import Settings
from stemapp.models import InputSource, SeparationJob, SeparationPreset, Track

log = logging.getLogger(__name__)

STEMS_DIRNAME = "stems"
MAX_NAME_LEN = 100
FETCH_DONE = "done"

# Windows でファイル名に使えない文字と制御文字（0x00〜0x1F）
_INVALID_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {
    f"LPT{i}" for i in range(1, 10)
}
# 取り除く「拡張子」とみなす形（英数字 1〜5 文字）。"Mr. Smith" の ". Smith" は拡張子にしない
_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,5}$")


def _is_reserved(name: str) -> bool:
    """予約名（拡張子付き・大文字小文字を区別しない）か。例: "con", "Nul.txt", "COM1.tar.gz"。"""
    return name.split(".", 1)[0].rstrip(" ").upper() in _RESERVED


def _trim(name: str) -> str:
    """前後の空白と、末尾のピリオド・空白を取り除く。"""
    return name.strip().rstrip(". ").strip()


def safe_folder_name(name: str | None, track_id: int) -> str:
    """Windows のフォルダ名に使える形にする（使えない文字は飛ばす）。

    - `\\ / : * ? " < > |` と制御文字を取り除く。
    - 前後の空白、末尾のピリオドと空白を取り除く。
    - 100 文字で切る（切った後にも末尾の整形をやり直す）。
    - 予約名（CON, PRN, AUX, NUL, COM1〜9, LPT1〜9。拡張子付きも）なら末尾に `_` を付ける。
    - 空になったら `track_<track_id>`。
    """
    cleaned = _trim(_INVALID_CHARS.sub("", name or ""))
    cleaned = _trim(cleaned[:MAX_NAME_LEN])
    if not cleaned:
        return f"track_{track_id}"
    if _is_reserved(cleaned):
        cleaned = cleaned[: MAX_NAME_LEN - 1] + "_"
    return cleaned


def strip_extension(name: str) -> str:
    """ファイル名から拡張子（英数字 1〜5 文字）を除く。"""
    return _EXTENSION.sub("", name)


def track_source_name(session: Session, track_id: int) -> str | None:
    """曲の元のファイル名（最初に取り込みに成功した INPUT_SOURCE。ファイルは拡張子を除く）。"""
    sources = session.scalars(
        select(InputSource)
        .where(InputSource.track_id == track_id)
        .where((InputSource.fetch_status == FETCH_DONE) | InputSource.fetch_status.is_(None))
        .order_by(InputSource.fetched_at.is_(None), InputSource.fetched_at, InputSource.source_id)
    ).all()
    for src in sources:
        if not src.original_name:
            continue
        if src.source_type == "url":
            return src.original_name
        return strip_extension(src.original_name)
    return None


def with_number(base: str, n: int) -> str:
    """2 以上なら `base (n)`。100 文字を超えないよう base を短くする。"""
    if n <= 1:
        return base
    suffix = f" ({n})"
    head = _trim(base[: MAX_NAME_LEN - len(suffix)]) or base[:1]
    return head + suffix


def pick_free_name(
    base: str, taken: Iterable[str], exists: Callable[[str], bool] | None = None
) -> str:
    """base, `base (2)`, `base (3)` … のうち、taken（大文字小文字を区別しない）に無く、
    exists(name) が False の最初の名前。"""
    used = {t.casefold() for t in taken}
    n = 1
    while True:
        name = with_number(base, n)
        if name.casefold() not in used and not (exists is not None and exists(name)):
            return name
        n += 1


def _parts(output_dir: str) -> tuple[str, ...]:
    return PurePosixPath(output_dir).parts


def track_folder_of_jobs(session: Session, track_id: int) -> str | None:
    """曲のジョブが既に使っている曲フォルダ名（stems/<ここ>/…）。無ければ None。"""
    for od in session.scalars(
        select(SeparationJob.output_dir)
        .where(SeparationJob.track_id == track_id, SeparationJob.output_dir.is_not(None))
        .order_by(SeparationJob.job_id)
    ):
        parts = _parts(od)
        if len(parts) >= 2 and parts[0] == STEMS_DIRNAME:
            return parts[1]
    return None


def _folders_of_other_tracks(session: Session, track_id: int) -> set[str]:
    out: set[str] = set()
    for od in session.scalars(
        select(SeparationJob.output_dir).where(
            SeparationJob.track_id != track_id, SeparationJob.output_dir.is_not(None)
        )
    ):
        parts = _parts(od)
        if len(parts) >= 2:
            out.add(parts[1])
    return out


def _job_folders_of_track(session: Session, track_id: int, exclude_job: int) -> set[str]:
    out: set[str] = set()
    for od in session.scalars(
        select(SeparationJob.output_dir).where(
            SeparationJob.track_id == track_id,
            SeparationJob.job_id != exclude_job,
            SeparationJob.output_dir.is_not(None),
        )
    ):
        parts = _parts(od)
        if len(parts) >= 3:
            out.add(parts[2])
    return out


class FolderPlanner:
    """フォルダ名を決める（DB と実際のフォルダの両方を見て重ならないようにする）。

    reserve したものを覚えておくので、移行の dry-run のように DB に書かずに続けて決められる。
    """

    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self._track_folder: dict[int, str] = {}  # reserve 済み（DB に未保存のものを含む）
        self._job_folders: dict[int, set[str]] = {}  # track_id → reserve 済みの分け方フォルダ

    @property
    def stems_root(self) -> Path:
        return self.settings.data_dir / STEMS_DIRNAME

    def track_folder(self, track_id: int) -> str:
        if track_id in self._track_folder:
            return self._track_folder[track_id]
        name = track_folder_of_jobs(self.session, track_id)
        if name is None:
            track = self.session.get(Track, track_id)
            base = safe_folder_name(track_source_name(self.session, track_id), track_id)
            if track is None:
                base = f"track_{track_id}"
            taken = _folders_of_other_tracks(self.session, track_id) | {
                v for k, v in self._track_folder.items() if k != track_id
            }
            name = pick_free_name(base, taken, lambda n: (self.stems_root / n).exists())
        self._track_folder[track_id] = name
        return name

    def job_folder(self, job: SeparationJob) -> str:
        """ジョブの保存先（データフォルダからの相対パス）。"""
        track_name = self.track_folder(job.track_id)
        preset = self.session.get(SeparationPreset, job.preset_id) if job.preset_id else None
        base = safe_folder_name(preset.code if preset else f"job_{job.job_id}", job.track_id)
        reserved = self._job_folders.setdefault(job.track_id, set())
        taken = _job_folders_of_track(self.session, job.track_id, job.job_id) | reserved
        track_dir = self.stems_root / track_name
        name = pick_free_name(base, taken, lambda n: (track_dir / n).exists())
        reserved.add(name)
        return f"{STEMS_DIRNAME}/{track_name}/{name}"


def assign_job_dir(session: Session, settings: Settings, job: SeparationJob) -> Path:
    """ジョブの保存フォルダを決めて output_dir に入れ、フォルダを作る（commit はしない）。

    既に決まっていればそれを使う。
    """
    if job.output_dir is None:
        job.output_dir = FolderPlanner(session, settings).job_folder(job)
    path = job_dir(settings, job.job_id, job.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_dir(settings: Settings, job_id: int, output_dir: str | None) -> Path:
    """ジョブの保存フォルダ。output_dir が NULL なら T13 より前の `data/stems/<job_id>`。"""
    if output_dir:
        return settings.data_dir / Path(*_parts(output_dir))
    return settings.data_dir / STEMS_DIRNAME / str(job_id)


def job_dir_of(session: Session, settings: Settings, job_id: int) -> Path:
    output_dir = session.scalar(
        select(SeparationJob.output_dir).where(SeparationJob.job_id == job_id)
    )
    return job_dir(settings, job_id, output_dir)


def remove_job_dir(settings: Settings, job_id: int, output_dir: str | None) -> None:
    """ジョブの保存フォルダを消し、空になった曲のフォルダも消す。"""
    path = job_dir(settings, job_id, output_dir)
    shutil.rmtree(path, ignore_errors=True)
    remove_if_empty(settings, path.parent)


def remove_if_empty(settings: Settings, folder: Path) -> None:
    """stems の下の（stems 自身ではない）空のフォルダを消す。"""
    root = (settings.data_dir / STEMS_DIRNAME).resolve()
    try:
        resolved = folder.resolve()
    except OSError:
        return
    if resolved == root or not resolved.is_relative_to(root):
        return
    try:
        if resolved.is_dir() and not any(resolved.iterdir()):
            os.rmdir(resolved)
    except OSError:
        log.debug("フォルダを消せませんでした: %s", resolved, exc_info=True)
