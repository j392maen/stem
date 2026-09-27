"""stem の保存フォルダ（`data/stems/<元のファイル名>/<分け方>/`）の名前を決める。

- 曲のフォルダ名は INPUT_SOURCE.original_name（最初に取り込みに成功したもの。ファイルなら
  拡張子を除く。INPUT_SOURCE が無いときは TRACK.title）を `safe_folder_name` で整形したもの。
  別の曲が同じ名前を使っていたら ` (2)`, ` (3)` … を付ける。
  一度決めたら変えない（その曲のジョブの SEPARATION_JOB.output_dir から読む）。
- ジョブのフォルダ名はプリセットの code。同じ曲の同じ code が使われていたら ` (2)` … を付ける。
- 決めた場所は SEPARATION_JOB.output_dir（データフォルダからの相対パス、/ 区切り）に保存する。
  NULL は T13 より前の `data/stems/<job_id>/`（`stemapp migrate-folders` で移す）。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from stemapp.config import Settings
from stemapp.models import InputSource, SeparationJob, SeparationPreset, Track

log = logging.getLogger(__name__)

STEMS_DIRNAME = "stems"
MAX_NAME_LEN = 100  # UTF-16 の単位（Windows のパスの長さの数え方。絵文字などは 2）
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


def utf16_len(text: str) -> int:
    """UTF-16 の単位での長さ（BMP 外の文字は 2）。"""
    return sum(2 if ord(c) > 0xFFFF else 1 for c in text)


def _is_joiner(c: str) -> bool:
    """直前の文字とくっついて1文字に見えるもの（結合文字・ZWJ・異体字セレクタ・肌の色）。"""
    cp = ord(c)
    return (
        unicodedata.combining(c) != 0
        or unicodedata.category(c) in ("Mn", "Me", "Mc")
        or cp == 0x200D
        or 0xFE00 <= cp <= 0xFE0F
        or 0xE0100 <= cp <= 0xE01EF
        or 0x1F3FB <= cp <= 0x1F3FF
        or 0xE0020 <= cp <= 0xE007F  # タグ文字（旗の絵文字）
    )


def truncate_utf16(text: str, limit: int) -> str:
    """UTF-16 の単位で limit 以内に切る。サロゲートペアや、結合文字・ZWJ でつながった
    まとまり（書記素）の途中では切らない（簡易判定）。"""
    if utf16_len(text) <= limit:
        return text
    used = 0
    end = 0
    for i, c in enumerate(text):
        w = 2 if ord(c) > 0xFFFF else 1
        if used + w > limit:
            break
        used += w
        end = i + 1
    # 切った位置の直後がくっつく文字なら、まとまりの先頭まで戻す
    while 0 < end < len(text) and (_is_joiner(text[end]) or text[end - 1] == "\u200d"):
        end -= 1
    # 国旗（地域指示子 2 つで1文字）の片方だけを残さない
    if 0 < end < len(text) and _is_regional(text[end - 1]) and _is_regional(text[end]):
        run = 0
        while end - run > 0 and _is_regional(text[end - run - 1]):
            run += 1
        if run % 2 == 1:
            end -= 1
    return text[:end]


def _is_regional(c: str) -> bool:
    return 0x1F1E6 <= ord(c) <= 0x1F1FF


def safe_folder_name(name: str | None, track_id: int) -> str:
    """Windows のフォルダ名に使える形にする（使えない文字は飛ばす）。

    - Unicode の NFC に正規化する。
    - `\\ / : * ? " < > |` と制御文字を取り除く。
    - 前後の空白、末尾のピリオドと空白を取り除く。
    - UTF-16 の単位で 100 以内に切る（切った後にも末尾の整形をやり直す）。
    - 予約名（CON, PRN, AUX, NUL, COM1〜9, LPT1〜9。拡張子付きも）なら末尾に `_` を付ける。
    - 数字だけなら末尾に `_` を付ける（T13 より前の `stems/<job_id>` と重ならないように）。
    - 空になったら `track_<track_id>`。
    """
    cleaned = unicodedata.normalize("NFC", name or "")
    cleaned = _trim(_INVALID_CHARS.sub("", cleaned))
    cleaned = _trim(truncate_utf16(cleaned, MAX_NAME_LEN))
    if not cleaned:
        return f"track_{track_id}"
    if _is_reserved(cleaned) or cleaned.isascii() and cleaned.isdigit():
        cleaned = _trim(truncate_utf16(cleaned, MAX_NAME_LEN - 1)) + "_"
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
    head = _trim(truncate_utf16(base, MAX_NAME_LEN - len(suffix))) or base[:1]
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
            raw = track_source_name(self.session, track_id)
            if raw is None and track is not None:
                raw = track.title  # INPUT_SOURCE が無い（または名前が無い）ときは曲名
            base = safe_folder_name(raw, track_id)
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


# T13 より前の `stems/<job_id>/` の中にあるフォルダ（これ以外のフォルダがあれば旧形式ではない）
LEGACY_SUBDIRS = frozenset({"stream", "peaks"})


def legacy_dir(settings: Settings, job_id: int) -> Path:
    return settings.data_dir / STEMS_DIRNAME / str(job_id)


def is_legacy_job_dir(session: Session, settings: Settings, job_id: int) -> bool:
    """`stems/<job_id>` が、そのジョブの T13 より前の保存フォルダだと判断できるか。

    - ほかのジョブの output_dir（新しい形の曲フォルダ）として使われていない。
    - 中身が旧形式（直下に master の FLAC など。フォルダは stream / peaks だけ）。
    どちらかに当てはまらなければ False（消さない・移さない）。
    """
    path = legacy_dir(settings, job_id)
    if not path.is_dir():
        return False
    rel = f"{STEMS_DIRNAME}/{job_id}"
    used = session.scalar(
        select(func.count())
        .select_from(SeparationJob)
        .where(or_(SeparationJob.output_dir == rel, SeparationJob.output_dir.like(rel + "/%")))
    )
    if used:
        return False
    try:
        return all(
            not child.is_dir() or child.name in LEGACY_SUBDIRS for child in path.iterdir()
        )
    except OSError:
        return False


def remove_job_dir(
    session: Session, settings: Settings, job_id: int, output_dir: str | None
) -> None:
    """ジョブの保存フォルダを消し、空になった曲のフォルダも消す。

    output_dir が NULL（T13 より前のジョブ）なら、`stems/<job_id>` が本当にそのジョブの
    旧形式のフォルダだと判断できるときだけ消す（別の曲のフォルダを消さないため）。
    """
    if output_dir is None:
        if is_legacy_job_dir(session, settings, job_id):
            shutil.rmtree(legacy_dir(settings, job_id), ignore_errors=True)
        elif legacy_dir(settings, job_id).exists():
            log.warning(
                "stems/%d は job %d の古い保存フォルダではないため消しません。", job_id, job_id
            )
        return
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
