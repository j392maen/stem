"""ファイルの取り込み（TRACK と INPUT_SOURCE の作成、二重登録の防止）。

流れ:
1. T02 の正規化（`normalize_audio`）で一時フォルダに normalized.wav を作り、audio_hash を得る。
2. 同じ audio_hash の TRACK があれば、新しい TRACK は作らず INPUT_SOURCE を1行足す（正規化した
   ファイルは捨てる）。
3. 無ければ TRACK を作り、正規化したファイルを `data/tracks/<track_id>/normalized.wav` に置く。

元のファイルは移動も削除もしない。
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.audio import FfmpegRunner, normalize_audio
from stemapp.config import Settings
from stemapp.library import data_relative, find_done_job, resolve_data_path
from stemapp.models import InputSource, Track

log = logging.getLogger(__name__)

SOURCE_FILE = "file"
SOURCE_URL = "url"
FETCH_DONE = "done"
FETCH_FAILED = "failed"

# ファイルを受け取り、タグ（キーは小文字）を返す関数。読めなければ空の dict。テストで差し替える。
TagReader = Callable[[Path], Mapping[str, str]]


def _merge_tags(target: dict[str, str], tags: object) -> None:
    if not isinstance(tags, dict):
        return
    for k, v in tags.items():
        key = str(k).lower()
        if key not in target and isinstance(v, str) and v.strip():
            target[key] = v.strip()


def parse_ffprobe_tags(stdout: str) -> dict[str, str]:
    """`ffprobe -of json` の出力からタグを取り出す（format のタグを優先、次に stream のタグ）。"""
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    tags: dict[str, str] = {}
    fmt = data.get("format")
    if isinstance(fmt, dict):
        _merge_tags(tags, fmt.get("tags"))
    streams = data.get("streams")
    if isinstance(streams, list):
        for st in streams:
            if isinstance(st, dict):
                _merge_tags(tags, st.get("tags"))
    return tags


def read_tags_ffprobe(path: Path) -> dict[str, str]:
    """ffprobe でタグを読む。ffprobe が無い・失敗したときは空。"""
    exe = shutil.which("ffprobe")
    if exe is None:
        return {}
    try:
        proc = subprocess.run(
            [
                exe, "-v", "error",
                "-show_entries", "format_tags:stream_tags",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("ffprobe でタグを読めませんでした: %s", e)
        return {}
    if proc.returncode != 0:
        return {}
    return parse_ffprobe_tags(proc.stdout)


@dataclass(frozen=True)
class ImportResult:
    track_id: int
    is_new: bool  # 新しく TRACK を作ったか（False なら既存の曲）
    has_done_full_job: bool  # 既存の曲で、完了済みの full ジョブがあるか
    title: str
    source_id: int


def _utcnow() -> datetime:
    return datetime.now(UTC)


def decide_title(
    tags: Mapping[str, str], file_name: str, title_hint: str | None = None
) -> str:
    """タイトルを決める: 指定 → タグ → ファイル名（拡張子なし）。"""
    for cand in (title_hint, tags.get("title")):
        if cand and cand.strip():
            return cand.strip()
    stem = Path(file_name).stem.strip()
    return stem or file_name or "無題"


def decide_artist(tags: Mapping[str, str], artist_hint: str | None = None) -> str | None:
    for cand in (artist_hint, tags.get("artist"), tags.get("album_artist")):
        if cand and cand.strip():
            return cand.strip()
    return None


def import_file(
    session: Session,
    settings: Settings,
    path: Path,
    original_name: str | None = None,
    *,
    source_type: str = SOURCE_FILE,
    url: str | None = None,
    fetched_at: datetime | None = None,
    title: str | None = None,
    artist: str | None = None,
    ffmpeg_runner: FfmpegRunner | None = None,
    tag_reader: TagReader | None = None,
) -> ImportResult:
    """音声ファイルを取り込む（commit まで行う）。

    title / artist を渡すとタグより優先する（URL 取得でページのタイトルを使うため）。
    正規化に失敗したら AudioError（DB は変更しない）。
    """
    path = Path(path)
    name = original_name or path.name
    tmp_dir = settings.cache_dir / "tmp" / uuid.uuid4().hex
    placed_dir: Path | None = None  # 失敗したときに消す（この呼び出しで置いた正規化ファイル）
    try:
        norm = normalize_audio(path, tmp_dir / "normalized.wav", runner=ffmpeg_runner)
        track = session.scalars(select(Track).where(Track.audio_hash == norm.audio_hash)).first()
        is_new = track is None
        if track is None:
            tags = (tag_reader or read_tags_ffprobe)(path)
            track = Track(
                title=decide_title(tags, name, title),
                artist=decide_artist(tags, artist),
                duration_sec=norm.duration_sec,
                audio_hash=norm.audio_hash,
            )
            session.add(track)
            session.flush()

        existing_file = (
            resolve_data_path(settings, track.normalized_path) if track.normalized_path else None
        )
        if existing_file is None or not existing_file.is_file():
            # 新規、または既存の曲なのに正規化ファイルが無くなっている場合は置く
            track_dir = settings.tracks_dir / str(track.track_id)
            track_dir.mkdir(parents=True, exist_ok=True)
            normalized_path = track_dir / "normalized.wav"
            shutil.move(str(norm.path), str(normalized_path))
            if is_new:
                placed_dir = track_dir
            track.normalized_path = data_relative(settings, normalized_path)

        source = InputSource(
            track_id=track.track_id,
            source_type=source_type,
            original_name=original_name if source_type != SOURCE_FILE else name,
            url=url,
            fetch_status=FETCH_DONE,
            fetched_at=fetched_at or _utcnow(),
        )
        session.add(source)
        session.flush()
        has_done = False if is_new else find_done_job(session, track.track_id) is not None
        result = ImportResult(
            track_id=track.track_id,
            is_new=is_new,
            has_done_full_job=has_done,
            title=track.title,
            source_id=source.source_id,
        )
        session.commit()
        if is_new:
            log.info("新しい曲を登録しました（track %d: %s）。", track.track_id, track.title)
        else:
            log.info("登録済みの曲です（track %d: %s）。", track.track_id, track.title)
        return result
    except Exception:
        session.rollback()
        if placed_dir is not None:
            shutil.rmtree(placed_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
