"""拍の解析の実行と保存（BEAT_GRID）、API 用の形への変換。

DB へは flush まで（commit は呼び出し側）。解析（時間がかかる）の間は DB に書かない。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stemapp.beats.base import BeatAnalysisError, BeatAnalyzer, BeatResult
from stemapp.beats.tempo import estimate_time_signature, tempo_segments
from stemapp.config import Settings
from stemapp.library import resolve_data_path
from stemapp.models import BeatGrid, SeparationJob, Track

log = logging.getLogger(__name__)

STAGE_BEATS = "拍を解析中"
NO_BEATS_MESSAGE = "拍が見つかりませんでした（無音・拍の無い曲など）。"


@dataclass(frozen=True)
class BeatsOutcome:
    grid: BeatGrid
    skipped: bool  # 既にあったので解析しなかった
    result: BeatResult | None = None
    seconds: float = 0.0


def get_grid(session: Session, track_id: int) -> BeatGrid | None:
    return session.get(BeatGrid, track_id)


def analyze_audio(
    session: Session, settings: Settings, track_id: int, analyzer: BeatAnalyzer
) -> BeatResult:
    """曲の normalized.wav を解析する（DB には書かない）。失敗したら例外。"""
    track = session.get(Track, track_id)
    if track is None:
        raise BeatAnalysisError(f"曲が見つかりません（track {track_id}）。")
    if not track.normalized_path:
        raise BeatAnalysisError(f"正規化した音声がありません（track {track_id}）。")
    path = resolve_data_path(settings, track.normalized_path)
    if not path.is_file():
        raise BeatAnalysisError(f"正規化した音声が見つかりません: {path}")
    result = analyzer.analyze(path)
    if not result.beats:
        raise BeatAnalysisError(NO_BEATS_MESSAGE)
    return result


def save_grid(session: Session, track_id: int, result: BeatResult) -> BeatGrid:
    """解析結果を BEAT_GRID に保存し（上書き）、その曲のジョブの拍の警告を消す（flush まで）。"""
    grid = session.get(BeatGrid, track_id)
    if grid is None:
        grid = BeatGrid(track_id=track_id)
        session.add(grid)
    grid.analyzer = result.analyzer[:100]
    grid.beats_json = [round(float(x), 4) for x in result.beats]
    grid.downbeats_json = [round(float(x), 4) for x in result.downbeats]
    grid.time_signature = estimate_time_signature(result.beats, result.downbeats)
    grid.created_at = datetime.now(UTC)
    session.execute(
        update(SeparationJob)
        .where(SeparationJob.track_id == track_id, SeparationJob.beat_warning.is_not(None))
        .values(beat_warning=None)
    )
    session.flush()
    return grid


def analyze_track(
    session: Session,
    settings: Settings,
    track_id: int,
    analyzer: BeatAnalyzer,
    *,
    force: bool = False,
) -> BeatsOutcome:
    """曲の拍を解析して保存する（flush まで）。既にあれば force でない限り解析しない。"""
    existing = get_grid(session, track_id)
    if existing is not None and not force:
        return BeatsOutcome(existing, skipped=True)
    t0 = time.perf_counter()
    result = analyze_audio(session, settings, track_id, analyzer)
    grid = save_grid(session, track_id, result)
    seconds = time.perf_counter() - t0
    log.info(
        "track %d: 拍 %d・小節の頭 %d・%d 拍子（%s, %s, %.1f 秒）",
        track_id, len(result.beats), len(result.downbeats), grid.time_signature,
        result.analyzer, result.device, seconds,
    )
    return BeatsOutcome(grid, skipped=False, result=result, seconds=seconds)


def beat_warning_text(exc: BaseException) -> str:
    return f"拍を解析できませんでした: {type(exc).__name__}: {exc}"[:1000]


def analyze_job_beats(
    session: Session,
    settings: Settings,
    job_id: int,
    analyzer: BeatAnalyzer,
    *,
    force: bool = False,
) -> str | None:
    """分割ジョブの後処理としての拍の解析。失敗しても例外を出さず、警告を JOB に残して返す。

    拍が無くても再生はできるため、失敗でジョブを failed にしない（commit まで行う）。
    拍が1つも見つからないときも警告にする。保存（commit）の失敗も警告にする。
    同じ曲の拍を別の処理が先に保存した（主キーの重複）ときは、既にあるものとして成功扱い。
    """
    job = session.get(SeparationJob, job_id)
    if job is None:
        return None
    track_id = job.track_id
    try:
        if get_grid(session, track_id) is not None and not force:
            return None
        result = analyze_audio(session, settings, track_id, analyzer)
        save_grid(session, track_id, result)
        session.commit()
    except IntegrityError:
        session.rollback()
        log.info("job %d: 同じ曲の拍が別の処理で先に保存されました（そちらを使います）。", job_id)
        return None
    except Exception as e:
        session.rollback()
        message = beat_warning_text(e)
        log.warning("job %d: %s", job_id, message, exc_info=not isinstance(e, BeatAnalysisError))
        session.execute(
            update(SeparationJob)
            .where(SeparationJob.job_id == job_id)
            .values(beat_warning=message)
        )
        session.commit()
        return message
    log.info(
        "job %d: 拍を解析しました（拍 %d、%s、%.1f 秒）。",
        job_id, len(result.beats), result.device, result.seconds,
    )
    return None


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:  # SQLite から読むとタイムゾーンが落ちる（保存は UTC）
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def beats_payload(grid: BeatGrid) -> dict[str, Any]:
    """GET /api/tracks/{id}/beats の中身。区間の BPM はここで計算する。"""
    beats = list(grid.beats_json or [])
    return {
        "track_id": grid.track_id,
        "analyzer": grid.analyzer,
        "time_signature": grid.time_signature,
        "beats": beats,
        "downbeats": list(grid.downbeats_json or []),
        "segments": [s.as_dict() for s in tempo_segments(beats)],
        "created_at": _iso(grid.created_at),
    }
