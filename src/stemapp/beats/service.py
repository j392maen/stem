"""拍の解析の実行と保存（BEAT_GRID）、API 用の形への変換。

DB へは flush まで（commit は呼び出し側）。解析（時間がかかる）の間は DB に書かない。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stemapp.beats.base import BeatAnalysisError, BeatAnalyzer, BeatResult
from stemapp.beats.edit import GridState, apply_edit
from stemapp.beats.tempo import estimate_time_signature, tempo_segments
from stemapp.config import Settings
from stemapp.library import resolve_data_path
from stemapp.models import BeatEdit, BeatGrid, SeparationJob, Track

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
    else:
        # 直した結果は、元に戻すで戻せるよう履歴に移し、新しい自動の結果を使う
        reset_edits(session, grid, op="reanalyze")
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


def auto_state(grid: BeatGrid) -> GridState:
    return GridState.from_dict({
        "beats": grid.beats_json, "downbeats": grid.downbeats_json,
        "time_signature": grid.time_signature,
    })


def is_edited(grid: BeatGrid) -> bool:
    return grid.edited_beats_json is not None


def effective_state(grid: BeatGrid) -> GridState:
    """有効な拍（直した結果があればそれ、無ければ自動の結果）。"""
    if not is_edited(grid):
        return auto_state(grid)
    return GridState.from_dict({
        "beats": grid.edited_beats_json, "downbeats": grid.edited_downbeats_json,
        "time_signature": grid.edited_time_signature or grid.time_signature,
    })


def _edited_dict(grid: BeatGrid) -> dict[str, Any] | None:
    return effective_state(grid).as_dict() if is_edited(grid) else None


def _set_edited(grid: BeatGrid, state: dict[str, Any] | None) -> None:
    if state is None:
        grid.edited_beats_json = None
        grid.edited_downbeats_json = None
        grid.edited_time_signature = None
        return
    s = GridState.from_dict(state)
    grid.edited_beats_json = list(s.beats)
    grid.edited_downbeats_json = list(s.downbeats)
    grid.edited_time_signature = s.time_signature


# 元に戻せる操作の数（曲ごと）。古いものから消す。1つ数 KB〜十数 KB
MAX_HISTORY = 100


def record_edit(
    session: Session, track_id: int, op: str, params: dict[str, Any] | None,
    before: dict[str, Any] | None,
) -> None:
    """操作の履歴を1つ足し、MAX_HISTORY を超えた古いものを消す（flush まで）。"""
    session.add(BeatEdit(track_id=track_id, op=op, params_json=params, before_json=before))
    session.flush()
    ids = session.scalars(
        select(BeatEdit.edit_id)
        .where(BeatEdit.track_id == track_id)
        .order_by(BeatEdit.edit_id.desc())
        .offset(MAX_HISTORY)
    ).all()
    if ids:
        session.execute(delete(BeatEdit).where(BeatEdit.edit_id.in_(ids)))
        session.flush()


def can_undo(session: Session, track_id: int) -> bool:
    return session.scalars(
        select(BeatEdit.edit_id).where(BeatEdit.track_id == track_id).limit(1)
    ).first() is not None


def edit_grid(
    session: Session, grid: BeatGrid, op: str, params: dict[str, Any]
) -> BeatGrid:
    """補正の操作を1つ行い、直した結果と履歴を保存する（flush まで）。失敗は BeatEditError。"""
    before = _edited_dict(grid)
    new = apply_edit(effective_state(grid), op, params)
    record_edit(session, grid.track_id, op, params, before)
    _set_edited(grid, new.as_dict())
    session.flush()
    return grid


def undo_edit(session: Session, grid: BeatGrid) -> bool:
    """直前の操作を取り消す（flush まで）。取り消すものが無ければ False。"""
    last = session.scalars(
        select(BeatEdit)
        .where(BeatEdit.track_id == grid.track_id)
        .order_by(BeatEdit.edit_id.desc())
        .limit(1)
    ).first()
    if last is None:
        return False
    _set_edited(grid, last.before_json)
    session.delete(last)
    session.flush()
    return True


def reset_edits(session: Session, grid: BeatGrid, op: str = "reset") -> bool:
    """自動の結果に戻す（元に戻すで取り消せるよう履歴に残す）。直していなければ何もしない。"""
    if not is_edited(grid):
        return False
    record_edit(session, grid.track_id, op, None, _edited_dict(grid))
    _set_edited(grid, None)
    session.flush()
    return True


def beats_payload(grid: BeatGrid, *, undo: bool = False) -> dict[str, Any]:
    """GET /api/tracks/{id}/beats の中身（有効な拍）。区間の BPM はここで計算する。

    edited: 直した結果を使っているか。can_undo: 元に戻せる操作があるか（undo で渡す）。
    auto_time_signature: 自動の結果の拍子。
    """
    state = effective_state(grid)
    beats = list(state.beats)
    return {
        "track_id": grid.track_id,
        "analyzer": grid.analyzer,
        "time_signature": state.time_signature,
        "auto_time_signature": grid.time_signature,
        "beats": beats,
        "downbeats": list(state.downbeats),
        "segments": [s.as_dict() for s in tempo_segments(beats)],
        "edited": is_edited(grid),
        "can_undo": undo,
        "created_at": _iso(grid.created_at),
    }
