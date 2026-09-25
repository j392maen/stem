"""分割ジョブの登録・キャンセル・後始末（DB 操作）。

状態の流れ: queued →（ワーカーが取り出す）→ running → done / failed / canceled。
queued のキャンセルはすぐ canceled。running のキャンセルは cancel_requested を立て、
ワーカーが子プロセスを終了させてから canceled にする。
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, or_, select, update
from sqlalchemy.orm import Session

from stemapp.config import Settings
from stemapp.library import find_done_job
from stemapp.models import SeparationJob, Track
from stemapp.separation.pipeline import delete_job_stems, job_tmp_dir, load_plan

log = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELED = "canceled"
ACTIVE_STATUSES: tuple[str, ...] = (QUEUED, RUNNING)
FINISHED_STATUSES: tuple[str, ...] = (DONE, FAILED, CANCELED)

INTERRUPTED_MESSAGE = "中断されました（アプリの再起動）"
STAGE_CANCELED = "キャンセルしました"
STAGE_STARTING = "起動中"


class JobError(RuntimeError):
    """ジョブ操作の失敗（メッセージは日本語）。"""


class JobNotFound(JobError):
    pass


class JobConflict(JobError):
    """今の状態ではできない操作。"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class EnqueueResult:
    job: SeparationJob
    created: bool
    # created=False のときの理由: "active"（分割待ち・分割中のジョブがある） / "done"（分割済み）
    reason: str | None = None


def active_job(
    session: Session, track_id: int, preset_id: int | None = None
) -> SeparationJob | None:
    """その曲の queued / running のジョブ（古いもの）。preset_id でプリセットを絞る。"""
    stmt = select(SeparationJob).where(
        SeparationJob.track_id == track_id, SeparationJob.status.in_(ACTIVE_STATUSES)
    )
    if preset_id is not None:
        stmt = stmt.where(SeparationJob.preset_id == preset_id)
    return session.scalars(stmt.order_by(SeparationJob.job_id)).first()


def enqueue_full_job(
    session: Session,
    track_id: int,
    preset_code: str | None = None,
    *,
    force: bool = False,
    run_on: str = "gpu",
) -> EnqueueResult:
    """分割ジョブ（full）を queued で登録する（commit まで行う）。

    - 同じプリセットで分割待ち・分割中のジョブがあれば、新しく作らずそれを返す（force でも同じ）。
    - 同じプリセットで分割済みなら、force でない限り新しく作らず完了済みのジョブを返す。
    プリセットが違えば、同じ曲に別の分け方のジョブを登録できる（聴き比べ用）。
    preset_code が None（既定のプリセット）のときは、どのプリセットのジョブでも同じ扱いにする。
    プリセットが無ければ SeparationError。
    """
    track = session.get(Track, track_id)
    if track is None:
        raise JobNotFound(f"曲が見つかりません（track {track_id}）。")
    plan = load_plan(session, preset_code)
    # プリセットを指定しないときは、どの分け方でも「分割待ち・分割済み」とみなす
    same_preset = plan.preset_id if preset_code is not None else None
    active = active_job(session, track_id, same_preset)
    if active is not None:
        return EnqueueResult(active, False, "active")
    if not force:
        done = find_done_job(session, track_id, same_preset)
        if done is not None:
            return EnqueueResult(done, False, "done")
    job = SeparationJob(
        track_id=track_id,
        job_kind="full",
        preset_id=plan.preset_id,
        status=QUEUED,
        run_on=run_on,
        progress=0.0,
        stage="分割待ち",
    )
    session.add(job)
    session.commit()
    log.info("ジョブを登録しました（job %d, track %d, %s）。", job.job_id, track_id, plan.code)
    return EnqueueResult(job, True)


def request_cancel(session: Session, job_id: int) -> SeparationJob:
    """キャンセルする。queued はすぐ canceled、running は cancel_requested を立てる。"""
    job = session.get(SeparationJob, job_id)
    if job is None:
        raise JobNotFound(f"ジョブが見つかりません（job {job_id}）。")
    now = _utcnow()
    res = session.execute(
        update(SeparationJob)
        .where(SeparationJob.job_id == job_id, SeparationJob.status == QUEUED)
        .values(status=CANCELED, stage=STAGE_CANCELED, finished_at=now, cancel_requested=True)
    )
    if res.rowcount == 0:  # type: ignore[attr-defined]
        # ワーカーが取り出した直後かもしれないので running も調べ直す
        res = session.execute(
            update(SeparationJob)
            .where(SeparationJob.job_id == job_id, SeparationJob.status == RUNNING)
            .values(cancel_requested=True)
        )
    session.commit()
    session.refresh(job)
    if res.rowcount == 0:  # type: ignore[attr-defined]
        raise JobConflict(f"このジョブは既に終わっています（状態: {job.status}）。")
    log.info("ジョブのキャンセルを受け付けました（job %d, %s）。", job_id, job.status)
    return job


def delete_job(session: Session, settings: Settings, job_id: int) -> SeparationJob:
    """終わったジョブ（done / failed / canceled）を消す。stem・配信用データ・ファイルも消える。

    分割待ち・分割中、配信用データの作成待ち・作成中のジョブは JobConflict。
    消したジョブ（DB からは消えた後の値）を返す。
    """
    job = session.get(SeparationJob, job_id)
    if job is None:
        raise JobNotFound(f"ジョブが見つかりません（job {job_id}）。")
    # 確かめてから消すまでの間にワーカーが取り出さないよう、条件付きで消す
    res = session.execute(
        delete(SeparationJob).where(
            SeparationJob.job_id == job_id,
            SeparationJob.status.not_in(ACTIVE_STATUSES),
            or_(
                SeparationJob.postprocess_status.is_(None),
                SeparationJob.postprocess_status.not_in(ACTIVE_STATUSES),
            ),
        )
    )
    if res.rowcount != 1:  # type: ignore[attr-defined]
        session.rollback()
        session.refresh(job)
        if job.status in ACTIVE_STATUSES:
            raise JobConflict(
                "分割待ち・分割中のジョブは削除できません。キャンセルしてから削除してください。"
            )
        raise JobConflict("配信用データを作成待ち・作成中です。終わってから削除してください。")
    session.commit()  # STEM・STEM_RENDITION・WAVEFORM・EXPORT は外部キーの CASCADE で消える
    session.expunge(job)
    shutil.rmtree(settings.stems_dir / str(job_id), ignore_errors=True)
    shutil.rmtree(job_tmp_dir(settings, job_id), ignore_errors=True)
    log.info("ジョブを削除しました（job %d, track %d）。", job_id, job.track_id)
    return job


def is_cancel_requested(session: Session, job_id: int) -> bool:
    value = session.scalar(
        select(SeparationJob.cancel_requested).where(SeparationJob.job_id == job_id)
    )
    return bool(value)


def discard_job_outputs(session: Session, settings: Settings, job_id: int) -> None:
    """ジョブの STEM 行、`data/stems/<job_id>`、一時フォルダを消す（commit まで行う）。"""
    delete_job_stems(session, job_id)
    session.commit()
    shutil.rmtree(settings.stems_dir / str(job_id), ignore_errors=True)
    shutil.rmtree(job_tmp_dir(settings, job_id), ignore_errors=True)


def clean_stale_tmp(session: Session, settings: Settings) -> list[str]:
    """running でないジョブの一時フォルダ（`data/cache/tmp/job-<id>`）を消す。"""
    tmp_root = settings.cache_dir / "tmp"
    if not tmp_root.is_dir():
        return []
    running = set(
        session.scalars(select(SeparationJob.job_id).where(SeparationJob.status == RUNNING))
    )
    removed: list[str] = []
    for d in tmp_root.glob("job-*"):
        try:
            job_id = int(d.name.removeprefix("job-"))
        except ValueError:
            continue
        if d.is_dir() and job_id not in running:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    return removed


def finish_job(
    session: Session,
    settings: Settings,
    job_id: int,
    status: str,
    *,
    message: str | None = None,
    stage: str | None = None,
) -> None:
    """ジョブを終わった状態（failed / canceled）にし、途中の成果物を消す。"""
    job = session.get(SeparationJob, job_id)
    if job is None:
        return
    job.status = status
    job.finished_at = _utcnow()
    if message is not None:
        job.error_message = message
    if stage is not None:
        job.stage = stage
    session.commit()
    discard_job_outputs(session, settings, job_id)


def recover_interrupted_jobs(session: Session, settings: Settings) -> list[int]:
    """running のまま残ったジョブを failed にし、途中の stems フォルダを消す（ワーカー起動時）。"""
    ids = list(
        session.scalars(select(SeparationJob.job_id).where(SeparationJob.status == RUNNING))
    )
    for job_id in ids:
        finish_job(session, settings, job_id, FAILED, message=INTERRUPTED_MESSAGE)
        log.warning("中断されたジョブを failed にしました（job %d）。", job_id)
    pp = session.execute(
        update(SeparationJob)
        .where(SeparationJob.postprocess_status == RUNNING)
        .values(postprocess_status=FAILED)
    )
    session.commit()
    count = int(pp.rowcount or 0)  # type: ignore[attr-defined]
    if count:
        log.warning("中断された配信用データの作り直しを failed にしました（%d 件）。", count)
    removed = clean_stale_tmp(session, settings)
    if removed:
        log.info("残っていた一時フォルダを消しました: %s", removed)
    return ids


def claim_next_job(session: Session) -> int | None:
    """いちばん古い queued のジョブを running にして job_id を返す。無ければ None。"""
    while True:
        job_id = session.scalar(
            select(SeparationJob.job_id)
            .where(SeparationJob.status == QUEUED)
            .order_by(SeparationJob.created_at, SeparationJob.job_id)
            .limit(1)
        )
        if job_id is None:
            return None
        res = session.execute(
            update(SeparationJob)
            .where(SeparationJob.job_id == job_id, SeparationJob.status == QUEUED)
            .values(status=RUNNING, stage=STAGE_STARTING, started_at=_utcnow(), progress=0.0)
        )
        session.commit()
        if res.rowcount == 1:  # type: ignore[attr-defined]
            return int(job_id)
        # 取り出す直前にキャンセルされた。次を探す


# --- 配信用データの作り直し ----------------------------------------------------------


@dataclass(frozen=True)
class PostprocessRequest:
    job: SeparationJob
    # created=False のときの理由: "ready"（欠けていない） / "active"（作り直し待ち・作成中）
    created: bool
    reason: str | None = None


def request_postprocess(
    session: Session, job_id: int, missing: list[str]
) -> PostprocessRequest:
    """done のジョブの配信用データの作り直しを登録する（ワーカーが処理する。commit まで）。

    missing は欠けている stem（`delivery.missing_delivery` の結果）。空なら登録しない。
    """
    job = session.get(SeparationJob, job_id)
    if job is None:
        raise JobNotFound(f"ジョブが見つかりません（job {job_id}）。")
    if job.status != DONE:
        raise JobConflict(
            f"分割が終わっていないジョブです（状態: {job.status}）。配信用データは作れません。"
        )
    if job.postprocess_status in ACTIVE_STATUSES:
        return PostprocessRequest(job, False, "active")
    if not missing:
        return PostprocessRequest(job, False, "ready")
    job.postprocess_status = QUEUED
    session.commit()
    log.info("配信用データの作り直しを登録しました（job %d, 欠け: %s）。", job_id, missing)
    return PostprocessRequest(job, True)


def claim_next_postprocess(session: Session) -> int | None:
    """作り直し待ち（postprocess_status=queued）のジョブを running にして job_id を返す。"""
    while True:
        job_id = session.scalar(
            select(SeparationJob.job_id)
            .where(SeparationJob.postprocess_status == QUEUED, SeparationJob.status == DONE)
            .order_by(SeparationJob.job_id)
            .limit(1)
        )
        if job_id is None:
            return None
        res = session.execute(
            update(SeparationJob)
            .where(SeparationJob.job_id == job_id, SeparationJob.postprocess_status == QUEUED)
            .values(postprocess_status=RUNNING)
        )
        session.commit()
        if res.rowcount == 1:  # type: ignore[attr-defined]
            return int(job_id)


def set_postprocess_status(session: Session, job_id: int, status: str) -> None:
    session.execute(
        update(SeparationJob)
        .where(SeparationJob.job_id == job_id)
        .values(postprocess_status=status)
    )
    session.commit()
