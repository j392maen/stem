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
from stemapp.exports.service import export_ids_for_jobs, remove_export_dirs
from stemapp.library import find_done_job
from stemapp.models import SeparationJob, Stem, Track
from stemapp.separation.pipeline import delete_job_stems, job_tmp_dir, load_plan
from stemapp.separation.refine import (
    ExportsBusy,
    active_exports_of_jobs,
    clean_orphan_refine_dirs,
    take_exports_of_jobs,
)
from stemapp.stem_folders import remove_job_dir
from stemapp.stem_view import root_job_id
from stemapp.tempo.service import invalidate_job_tempo, remove_job_tempo_dirs

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
    """その曲の queued / running の分割（full）ジョブ（古いもの）。preset_id でプリセットを絞る。

    詳細分割（refine）のジョブは含めない。
    """
    stmt = select(SeparationJob).where(
        SeparationJob.track_id == track_id,
        SeparationJob.job_kind == "full",
        SeparationJob.status.in_(ACTIVE_STATUSES),
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


def refine_descendants(session: Session, job_id: int) -> list[SeparationJob]:
    """そのジョブの stem を詳細分割したジョブ（子の子も。深い順）。"""
    out: list[SeparationJob] = []
    frontier = [job_id]
    seen = {job_id}
    while frontier:
        found = [
            j
            for j in session.scalars(
                select(SeparationJob)
                .where(
                    SeparationJob.job_kind == "refine",
                    SeparationJob.input_stem_id.in_(
                        select(Stem.stem_id).where(Stem.job_id.in_(frontier))
                    ),
                )
                .order_by(SeparationJob.job_id)
            )
            if j.job_id not in seen
        ]
        seen.update(j.job_id for j in found)
        out.extend(found)
        frontier = [j.job_id for j in found]
    return list(reversed(out))


def delete_job(session: Session, settings: Settings, job_id: int) -> SeparationJob:
    """終わったジョブ（done / failed / canceled）を消す。stem・配信用データ・ファイルも消える。

    その stem を詳細分割したジョブ（refine）も一緒に消す。
    分割待ち・分割中、配信用データの作成待ち・作成中のジョブ（詳細分割を含む）は JobConflict。
    消したジョブ（DB からは消えた後の値）を返す。
    """
    job = session.get(SeparationJob, job_id)
    if job is None:
        raise JobNotFound(f"ジョブが見つかりません（job {job_id}）。")
    children = refine_descendants(session, job_id)
    if children and (
        job.status in ACTIVE_STATUSES or job.postprocess_status in ACTIVE_STATUSES
    ):
        raise JobConflict("分割中・配信用データの作成中のジョブは削除できません。")
    if any(c.status in ACTIVE_STATUSES for c in children):
        raise JobConflict(
            "この分け方の stem を「もっと分ける」処理が分割待ち・分割中です。"
            "キャンセルしてから削除してください。"
        )
    refine_ids = [c.job_id for c in children] + ([job_id] if job.job_kind == "refine" else [])
    if active_exports_of_jobs(session, refine_ids):
        # 子の stem が消えると EXPORT_ITEM だけが消え、中身の欠けた書き出しが作られてしまう
        raise JobConflict(
            "この stem を使った書き出しが作成待ち・作成中です。書き出しが終わってから"
            "削除してください。"
        )
    for child in children:
        delete_job(session, settings, child.job_id)
    output_dir = job.output_dir
    export_ids = export_ids_for_jobs(session, [job_id])
    # 詳細分割を戻すと元の分け方の葉が変わる（速度変更のキャッシュを消す。done のときだけ）
    tempo_owner = (
        root_job_id(session, job_id)
        if job.job_kind == "refine" and job.status == DONE
        else None
    )
    if job.job_kind == "refine":
        # 詳細分割の子を使った書き出しは full ジョブの行に付くので、stem から探して一緒に消す
        try:
            export_ids += take_exports_of_jobs(session, [job_id])
        except ExportsBusy as e:
            session.rollback()
            raise JobConflict(str(e)) from e
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
    remove_job_dir(session, settings, job_id, output_dir)
    shutil.rmtree(job_tmp_dir(settings, job_id), ignore_errors=True)
    remove_export_dirs(settings, export_ids)  # 書き出したファイル（data/exports/<id>）
    remove_job_tempo_dirs(settings, [job_id])  # 速度を変えた音声（data/cache/tempo/<job_id>）
    if tempo_owner is not None and tempo_owner != job_id:
        invalidate_job_tempo(session, settings, tempo_owner)
    log.info("ジョブを削除しました（job %d, track %d）。", job_id, job.track_id)
    return job


def is_cancel_requested(session: Session, job_id: int) -> bool:
    value = session.scalar(
        select(SeparationJob.cancel_requested).where(SeparationJob.job_id == job_id)
    )
    return bool(value)


def discard_job_outputs(session: Session, settings: Settings, job_id: int) -> None:
    """ジョブの STEM 行、保存フォルダ、一時フォルダを消す（commit まで行う）。

    保存フォルダの名前（SEPARATION_JOB.output_dir）も空に戻す。
    """
    output_dir = session.scalar(
        select(SeparationJob.output_dir).where(SeparationJob.job_id == job_id)
    )
    delete_job_stems(session, job_id)
    session.execute(
        update(SeparationJob).where(SeparationJob.job_id == job_id).values(output_dir=None)
    )
    session.commit()
    remove_job_dir(session, settings, job_id, output_dir)
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
    orphans = clean_orphan_refine_dirs(session, settings)
    if orphans:
        log.info("使われていない詳細分割のフォルダを消しました: %s", orphans)
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
