"""速度変更（ピッチを保つ方式）の伸縮済み音声: 登録、実行、キャンセル、キャッシュの片付け。

- 置き場所: `data/cache/tempo/<job_id>/<倍率を小数3桁>/<stem の code>.webm`
  （stem の保存フォルダとは別）。
- DB: TEMPO_RENDER（ジョブ × 倍率。作成の状態と最後に使った時刻）と
  TEMPO_RENDITION（stem ごとのファイル）。
- 作成はワーカーが1件ずつ行う（分割・配信用データの作り直しより後回し。GPU は使わない）。
  各 stem の master（FLAC）を並列（`tempo_workers`、既定は論理コア数の半分・最大 8）に伸縮する。
  全 stem の長さは「master の長さ ÷ 倍率」（いちばん短い stem に合わせる）にそろえる。
- キャッシュの上限: 1曲あたりの倍率の数（`tempo_cache_per_track`）と
  全体の容量（`tempo_cache_max_mb`）。
  超えたら last_used_at の古いものから消す。曲・ジョブを消したらフォルダごと消す。
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile as sf
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from stemapp.config import Settings
from stemapp.delivery import DEFAULT_STREAM_FORMAT, PURPOSE_MASTER, StreamFormat
from stemapp.library import data_relative, resolve_data_path
from stemapp.models import (
    SeparationJob,
    Stem,
    StemRendition,
    StemType,
    TempoRender,
    TempoRendition,
)
from stemapp.tempo.stretch import StretchCanceled, Stretcher

log = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELED = "canceled"
ACTIVE_STATUSES: tuple[str, ...] = (QUEUED, RUNNING)
FINISHED_STATUSES: tuple[str, ...] = (DONE, FAILED, CANCELED)

MIN_RATIO = 0.5
MAX_RATIO = 2.0
PITCH_KEEP = "keep"

STAGE_QUEUED = "作成待ち"
STAGE_CANCELED = "キャンセルしました"
INTERRUPTED_MESSAGE = "中断されました（アプリの再起動）"
PROGRESS_STEP = 0.01  # DB に書く進み具合の細かさ
WATCH_SEC = 0.25  # 実行中にキャンセルを調べ、進み具合を書く間隔
MAX_WORKERS = 8


class TempoError(RuntimeError):
    """速度変更の操作の失敗（メッセージは日本語）。"""


class TempoNotFound(TempoError):
    pass


class TempoConflict(TempoError):
    """今の状態ではできない操作。"""


class TempoInvalid(TempoError):
    """倍率などの指定の誤り。"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _aware(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.min.replace(tzinfo=UTC)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


# --- 倍率とパス -------------------------------------------------------------------------


def normalize_ratio(ratio: float) -> float:
    """倍率を小数3桁に丸める。範囲外・1.000（元の速度）は TempoInvalid。"""
    try:
        r = round(float(ratio), 3)
    except (TypeError, ValueError) as e:
        raise TempoInvalid("速度の倍率が正しくありません。") from e
    if not (MIN_RATIO <= r <= MAX_RATIO):
        raise TempoInvalid(f"速度の倍率は {MIN_RATIO:.2f}〜{MAX_RATIO:.2f} で指定してください。")
    if r == 1.0:
        raise TempoInvalid("元の速度（1.000 倍）は伸縮しなくても再生できます。")
    return r


def ratio_key(ratio: float) -> str:
    """フォルダ名（小数3桁。例 1.100）。"""
    return f"{ratio:.3f}"


def tempo_root(settings: Settings) -> Path:
    return settings.cache_dir / "tempo"


def job_tempo_dir(settings: Settings, job_id: int) -> Path:
    return tempo_root(settings) / str(job_id)


def render_dir(settings: Settings, job_id: int, ratio: float) -> Path:
    return job_tempo_dir(settings, job_id) / ratio_key(ratio)


def remove_job_tempo_dirs(settings: Settings, job_ids: list[int]) -> None:
    """ジョブの伸縮済み音声のフォルダを消す（ジョブ・曲を消したとき。行は CASCADE で消える）。"""
    for job_id in job_ids:
        shutil.rmtree(job_tempo_dir(settings, job_id), ignore_errors=True)


def worker_count(settings: Settings, stems: int) -> int:
    n = settings.tempo_workers
    if n <= 0:
        n = min(MAX_WORKERS, max(1, (os.cpu_count() or 2) // 2))
    return max(1, min(n, stems))


# --- 登録・キャンセル ------------------------------------------------------------------


@dataclass(frozen=True)
class RenderRequest:
    render: TempoRender
    created: bool  # 作成を登録した（False: 作成済み・作成待ち・作成中のものを返した）


def get_render(session: Session, render_id: int) -> TempoRender:
    render = session.get(TempoRender, render_id)
    if render is None:
        raise TempoNotFound("速度を変えた音声が見つかりません。")
    return render


def request_render(session: Session, job_id: int, ratio: float) -> RenderRequest:
    """ジョブの全 stem を ratio 倍に伸縮する作成を登録する（commit まで）。

    作成済みなら最後に使った時刻を今にして返す。作成待ち・作成中ならそれを返す。
    失敗・キャンセルしたものは作成待ちに戻す。
    """
    r = normalize_ratio(ratio)
    job = session.get(SeparationJob, job_id)
    if job is None:
        raise TempoNotFound("ジョブが見つかりません。")
    if job.status != DONE:
        raise TempoConflict("分割が終わっていないジョブです。速度を変えた音声は作れません。")
    # ワーカーの片付け（キャッシュの上限）や別の要求と同時になっても 500 にしないよう、
    # 行の更新は条件付きの update の件数で判定し、作成と重なったら読み直す
    for _attempt in range(3):
        now = _utcnow()
        render = session.scalars(
            select(TempoRender).where(TempoRender.job_id == job_id, TempoRender.ratio == r)
        ).first()
        if render is None:
            # 別の処理が消した行の古いオブジェクトを手放す（同じ番号が使い回されることがある）
            for obj in [o for o in session.identity_map.values() if isinstance(o, TempoRender)]:
                session.expunge(obj)
            render = TempoRender(
                job_id=job_id, ratio=r, pitch_mode=PITCH_KEEP, status=QUEUED, progress=0.0,
                stage=STAGE_QUEUED, cancel_requested=False, created_at=now, last_used_at=now,
            )
            session.add(render)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()  # 同じ倍率を同時に登録した: 読み直す
                continue
            log.info("速度を変えた音声の作成を登録しました（job %d, %s 倍）。",
                     job_id, ratio_key(r))
            return RenderRequest(render, True)
        render_id = render.render_id
        res = session.execute(
            update(TempoRender)
            .where(
                TempoRender.render_id == render_id,
                TempoRender.status.in_((DONE, QUEUED, RUNNING)),
            )
            .values(last_used_at=now)
        )
        if res.rowcount == 1:  # type: ignore[attr-defined]
            session.commit()
            session.expire_all()
            fresh = session.get(TempoRender, render_id)
            if fresh is not None:
                return RenderRequest(fresh, False)
            continue
        # 失敗・キャンセルしたもの: 作成待ちに戻す
        res = session.execute(
            update(TempoRender)
            .where(
                TempoRender.render_id == render_id,
                TempoRender.status.in_((FAILED, CANCELED)),
            )
            .values(
                status=QUEUED, progress=0.0, stage=STAGE_QUEUED, error_message=None,
                cancel_requested=False, created_at=now, started_at=None, finished_at=None,
                last_used_at=now,
            )
        )
        session.commit()
        session.expire_all()
        if res.rowcount == 1:  # type: ignore[attr-defined]
            fresh = session.get(TempoRender, render_id)
            if fresh is not None:
                log.info("速度を変えた音声の作成を登録し直しました（render %d）。", render_id)
                return RenderRequest(fresh, True)
        # 行が消えた・状態が変わった: 読み直す
    raise TempoConflict("ほかの操作と重なりました。もう一度お試しください。")


def cancel_render(session: Session, settings: Settings, render_id: int) -> TempoRender:
    """作成をキャンセルする。作成待ちはすぐ canceled、作成中は cancel_requested を立てる。"""
    render = get_render(session, render_id)
    res = session.execute(
        update(TempoRender)
        .where(TempoRender.render_id == render_id, TempoRender.status == QUEUED)
        .values(status=CANCELED, stage=STAGE_CANCELED, finished_at=_utcnow(), cancel_requested=True)
    )
    if res.rowcount == 0:  # type: ignore[attr-defined]
        res = session.execute(
            update(TempoRender)
            .where(TempoRender.render_id == render_id, TempoRender.status == RUNNING)
            .values(cancel_requested=True)
        )
    session.commit()
    session.refresh(render)
    if res.rowcount == 0:  # type: ignore[attr-defined]
        raise TempoConflict(f"この作成は既に終わっています（状態: {render.status}）。")
    return render


def claim_next_render(session: Session) -> int | None:
    """いちばん古い作成待ちを running にして render_id を返す。無ければ None。"""
    while True:
        render_id = session.scalar(
            select(TempoRender.render_id)
            .where(TempoRender.status == QUEUED)
            .order_by(TempoRender.created_at, TempoRender.render_id)
            .limit(1)
        )
        if render_id is None:
            return None
        res = session.execute(
            update(TempoRender)
            .where(TempoRender.render_id == render_id, TempoRender.status == QUEUED)
            .values(status=RUNNING, stage="準備中", started_at=_utcnow(), progress=0.0)
        )
        session.commit()
        if res.rowcount == 1:  # type: ignore[attr-defined]
            return int(render_id)


def requeue_render(session: Session, render_id: int) -> None:
    """ワーカーの停止で中断した作成を作成待ちに戻す（次に起動したときにやり直す）。"""
    session.execute(
        update(TempoRender)
        .where(TempoRender.render_id == render_id, TempoRender.status == RUNNING)
        .values(status=QUEUED, stage=STAGE_QUEUED, progress=0.0)
    )
    session.commit()


# --- 実行 ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceStem:
    stem_id: int
    code: str
    master: Path


def leaf_sources(session: Session, settings: Settings, job_id: int) -> list[SourceStem]:
    """伸縮する stem（子に分かれていない stem = 画面で鳴らす stem）と master のパス（木の順）。

    詳細分割（T07）の子も含む（`stemapp.stem_view` の木。画面と同じ葉）。
    """
    from stemapp.stem_view import build_view

    rows = build_view(session, job_id).rows
    parents = {s.parent_stem_id for s, _ in rows if s.parent_stem_id is not None}
    leaves = [(s, t) for s, t in rows if s.stem_id not in parents]
    masters = {
        r.stem_id: r
        for r in session.scalars(
            select(StemRendition).where(
                StemRendition.stem_id.in_([s.stem_id for s, _ in leaves]),
                StemRendition.purpose == PURPOSE_MASTER,
            )
        )
    }
    return [
        SourceStem(s.stem_id, t.code, resolve_data_path(settings, masters[s.stem_id].file_path))
        for s, t in leaves
        if s.stem_id in masters
    ]


def output_frames(sources: list[SourceStem], ratio: float) -> int:
    """伸縮後の長さ（44.1kHz のサンプル数）。stem の長さが違えばいちばん短いものに合わせる。"""
    lengths = [sf.info(str(s.master)).frames for s in sources]
    return max(1, min(round(n / ratio) for n in lengths))


class _RenderGone(Exception):
    """書き込もうとした TEMPO_RENDER の行が消えていた。"""


class _Stop:
    """ワーカースレッドから見る「止めるか」（メインスレッドが DB を見て立てる）。"""

    def __init__(self) -> None:
        self.event = threading.Event()

    def __call__(self) -> bool:
        return self.event.is_set()


def _cancel_wanted(session: Session, render_id: int) -> bool:
    """キャンセルの依頼があるか。行が消えた（曲・ジョブが消された）ときも True。"""
    value = session.scalar(
        select(TempoRender.cancel_requested).where(TempoRender.render_id == render_id)
    )
    return value is None or bool(value)


def run_render(
    settings: Settings,
    session_factory: sessionmaker[Session],
    render_id: int,
    stretcher: Stretcher,
    should_stop: Callable[[], bool] | None = None,
    fmt: StreamFormat = DEFAULT_STREAM_FORMAT,
) -> str:
    """running の作成を1件実行する。結果の状態（done / failed / canceled / queued）を返す。

    should_stop（引数なしで bool を返す）が True になったら（ワーカーの停止）作成待ちに戻す。
    """
    with session_factory() as session:
        render = session.get(TempoRender, render_id)
        if render is None:
            return CANCELED
        job_id, ratio = render.job_id, render.ratio
        out_dir = render_dir(settings, job_id, ratio)
        shutil.rmtree(out_dir, ignore_errors=True)
        try:
            sources = leaf_sources(session, settings, job_id)
            if not sources:
                raise TempoError("このジョブには伸縮できる stem がありません。")
            frames = output_frames(sources, ratio)
        except Exception as e:
            log.exception("速度を変えた音声を作れませんでした（render %d）", render_id)
            _finish(session, settings, render_id, FAILED, message=_message(e))
            return FAILED
        out_dir.mkdir(parents=True, exist_ok=True)
        stop = _Stop()
        fractions = [0.0] * len(sources)
        workers = worker_count(settings, len(sources))
        log.info(
            "速度を変えた音声を作ります（render %d, job %d, %s 倍, stem %d, 並列 %d）。",
            render_id, job_id, ratio_key(ratio), len(sources), workers,
        )

        def task(i: int, src: SourceStem) -> Path:
            dst = out_dir / f"{src.code}.{fmt.extension}"

            def progress(f: float) -> None:
                fractions[i] = f

            stretcher(src.master, dst, ratio, frames, progress, stop)
            return dst

        outcome = DONE
        error: BaseException | None = None
        last_written = -1.0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stretch") as pool:
            futures: list[Future[Path]] = [
                pool.submit(task, i, s) for i, s in enumerate(sources)
            ]
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=WATCH_SEC, return_when=FIRST_EXCEPTION)
                failed = [f for f in finished if f.exception() is not None]
                if failed and error is None:
                    error = failed[0].exception()
                    stop.event.set()
                if not stop.event.is_set():
                    if should_stop is not None and should_stop():
                        outcome = QUEUED
                        stop.event.set()
                    elif _cancel_wanted(session, render_id):
                        outcome = CANCELED
                        stop.event.set()
                p = sum(fractions) / len(fractions)
                if not stop.event.is_set() and p - last_written >= PROGRESS_STEP:
                    last_written = p
                    session.execute(
                        update(TempoRender)
                        .where(TempoRender.render_id == render_id)
                        .values(progress=round(p * 0.99, 4), stage="作成中")
                    )
                    session.commit()
        if error is not None and not isinstance(error, StretchCanceled):
            outcome = FAILED
        if outcome == DONE and _cancel_wanted(session, render_id):
            outcome = CANCELED
        if outcome != DONE:
            shutil.rmtree(out_dir, ignore_errors=True)
            if outcome == QUEUED:
                requeue_render(session, render_id)
                log.info("停止の指示で中断しました。作成待ちに戻します（render %d）。", render_id)
            elif outcome == CANCELED:
                _finish(session, settings, render_id, CANCELED, stage=STAGE_CANCELED)
                log.info("速度を変えた音声の作成をキャンセルしました（render %d）。", render_id)
            else:
                log.error("速度を変えた音声を作れませんでした（render %d）: %s", render_id, error)
                _finish(session, settings, render_id, FAILED, message=_message(error))
            return outcome

        total = 0
        try:
            session.execute(
                delete(TempoRendition).where(TempoRendition.render_id == render_id)
            )
            for i, src in enumerate(sources):
                dst = futures[i].result()
                size = dst.stat().st_size
                total += size
                session.add(
                    TempoRendition(
                        render_id=render_id,
                        stem_id=src.stem_id,
                        codec=fmt.codec,
                        bitrate_kbps=fmt.bitrate_kbps,
                        file_path=data_relative(settings, dst),
                        bytes=size,
                    )
                )
            now = _utcnow()
            res = session.execute(
                update(TempoRender)
                .where(TempoRender.render_id == render_id)
                .values(
                    status=DONE, progress=1.0, stage="完了", finished_at=now, last_used_at=now,
                    dir_path=data_relative(settings, out_dir), bytes=total, frames=frames,
                    error_message=None,
                )
            )
            if res.rowcount != 1:  # type: ignore[attr-defined]
                raise _RenderGone
            session.commit()
        except Exception as e:
            # 書き込む直前に行が消された（配信用データの作り直し・詳細分割・ジョブの削除で
            # invalidate_job_tempo が呼ばれた）: 作ったファイルは古い音声なので捨てる。
            # 行と一緒にフォルダも消されるので、ファイルが無い（FileNotFoundError）こともある
            session.rollback()
            gone = session.get(TempoRender, render_id) is None
            shutil.rmtree(out_dir, ignore_errors=True)
            if not gone and not isinstance(e, (IntegrityError, _RenderGone)):
                log.exception("速度を変えた音声を保存できませんでした（render %d）", render_id)
                _finish(session, settings, render_id, FAILED, message=_message(e))
                return FAILED
            try:
                out_dir.parent.rmdir()  # ジョブのフォルダが空なら消す
            except OSError:
                pass
            _finish(session, settings, render_id, CANCELED, stage=STAGE_CANCELED)
            log.info("作成中に速度変更のキャッシュが消されたため捨てました（render %d）。", render_id)
            return CANCELED
        log.info(
            "速度を変えた音声を作りました（render %d, %.1f MB）。", render_id, total / 1024 / 1024
        )
        enforce_cache_limits(session, settings, protect={render_id})
        return DONE


def _message(e: BaseException | None) -> str:
    text = str(e) if e is not None else ""
    return text or "速度を変えた音声を作れませんでした。"


def _finish(
    session: Session, settings: Settings, render_id: int, status: str,
    *, message: str | None = None, stage: str | None = None,
) -> None:
    """作成を終わった状態（failed / canceled）にし、ファイルと TEMPO_RENDITION を消す。"""
    session.rollback()
    render = session.get(TempoRender, render_id)
    if render is None:
        return
    session.execute(delete(TempoRendition).where(TempoRendition.render_id == render_id))
    render.status = status
    render.finished_at = _utcnow()
    render.dir_path = None
    render.bytes = None
    if message is not None:
        render.error_message = message
    render.stage = stage if stage is not None else ("失敗しました" if status == FAILED else None)
    session.commit()
    shutil.rmtree(render_dir(settings, render.job_id, render.ratio), ignore_errors=True)


# --- キャッシュの片付け -------------------------------------------------------------------


def invalidate_job_tempo(session: Session, settings: Settings, job_id: int) -> int:
    """ジョブの速度変更のキャッシュ（TEMPO_RENDER の全行と `data/cache/tempo/<job_id>`）を消す。

    stem の音声や構成が変わった後（配信用データの作り直し、詳細分割など）に呼ぶ。古い音声を
    伸縮したものを使い続けないため。作成中のものは行が消えるので、ワーカーが止めて片付ける。
    commit まで行う。消した行の数を返す。
    """
    ids = list(
        session.scalars(select(TempoRender.render_id).where(TempoRender.job_id == job_id))
    )
    if ids:
        session.execute(delete(TempoRendition).where(TempoRendition.render_id.in_(ids)))
        session.execute(delete(TempoRender).where(TempoRender.render_id.in_(ids)))
    session.commit()
    shutil.rmtree(job_tempo_dir(settings, job_id), ignore_errors=True)
    if ids:
        log.info("job %d の速度変更のキャッシュを消しました（%d 件）。", job_id, len(ids))
    return len(ids)


def remove_render(session: Session, settings: Settings, render: TempoRender) -> None:
    """作成済みのもの（行とファイル）を消す（commit まで）。"""
    job_id, ratio = render.job_id, render.ratio
    session.execute(delete(TempoRendition).where(TempoRendition.render_id == render.render_id))
    session.delete(render)
    session.commit()
    shutil.rmtree(render_dir(settings, job_id, ratio), ignore_errors=True)


def enforce_cache_limits(
    session: Session, settings: Settings, protect: set[int] | None = None
) -> list[int]:
    """上限（1曲あたりの倍率の数・全体の容量）を超えた作成済みのものを、使っていない順に消す。

    protect の render_id は消さない（いま作った・使っているもの）。消した render_id を返す。
    """
    protect = protect or set()
    rows = session.execute(
        select(TempoRender, SeparationJob.track_id)
        .join(SeparationJob, SeparationJob.job_id == TempoRender.job_id)
        .where(TempoRender.status == DONE)
    ).all()
    # 新しく使った順
    rows.sort(key=lambda x: (_aware(x[0].last_used_at), x[0].render_id), reverse=True)
    victims: list[TempoRender] = []
    per_track: dict[int, int] = {}
    keep: list[TempoRender] = []
    limit = max(1, settings.tempo_cache_per_track)
    for render, track_id in rows:
        n = per_track.get(track_id, 0)
        if n >= limit and render.render_id not in protect:
            victims.append(render)
            continue
        per_track[track_id] = n + 1
        keep.append(render)
    max_bytes = max(0, settings.tempo_cache_max_mb) * 1024 * 1024
    total = sum(r.bytes or 0 for r in keep)
    for render in reversed(keep):  # 古い順
        if total <= max_bytes:
            break
        if render.render_id in protect:
            continue
        victims.append(render)
        total -= render.bytes or 0
    removed = []
    for render in victims:
        removed.append(render.render_id)
        log.info(
            "速度を変えた音声のキャッシュを消します（render %d, job %d, %s 倍）。",
            render.render_id, render.job_id, ratio_key(render.ratio),
        )
        remove_render(session, settings, render)
    return removed


def clean_orphan_dirs(session: Session, settings: Settings) -> list[str]:
    """DB に作成済み・作成中の行が無いフォルダ（`data/cache/tempo/<job_id>/<倍率>`）を消す。"""
    root = tempo_root(settings)
    if not root.is_dir():
        return []
    alive = {
        (job_id, ratio_key(ratio))
        for job_id, ratio in session.execute(
            select(TempoRender.job_id, TempoRender.ratio).where(
                TempoRender.status.in_((DONE, RUNNING))
            )
        )
    }
    removed: list[str] = []
    for job_dir in root.iterdir():
        try:
            job_id = int(job_dir.name)
        except ValueError:
            continue
        if not job_dir.is_dir():
            continue
        for d in job_dir.iterdir():
            if (job_id, d.name) not in alive:
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                else:
                    d.unlink(missing_ok=True)
                removed.append(f"{job_id}/{d.name}")
        if not any(job_dir.iterdir()):
            job_dir.rmdir()
    return removed


def recover_interrupted_renders(session: Session, settings: Settings) -> list[int]:
    """running のまま残った作成を failed にし（ワーカー起動時）、行の無いフォルダを消す。"""
    ids = list(
        session.scalars(select(TempoRender.render_id).where(TempoRender.status == RUNNING))
    )
    for render_id in ids:
        _finish(session, settings, render_id, FAILED, message=INTERRUPTED_MESSAGE)
    removed = clean_orphan_dirs(session, settings)
    if removed:
        log.info("残っていた速度変更のフォルダを消しました: %s", removed)
    enforce_cache_limits(session, settings)
    return ids


# --- 画面に返す形 -----------------------------------------------------------------------


def render_to_dict(session: Session, render: TempoRender) -> dict[str, Any]:
    files: dict[str, str] = {}
    if render.status == DONE:
        for code, stem_id in session.execute(
            select(StemType.code, TempoRendition.stem_id)
            .join(Stem, Stem.stem_id == TempoRendition.stem_id)
            .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
            .where(TempoRendition.render_id == render.render_id)
        ):
            files[code] = f"/api/files/tempo/{render.render_id}/{stem_id}"
    return {
        "render_id": render.render_id,
        "job_id": render.job_id,
        "ratio": render.ratio,
        "ratio_key": ratio_key(render.ratio),
        "pitch_mode": render.pitch_mode,
        "status": render.status,
        "progress": render.progress,
        "stage": render.stage,
        "error_message": render.error_message,
        "cancel_requested": bool(render.cancel_requested),
        "bytes": render.bytes,
        "frames": render.frames,
        "files": files,
    }
