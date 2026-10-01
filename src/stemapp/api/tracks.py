"""曲・ジョブ・stem の API。"""

from __future__ import annotations

import logging
import shutil
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.api.common import (
    SessionDep,
    iso,
    job_to_dict,
    not_found,
    preset_codes,
    sse_response,
)
from stemapp.beats.edit import BeatEditError
from stemapp.beats.service import (
    beats_payload,
    can_undo,
    edit_grid,
    get_grid,
    reset_edits,
    undo_edit,
)
from stemapp.config import Settings
from stemapp.delivery import missing_delivery
from stemapp.exports import export_ids_for_jobs, remove_export_dirs
from stemapp.jobs import (
    ACTIVE_STATUSES,
    DONE,
    FAILED,
    FINISHED_STATUSES,
    QUEUED,
    RUNNING,
    JobConflict,
    JobNotFound,
    delete_job,
    enqueue_full_job,
    request_cancel,
    request_postprocess,
)
from stemapp.models import (
    BeatGrid,
    InputSource,
    SeparationJob,
    SeparationPreset,
    Stem,
    StemRendition,
    StemType,
    Track,
    Waveform,
)
from stemapp.separation.pipeline import SeparationError
from stemapp.separation.refine import (
    RefineConflict,
    RefineInvalid,
    RefineNotFound,
    TypeIndex,
    check_refine,
    enqueue_refine_job,
    load_methods,
    method_applies,
    refine_payload,
)
from stemapp.stem_folders import remove_job_dir
from stemapp.stem_view import build_view
from stemapp.tempo.service import remove_job_tempo_dirs

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tracks"])

# --- 曲 ---------------------------------------------------------------------------


def _jobs_by_track(session: Session) -> dict[int, list[SeparationJob]]:
    """曲ごとのジョブ（新しい順）。"""
    out: dict[int, list[SeparationJob]] = {}
    for job in session.scalars(select(SeparationJob).order_by(SeparationJob.job_id.desc())):
        out.setdefault(job.track_id, []).append(job)
    return out


def _track_summary(
    track: Track, jobs: list[SeparationJob], presets: dict[int, SeparationPreset]
) -> dict[str, Any]:
    """jobs はその曲のジョブ（新しい順）。

    一覧の状態（latest_job・active_job・active_count）は分割（full）のジョブだけで決める
    （詳細分割は曲の中の1つの stem の処理なので、一覧には出さない）。
    """
    jobs = [j for j in jobs if j.job_kind == "full"]
    done = [j for j in jobs if j.status == "done"]

    def experimental(j: SeparationJob) -> bool:
        p = presets.get(j.preset_id) if j.preset_id is not None else None
        return bool(p is not None and p.is_experimental)

    # 実験プリセット（聴き比べ用）のジョブは、ほかに完了したジョブがあれば既定にしない
    playable = next((j for j in done if not experimental(j)), done[0] if done else None)
    # 分割中のもの、無ければ次に分割されるもの（いちばん古い分割待ち）
    running = [j for j in jobs if j.status == RUNNING]
    queued = [j for j in jobs if j.status == QUEUED]
    active = running[0] if running else (queued[-1] if queued else None)
    return {
        "track_id": track.track_id,
        "title": track.title,
        "artist": track.artist,
        "duration_sec": track.duration_sec,
        "created_at": iso(track.created_at),
        "latest_job": job_to_dict(jobs[0], presets) if jobs else None,
        # 再生に使う既定のジョブ（完了した full ジョブのうち新しいもの。実験プリセットは後回し）
        "playable_job_id": playable.job_id if playable is not None else None,
        # 分割待ち・分割中のジョブ（同じ曲に複数の分け方があるとき、一覧はこれを優先して出す）
        "active_job": job_to_dict(active, presets) if active is not None else None,
        "active_count": len(running) + len(queued),
    }


@router.get("/tracks")
def list_tracks(session: SessionDep) -> dict[str, Any]:
    presets = preset_codes(session)
    jobs = _jobs_by_track(session)
    tracks = session.scalars(select(Track).order_by(Track.track_id.desc())).all()
    return {"tracks": [_track_summary(t, jobs.get(t.track_id, []), presets) for t in tracks]}


@router.get("/tracks/{track_id}")
def get_track(track_id: int, session: SessionDep) -> dict[str, Any]:
    track = session.get(Track, track_id)
    if track is None:
        raise not_found("曲")
    presets = preset_codes(session)
    jobs = session.scalars(
        select(SeparationJob)
        .where(SeparationJob.track_id == track_id)
        .order_by(SeparationJob.job_id.desc())
    ).all()
    sources = session.scalars(
        select(InputSource)
        .where(InputSource.track_id == track_id)
        .order_by(InputSource.source_id)
    ).all()
    out = _track_summary(track, list(jobs), presets)
    levels = _stem_levels(session, [j.job_id for j in jobs if j.status == "done"])
    out["jobs"] = [
        {**job_to_dict(j, presets), "stem_rms_db": levels.get(j.job_id, {})} for j in jobs
    ]
    out["sources"] = [
        {
            "source_id": s.source_id,
            "source_type": s.source_type,
            "original_name": s.original_name,
            "url": s.url,
            "fetched_at": iso(s.fetched_at),
        }
        for s in sources
    ]
    return out


def _stem_levels(session: Session, job_ids: list[int]) -> dict[int, dict[str, float | None]]:
    """ジョブごとの stem の RMS（dBFS）。{job_id: {stem の code: rms_db}}（表示順）。"""
    out: dict[int, dict[str, float | None]] = {}
    if not job_ids:
        return out
    rows = session.execute(
        select(Stem.job_id, StemType.code, Stem.rms_db)
        .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
        .where(Stem.job_id.in_(job_ids))
        .order_by(Stem.job_id, StemType.display_order)
    ).all()
    for job_id, code, level in rows:
        out.setdefault(job_id, {})[code] = round(level, 2) if level is not None else None
    return out


@router.delete("/tracks/{track_id}")
def delete_track(
    track_id: int, request: Request, session: SessionDep
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    track = session.get(Track, track_id)
    if track is None:
        raise not_found("曲")
    jobs = session.scalars(select(SeparationJob).where(SeparationJob.track_id == track_id)).all()
    if any(j.status in ACTIVE_STATUSES for j in jobs):
        raise HTTPException(
            status_code=409,
            detail="分割待ち・分割中のジョブがあるため削除できません。キャンセルしてから削除してください。",
        )
    job_ids = [j.job_id for j in jobs]
    job_dirs = [(j.job_id, j.output_dir) for j in jobs]
    export_ids = export_ids_for_jobs(session, job_ids)
    session.delete(track)  # JOB・STEM・INPUT_SOURCE・EXPORT などは外部キーの CASCADE で消える
    session.commit()
    remove_export_dirs(settings, export_ids)
    shutil.rmtree(settings.tracks_dir / str(track_id), ignore_errors=True)
    for job_id, output_dir in job_dirs:
        remove_job_dir(session, settings, job_id, output_dir)
    remove_job_tempo_dirs(settings, job_ids)  # 速度を変えた音声（行は CASCADE で消える）
    log.info("曲を削除しました（track %d, job %s）。", track_id, job_ids)
    return {"deleted": True, "track_id": track_id, "job_ids": job_ids}


# --- 拍 ---------------------------------------------------------------------------

# 作り直し（postprocess）の「欠けているもの」に入れる、拍が無いことの印
MISSING_BEATS = "beats"


@router.get("/tracks/{track_id}/beats")
def get_beats(track_id: int, session: SessionDep) -> dict[str, Any]:
    """有効な拍（直した結果があればそれ）・小節の頭・拍子・区間ごとの BPM、直したか（edited）、
    元に戻せるか（can_undo）。まだ解析していなければ 404。"""
    return beats_payload(_grid_or_404(session, track_id), undo=can_undo(session, track_id))


def _grid_or_404(session: Session, track_id: int) -> BeatGrid:
    if session.get(Track, track_id) is None:
        raise not_found("曲")
    grid = get_grid(session, track_id)
    if grid is None:
        raise HTTPException(status_code=404, detail="この曲の拍はまだ解析されていません。")
    return grid


EditOp = Literal["downbeat", "double", "half", "meter", "shift", "tap", "cues"]
Sec = Annotated[float, Field(ge=0.0, le=24 * 60 * 60, allow_inf_nan=False)]


class BeatEditRequest(BaseModel):
    """拍の補正の操作。計算はサーバー側（`stemapp.beats.edit`）。

    range: segment（再生位置を含む区間。既定）/ all（曲全体）/ loop（loop_start〜loop_end）。
    操作ごとの引数: meter は beats_per_bar、shift は delta_sec、tap は taps（曲の時刻の列）、
    cues は cue_start・cue_end・bars（beats_per_bar は省略可）。
    """

    op: EditOp
    position: Sec = 0.0
    range: Literal["segment", "all", "loop"] = "segment"
    loop_start: Sec | None = None
    loop_end: Sec | None = None
    beats_per_bar: int | None = Field(default=None, ge=2, le=12)
    delta_sec: float | None = Field(default=None, ge=-1.0, le=1.0)
    taps: list[Sec] | None = Field(default=None, max_length=64)  # 有限・0 以上（Sec）
    cue_start: Sec | None = None
    cue_end: Sec | None = None
    bars: int | None = Field(default=None, ge=1, le=1024)

    @model_validator(mode="after")
    def _needed_args(self) -> BeatEditRequest:
        need = {
            "meter": ("beats_per_bar",),
            "shift": ("delta_sec",),
            "tap": ("taps",),
            "cues": ("cue_start", "cue_end", "bars"),
        }.get(self.op, ())
        missing = [k for k in need if getattr(self, k) is None]
        if missing:
            raise ValueError(f"{self.op} には {', '.join(missing)} が必要です。")
        return self


# 曲の長さを超える時刻を受け付ける余裕（秒）。曲の長さが無いときは自動の拍の最後からの余裕
TIME_MARGIN_SEC = 1.0
NO_DURATION_MARGIN_SEC = 5.0


def _check_times(session: Session, grid: BeatGrid, body: BeatEditRequest) -> None:
    """時刻（再生位置・ループ・キュー・タップ）が曲の長さの中にあるか。外れていれば 400。"""
    track = session.get(Track, grid.track_id)
    if track is not None and track.duration_sec:
        limit = float(track.duration_sec) + TIME_MARGIN_SEC
    else:
        auto = list(grid.beats_json or [])
        limit = (max(auto) if auto else 0.0) + NO_DURATION_MARGIN_SEC
    times = [body.position, body.loop_start, body.loop_end, body.cue_start, body.cue_end]
    times += list(body.taps or [])
    if any(t is not None and t > limit for t in times):
        raise HTTPException(status_code=400, detail="曲の長さを超える時刻は指定できません。")


@router.post("/tracks/{track_id}/beats/edit")
def edit_beats(track_id: int, body: BeatEditRequest, session: SessionDep) -> dict[str, Any]:
    """拍を補正して保存し、有効な拍を返す。補正できないときは 400（理由は detail）。"""
    grid = _grid_or_404(session, track_id)
    _check_times(session, grid, body)
    params = body.model_dump(exclude_none=True, exclude={"op"})
    try:
        edit_grid(session, grid, body.op, params)
    except BeatEditError as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e
    session.commit()
    return beats_payload(grid, undo=True)


@router.post("/tracks/{track_id}/beats/undo")
def undo_beats(track_id: int, session: SessionDep) -> dict[str, Any]:
    """直前の補正を取り消す。取り消すものが無ければ 409。"""
    grid = _grid_or_404(session, track_id)
    if not undo_edit(session, grid):
        raise HTTPException(status_code=409, detail="元に戻せる操作がありません。")
    session.commit()
    return beats_payload(grid, undo=can_undo(session, track_id))


@router.post("/tracks/{track_id}/beats/reset")
def reset_beats(track_id: int, session: SessionDep) -> dict[str, Any]:
    """自動の結果に戻す（元に戻すで取り消せる）。直していなければ何もしない。"""
    grid = _grid_or_404(session, track_id)
    reset_edits(session, grid)
    session.commit()
    return beats_payload(grid, undo=can_undo(session, track_id))


BEATS_MESSAGES = {
    None: "拍の解析を登録しました。",
    "active": "配信用データ・拍を作成待ち・作成中です。",
}


@router.post("/tracks/{track_id}/beats")
def reanalyze_beats(track_id: int, response: Response, session: SessionDep) -> dict[str, Any]:
    """拍を解析し直す（ワーカーが「作り直し」として処理する）。登録したら 202。

    今の結果は消してから登録する（解析が終わるまで拍の無い曲として表示される）。
    作り直しが作成待ち・作成中のときも拍は消す（ワーカーは作り直しの最後に拍の有無を見るので、
    その作り直しの中で解析される）。分割が終わっていない曲は 409。
    """
    if session.get(Track, track_id) is None:
        raise not_found("曲")
    job = session.scalars(
        select(SeparationJob)
        .where(
            SeparationJob.track_id == track_id,
            SeparationJob.job_kind == "full",
            SeparationJob.status == DONE,
        )
        .order_by(SeparationJob.job_id.desc())
    ).first()
    if job is None:
        raise HTTPException(
            status_code=409, detail="分割が終わっていない曲です。分割が終わると拍も解析されます。"
        )
    grid = session.get(BeatGrid, track_id)
    if grid is not None:
        # 直した結果は履歴に移す（解析が終わった後、元に戻すで戻せる）
        reset_edits(session, grid, op="reanalyze")
        session.delete(grid)
        session.commit()
    if job.postprocess_status in ACTIVE_STATUSES:
        res_job, created, reason = job, False, "active"
    else:
        res = request_postprocess(session, job.job_id, [MISSING_BEATS])
        res_job, created, reason = res.job, res.created, res.reason
    response.status_code = 202 if created else 200
    return {
        "created": created,
        "reason": reason,
        "message": BEATS_MESSAGES.get(reason, ""),
        "job": job_to_dict(res_job, preset_codes(session)),
    }


def _missing_for_postprocess(session: Session, job: SeparationJob) -> list[str]:
    """作り直しで作るもの（配信用データが欠けた stem と、拍が無ければ "beats"）。"""
    if job.status != DONE:
        return []
    missing = missing_delivery(session, job.job_id)
    if get_grid(session, job.track_id) is None:
        missing.append(MISSING_BEATS)
    return missing


# --- ジョブ -------------------------------------------------------------------------


class JobRequest(BaseModel):
    preset: str | None = None
    force: bool = False


ENQUEUE_MESSAGES = {
    None: "分割ジョブを登録しました。",
    "active": "この曲はこの分け方で分割待ち・分割中です。",
    "done": "この曲はこの分け方で分割済みです（分割し直すには force を指定してください）。",
}


@router.post("/tracks/{track_id}/jobs")
def create_job(
    track_id: int,
    response: Response,
    session: SessionDep,
    body: JobRequest | None = None,
) -> dict[str, Any]:
    body = body or JobRequest()
    try:
        res = enqueue_full_job(session, track_id, body.preset, force=body.force)
    except JobNotFound as e:
        raise not_found("曲") from e
    except SeparationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    response.status_code = 201 if res.created else 200
    return {
        "created": res.created,
        "reason": res.reason,
        "message": ENQUEUE_MESSAGES.get(res.reason, ""),
        "job": job_to_dict(res.job, preset_codes(session)),
    }


def _get_job(session: Session, job_id: int) -> SeparationJob:
    job = session.get(SeparationJob, job_id)
    if job is None:
        raise not_found("ジョブ")
    return job


@router.get("/jobs/{job_id}")
def get_job(job_id: int, session: SessionDep) -> dict[str, Any]:
    return job_to_dict(_get_job(session, job_id), preset_codes(session))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, session: SessionDep) -> dict[str, Any]:
    try:
        job = request_cancel(session, job_id)
    except JobNotFound as e:
        raise not_found("ジョブ") from e
    except JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return job_to_dict(job, preset_codes(session))


@router.delete("/jobs/{job_id}")
def delete_job_api(job_id: int, request: Request, session: SessionDep) -> dict[str, Any]:
    """ジョブ（1つの分け方）を消す。stem のファイルも消える。曲とほかのジョブは残る。

    分割待ち・分割中（配信用データの作成中を含む）は 409。
    """
    settings: Settings = request.app.state.settings
    try:
        job = delete_job(session, settings, job_id)
    except JobNotFound as e:
        raise not_found("ジョブ") from e
    except JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"deleted": True, "job_id": job_id, "track_id": job.track_id}


POSTPROCESS_MESSAGES = {
    None: "配信用データの作成を登録しました。",
    "active": "配信用データを作成待ち・作成中です。",
    "ready": "配信用データはそろっています。",
}


@router.post("/jobs/{job_id}/postprocess")
def postprocess_job(job_id: int, response: Response, session: SessionDep) -> dict[str, Any]:
    """配信用データ（stream rendition・peaks）や拍が欠けているとき、ワーカーで作り直す。

    登録したら 202、欠けていない・作成待ちなら 200。done 以外のジョブは 409。
    missing には欠けている stem の code と、拍が無ければ "beats" が入る。
    """
    job = _get_job(session, job_id)
    missing = _missing_for_postprocess(session, job)
    try:
        res = request_postprocess(session, job_id, missing)
    except JobNotFound as e:
        raise not_found("ジョブ") from e
    except JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    response.status_code = 202 if res.created else 200
    return {
        "created": res.created,
        "reason": res.reason,
        "message": POSTPROCESS_MESSAGES.get(res.reason, ""),
        "missing": missing,
        "job": job_to_dict(res.job, preset_codes(session)),
    }


@router.get("/jobs/{job_id}/events")
def job_events(job_id: int, request: Request) -> StreamingResponse:
    """Server-Sent Events。progress・stage・status が変わるたびに `event: job` を送る。

    終わった状態（done / failed / canceled）を送ったら閉じる。
    """
    factory = request.app.state.session_factory
    with factory() as session:
        _get_job(session, job_id)

    def load() -> dict[str, Any] | None:
        with factory() as s:
            job = s.get(SeparationJob, job_id)
            return job_to_dict(job, preset_codes(s)) if job is not None else None

    return sse_response(
        request, load, event="job", gone_message="ジョブが削除されました。",
        finished=FINISHED_STATUSES,
    )


# --- stem ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/stems")
def job_stems(job_id: int, session: SessionDep) -> dict[str, Any]:
    """分け方（ジョブ）の stem。詳細分割（refine）の子も木の順（親の直後に子）で入る。

    stem ごとの refine_methods は「もっと分ける」の方法（子を持たない stem だけ。available が
    false なら reason に理由）、refined_by は子を作った詳細分割のジョブ（戻すときに消すもの）。
    refine_jobs は分割待ち・分割中・失敗した詳細分割（stem ごとに新しいもの1つ）。
    """
    job = _get_job(session, job_id)
    view = build_view(session, job_id)
    rows = view.rows
    code_of = {s.stem_id: t.code for s, t in rows}
    stem_ids = list(code_of)
    renditions: dict[int, list[StemRendition]] = {}
    for r in session.scalars(
        select(StemRendition)
        .where(StemRendition.stem_id.in_(stem_ids))
        .order_by(StemRendition.rendition_id)
    ):
        renditions.setdefault(r.stem_id, []).append(r)
    waves: dict[int, list[Waveform]] = {}
    for w in session.scalars(
        select(Waveform).where(Waveform.stem_id.in_(stem_ids)).order_by(Waveform.samples_per_px)
    ):
        waves.setdefault(w.stem_id, []).append(w)

    types = TypeIndex.load(session)
    methods = load_methods(session, types)
    methods_by_id = {m.model_id: m for m in methods.values()}
    refine_rows = (
        list(
            session.scalars(
                select(SeparationJob)
                .where(
                    SeparationJob.job_kind == "refine",
                    SeparationJob.input_stem_id.in_(stem_ids),
                )
                .order_by(SeparationJob.job_id)
            )
        )
        if stem_ids
        else []
    )
    active_refines = [j for j in refine_rows if j.status in ACTIVE_STATUSES]
    busy_stems = {j.input_stem_id for j in active_refines}
    can_refine = job.job_kind == "full" and job.status == DONE and job.output_dir is not None
    # 分けられない理由を画面に出す（ボタンを黙って消さない）
    refine_note = (
        "詳細分割（もっと分ける）するには、先に保存フォルダの移行"
        "（stemapp migrate-folders）が必要です。"
        if job.job_kind == "full" and job.status == DONE and job.output_dir is None
        else None
    )

    def methods_of(stem: Stem, stype: StemType) -> list[dict[str, Any]]:
        if not can_refine or view.children_of(stem.stem_id):
            return []
        out = []
        for m in methods.values():
            if not method_applies(m, stype.code, types):
                continue
            item = refine_payload(m, types, stype.code)
            if stem.stem_id in busy_stems:
                item.update(available=False, reason="分割待ち・分割中です。")
            else:
                chk = check_refine(
                    session, view, stem.stem_id, m, types,
                    active_jobs=active_refines, methods_by_id=methods_by_id,
                )
                item.update(available=chk.ok, reason=chk.reason)
            out.append(item)
        return out

    def refined_by(stem_id: int) -> dict[str, Any] | None:
        rj = view.refined_by.get(stem_id)
        if rj is None:
            return None
        m = methods_by_id.get(rj.refine_model_id or -1)
        return {
            "job_id": rj.job_id,
            "model": m.filename if m else None,
            "display_name": m.display_name if m else None,
            "warning": rj.warning,
        }

    stems = []
    for s, t in rows:
        stems.append(
            {
                "stem_id": s.stem_id,
                "job_id": s.job_id,
                "stem_type_id": t.stem_type_id,
                "code": t.code,
                "display_name": t.display_name,
                "color": t.color,
                "parent_stem_id": s.parent_stem_id,
                "parent_code": code_of.get(s.parent_stem_id) if s.parent_stem_id else None,
                "is_residual": s.is_residual,
                "rms_db": s.rms_db,
                "is_silent": s.is_silent,
                "renditions": [
                    {
                        "rendition_id": r.rendition_id,
                        "purpose": r.purpose,
                        "codec": r.codec,
                        "bitrate_kbps": r.bitrate_kbps,
                        "bytes": r.bytes,
                        "url": f"/api/files/renditions/{r.rendition_id}",
                    }
                    for r in renditions.get(s.stem_id, [])
                ],
                "peaks": [
                    {
                        "samples_per_px": w.samples_per_px,
                        "url": f"/api/files/peaks/{s.stem_id}/{w.samples_per_px}",
                    }
                    for w in waves.get(s.stem_id, [])
                ],
                "refine_methods": methods_of(s, t),
                "refined_by": refined_by(s.stem_id),
            }
        )
    # stem ごとに新しい詳細分割（done 以外）。done の結果は子として木に入っている
    latest: dict[int, SeparationJob] = {}
    for rj in refine_rows:
        if rj.input_stem_id is not None:
            latest[rj.input_stem_id] = rj
    presets = preset_codes(session)
    refine_jobs = [
        {
            **job_to_dict(rj, presets),
            "input_code": code_of.get(rj.input_stem_id or -1),
            "model": (
                methods_by_id[rj.refine_model_id].filename
                if rj.refine_model_id in methods_by_id
                else None
            ),
        }
        for rj in latest.values()
        if rj.status in (QUEUED, RUNNING, FAILED)
    ]
    missing = missing_delivery(session, job_id) if job.status == "done" else []
    return {
        "job_id": job.job_id,
        "track_id": job.track_id,
        "status": job.status,
        "output_gain_db": job.output_gain_db,
        "postprocess_status": job.postprocess_status,
        "beat_warning": job.beat_warning,
        # 配信用データ（stream と全解像度の peaks）がそろっているか
        "delivery_ready": job.status == "done" and not missing,
        "delivery_missing": missing,
        "stems": stems,
        "refine_jobs": refine_jobs,
        "refine_note": refine_note,
    }


# --- 詳細分割（もっと分ける） ------------------------------------------------------------


class RefineRequest(BaseModel):
    # 方法（MODEL.filename。例 "MDX23C-DrumSep-aufr33-jarredou.ckpt"。HPSS は "hpss"）
    model: str = Field(min_length=1, max_length=300)
    force: bool = False


REFINE_MESSAGES = {
    None: "「もっと分ける」を登録しました。",
    "active": "この stem はこの方法で分割待ち・分割中です。",
    "done": "この stem はこの方法で分けてあります（分け直すには force を指定してください）。",
}


@router.post("/stems/{stem_id}/refine")
def refine_stem(
    stem_id: int, body: RefineRequest, response: Response, session: SessionDep
) -> dict[str, Any]:
    """stem をさらに分けるジョブを登録する（登録したら 201、既にあれば 200）。

    分けられない stem・方法は 400、今の状態ではできない（別の方法で分けてある等）は 409。
    子を消して分ける前に戻すには、refined_by のジョブを DELETE /api/jobs/{job_id} で消す。
    """
    try:
        res = enqueue_refine_job(session, stem_id, body.model, force=body.force)
    except RefineNotFound as e:
        raise not_found("stem") from e
    except RefineInvalid as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RefineConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    response.status_code = 201 if res.created else 200
    return {
        "created": res.created,
        "reason": res.reason,
        "message": REFINE_MESSAGES.get(res.reason, ""),
        "job": job_to_dict(res.job, preset_codes(session)),
    }
