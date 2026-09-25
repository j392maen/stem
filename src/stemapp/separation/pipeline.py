"""分離パイプライン（SPEC 5章）。

手順は DB の SEPARATION_PRESET / PRESET_STEP から組み立てる。流れ:

1. input=mixture のステップを step_order 順に実行する。
   - role=multistem の出力（vocals, drums, bass, guitar, piano, other）を ensemble_weight で平均。
   - vocals は、multistem の vocals と role=vocals の出力を ensemble_weight で重み付き平均。
2. input=vocals のステップ（role=karaoke）を vocals に適用し、lead を重み付き平均で得る。
3. backing = vocals − lead（残差）。
4. other += mixture − Σ上位 stem。これで上位 stem の合計が元の曲に一致する。

`run_plan` は DB を触らない計算部分（bench でも使う）。`separate_file` が正規化・DB 登録・
ファイル保存まで行う。
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.audio import (
    FfmpegRunner,
    count_clipped,
    normalize_audio,
    rms_db,
    write_flac24,
    write_wav_float,
)
from stemapp.config import Settings
from stemapp.models import (
    Model,
    PresetStep,
    SeparationJob,
    SeparationPreset,
    Stem,
    StemRendition,
    StemType,
    Track,
)
from stemapp.separation.base import (
    DEVICE_CPU,
    DEVICE_CUDA,
    OPT_CHUNK_SCALE,
    Separator,
    is_oom_error,
)

log = logging.getLogger(__name__)

SILENT_THRESHOLD_DB = -60.0

ROLE_MULTISTEM = "multistem"
ROLE_VOCALS = "vocals"
ROLE_KARAOKE = "karaoke"
INPUT_MIXTURE = "mixture"
INPUT_VOCALS = "vocals"

# 出力名（separation/base.py 参照）
VOCALS = "vocals"
OTHER = "other"
LEAD = "lead_vocal"
BACKING = "backing_vocal"
RESIDUAL_STEMS: frozenset[str] = frozenset({BACKING})

ProgressCallback = Callable[[float, str], None]


class SeparationError(RuntimeError):
    """分割の失敗（メッセージは日本語）。"""


# --- 手順 ------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSpec:
    order: int
    model_filename: str
    model_name: str
    input: str  # mixture / vocals
    role: str  # multistem / vocals / karaoke
    weight: float = 1.0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PresetPlan:
    code: str
    display_name: str
    steps: list[StepSpec]
    preset_id: int | None = None


def load_plan(session: Session, code: str | None = None) -> PresetPlan:
    """DB からプリセットの手順を読む。code が None なら is_default のもの。"""
    stmt = select(SeparationPreset)
    if code is None:
        stmt = stmt.where(SeparationPreset.is_default.is_(True))
    else:
        stmt = stmt.where(SeparationPreset.code == code)
    preset = session.scalars(stmt).first()
    if preset is None:
        what = "既定のプリセット" if code is None else f"プリセット「{code}」"
        raise SeparationError(f"{what}が DB にありません（stemapp init-db を実行してください）。")
    rows = session.execute(
        select(PresetStep, Model)
        .join(Model, Model.model_id == PresetStep.model_id)
        .where(PresetStep.preset_id == preset.preset_id)
        .order_by(PresetStep.step_order)
    ).all()
    steps = [
        StepSpec(
            order=s.step_order,
            model_filename=m.filename,
            model_name=m.display_name,
            input=s.input,
            role=s.role,
            weight=float(s.ensemble_weight),
            options=dict(s.options_json or {}),
        )
        for s, m in rows
    ]
    plan = PresetPlan(preset.code, preset.display_name, steps, preset.preset_id)
    validate_plan(plan)
    return plan


def validate_plan(plan: PresetPlan) -> None:
    """パイプラインが扱える手順か確かめる。"""
    roles = [s.role for s in plan.steps]
    if ROLE_MULTISTEM not in roles:
        raise SeparationError(f"プリセット「{plan.code}」に multistem のステップがありません。")
    if ROLE_KARAOKE not in roles:
        raise SeparationError(f"プリセット「{plan.code}」に karaoke のステップがありません。")
    for s in plan.steps:
        want = INPUT_VOCALS if s.role == ROLE_KARAOKE else INPUT_MIXTURE
        if s.role not in (ROLE_MULTISTEM, ROLE_VOCALS, ROLE_KARAOKE):
            raise SeparationError(f"ステップ {s.order} の role「{s.role}」には対応していません。")
        if s.input != want:
            raise SeparationError(
                f"ステップ {s.order}（{s.role}）の input は {want} である必要があります"
                f"（今は {s.input}）。"
            )
        if s.weight <= 0:
            raise SeparationError(f"ステップ {s.order} の ensemble_weight は正の数にしてください。")


# --- OOM への対応 -------------------------------------------------------------------


@dataclass(frozen=True)
class OomPolicy:
    """メモリ不足のとき、チャンクを shrink_factor 倍に max_shrinks 回まで縮め、だめなら CPU。"""

    max_shrinks: int = 2
    shrink_factor: float = 0.5
    cpu_fallback: bool = True


@dataclass
class StepResult:
    order: int
    model_filename: str
    role: str
    seconds: float
    device: str
    chunk_scale: float
    peak_memory_mb: float | None
    attempts: list[str] = field(default_factory=list)  # やり直しの記録（日本語）


def run_step(
    separator: Separator,
    step: StepSpec,
    wav_path: Path,
    device: str,
    policy: OomPolicy,
) -> tuple[dict[str, np.ndarray], StepResult]:
    """1ステップを実行する。OOM ならチャンク縮小 → CPU の順にやり直す。"""
    scale = 1.0
    dev = device
    shrinks = 0
    attempts: list[str] = []
    separator.reset_peak_memory()
    t0 = time.perf_counter()
    while True:
        opts = dict(step.options)
        if scale != 1.0:
            opts[OPT_CHUNK_SCALE] = scale
        try:
            out = separator.separate(wav_path, step.model_filename, opts, dev, step.role)
            break
        except Exception as e:
            if not is_oom_error(e) or dev != DEVICE_CUDA:
                raise
            if shrinks < policy.max_shrinks:
                shrinks += 1
                new_scale = scale * policy.shrink_factor
                msg = (
                    f"ステップ {step.order}（{step.model_filename}）で GPU メモリ不足。"
                    f"チャンクを {scale:g} 倍 → {new_scale:g} 倍に縮めて再試行します。"
                )
                scale = new_scale
            elif policy.cpu_fallback:
                msg = (
                    f"ステップ {step.order}（{step.model_filename}）でチャンクを {scale:g} 倍まで"
                    "縮めても GPU メモリ不足。CPU で実行します（時間がかかります）。"
                )
                dev = DEVICE_CPU
                scale = 1.0
            else:
                raise
            log.warning(msg)
            attempts.append(msg)
    result = StepResult(
        order=step.order,
        model_filename=step.model_filename,
        role=step.role,
        seconds=time.perf_counter() - t0,
        device=dev,
        chunk_scale=scale,
        peak_memory_mb=separator.peak_memory_mb(),
        attempts=attempts,
    )
    return out, result


# --- 計算部分 ----------------------------------------------------------------------


@dataclass
class SeparationOutput:
    # stem 名 → (samples, 2) float32。上位 stem と lead_vocal / backing_vocal。
    stems: dict[str, np.ndarray]
    top_level: list[str]  # 上位 stem の名前（合計が mixture になるもの）
    steps: list[StepResult]
    seconds: float

    @property
    def peak_memory_mb(self) -> float | None:
        peaks = [s.peak_memory_mb for s in self.steps if s.peak_memory_mb is not None]
        return max(peaks) if peaks else None


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """長さを n サンプルにそろえる（モデルの出力が数サンプルずれることがある）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - x.shape[0], x.shape[1]))], axis=0)


class _WeightedSum:
    def __init__(self) -> None:
        self.total: np.ndarray | None = None
        self.weight = 0.0

    def add(self, x: np.ndarray, w: float) -> None:
        self.total = x * w if self.total is None else self.total + x * w
        self.weight += w

    def mean(self) -> np.ndarray:
        assert self.total is not None and self.weight > 0
        return self.total / self.weight


def _require(out: dict[str, np.ndarray], name: str, step: StepSpec) -> np.ndarray:
    if name not in out:
        raise SeparationError(
            f"ステップ {step.order}（{step.model_filename}）の出力に {name} がありません"
            f"（出力: {', '.join(sorted(out))}）。"
        )
    return out[name]


def run_plan(
    mix: np.ndarray,
    plan: PresetPlan,
    separator: Separator,
    *,
    workdir: Path,
    device: str = DEVICE_CUDA,
    mix_path: Path | None = None,
    progress: ProgressCallback | None = None,
    oom_policy: OomPolicy | None = None,
) -> SeparationOutput:
    """手順どおりに分離し、残差補正まで済んだ stem を返す（DB は触らない）。

    progress には 0〜1 と日本語の段階名を渡す。
    """
    validate_plan(plan)
    policy = oom_policy or OomPolicy()
    t0 = time.perf_counter()
    n = mix.shape[0]
    mix64 = np.asarray(mix, dtype=np.float64)
    workdir.mkdir(parents=True, exist_ok=True)
    if mix_path is None:
        mix_path = workdir / "mixture.wav"
        write_wav_float(mix_path, mix)

    ordered = [s for s in plan.steps if s.input == INPUT_MIXTURE] + [
        s for s in plan.steps if s.input == INPUT_VOCALS
    ]
    total = len(ordered)

    def report(i: int, stage: str) -> None:
        if progress is not None:
            progress(i / total, stage)

    multi: dict[str, _WeightedSum] = {}
    multi_order: list[str] = []
    vocals_sum = _WeightedSum()
    lead_sum = _WeightedSum()
    vocals_path: Path | None = None
    results: list[StepResult] = []

    for i, step in enumerate(ordered):
        if step.input == INPUT_VOCALS and vocals_path is None:
            vocals_path = workdir / "vocals.wav"
            write_wav_float(vocals_path, vocals_sum.mean())
        report(i, f"分離中（{i + 1}/{total}）: {step.model_name}")
        wav = mix_path if step.input == INPUT_MIXTURE else vocals_path
        assert wav is not None
        out, res = run_step(separator, step, wav, device, policy)
        results.append(res)
        log.info(
            "ステップ %d %s: %.1f 秒（%s, チャンク %g 倍）",
            step.order, step.model_filename, res.seconds, res.device, res.chunk_scale,
        )
        if step.role == ROLE_MULTISTEM:
            for name, arr in out.items():
                if name not in multi:
                    multi[name] = _WeightedSum()
                    multi_order.append(name)
                multi[name].add(_fit(arr, n), step.weight)
            vocals_sum.add(_fit(_require(out, VOCALS, step), n), step.weight)
        elif step.role == ROLE_VOCALS:
            vocals_sum.add(_fit(_require(out, VOCALS, step), n), step.weight)
        else:  # karaoke
            lead_sum.add(_fit(_require(out, LEAD, step), n), step.weight)
        del out

    if OTHER not in multi:
        raise SeparationError("multistem の出力に other がありません（残差を足す先が無い）。")
    report(total, "仕上げ中")
    top: dict[str, np.ndarray] = {name: multi[name].mean() for name in multi_order}
    top[VOCALS] = vocals_sum.mean()
    # 残差補正: 上位 stem の合計を元の曲に一致させる
    top[OTHER] = top[OTHER] + (mix64 - sum(top.values()))
    lead = lead_sum.mean()
    stems: dict[str, np.ndarray] = {name: arr.astype(np.float32) for name, arr in top.items()}
    stems[LEAD] = lead.astype(np.float32)
    stems[BACKING] = (top[VOCALS] - lead).astype(np.float32)
    return SeparationOutput(
        stems=stems,
        top_level=list(multi_order),
        steps=results,
        seconds=time.perf_counter() - t0,
    )


# --- DB 登録とファイル保存 ------------------------------------------------------------


@dataclass(frozen=True)
class StemInfo:
    code: str
    display_name: str
    parent_code: str | None
    is_residual: bool
    rms_db: float | None
    is_silent: bool
    file_path: Path
    clipped_samples: int = 0


@dataclass
class SeparateResult:
    track_id: int
    job_id: int
    skipped: bool  # 分割済みだったので何もしなかった
    stems: list[StemInfo]
    seconds: float
    steps: list[StepResult] = field(default_factory=list)


def data_relative(settings: Settings, path: Path) -> str:
    """DB に保存するパス（データフォルダからの相対、/ 区切り）。"""
    return path.resolve().relative_to(settings.data_dir.resolve()).as_posix()


def resolve_data_path(settings: Settings, stored: str) -> Path:
    """DB に保存したパスを実際のパスにする（相対ならデータフォルダ基準）。"""
    p = Path(stored)
    return p if p.is_absolute() else settings.data_dir / p


def _utcnow() -> datetime:
    return datetime.now(UTC)


def find_done_job(session: Session, track_id: int) -> SeparationJob | None:
    return session.scalars(
        select(SeparationJob)
        .where(
            SeparationJob.track_id == track_id,
            SeparationJob.job_kind == "full",
            SeparationJob.status == "done",
        )
        .order_by(SeparationJob.job_id.desc())
    ).first()


def stems_of_job(session: Session, settings: Settings, job_id: int) -> list[StemInfo]:
    """ジョブの stem を表示順で返す。"""
    rows = session.execute(
        select(Stem, StemType)
        .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
        .where(Stem.job_id == job_id)
        .order_by(StemType.display_order)
    ).all()
    by_id = {s.stem_id: t.code for s, t in rows}
    infos: list[StemInfo] = []
    for s, t in rows:
        rend = session.scalars(
            select(StemRendition).where(
                StemRendition.stem_id == s.stem_id, StemRendition.purpose == "master"
            )
        ).first()
        infos.append(
            StemInfo(
                code=t.code,
                display_name=t.display_name,
                parent_code=by_id.get(s.parent_stem_id) if s.parent_stem_id else None,
                is_residual=s.is_residual,
                rms_db=s.rms_db,
                is_silent=s.is_silent,
                file_path=resolve_data_path(settings, rend.file_path) if rend else Path(),
            )
        )
    return infos


def _save_stems(
    session: Session,
    settings: Settings,
    job: SeparationJob,
    output: SeparationOutput,
) -> list[StemInfo]:
    types = {t.code: t for t in session.scalars(select(StemType))}
    id_to_code = {t.stem_type_id: t.code for t in types.values()}
    missing = [c for c in output.stems if c not in types]
    if missing:
        raise SeparationError(f"STEM_TYPE に無い stem があります: {', '.join(missing)}")

    # 親を先に登録する（STEM_TYPE の親子関係に従う）
    def depth(code: str) -> int:
        d, t = 0, types[code]
        while t.parent_id is not None:
            d, t = d + 1, types[id_to_code[t.parent_id]]
        return d

    codes = sorted(output.stems, key=lambda c: (depth(c), types[c].display_order))
    out_dir = settings.stems_dir / str(job.job_id)
    stems: dict[str, Stem] = {}
    infos: list[StemInfo] = []
    for code in codes:
        arr = output.stems[code]
        t = types[code]
        parent_code = id_to_code.get(t.parent_id) if t.parent_id is not None else None
        if parent_code is not None and parent_code not in stems:
            raise SeparationError(f"{code} の親 stem（{parent_code}）がありません。")
        level = rms_db(arr)
        clipped = count_clipped(arr)
        if clipped:
            log.warning("stem %s: クリップ（|x|>1）が %d サンプルあります。", code, clipped)
        path = out_dir / f"{code}.flac"
        write_flac24(path, arr)
        stem = Stem(
            job_id=job.job_id,
            stem_type_id=t.stem_type_id,
            parent_stem_id=stems[parent_code].stem_id if parent_code else None,
            is_residual=code in RESIDUAL_STEMS,
            rms_db=level,
            is_silent=level < SILENT_THRESHOLD_DB,
        )
        session.add(stem)
        session.flush()
        session.add(
            StemRendition(
                stem_id=stem.stem_id,
                purpose="master",
                codec="flac",
                bitrate_kbps=None,
                file_path=data_relative(settings, path),
                bytes=path.stat().st_size,
            )
        )
        stems[code] = stem
        infos.append(
            StemInfo(
                code=code,
                display_name=t.display_name,
                parent_code=parent_code,
                is_residual=stem.is_residual,
                rms_db=level,
                is_silent=stem.is_silent,
                file_path=path,
                clipped_samples=clipped,
            )
        )
    session.flush()
    return infos


def separate_file(
    session: Session,
    settings: Settings,
    src: Path,
    separator: Separator,
    *,
    preset_code: str | None = None,
    force: bool = False,
    device: str = DEVICE_CUDA,
    title: str | None = None,
    progress: ProgressCallback | None = None,
    oom_policy: OomPolicy | None = None,
    ffmpeg_runner: FfmpegRunner | None = None,
) -> SeparateResult:
    """1曲を正規化・分割し、ファイルと DB に保存する。

    同じ audio_hash の TRACK に完了済みの full ジョブがあれば、force でない限り分割しない。
    失敗したら JOB を failed にして SeparationError を投げる。
    """
    t0 = time.perf_counter()
    plan = load_plan(session, preset_code)

    tmp_dir = settings.cache_dir / "tmp" / uuid.uuid4().hex
    try:
        norm = normalize_audio(src, tmp_dir / "normalized.wav", runner=ffmpeg_runner)
        track = session.scalars(select(Track).where(Track.audio_hash == norm.audio_hash)).first()
        if track is not None and not force:
            done = find_done_job(session, track.track_id)
            if done is not None:
                log.info("分割済みの曲です（track %d, job %d）。", track.track_id, done.job_id)
                return SeparateResult(
                    track_id=track.track_id,
                    job_id=done.job_id,
                    skipped=True,
                    stems=stems_of_job(session, settings, done.job_id),
                    seconds=time.perf_counter() - t0,
                )

        if track is None:
            track = Track(
                title=title or src.stem,
                audio_hash=norm.audio_hash,
                duration_sec=norm.duration_sec,
            )
            session.add(track)
            session.flush()
        track_dir = settings.tracks_dir / str(track.track_id)
        track_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = track_dir / "normalized.wav"
        shutil.move(str(norm.path), str(normalized_path))
        track.normalized_path = data_relative(settings, normalized_path)

        job = SeparationJob(
            track_id=track.track_id,
            job_kind="full",
            preset_id=plan.preset_id,
            status="running",
            run_on="cpu" if device == DEVICE_CPU else "gpu",
            progress=0.0,
            stage="準備中",
            started_at=_utcnow(),
        )
        session.add(job)
        session.commit()

        def set_progress(p: float, stage: str) -> None:
            job.progress = round(min(max(p, 0.0), 1.0), 4)
            job.stage = stage[:100]
            session.commit()
            if progress is not None:
                progress(job.progress, stage)

        try:
            set_progress(0.02, "分離の準備中")
            output = run_plan(
                norm.data,
                plan,
                separator,
                workdir=tmp_dir,
                device=device,
                mix_path=normalized_path,
                progress=lambda p, s: set_progress(0.02 + 0.88 * p, s),
                oom_policy=oom_policy,
            )
            set_progress(0.92, "stem を保存中")
            if any(r.device == DEVICE_CPU for r in output.steps) and device != DEVICE_CPU:
                job.run_on = "cpu"
            infos = _save_stems(session, settings, job, output)
            job.status = "done"
            job.progress = 1.0
            job.stage = "完了"
            job.finished_at = _utcnow()
            session.commit()
            if progress is not None:
                progress(1.0, "完了")
        except Exception as e:
            session.rollback()
            stage = job.stage or ""
            job.status = "failed"
            job.finished_at = _utcnow()
            job.error_message = f"分割に失敗しました（{stage}）: {type(e).__name__}: {e}"
            session.commit()
            shutil.rmtree(settings.stems_dir / str(job.job_id), ignore_errors=True)
            log.error("job %d 失敗: %s", job.job_id, job.error_message)
            raise SeparationError(job.error_message) from e

        return SeparateResult(
            track_id=track.track_id,
            job_id=job.job_id,
            skipped=False,
            stems=infos,
            seconds=time.perf_counter() - t0,
            steps=output.steps,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
