"""分離パイプライン（SPEC 5章）。

手順は DB の SEPARATION_PRESET / PRESET_STEP から組み立てる。流れ:

1. input=mixture のステップを step_order 順に実行する。
   - role=multistem の出力（vocals, drums, bass, guitar, piano, other）を ensemble_weight で平均。
   - vocals は、multistem の vocals と role=vocals の出力を ensemble_weight で重み付き平均。
   - role=karaoke で input=mixture なら、元の曲に karaoke をかけ、lead を重み付き平均で得る。
2. 残差 R = mixture − Σ上位 stem を、プリセットの選択肢 residual_to に従って足す。
   - other（既定）: other += R
   - vocals: vocals += R
   - split: R を「vocals の平均で生じた差（multistem の vocals − 平均の vocals）」と残りに分け、
     前者を vocals、後者を other に足す。
   どの方式でも上位 stem の合計は元の曲に一致する。
3. input=vocals のステップ（role=karaoke）を、残差を足した後の vocals に適用し、lead を得る。
4. backing = vocals − lead（残差）。lead + backing = vocals。

`run_plan` は DB を触らない計算部分（bench でも使う）。`separate_track` が取り込み済みの曲を
分割して DB 登録・ファイル保存まで行う。`separate_file` は取り込み（`stemapp.ingest`）→
`separate_track` を順に呼ぶ。
"""

from __future__ import annotations

import gc
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from stemapp.audio import (
    SILENCE_FLOOR_DB,
    FfmpegRunner,
    count_clipped,
    read_audio,
    rms_db,
    write_flac24,
    write_wav_float,
)
from stemapp.config import Settings
from stemapp.ingest.service import TagReader, import_file
from stemapp.library import data_relative, find_done_job, resolve_data_path
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

# プリセット全体の選択肢（SEPARATION_PRESET.options_json）
OPT_RESIDUAL_TO = "residual_to"
RESIDUAL_TO_OTHER = "other"
RESIDUAL_TO_VOCALS = "vocals"
RESIDUAL_TO_SPLIT = "split"
RESIDUAL_TO_CHOICES: tuple[str, ...] = (RESIDUAL_TO_OTHER, RESIDUAL_TO_VOCALS, RESIDUAL_TO_SPLIT)
PRESET_OPTION_KEYS: frozenset[str] = frozenset({OPT_RESIDUAL_TO})

ProgressCallback = Callable[[float, str], None]


class SeparationError(RuntimeError):
    """分割の失敗（メッセージは日本語）。"""


class JobAbandoned(SeparationError):
    """ジョブが running でなくなった・キャンセルされたので、結果を書き込まずにやめた。"""


def job_tmp_dir(settings: Settings, job_id: int) -> Path:
    """分割中の一時ファイルの置き場所（ジョブごと。中断後にワーカーが片付ける）。"""
    return settings.cache_dir / "tmp" / f"job-{job_id}"


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
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def residual_to(self) -> str:
        return str(self.options.get(OPT_RESIDUAL_TO, RESIDUAL_TO_OTHER))


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
    plan = PresetPlan(
        preset.code,
        preset.display_name,
        steps,
        preset.preset_id,
        options=dict(preset.options_json or {}),
    )
    validate_plan(plan)
    return plan


def validate_plan(plan: PresetPlan) -> None:
    """パイプラインが扱える手順・選択肢か確かめる。"""
    unknown = sorted(set(plan.options) - PRESET_OPTION_KEYS)
    if unknown:
        raise SeparationError(
            f"プリセット「{plan.code}」の選択肢 {', '.join(unknown)} には対応していません。"
        )
    if plan.residual_to not in RESIDUAL_TO_CHOICES:
        raise SeparationError(
            f"プリセット「{plan.code}」の residual_to「{plan.residual_to}」には対応していません"
            f"（{' / '.join(RESIDUAL_TO_CHOICES)}）。"
        )
    roles = [s.role for s in plan.steps]
    if ROLE_MULTISTEM not in roles:
        raise SeparationError(f"プリセット「{plan.code}」に multistem のステップがありません。")
    if ROLE_KARAOKE not in roles:
        raise SeparationError(f"プリセット「{plan.code}」に karaoke のステップがありません。")
    for s in plan.steps:
        if s.role not in (ROLE_MULTISTEM, ROLE_VOCALS, ROLE_KARAOKE):
            raise SeparationError(f"ステップ {s.order} の role「{s.role}」には対応していません。")
        allowed = (INPUT_MIXTURE, INPUT_VOCALS) if s.role == ROLE_KARAOKE else (INPUT_MIXTURE,)
        if s.input not in allowed:
            raise SeparationError(
                f"ステップ {s.order}（{s.role}）の input は {' / '.join(allowed)} のどれかに"
                f"してください（今は {s.input}）。"
            )
        if s.weight <= 0:
            raise SeparationError(f"ステップ {s.order} の ensemble_weight は正の数にしてください。")
    karaoke_inputs = {s.input for s in plan.steps if s.role == ROLE_KARAOKE}
    if len(karaoke_inputs) > 1:
        raise SeparationError(
            f"プリセット「{plan.code}」の karaoke の input は、全ステップで同じにしてください"
            "（mixture と vocals を混ぜられません）。"
        )


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


def free_gpu_memory() -> None:
    """使われなくなった GPU メモリを解放する（torch が無い環境では gc だけ）。"""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        oom = False
        try:
            out = separator.separate(wav_path, step.model_filename, opts, dev, step.role)
            break
        except Exception as e:
            if not is_oom_error(e) or dev != DEVICE_CUDA:
                raise
            oom = True
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
        if oom:
            # 例外（とその traceback が持つ GPU テンソル）を手放してから解放する
            free_gpu_memory()
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
    # 補正前の残差（mixture − 上位 stem の生出力の合計）と mixture の RMS（dBFS）
    residual_rms_db: float = SILENCE_FLOOR_DB
    mixture_rms_db: float = SILENCE_FLOOR_DB

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

    mixture_steps = [s for s in plan.steps if s.input == INPUT_MIXTURE]
    vocal_steps = [s for s in plan.steps if s.input == INPUT_VOCALS]
    total = len(mixture_steps) + len(vocal_steps)

    def report(i: int, stage: str) -> None:
        if progress is not None:
            progress(i / total, stage)

    multi: dict[str, _WeightedSum] = {}
    multi_order: list[str] = []
    vocals_sum = _WeightedSum()
    lead_sum = _WeightedSum()
    results: list[StepResult] = []

    def run_one(i: int, step: StepSpec, wav: Path) -> None:
        report(i, f"分離中（{i + 1}/{total}）: {step.model_name}")
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

    # 1. 元の曲にかけるステップ
    for i, step in enumerate(mixture_steps):
        run_one(i, step, mix_path)

    if OTHER not in multi:
        raise SeparationError("multistem の出力に other がありません（残差を足す先が無い）。")
    top: dict[str, np.ndarray] = {name: multi[name].mean() for name in multi_order}
    top[VOCALS] = vocals_sum.mean()
    # 2. 残差補正: 上位 stem の合計を元の曲に一致させる
    residual = mix64 - sum(top.values())
    residual_db = rms_db(residual)
    mixture_db = rms_db(mix64)
    route = plan.residual_to
    log.info(
        "補正前の残差: %.1f dBFS（mixture %.1f dBFS、差 %.1f dB）。行き先: %s",
        residual_db, mixture_db, mixture_db - residual_db, route,
    )
    if route == RESIDUAL_TO_VOCALS:
        top[VOCALS] = top[VOCALS] + residual
    elif route == RESIDUAL_TO_SPLIT:
        # vocals の平均で生じた差（multistem の vocals − 平均の vocals）は vocals へ、
        # 残り（multistem 自身の合計のずれ）は other へ
        vocal_part = multi[VOCALS].mean() - top[VOCALS]
        log.info(
            "残差の内訳: ボーカルの平均による差 %.1f dBFS、その他 %.1f dBFS",
            rms_db(vocal_part), rms_db(residual - vocal_part),
        )
        top[VOCALS] = top[VOCALS] + vocal_part
        top[OTHER] = top[OTHER] + (residual - vocal_part)
        del vocal_part
    else:
        top[OTHER] = top[OTHER] + residual
    del residual

    # 3. 残差を足した後の vocals にかけるステップ
    if vocal_steps:
        vocals_path = workdir / "vocals.wav"
        write_wav_float(vocals_path, top[VOCALS].astype(np.float32))
        for i, step in enumerate(vocal_steps, start=len(mixture_steps)):
            run_one(i, step, vocals_path)

    report(total, "仕上げ中")
    lead = lead_sum.mean()
    stems: dict[str, np.ndarray] = {name: arr.astype(np.float32) for name, arr in top.items()}
    stems[LEAD] = lead.astype(np.float32)
    stems[BACKING] = (top[VOCALS] - lead).astype(np.float32)
    return SeparationOutput(
        stems=stems,
        top_level=list(multi_order),
        steps=results,
        seconds=time.perf_counter() - t0,
        residual_rms_db=residual_db,
        mixture_rms_db=mixture_db,
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


def _utcnow() -> datetime:
    return datetime.now(UTC)


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
    # 表示順（親の直後に子）で返す
    return sorted(infos, key=lambda i: types[i.code].display_order)


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
    tag_reader: TagReader | None = None,
) -> SeparateResult:
    """1曲を取り込み（`import_file`）、分割してファイルと DB に保存する。

    同じ audio_hash の TRACK に同じプリセットの完了済み full ジョブがあれば、force でない限り
    分割しない。
    失敗したら JOB を failed にして SeparationError を投げる。
    """
    t0 = time.perf_counter()
    load_plan(session, preset_code)  # プリセットの誤りは取り込む前に知らせる
    imported = import_file(
        session,
        settings,
        src,
        title=title,
        ffmpeg_runner=ffmpeg_runner,
        tag_reader=tag_reader,
    )
    return separate_track(
        session,
        settings,
        imported.track_id,
        separator,
        preset_code=preset_code,
        force=force,
        device=device,
        progress=progress,
        oom_policy=oom_policy,
        started=t0,
    )


PostProcess = Callable[[Session, Settings, int, ProgressCallback], None]
"""分割結果の保存後に同じジョブの中で行う処理（配信用データの作成など）。

引数は (session, settings, job_id, progress)。progress には 0〜1 と段階名を渡す。
DB は flush まででよい（commit はジョブの完了時にまとめて行う）。
"""

LIMIT_PEAK = 0.999


def limit_output(output: SeparationOutput, mix: np.ndarray) -> float:
    """mixture と全 stem の最大絶対値が 1 を超えていたら、全 stem に同じ倍率をかける。

    倍率は 0.999 / 最大値。stem 同士のバランスと「合計＝mixture × 倍率」の関係が保たれる。
    かけた倍率（1.0 なら何もしていない）を返す。
    """
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    for arr in output.stems.values():
        if arr.size:
            peak = max(peak, float(np.max(np.abs(arr))))
    if peak <= 1.0:
        return 1.0
    gain = LIMIT_PEAK / peak
    for name, arr in output.stems.items():
        output.stems[name] = (arr.astype(np.float64) * gain).astype(np.float32)
    log.info("最大値 %.3f が 1 を超えるため、全 stem に %.4f 倍（%.2f dB）をかけます。",
             peak, gain, gain_to_db(gain))
    return gain


def gain_to_db(gain: float) -> float:
    return 20.0 * float(np.log10(gain)) if gain > 0 else SILENCE_FLOOR_DB


def delete_job_stems(session: Session, job_id: int) -> None:
    """ジョブの STEM 行を消す（STEM_RENDITION と WAVEFORM は外部キーの CASCADE で消える）。"""
    session.execute(
        update(SeparationJob)
        .where(SeparationJob.input_stem_id.in_(select(Stem.stem_id).where(Stem.job_id == job_id)))
        .values(input_stem_id=None)
    )
    # 子（parent_stem_id あり）から消す
    session.execute(delete(Stem).where(Stem.job_id == job_id, Stem.parent_stem_id.is_not(None)))
    session.execute(delete(Stem).where(Stem.job_id == job_id))


def _prepare_job(
    session: Session,
    track: Track,
    plan: PresetPlan,
    device: str,
    job_id: int | None,
) -> SeparationJob:
    """ジョブを running にする。job_id が無ければ新しく作る。"""
    run_on = "cpu" if device == DEVICE_CPU else "gpu"
    if job_id is None:
        job = SeparationJob(
            track_id=track.track_id,
            job_kind="full",
            preset_id=plan.preset_id,
            status="running",
            run_on=run_on,
        )
        session.add(job)
    else:
        found = session.get(SeparationJob, job_id)
        if found is None:
            raise SeparationError(f"ジョブが見つかりません（job {job_id}）。")
        if found.status not in ("queued", "running"):
            raise SeparationError(
                f"ジョブ {job_id} は実行できない状態です（{found.status}）。"
            )
        job = found
        job.status = "running"
        job.run_on = run_on
    job.progress = 0.0
    job.stage = "準備中"
    job.started_at = job.started_at or _utcnow()
    job.error_message = None
    job.output_gain_db = 0.0
    session.commit()
    return job


def separate_track(
    session: Session,
    settings: Settings,
    track_id: int,
    separator: Separator,
    *,
    preset_code: str | None = None,
    force: bool = False,
    device: str = DEVICE_CUDA,
    progress: ProgressCallback | None = None,
    oom_policy: OomPolicy | None = None,
    started: float | None = None,
    job_id: int | None = None,
    postprocess: PostProcess | None = None,
) -> SeparateResult:
    """取り込み済みの曲を分割する。同じプリセットの完了済み full ジョブがあれば、force でない限り
    分割しない（プリセットが違えば、同じ曲に複数の分け方を持てる）。preset_code が None なら
    どのプリセットの完了済みジョブでも分割しない。

    job_id を渡すと、登録済みのジョブ（queued / running）を実行する（ワーカー用）。
    このときプリセットはジョブのものを使い、分割済みかどうかは調べない。
    postprocess を渡すと、stem の保存後に「配信用データを作成中」の段階として実行する。
    """
    t0 = time.perf_counter() if started is None else started
    if job_id is not None:
        queued = session.get(SeparationJob, job_id)
        if queued is None:
            raise SeparationError(f"ジョブが見つかりません（job {job_id}）。")
        preset = session.get(SeparationPreset, queued.preset_id) if queued.preset_id else None
        preset_code = preset.code if preset is not None else None
        track_id = queued.track_id
        force = True
    plan = load_plan(session, preset_code)
    track = session.get(Track, track_id)
    if track is None:
        raise SeparationError(f"曲が見つかりません（track {track_id}）。")
    if not force:
        same_preset = plan.preset_id if preset_code is not None else None
        done = find_done_job(session, track.track_id, same_preset)
        if done is not None:
            log.info("分割済みの曲です（track %d, job %d）。", track.track_id, done.job_id)
            return SeparateResult(
                track_id=track.track_id,
                job_id=done.job_id,
                skipped=True,
                stems=stems_of_job(session, settings, done.job_id),
                seconds=time.perf_counter() - t0,
            )
    if not track.normalized_path:
        raise SeparationError(f"正規化した音声がありません（track {track_id}）。")
    normalized_path = resolve_data_path(settings, track.normalized_path)
    try:
        mix = read_audio(normalized_path)
    except Exception as e:
        raise SeparationError(f"正規化した音声を読めません: {normalized_path}: {e}") from e

    tmp_dir: Path | None = None
    try:
        job = _prepare_job(session, track, plan, device, job_id)
        tmp_dir = job_tmp_dir(settings, job.job_id)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        def ensure_still_ours() -> None:
            """ジョブが running のままで、キャンセルされていないか確かめる。

            ワーカーが先に終わった後などに残った処理が、結果を書き込まないようにする。
            """
            row = session.execute(
                select(SeparationJob.status, SeparationJob.cancel_requested).where(
                    SeparationJob.job_id == job.job_id
                )
            ).one_or_none()
            if row is None or row.status != "running" or row.cancel_requested:
                state = "削除済み" if row is None else row.status
                if row is not None and row.cancel_requested:
                    state += "・キャンセル依頼あり"
                raise JobAbandoned(f"ジョブ {job.job_id} は続けられない状態です（{state}）。")

        def set_progress(p: float, stage: str) -> None:
            ensure_still_ours()
            job.progress = round(min(max(p, 0.0), 1.0), 4)
            job.stage = stage[:100]
            session.commit()
            if progress is not None:
                progress(job.progress, stage)

        try:
            set_progress(0.02, "分離の準備中")
            output = run_plan(
                mix,
                plan,
                separator,
                workdir=tmp_dir,
                device=device,
                mix_path=normalized_path,
                progress=lambda p, s: set_progress(0.02 + 0.86 * p, s),
                oom_policy=oom_policy,
            )
            set_progress(0.9, "stem を保存中")
            if any(r.device == DEVICE_CPU for r in output.steps) and device != DEVICE_CPU:
                job.run_on = "cpu"
            gain = limit_output(output, mix)
            job.output_gain_db = round(gain_to_db(gain), 4) if gain != 1.0 else 0.0
            job.residual_rms_db = round(output.residual_rms_db, 2)
            job.mixture_rms_db = round(output.mixture_rms_db, 2)
            infos = _save_stems(session, settings, job, output)
            if postprocess is not None:
                set_progress(0.92, "配信用データを作成中")
                postprocess(
                    session,
                    settings,
                    job.job_id,
                    lambda p, s: set_progress(0.92 + 0.07 * p, s),
                )
            # running のままでキャンセルされていないときだけ done にする（同じトランザクション）
            res = session.execute(
                update(SeparationJob)
                .where(
                    SeparationJob.job_id == job.job_id,
                    SeparationJob.status == "running",
                    SeparationJob.cancel_requested.is_(False),
                )
                .values(status="done", progress=1.0, stage="完了", finished_at=_utcnow())
            )
            if res.rowcount != 1:  # type: ignore[attr-defined]
                raise JobAbandoned(f"ジョブ {job.job_id} は完了を書き込める状態ではありません。")
            session.commit()
            session.refresh(job)
            if progress is not None:
                progress(1.0, "完了")
        except JobAbandoned:
            # 状態は他（ワーカー・キャンセル）が決める。自分が作った成果物だけ消す
            session.rollback()
            delete_job_stems(session, job.job_id)
            session.execute(
                update(SeparationJob)
                .where(
                    SeparationJob.job_id == job.job_id,
                    SeparationJob.status == "running",
                    SeparationJob.cancel_requested.is_(True),
                )
                .values(status="canceled", stage="キャンセルしました", finished_at=_utcnow())
            )
            session.commit()
            shutil.rmtree(settings.stems_dir / str(job.job_id), ignore_errors=True)
            log.warning("job %d の処理をやめました（結果は書き込みません）。", job.job_id)
            raise
        except Exception as e:
            session.rollback()
            stage = job.stage or ""
            delete_job_stems(session, job.job_id)
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
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
