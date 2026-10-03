"""詳細分割（「もっと分ける」、SEPARATION_JOB.job_kind=refine）。

分割済みの stem（親）を、1つの「方法」でさらに分ける。方法は MODEL の行で表す:
- 子の STEM_TYPE（refine_model_id がその MODEL の行）が、その方法で作る子。
  例: DrumSep → kick, snare, toms, hihat, ride, crash。男女 → male, female。息 → breath。
  HPSS（architecture="hpss"、信号処理）→ sustained, transient。
  Mega 53（architecture="msst_bs_roformer"）→ strings, brass, woodwind, synth, percussion。
  Mega 53 は曲によって多くの出力がほぼ無音になるので、実質的に無音の子（RMS が
  SILENT_THRESHOLD_DB 未満）は作らず、その分は残りに入れる（T07b）。
- 子の STEM_TYPE の親（例 drums、vocals、other）か、その子孫の型の stem に使える
  （vocals の方法は lead_vocal / backing_vocal にも使える）。
- 「残り」の STEM_TYPE（`<親の code>_rest`）がある型の stem だけ分けられる。
  残り = 親 − 名前の付いた子の合計。子の合計は親に一致する（SPEC 4章）。

分けられるのは、画面の木（`stemapp.stem_view`）で子を持たない stem だけ（vocals は分割時に
lead_vocal / backing_vocal に分かれているので、その2つを分ける）。1つの stem を分けた結果は
1組だけ。同じ方法の結果があれば再実行しない（force で作り直す）。別の方法の結果があるときは
force で置き換える。木の中で STEM_TYPE の code が重なる分け方（例: lead と backing の両方を
男女で分ける）はできない（画面・組み合わせが code で stem を指すため）。

子の保存先は、分けた stem のジョブの保存フォルダの下の `<親の code>/`（例
`stems/曲/standard/drums/kick.flac`）。配信用データも同じフォルダの `stream/`・`peaks/`。
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from stemapp.audio import read_audio, rms_db, write_flac24, write_wav_float
from stemapp.config import Settings
from stemapp.exports.service import remove_export_dirs
from stemapp.library import data_relative, resolve_data_path
from stemapp.models import Model, SeparationJob, Stem, StemRendition, StemType
from stemapp.separation.base import DEVICE_CPU, DEVICE_CUDA, Separator
from stemapp.separation.hpss import split_sustained_transient
from stemapp.separation.pipeline import (
    SILENT_THRESHOLD_DB,
    JobAbandoned,
    OomPolicy,
    PostProcess,
    ProgressCallback,
    SeparationError,
    StepResult,
    StepSpec,
    delete_job_stems,
    job_tmp_dir,
    run_step,
)
from stemapp.stem_folders import job_dir, pick_free_name, remove_job_dir
from stemapp.stem_view import (
    JOB_KIND_FULL,
    JOB_KIND_REFINE,
    StemView,
    build_view,
    root_job_id,
)

log = logging.getLogger(__name__)

REST_SUFFIX = "_rest"
ARCH_HPSS = "hpss"
ARCH_MSST = "msst_bs_roformer"
ROLE_REFINE = "refine"
INPUT_STEM = "stem"

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELED = "canceled"
ACTIVE = (QUEUED, RUNNING)

HpssFunc = Callable[..., dict[str, np.ndarray]]


class RefineInvalid(SeparationError):
    """その stem・方法では分けられない（要求の誤り）。"""


class RefineConflict(SeparationError):
    """今の状態では分けられない（別の分け方がある、分割中など）。"""


class RefineNotFound(SeparationError):
    pass


def rest_code(code: str) -> str:
    """親の code から「残り」の STEM_TYPE の code を作る。"""
    return code + REST_SUFFIX


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --- 方法 ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefineMethod:
    model_id: int
    filename: str  # MODEL.filename（API ではこれで方法を指す）
    display_name: str
    architecture: str | None
    parent_code: str  # 子の STEM_TYPE の親
    child_codes: tuple[str, ...]  # 名前の付いた子（表示順）
    experimental: bool = False  # 「実験」扱い（MODEL.is_experimental）

    @property
    def is_hpss(self) -> bool:
        return self.architecture == ARCH_HPSS

    @property
    def uses_gpu(self) -> bool:
        return not self.is_hpss

    @property
    def drops_silent(self) -> bool:
        """実質的に無音の子を作らず残りに入れるか（Mega 53）。"""
        return self.architecture == ARCH_MSST


@dataclass
class TypeIndex:
    by_code: dict[str, StemType]
    by_id: dict[int, StemType]

    @classmethod
    def load(cls, session: Session) -> TypeIndex:
        types = session.scalars(select(StemType).order_by(StemType.display_order)).all()
        return cls({t.code: t for t in types}, {t.stem_type_id: t for t in types})

    def ancestors(self, code: str) -> list[str]:
        """自分と祖先の code（近い順）。"""
        out: list[str] = []
        t = self.by_code.get(code)
        while t is not None and t.code not in out:
            out.append(t.code)
            t = self.by_id.get(t.parent_id) if t.parent_id is not None else None
        return out


def load_methods(session: Session, types: TypeIndex | None = None) -> dict[str, RefineMethod]:
    """詳細分割の方法（MODEL.filename → 方法）。子の STEM_TYPE から組み立てる。

    並び（dict の順）は画面の一覧の順で、同じ stem に使える方法のうち先頭が既定。
    MODEL.refine_order の小さい順（NULL は後ろ）、同じなら子の STEM_TYPE の表示順。
    """
    types = types or TypeIndex.load(session)
    groups: dict[int, list[StemType]] = {}
    for t in types.by_code.values():
        if t.refine_model_id is not None:
            groups.setdefault(t.refine_model_id, []).append(t)
    found: list[tuple[tuple[int, int, int], RefineMethod]] = []
    for model_id, kids in groups.items():
        model = session.get(Model, model_id)
        parents = {k.parent_id for k in kids}
        if model is None or len(parents) != 1 or None in parents:
            log.warning("詳細分割の方法を組み立てられません（model %s）。", model_id)
            continue
        parent = types.by_id[next(iter(parents))]  # type: ignore[index]
        kids.sort(key=lambda k: k.display_order)
        order = model.refine_order
        key = (0 if order is not None else 1, order or 0, kids[0].display_order)
        found.append((key, RefineMethod(
            model_id=model.model_id,
            filename=model.filename,
            display_name=model.display_name,
            architecture=model.architecture,
            parent_code=parent.code,
            child_codes=tuple(k.code for k in kids),
            experimental=bool(model.is_experimental),
        )))
    found.sort(key=lambda kv: kv[0])
    return {m.filename: m for _, m in found}


def method_applies(method: RefineMethod, stem_code: str, types: TypeIndex) -> bool:
    """その型の stem にこの方法が使えるか（型の親子と「残り」の型の有無だけを見る）。"""
    if stem_code in method.child_codes:
        return False
    return method.parent_code in types.ancestors(stem_code) and (
        rest_code(stem_code) in types.by_code
    )


def output_codes(method: RefineMethod, stem_code: str) -> tuple[str, ...]:
    """その stem をこの方法で分けたときにできる子の code（残りを含む）。"""
    return (*method.child_codes, rest_code(stem_code))


# --- 登録 ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefineCheck:
    """ある stem をある方法で分けられるか。"""

    ok: bool
    reason: str | None = None  # 分けられないときの理由（日本語）


def refine_jobs_of_stem(session: Session, stem_id: int) -> list[SeparationJob]:
    return list(
        session.scalars(
            select(SeparationJob)
            .where(
                SeparationJob.job_kind == JOB_KIND_REFINE,
                SeparationJob.input_stem_id == stem_id,
            )
            .order_by(SeparationJob.job_id)
        )
    )


def check_refine(
    session: Session,
    view: StemView,
    stem_id: int,
    method: RefineMethod,
    types: TypeIndex,
    *,
    replacing: bool = False,
    active_jobs: list[SeparationJob] | None = None,
    methods_by_id: dict[int, RefineMethod] | None = None,
) -> RefineCheck:
    """木（view）の中の stem を method で分けられるか。replacing なら今の子を置き換える前提。"""
    found = view.find(stem_id)
    if found is None:
        return RefineCheck(False, "この分け方の stem ではありません。")
    stem, stype = found
    if not method_applies(method, stype.code, types):
        return RefineCheck(False, f"「{stype.display_name}」はこの方法では分けられません。")
    own = view.refined_by.get(stem_id)
    children = view.children_of(stem_id)
    if children and own is None:
        return RefineCheck(False, f"「{stype.display_name}」は既に子に分かれています。")
    if own is not None and not replacing:
        return RefineCheck(False, f"「{stype.display_name}」は既に分けてあります。")
    mine = {s.stem_id for s, _ in view.descendants_of(stem_id)}
    new_codes = output_codes(method, stype.code)
    others = {t.code: t for s, t in view.rows if s.stem_id not in mine}
    clash = [others[c].display_name for c in new_codes if c in others]
    if clash:
        return RefineCheck(
            False,
            f"この分け方には既に「{'・'.join(clash)}」があります（先にそれを戻してください）。",
        )
    # 分割待ち・分割中のほかの stem の結果と重ならないか
    by_id = methods_by_id or {m.model_id: m for m in load_methods(session, types).values()}
    for job in active_jobs if active_jobs is not None else _active_refines(session, view):
        if job.input_stem_id == stem_id or job.refine_model_id is None:
            continue
        other = view.find(job.input_stem_id or -1)
        other_method = by_id.get(job.refine_model_id)
        if other is None or other_method is None:
            continue
        codes = set(output_codes(other_method, other[1].code))
        dup = [c for c in new_codes if c in codes]
        if dup:
            names = "・".join(types.by_code[c].display_name for c in dup if c in types.by_code)
            return RefineCheck(
                False,
                f"「{other[1].display_name}」を分けている途中です（{names}が重なります）。",
            )
    return RefineCheck(True)


def _active_refines(session: Session, view: StemView) -> list[SeparationJob]:
    ids = view.stem_ids
    if not ids:
        return []
    return list(
        session.scalars(
            select(SeparationJob).where(
                SeparationJob.job_kind == JOB_KIND_REFINE,
                SeparationJob.status.in_(ACTIVE),
                SeparationJob.input_stem_id.in_(ids),
            )
        )
    )


@dataclass(frozen=True)
class RefineEnqueueResult:
    job: SeparationJob
    created: bool
    # created=False の理由: "active"（分割待ち・分割中） / "done"（同じ方法で分けてある）
    reason: str | None = None


def enqueue_refine_job(
    session: Session,
    stem_id: int,
    model_filename: str,
    *,
    force: bool = False,
) -> RefineEnqueueResult:
    """stem の詳細分割ジョブを queued で登録する（commit まで）。

    - その stem の分割待ち・分割中の refine があれば新しく作らない（同じ方法ならそれを返す、
      違う方法なら RefineConflict）。
    - 同じ方法の結果があれば force でない限り作らない（"done"）。違う方法の結果があれば
      force でない限り RefineConflict。force のときは、終わったら今の子を置き換える。
    """
    stem = session.get(Stem, stem_id)
    if stem is None:
        raise RefineNotFound("stem が見つかりません。")
    types = TypeIndex.load(session)
    methods = load_methods(session, types)
    method = methods.get(model_filename)
    if method is None:
        raise RefineInvalid(f"詳細分割の方法「{model_filename}」はありません。")
    stype = types.by_id.get(stem.stem_type_id)
    if stype is None or not method_applies(method, stype.code, types):
        name = stype.display_name if stype is not None else "この stem"
        raise RefineInvalid(f"「{name}」は「{method.display_name}」では分けられません。")
    root_id = root_job_id(session, stem.job_id)
    root = session.get(SeparationJob, root_id) if root_id is not None else None
    if root is None or root.job_kind != JOB_KIND_FULL or root.status != DONE:
        raise RefineConflict("分割が終わっていない stem は分けられません。")
    owner = session.get(SeparationJob, stem.job_id)
    if owner is None or owner.status != DONE:
        raise RefineConflict("分割が終わっていない stem は分けられません。")
    if owner.output_dir is None:
        raise RefineConflict(
            "古い保存フォルダの曲です。先に `stemapp migrate-folders` で"
            "保存フォルダを移してください。"
        )
    jobs = refine_jobs_of_stem(session, stem_id)
    active = [j for j in jobs if j.status in ACTIVE]
    if active:
        if active[0].refine_model_id == method.model_id:
            return RefineEnqueueResult(active[0], False, "active")
        raise RefineConflict("この stem は別の方法で分割待ち・分割中です。")
    view = build_view(session, root.job_id)
    current = view.refined_by.get(stem_id)
    if current is not None and not force:
        if current.refine_model_id == method.model_id:
            return RefineEnqueueResult(current, False, "done")
        raise RefineConflict(
            "この stem は既に別の方法で分けてあります（置き換えるには force を指定してください）。"
        )
    if current is not None and active_exports_of_jobs(session, [current.job_id]):
        raise ExportsBusy(
            "今の子を使った書き出しが作成待ち・作成中です。書き出しが終わってから分け直してください。"
        )
    check = check_refine(
        session, view, stem_id, method, types,
        replacing=True,
        methods_by_id={m.model_id: m for m in methods.values()},
    )
    if not check.ok:
        raise RefineConflict(check.reason or "この stem は分けられません。")
    # 終わった（失敗・キャンセル）ジョブの行は片付ける（成果物は残っていない）
    for j in jobs:
        if j.status in (FAILED, CANCELED):
            session.delete(j)
    job = SeparationJob(
        track_id=root.track_id,
        job_kind=JOB_KIND_REFINE,
        input_stem_id=stem_id,
        refine_model_id=method.model_id,
        status=QUEUED,
        run_on="cpu" if method.is_hpss else "gpu",
        progress=0.0,
        stage="分割待ち",
    )
    session.add(job)
    session.commit()
    log.info(
        "詳細分割を登録しました（job %d, stem %d, %s）。", job.job_id, stem_id, method.filename
    )
    return RefineEnqueueResult(job, True)


# --- 計算 ---------------------------------------------------------------------------


@dataclass
class RefineOutput:
    stems: dict[str, np.ndarray]  # 子の code → (samples, 2) float32（残りを含む）
    rest: str  # 残りの code
    steps: list[StepResult] = field(default_factory=list)
    seconds: float = 0.0
    clipped: dict[str, int] = field(default_factory=dict)  # |x|>1 を丸めたサンプル数
    dropped: list[str] = field(default_factory=list)  # 無音なので作らなかった子（残りに入れた）

    @property
    def peak_memory_mb(self) -> float | None:
        peaks = [s.peak_memory_mb for s in self.steps if s.peak_memory_mb is not None]
        return max(peaks) if peaks else None


FULL_SCALE_24 = float(2**23)
MAX_24 = (2**23 - 1) / FULL_SCALE_24  # 24bit で表せるいちばん大きい値


def quantize24(x: np.ndarray) -> np.ndarray:
    """FLAC 24bit に保存して読み直したときと同じ値（float64）にする。

    libsndfile は x × 2^23 を四捨五入し、-2^23〜2^23-1 に収める（1.0 は 2^23-1 になる）。
    """
    k = np.clip(np.round(np.asarray(x, dtype=np.float64) * FULL_SCALE_24), -FULL_SCALE_24,
                FULL_SCALE_24 - 1)
    return k / FULL_SCALE_24


def count_outside24(x: np.ndarray) -> int:
    """24bit の範囲（-1〜MAX_24）に収まらないサンプルの数（四捨五入で収まるものは数えない）。"""
    a = np.asarray(x, dtype=np.float64)
    hi = MAX_24 + 0.5 / FULL_SCALE_24
    lo = -1.0 - 0.5 / FULL_SCALE_24
    return int(np.count_nonzero((a >= hi) | (a < lo)))


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - x.shape[0], x.shape[1]))], axis=0)


def run_refine(
    parent: np.ndarray,
    method: RefineMethod,
    parent_code: str,
    separator: Separator | None,
    *,
    workdir: Path,
    device: str = DEVICE_CUDA,
    hpss: HpssFunc = split_sustained_transient,
    progress: ProgressCallback | None = None,
    oom_policy: OomPolicy | None = None,
) -> RefineOutput:
    """親の音を method で分け、残り（親 − 子の合計）まで作る（DB は触らない）。

    親と名前の付いた子を、FLAC 24bit に保存したときと同じ値（±1 に丸め、2^-23 の刻み）に
    してから残りを計算する。残りも同じ刻みに乗るので、保存した後も「子の合計＝親」がぴったり
    成り立つ。残りが 24bit の範囲（±1）を超えたときだけ一致しなくなる（clipped に残す。
    ジョブでは JOB.warning にも書く）。
    """
    t0 = time.perf_counter()
    n = parent.shape[0]
    parent64 = quantize24(parent)  # 保存済みの親（master）なら何も変わらない
    steps: list[StepResult] = []

    def report(p: float, stage: str) -> None:
        if progress is not None:
            progress(min(max(p, 0.0), 1.0), stage)

    if method.is_hpss:
        report(0.0, "持続音と短い音に分けています")
        t_hpss = time.perf_counter()
        out = hpss(parent, progress=lambda p, s: report(0.95 * p, s))
        log.info("HPSS: %.1f 秒（CPU）", time.perf_counter() - t_hpss)
    else:
        if separator is None:
            raise SeparationError("分離器がありません。")
        workdir.mkdir(parents=True, exist_ok=True)
        wav = workdir / "parent.wav"
        write_wav_float(wav, parent)
        step = StepSpec(
            order=1,
            model_filename=method.filename,
            model_name=method.display_name,
            input=INPUT_STEM,
            role=ROLE_REFINE,
        )
        # 分離器が要るファイル（初回の重みのダウンロード等）を先に用意する。
        # ダウンロードがあれば、この段階の進み具合の前半（0〜0.5）をそれに使う
        start = 0.0
        prepare = getattr(separator, "prepare_model", None)
        if callable(prepare):

            def on_prepare(p: float, s: str) -> None:
                nonlocal start
                start = 0.5 * min(max(p, 0.0), 1.0)
                report(start, s)

            prepare(method.filename, on_prepare)
        report(start, f"分離中: {method.display_name}")
        out, res = run_step(separator, step, wav, device, oom_policy or OomPolicy())
        steps.append(res)
        log.info(
            "%s: %.1f 秒（%s, チャンク %g 倍, GPU 最大 %s MB）",
            method.filename, res.seconds, res.device, res.chunk_scale,
            f"{res.peak_memory_mb:.0f}" if res.peak_memory_mb is not None else "-",
        )
    report(0.96, "残りを計算中")
    missing = [c for c in method.child_codes if c not in out]
    if missing:
        raise SeparationError(
            f"{method.filename} の出力に {', '.join(missing)} がありません"
            f"（出力: {', '.join(sorted(out))}）。"
        )
    stems: dict[str, np.ndarray] = {}
    clipped: dict[str, int] = {}
    dropped: list[str] = []
    total = np.zeros_like(parent64)
    for code in method.child_codes:
        arr = _fit(out[code], n)
        if method.drops_silent and rms_db(arr) < SILENT_THRESHOLD_DB:
            dropped.append(code)  # 子にしない（この分は残りに入る）
            continue
        c = count_outside24(arr)
        if c:
            clipped[code] = c
        # 保存する値（24bit）にそろえてから足す。はみ出た分は残りに入る
        q = quantize24(arr)
        stems[code] = q.astype(np.float32)  # 24bit の刻みは float32 で正確に表せる
        total += q
    del out
    rest = rest_code(parent_code)
    rest_arr = parent64 - total  # 刻みの上の値どうしの差なので、これも刻みに乗る（丸め誤差なし）
    c = count_outside24(rest_arr)
    if c:
        clipped[rest] = c
        log.warning(
            "残り（%s）が ±1 を超えるサンプルが %d あります（合計は一致しません）。", rest, c
        )
    stems[rest] = rest_arr.astype(np.float32)
    if clipped:
        log.warning("詳細分割で ±1 を超えた stem: %s", clipped)
    if dropped:
        log.info("無音なので作らなかった子（残りに入れた）: %s", ", ".join(dropped))
    report(1.0, "分割しました")
    return RefineOutput(
        stems=stems,
        rest=rest,
        steps=steps,
        seconds=time.perf_counter() - t0,
        clipped=clipped,
        dropped=dropped,
    )


# --- 実行（ワーカーの子プロセス） ---------------------------------------------------------


@dataclass
class RefineResult:
    job_id: int
    stem_ids: dict[str, int]
    seconds: float
    steps: list[StepResult] = field(default_factory=list)
    replaced_job_ids: list[int] = field(default_factory=list)


def refine_dir_name(session: Session, settings: Settings, owner: SeparationJob, code: str) -> str:
    """子の保存先（データフォルダからの相対パス）。`<親のジョブのフォルダ>/<親の code>`。

    ほかのジョブが使っている・既にあるときは ` (2)` … を付ける。
    """
    assert owner.output_dir is not None
    base_rel = PurePosixPath(owner.output_dir)
    taken = {
        PurePosixPath(od).name
        for od in session.scalars(
            select(SeparationJob.output_dir).where(
                SeparationJob.output_dir.like(owner.output_dir + "/%")
            )
        )
        if od and PurePosixPath(od).parent == base_rel
    }
    # 親のジョブのフォルダにある stream / peaks とは重ならないように
    taken |= {"stream", "peaks"}
    base_dir = job_dir(settings, owner.job_id, owner.output_dir)
    name = pick_free_name(code, taken, lambda n: (base_dir / n).exists())
    return f"{owner.output_dir}/{name}"


def run_refine_job(
    session: Session,
    settings: Settings,
    job_id: int,
    separator: Separator | None,
    *,
    device: str = DEVICE_CUDA,
    hpss: HpssFunc = split_sustained_transient,
    postprocess: PostProcess | None = None,
    progress: ProgressCallback | None = None,
    oom_policy: OomPolicy | None = None,
) -> RefineResult:
    """登録済みの refine ジョブ（queued / running）を実行する。

    失敗したら JOB を failed にし、作りかけ（STEM 行・フォルダ）を消して SeparationError。
    キャンセル・ワーカーの中断に気づいたら JobAbandoned（状態はワーカーが決める）。
    終わったら、同じ stem を前に分けた結果（force で置き換えるもの）を消す。
    """
    t0 = time.perf_counter()
    job = session.get(SeparationJob, job_id)
    if job is None:
        raise SeparationError(f"ジョブが見つかりません（job {job_id}）。")
    if job.job_kind != JOB_KIND_REFINE:
        raise SeparationError(f"ジョブ {job_id} は詳細分割ではありません。")
    if job.status not in ACTIVE:
        raise SeparationError(f"ジョブ {job_id} は実行できない状態です（{job.status}）。")
    stem = session.get(Stem, job.input_stem_id) if job.input_stem_id is not None else None
    if stem is None:
        raise SeparationError("分ける stem が見つかりません（削除された可能性があります）。")
    stype = session.get(StemType, stem.stem_type_id)
    owner = session.get(SeparationJob, stem.job_id)
    model = session.get(Model, job.refine_model_id) if job.refine_model_id else None
    if stype is None or owner is None or model is None:
        raise SeparationError("分ける stem・方法の情報がそろっていません。")
    if owner.output_dir is None:
        raise SeparationError("分ける stem の保存フォルダが決まっていません（古い形式）。")
    method = load_methods(session).get(model.filename)
    if method is None:
        raise SeparationError(f"詳細分割の方法「{model.filename}」がありません。")
    master = session.scalars(
        select(StemRendition).where(
            StemRendition.stem_id == stem.stem_id, StemRendition.purpose == "master"
        )
    ).first()
    if master is None:
        raise SeparationError("分ける stem の音声（master）がありません。")
    parent_path = resolve_data_path(settings, master.file_path)
    try:
        parent = read_audio(parent_path)
    except Exception as e:
        raise SeparationError(f"分ける stem の音声を読めません: {parent_path}: {e}") from e

    job.status = RUNNING
    job.run_on = "cpu" if (method.is_hpss or device == DEVICE_CPU) else "gpu"
    job.progress = 0.0
    job.stage = "準備中"
    job.started_at = job.started_at or _utcnow()
    job.error_message = None
    session.commit()
    stem_id = stem.stem_id
    tmp_dir = job_tmp_dir(settings, job_id)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    out_rel: str | None = None

    def ensure_still_ours() -> None:
        row = session.execute(
            select(SeparationJob.status, SeparationJob.cancel_requested).where(
                SeparationJob.job_id == job_id
            )
        ).one_or_none()
        if row is None or row.status != RUNNING or row.cancel_requested:
            state = "削除済み" if row is None else row.status
            if row is not None and row.cancel_requested:
                state += "・キャンセル依頼あり"
            raise JobAbandoned(f"ジョブ {job_id} は続けられない状態です（{state}）。")

    def set_progress(p: float, stage: str) -> None:
        ensure_still_ours()
        job.progress = round(min(max(p, 0.0), 1.0), 4)
        job.stage = stage[:100]
        session.commit()
        if progress is not None:
            progress(job.progress, stage)

    try:
        try:
            set_progress(0.02, "分離の準備中")
            output = run_refine(
                parent,
                method,
                stype.code,
                separator,
                workdir=tmp_dir,
                device=device,
                hpss=hpss,
                progress=lambda p, s: set_progress(0.02 + 0.83 * p, s),
                oom_policy=oom_policy,
            )
            if any(r.device == DEVICE_CPU for r in output.steps):
                job.run_on = "cpu"
            job.warning = rest_warning(output, types_display(session))
            set_progress(0.88, "stem を保存中")
            job.output_dir = refine_dir_name(session, settings, owner, stype.code)
            out_rel = job.output_dir
            session.commit()
            ids = _save_children(session, settings, job, stem, output)
            if postprocess is not None:
                set_progress(0.9, "配信用データを作成中")
                postprocess(session, settings, job_id, lambda p, s: set_progress(0.9 + 0.09 * p, s))
            ensure_still_ours()
            # 前に同じ stem を分けた結果（置き換えるもの）を、完了と同じトランザクションで消す
            old_jobs = [
                j for j in refine_jobs_of_stem(session, stem_id)
                if j.job_id != job_id and j.status not in ACTIVE
            ]
            old = [(j.job_id, j.output_dir) for j in old_jobs]
            # 古い子を使った書き出し: 作成待ち・作成中なら置き換えない（失敗にする）、
            # 作成済みなら一緒に消す（EXPORT_ITEM だけが消えて中身の欠けた書き出しが残らないように）
            old_exports = take_exports_of_jobs(session, [j for j, _ in old])
            for old_id, _ in old:
                delete_job_stems(session, old_id)
                session.execute(delete(SeparationJob).where(SeparationJob.job_id == old_id))
            res = session.execute(
                update(SeparationJob)
                .where(
                    SeparationJob.job_id == job_id,
                    SeparationJob.status == RUNNING,
                    SeparationJob.cancel_requested.is_(False),
                )
                .values(status=DONE, progress=1.0, stage="完了", finished_at=_utcnow())
            )
            if res.rowcount != 1:  # type: ignore[attr-defined]
                raise JobAbandoned(f"ジョブ {job_id} は完了を書き込める状態ではありません。")
            session.commit()
            session.refresh(job)
            for old_id, old_dir in old:
                remove_job_dir(session, settings, old_id, old_dir)
                shutil.rmtree(job_tmp_dir(settings, old_id), ignore_errors=True)
            remove_export_dirs(settings, old_exports)
            # stem の構成（画面で鳴らす葉）が変わったので、分け方の速度変更のキャッシュは使わない
            invalidate_root_tempo(session, settings, stem.job_id)
            if progress is not None:
                progress(1.0, "完了")
        except JobAbandoned:
            session.rollback()
            delete_job_stems(session, job_id)
            session.execute(
                update(SeparationJob)
                .where(
                    SeparationJob.job_id == job_id,
                    SeparationJob.status == RUNNING,
                    SeparationJob.cancel_requested.is_(True),
                )
                .values(status=CANCELED, stage="キャンセルしました", finished_at=_utcnow())
            )
            session.execute(
                update(SeparationJob).where(SeparationJob.job_id == job_id).values(output_dir=None)
            )
            session.commit()
            remove_job_dir(session, settings, job_id, out_rel)
            log.warning("job %d の処理をやめました（結果は書き込みません）。", job_id)
            raise
        except Exception as e:
            session.rollback()
            stage = job.stage or ""
            delete_job_stems(session, job_id)
            job.status = FAILED
            job.finished_at = _utcnow()
            job.error_message = f"詳細分割に失敗しました（{stage}）: {type(e).__name__}: {e}"
            job.output_dir = None
            session.commit()
            remove_job_dir(session, settings, job_id, out_rel)
            log.error("job %d 失敗: %s", job_id, job.error_message)
            raise SeparationError(job.error_message) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return RefineResult(
        job_id=job_id,
        stem_ids=ids,
        seconds=time.perf_counter() - t0,
        steps=output.steps,
        replaced_job_ids=[j for j, _ in old],
    )


def _save_children(
    session: Session,
    settings: Settings,
    job: SeparationJob,
    parent: Stem,
    output: RefineOutput,
) -> dict[str, int]:
    """子を FLAC（24bit）で保存し、STEM・STEM_RENDITION（master）を登録する（flush まで）。"""
    types = {t.code: t for t in session.scalars(select(StemType))}
    missing = [c for c in output.stems if c not in types]
    if missing:
        raise SeparationError(f"STEM_TYPE に無い stem があります: {', '.join(missing)}")
    out_dir = job_dir(settings, job.job_id, job.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids: dict[str, int] = {}
    for code, arr in output.stems.items():
        level = rms_db(arr)
        path = out_dir / f"{code}.flac"
        write_flac24(path, arr)
        stem = Stem(
            job_id=job.job_id,
            stem_type_id=types[code].stem_type_id,
            parent_stem_id=parent.stem_id,
            is_residual=code == output.rest,
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
        ids[code] = stem.stem_id
    session.flush()
    return ids


def refine_payload(method: RefineMethod, types: TypeIndex, stem_code: str) -> dict[str, Any]:
    """画面に出す方法の説明（API 用）。"""
    return {
        "model": method.filename,
        "display_name": method.display_name,
        "gpu": method.uses_gpu,
        "experimental": method.experimental,
        "children": [
            {"code": c, "display_name": types.by_code[c].display_name}
            for c in output_codes(method, stem_code)
            if c in types.by_code
        ],
    }


class ExportsBusy(RefineConflict):
    """消す stem を使う書き出しが作成待ち・作成中。"""


def take_exports_of_jobs(session: Session, job_ids: list[int]) -> list[int]:
    """ジョブの stem を使う書き出し（EXPORT）を DB から消し、export_id を返す（commit しない）。

    書き出しは full ジョブの行（EXPORT.job_id）に付くので、stem（EXPORT_ITEM）から探す。
    作成待ち・作成中のものがあれば何も消さずに ExportsBusy。ファイルは呼び出し側が commit の後に
    `remove_export_dirs` で消す。
    """
    from stemapp.exports.service import exports_using_stems

    if not job_ids:
        return []
    stem_ids = list(session.scalars(select(Stem.stem_id).where(Stem.job_id.in_(job_ids))))
    exps = exports_using_stems(session, stem_ids)
    if any(e.status in ACTIVE for e in exps):
        raise ExportsBusy(
            "この stem を使った書き出しが作成待ち・作成中です。"
            "書き出しが終わってからやり直してください。"
        )
    ids = [e.export_id for e in exps]
    for e in exps:
        session.delete(e)
    session.flush()
    return ids


def types_display(session: Session) -> dict[str, str]:
    return {t.code: t.display_name for t in session.scalars(select(StemType))}


def rest_warning(output: RefineOutput, names: dict[str, str]) -> str | None:
    """残りが 24bit の範囲を超えた（保存後に子の合計が親と一致しない）ときの注意書き。"""
    n = output.clipped.get(output.rest, 0)
    if not n:
        return None
    return (
        f"{names.get(output.rest, output.rest)}が 24bit の範囲（±1）を超えたため、"
        f"{n} サンプルで子の合計が親と一致しません（音割れしている可能性があります）。"
    )


def active_exports_of_jobs(session: Session, job_ids: list[int]) -> list[int]:
    """ジョブの stem を使う、作成待ち・作成中の書き出しの export_id。"""
    from stemapp.exports.service import exports_using_stems

    if not job_ids:
        return []
    stem_ids = list(session.scalars(select(Stem.stem_id).where(Stem.job_id.in_(job_ids))))
    return [e.export_id for e in exports_using_stems(session, stem_ids) if e.status in ACTIVE]


# 詳細分割のフォルダに入っているもの（これ以外があれば消さない）
_REFINE_FILE_SUFFIXES = frozenset({".flac", ".webm", ".m4a", ".stpk"})
_DELIVERY_DIRS = frozenset({"stream", "peaks"})


def _looks_like_refine_dir(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_dir():
                if child.name not in _DELIVERY_DIRS:
                    return False
                if any(
                    not f.is_file() or f.suffix.lower() not in _REFINE_FILE_SUFFIXES
                    for f in child.iterdir()
                ):
                    return False
            elif child.suffix.lower() not in _REFINE_FILE_SUFFIXES:
                return False
    except OSError:
        return False
    return True


def clean_orphan_refine_dirs(session: Session, settings: Settings) -> list[str]:
    """どのジョブも使っていない詳細分割のフォルダ（分け方のフォルダの下）を消す。

    置き換え（force）の完了を commit した後、古い子のフォルダを消す前に止められたときなどに残る。
    ワーカーの起動時（実行中のジョブが無いとき）に呼ぶ。中身が stem の音声・配信用データだけの
    フォルダしか消さない。消したもの（データフォルダからの相対パス）を返す。
    """
    used = {
        od for od in session.scalars(
            select(SeparationJob.output_dir).where(SeparationJob.output_dir.is_not(None))
        ) if od
    }
    running = session.scalar(
        select(SeparationJob.job_id).where(SeparationJob.status == RUNNING).limit(1)
    )
    if running is not None:
        return []
    removed: list[str] = []
    for od in sorted(
        session.scalars(
            select(SeparationJob.output_dir).where(
                SeparationJob.job_kind == JOB_KIND_FULL, SeparationJob.output_dir.is_not(None)
            )
        )
    ):
        base = job_dir(settings, 0, od)
        if not base.is_dir():
            continue
        for sub in base.iterdir():
            if not sub.is_dir() or sub.name in _DELIVERY_DIRS:
                continue
            rel = f"{od}/{sub.name}"
            if rel in used or not _looks_like_refine_dir(sub):
                continue
            shutil.rmtree(sub, ignore_errors=True)
            removed.append(rel)
    return removed


def invalidate_root_tempo(session: Session, settings: Settings, job_id: int) -> None:
    """job_id（full か refine）の元の分け方（full ジョブ）の速度変更のキャッシュを消す（T11）。

    詳細分割で葉が変わる（完了・戻す・置き換え）と、伸縮した音声の組が古くなるため。
    """
    from stemapp.tempo.service import invalidate_job_tempo

    root = root_job_id(session, job_id)
    if root is not None:
        invalidate_job_tempo(session, settings, root)
