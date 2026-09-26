"""書き出し（EXPORT / EXPORT_ITEM）の DB 操作: 対象の決定、登録、実行、片付け。

- 対象の stem と、その master（STEM_RENDITION purpose=master）のパスは必ず DB から引く。
- 書き出したファイルは `data/exports/<export_id>/<ファイル名>` に置く。
- 片付け: 作ってから `export_ttl_hours` たったもの、合計が `export_max_mb` を超えた分
  （古い順。いちばん新しい1件は残す）、DB に行の無いフォルダを消す。
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stemapp.audio import FfmpegRunner
from stemapp.config import Settings
from stemapp.exports.naming import export_base_name, mix_label, safe_filename
from stemapp.exports.render import (
    EXPORT_TYPES,
    FORMATS,
    TYPE_ALL,
    TYPE_MIX,
    TYPE_SINGLE,
    ExportError,
    RenderResult,
    SourceStem,
    render_mix,
    render_single,
    render_zip,
)
from stemapp.library import data_relative, resolve_data_path
from stemapp.models import (
    Export,
    ExportItem,
    ListenPreset,
    ListenPresetItem,
    SeparationJob,
    Stem,
    StemGroupMember,
    StemRendition,
    StemType,
    Track,
)

log = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
ACTIVE_STATUSES: tuple[str, ...] = (QUEUED, RUNNING)

INTERRUPTED_MESSAGE = "中断されました（アプリの再起動）。もう一度書き出してください。"
ZIP_LABEL = "stems"
ALL_LABEL = "全部"
PROGRESS_STEP = 0.02  # DB に書く進み具合の細かさ


class ExportNotFound(ExportError):
    """ジョブ・組み合わせ・書き出しが無い（404）。"""


class ExportConflict(ExportError):
    """今の状態ではできない（409）。"""


class ExportInvalid(ExportError):
    """指定の誤り（400）。"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


# --- 対象の決定 -----------------------------------------------------------------------


@dataclass(frozen=True)
class ExportRequest:
    export_type: str
    format: str
    stem_code: str | None = None  # single
    parents_only: bool = False  # all: 子に分かれていても親（いちばん上の stem）だけにする
    listen_preset_id: int | None = None  # mix: 組み合わせプリセット
    stems: Sequence[tuple[str, float]] = ()  # mix: (stem の code, gain_db)


@dataclass(frozen=True)
class JobStem:
    stem_id: int
    code: str
    display_name: str
    stem_type_id: int
    parent_stem_id: int | None
    display_order: int


@dataclass
class JobTree:
    """ジョブの stem の木（親子）。"""

    stems: list[JobStem]
    by_code: dict[str, JobStem] = field(init=False)
    children: dict[int, list[JobStem]] = field(init=False)

    def __post_init__(self) -> None:
        self.stems.sort(key=lambda s: (s.display_order, s.stem_id))
        self.by_code = {s.code: s for s in self.stems}
        self.children = {}
        for s in self.stems:
            if s.parent_stem_id is not None:
                self.children.setdefault(s.parent_stem_id, []).append(s)

    def leaves_of(self, stem: JobStem) -> list[JobStem]:
        kids = self.children.get(stem.stem_id, [])
        if not kids:
            return [stem]
        return [leaf for k in kids for leaf in self.leaves_of(k)]

    def leaves(self) -> list[JobStem]:
        return [s for s in self.stems if not self.children.get(s.stem_id)]

    def tops(self) -> list[JobStem]:
        return [s for s in self.stems if s.parent_stem_id is None]


def load_tree(session: Session, job_id: int) -> JobTree:
    rows = session.execute(
        select(Stem, StemType)
        .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
        .where(Stem.job_id == job_id)
    ).all()
    return JobTree(
        [
            JobStem(
                stem_id=s.stem_id,
                code=t.code,
                display_name=t.display_name,
                stem_type_id=t.stem_type_id,
                parent_stem_id=s.parent_stem_id,
                display_order=t.display_order,
            )
            for s, t in rows
        ]
    )


@dataclass(frozen=True)
class PlannedItem:
    stem: JobStem
    gain_db: float = 0.0


@dataclass(frozen=True)
class ExportPlan:
    job_id: int
    track_id: int
    export_type: str
    format: str
    listen_preset_id: int | None
    items: list[PlannedItem]
    filename: str


def _ordered(tree: JobTree, gains: dict[str, float]) -> list[PlannedItem]:
    order = {s.code: i for i, s in enumerate(tree.stems)}
    return [
        PlannedItem(tree.by_code[c], gains[c]) for c in sorted(gains, key=lambda c: order[c])
    ]


def _preset_items(
    session: Session, tree: JobTree, preset_id: int
) -> tuple[ListenPreset, list[PlannedItem]]:
    """組み合わせプリセットの葉と gain_db（画面の presetToSelection と同じ規則）。"""
    preset = session.get(ListenPreset, preset_id)
    if preset is None or preset.hidden:
        raise ExportNotFound("組み合わせが見つかりません。")
    code_of = {t.stem_type_id: t.code for t in session.scalars(select(StemType))}
    members: dict[int, list[int]] = {}
    for m in session.scalars(select(StemGroupMember)):
        members.setdefault(m.group_id, []).append(m.stem_type_id)
    gains: dict[str, float] = {}
    for it in session.scalars(
        select(ListenPresetItem)
        .where(ListenPresetItem.listen_preset_id == preset_id)
        .order_by(ListenPresetItem.item_id)
    ):
        type_ids = [it.stem_type_id] if it.stem_type_id is not None else members.get(
            it.group_id or -1, []
        )
        for type_id in type_ids:
            stem = tree.by_code.get(code_of.get(type_id, ""))
            if stem is None:
                continue
            for leaf in tree.leaves_of(stem):
                gains[leaf.code] = float(it.gain_db or 0.0)
    return preset, _ordered(tree, gains)


def plan_export(session: Session, job_id: int, req: ExportRequest) -> ExportPlan:
    """書き出しの対象とファイル名を決める。誤りは ExportNotFound / Conflict / Invalid。"""
    if req.export_type not in EXPORT_TYPES:
        raise ExportInvalid(f"書き出しの種類が正しくありません（{req.export_type}）。")
    if req.format not in FORMATS:
        raise ExportInvalid(f"形式が正しくありません（{req.format}）。")
    job = session.get(SeparationJob, job_id)
    if job is None:
        raise ExportNotFound("ジョブが見つかりません。")
    if job.status != DONE:
        raise ExportConflict("分割が終わっていないため書き出せません。")
    track = session.get(Track, job.track_id)
    title = track.title if track is not None else ""
    tree = load_tree(session, job_id)
    if not tree.stems:
        raise ExportConflict("このジョブには stem がありません。")

    preset_id: int | None = None
    if req.export_type == TYPE_SINGLE:
        if not req.stem_code:
            raise ExportInvalid("書き出す stem を選んでください。")
        stem = tree.by_code.get(req.stem_code)
        if stem is None:
            raise ExportInvalid(f"この曲に stem「{req.stem_code}」はありません。")
        items = [PlannedItem(stem)]
        filename = safe_filename(export_base_name(title, stem.display_name), req.format)
    elif req.export_type == TYPE_ALL:
        chosen = tree.tops() if req.parents_only else tree.leaves()
        items = [PlannedItem(s) for s in chosen]
        filename = safe_filename(export_base_name(title, ZIP_LABEL), "zip")
    else:  # mix
        if req.listen_preset_id is not None:
            preset, items = _preset_items(session, tree, req.listen_preset_id)
            preset_id = preset.listen_preset_id
            label = preset.name
            if not items:
                raise ExportInvalid("この曲には、この組み合わせの stem がありません。")
        else:
            gains: dict[str, float] = {}
            named: list[JobStem] = []
            for code, gain_db in req.stems:
                stem = tree.by_code.get(code)
                if stem is None:
                    raise ExportInvalid(f"この曲に stem「{code}」はありません。")
                if stem not in named:
                    named.append(stem)
                for leaf in tree.leaves_of(stem):
                    gains[leaf.code] = float(gain_db)
            if not gains:
                raise ExportInvalid("ミックスする stem を1つ以上選んでください。")
            items = _ordered(tree, gains)
            everything = {s.code for s in tree.leaves()} == set(gains)
            if everything and not any(gains.values()):
                label = ALL_LABEL
            else:
                named.sort(key=lambda s: tree.stems.index(s))
                label = mix_label([s.display_name for s in named])
        filename = safe_filename(export_base_name(title, label), req.format)
    return ExportPlan(
        job_id=job_id,
        track_id=job.track_id,
        export_type=req.export_type,
        format=req.format,
        listen_preset_id=preset_id,
        items=items,
        filename=filename,
    )


def create_export(session: Session, plan: ExportPlan) -> Export:
    """EXPORT（queued）と EXPORT_ITEM を登録する（commit まで）。"""
    exp = Export(
        job_id=plan.job_id,
        listen_preset_id=plan.listen_preset_id,
        export_type=plan.export_type,
        format=plan.format,
        status=QUEUED,
        progress=0.0,
        stage="書き出し待ち",
        filename=plan.filename,
    )
    session.add(exp)
    session.flush()
    for it in plan.items:
        session.add(
            ExportItem(export_id=exp.export_id, stem_id=it.stem.stem_id, gain_db=it.gain_db)
        )
    session.commit()
    log.info(
        "書き出しを登録しました（export %d, job %d, %s, %s）。",
        exp.export_id, plan.job_id, plan.export_type, plan.format,
    )
    return exp


# --- 実行 -------------------------------------------------------------------------------


def _track_title(session: Session, job_id: int) -> str:
    job = session.get(SeparationJob, job_id)
    track = session.get(Track, job.track_id) if job is not None else None
    return track.title if track is not None else ""


def _sources(
    session: Session, settings: Settings, job_id: int, items: Sequence[tuple[int, float]]
) -> list[SourceStem]:
    """items（stem_id, gain_db）を書き出す stem にする。master のパスは STEM_RENDITION から引く。"""
    tree = load_tree(session, job_id)
    by_id = {s.stem_id: s for s in tree.stems}
    masters = {
        r.stem_id: r
        for r in session.scalars(
            select(StemRendition).where(
                StemRendition.stem_id.in_([stem_id for stem_id, _ in items]),
                StemRendition.purpose == "master",
            )
        )
    }
    order = {s.stem_id: i for i, s in enumerate(tree.stems)}
    out: list[SourceStem] = []
    for stem_id, gain_db in sorted(items, key=lambda i: order.get(i[0], 0)):
        stem = by_id.get(stem_id)
        rend = masters.get(stem_id)
        if stem is None or rend is None:
            raise ExportError("書き出す stem の元の音声（master）が見つかりません。")
        out.append(
            SourceStem(
                code=stem.code,
                display_name=stem.display_name,
                path=resolve_data_path(settings, rend.file_path),
                gain_db=float(gain_db or 0.0),
            )
        )
    if not out:
        raise ExportError("書き出す stem がありません。")
    return out


def export_dir(settings: Settings, export_id: int) -> Path:
    return settings.exports_dir / str(export_id)


def _render(
    title: str,
    export_type: str,
    fmt: str,
    filename: str,
    sources: list[SourceStem],
    dst_dir: Path,
    runner: FfmpegRunner | None,
    progress: Callable[[float, str], None] | None,
) -> RenderResult:
    prog = progress or (lambda _p, _s: None)
    dst = dst_dir / filename
    if export_type == TYPE_SINGLE:
        return render_single(sources[0], dst, fmt, runner, prog)
    if export_type == TYPE_MIX:
        return render_mix(sources, dst, fmt, runner, prog)
    entries = [(s, safe_filename(export_base_name(title, s.display_name), fmt)) for s in sources]
    return render_zip(entries, dst, fmt, runner, prog)


def render_export(
    session: Session,
    settings: Settings,
    exp: Export,
    dst_dir: Path,
    *,
    runner: FfmpegRunner | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> RenderResult:
    """exp（DB の行と EXPORT_ITEM）の内容を dst_dir/<exp.filename> に書き出す（DB は変えない）。"""
    title = _track_title(session, exp.job_id)
    items = session.scalars(select(ExportItem).where(ExportItem.export_id == exp.export_id)).all()
    sources = _sources(session, settings, exp.job_id, [(i.stem_id, i.gain_db) for i in items])
    filename = exp.filename or safe_filename(export_base_name(title, exp.export_type), exp.format)
    return _render(
        title, exp.export_type, exp.format, filename, sources, dst_dir, runner, progress
    )


def render_plan(
    session: Session,
    settings: Settings,
    plan: ExportPlan,
    dst_dir: Path,
    *,
    runner: FfmpegRunner | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> RenderResult:
    """DB に登録せずに書き出す（CLI 用）。dst_dir/<plan.filename> に書く。"""
    title = _track_title(session, plan.job_id)
    sources = _sources(
        session, settings, plan.job_id, [(i.stem.stem_id, i.gain_db) for i in plan.items]
    )
    return _render(
        title, plan.export_type, plan.format, plan.filename, sources, dst_dir, runner, progress
    )


def _update(
    factory: sessionmaker[Session], export_id: int, **values: object
) -> bool:
    """行を書き換える。行が無ければ（削除された）False。"""
    with factory() as s:
        exp = s.get(Export, export_id)
        if exp is None:
            return False
        for k, v in values.items():
            setattr(exp, k, v)
        s.commit()
        return True


class _Deleted(Exception):
    """書き出し中に行が消された（曲の削除など）。"""


def run_export(
    settings: Settings,
    factory: sessionmaker[Session],
    export_id: int,
    runner: FfmpegRunner | None = None,
) -> str | None:
    """queued の書き出しを1件実行する。終わった状態（done / failed）か、行が無ければ None。"""
    with factory() as s:
        exp = s.get(Export, export_id)
        if exp is None or exp.status != QUEUED:
            return None
        exp.status = RUNNING
        exp.stage = "書き出し中"
        s.commit()
    out_dir = export_dir(settings, export_id)
    last = [-1.0, ""]

    def progress(p: float, stage: str) -> None:
        if p - last[0] < PROGRESS_STEP and stage == last[1]:
            return
        last[0], last[1] = p, stage
        if not _update(factory, export_id, progress=min(max(p, 0.0), 0.99), stage=stage[:100]):
            raise _Deleted

    try:
        with factory() as s:
            exp = s.get(Export, export_id)
            if exp is None:
                raise _Deleted
            result = render_export(s, settings, exp, out_dir, runner=runner, progress=progress)
        ok = _update(
            factory, export_id,
            status=DONE, progress=1.0, stage="完了",
            output_path=data_relative(settings, result.path),
            bytes=result.bytes, mix_gain_db=result.mix_gain_db, finished_at=_utcnow(),
        )
        if not ok:
            raise _Deleted
        log.info("書き出しました（export %d, %d バイト）。", export_id, result.bytes)
        return DONE
    except _Deleted:
        shutil.rmtree(out_dir, ignore_errors=True)
        log.info("書き出し中に削除されました（export %d）。", export_id)
        return None
    except Exception as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        message = str(e) if isinstance(e, ExportError) else f"書き出しに失敗しました: {e}"
        if not isinstance(e, ExportError):
            log.exception("書き出しで想定外のエラー（export %d）", export_id)
        _update(
            factory, export_id, status=FAILED, stage="失敗", error_message=message,
            finished_at=_utcnow(),
        )
        return FAILED


# --- 片付け -------------------------------------------------------------------------------


def export_ids_for_jobs(session: Session, job_ids: Iterable[int]) -> list[int]:
    ids = list(job_ids)
    if not ids:
        return []
    return list(session.scalars(select(Export.export_id).where(Export.job_id.in_(ids))))


def remove_export_dirs(settings: Settings, export_ids: Iterable[int]) -> None:
    for export_id in export_ids:
        shutil.rmtree(export_dir(settings, export_id), ignore_errors=True)


def _delete(session: Session, settings: Settings, exps: Sequence[Export]) -> list[int]:
    ids = [e.export_id for e in exps]
    for e in exps:
        session.delete(e)  # EXPORT_ITEM は外部キーの CASCADE で消える
    session.commit()
    remove_export_dirs(settings, ids)
    return ids


def recover_interrupted_exports(session: Session, settings: Settings) -> list[int]:
    """起動時: 前回の queued / running（アプリの終了で中断）を failed にし、作りかけを消す。"""
    exps = session.scalars(select(Export).where(Export.status.in_(ACTIVE_STATUSES))).all()
    for e in exps:
        e.status = FAILED
        e.stage = "失敗"
        e.error_message = INTERRUPTED_MESSAGE
        e.finished_at = _utcnow()
    session.commit()
    ids = [e.export_id for e in exps]
    remove_export_dirs(settings, ids)
    return ids


def cleanup_exports(
    session: Session, settings: Settings, now: datetime | None = None
) -> list[int]:
    """期限切れ・容量超えの書き出しと、DB に無いフォルダを消す。消した export_id を返す。"""
    now = now or _utcnow()
    limit = now - timedelta(hours=settings.export_ttl_hours)
    finished = session.scalars(
        select(Export)
        .where(Export.status.not_in(ACTIVE_STATUSES))
        .order_by(Export.created_at, Export.export_id)
    ).all()
    expired = [e for e in finished if _aware(e.created_at) < limit]
    removed = _delete(session, settings, expired)

    done = [e for e in finished if e.status == DONE and e.export_id not in removed]
    total = sum(e.bytes or 0 for e in done)
    max_bytes = settings.export_max_mb * 1024 * 1024
    over: list[Export] = []
    # いちばん新しい1件は残す（作った直後にダウンロードできなくならないように）
    for e in done[:-1]:
        if total <= max_bytes:
            break
        over.append(e)
        total -= e.bytes or 0
    removed += _delete(session, settings, over)

    root = settings.exports_dir
    known = {str(i) for i in session.scalars(select(Export.export_id))}
    for d in root.iterdir():
        if d.is_dir() and d.name not in known:
            shutil.rmtree(d, ignore_errors=True)
    if removed:
        log.info("書き出しを片付けました（export %s）。", removed)
    return removed
