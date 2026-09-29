"""画面・書き出しに見せる stem の木（分け方＝full ジョブの stem と、詳細分割の子）。

詳細分割（job_kind=refine）の子 stem は、refine ジョブの STEM として登録される
（STEM.job_id = refine ジョブ、parent_stem_id = 分けた stem）。full ジョブを開いたときは、
その stem を分けた完了済みの refine ジョブの子も同じ木に入れる（子の子も同じ）。

並びは木の順（親の直後にその子。兄弟は STEM_TYPE.display_order の順）。
code（STEM_TYPE.code）は木の中で重ならない（refine の登録時に確かめる）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.models import SeparationJob, Stem, StemType

JOB_KIND_FULL = "full"
JOB_KIND_REFINE = "refine"
DONE = "done"


@dataclass
class StemView:
    """full ジョブ（分け方）の stem の木。"""

    job_id: int
    rows: list[tuple[Stem, StemType]]
    # 分けた stem の stem_id → その子を作った完了済みの refine ジョブ
    refined_by: dict[int, SeparationJob] = field(default_factory=dict)

    @property
    def stem_ids(self) -> list[int]:
        return [s.stem_id for s, _ in self.rows]

    def children_of(self, stem_id: int) -> list[tuple[Stem, StemType]]:
        return [(s, t) for s, t in self.rows if s.parent_stem_id == stem_id]

    def descendants_of(self, stem_id: int) -> list[tuple[Stem, StemType]]:
        out: list[tuple[Stem, StemType]] = []
        for s, t in self.children_of(stem_id):
            out.append((s, t))
            out.extend(self.descendants_of(s.stem_id))
        return out

    def find(self, stem_id: int) -> tuple[Stem, StemType] | None:
        return next(((s, t) for s, t in self.rows if s.stem_id == stem_id), None)


def root_job_id(session: Session, job_id: int) -> int | None:
    """refine ジョブなら、元をたどった full ジョブの job_id。full ならそのまま。"""
    seen: set[int] = set()
    current: int | None = job_id
    while current is not None and current not in seen:
        seen.add(current)
        job = session.get(SeparationJob, current)
        if job is None:
            return None
        if job.job_kind != JOB_KIND_REFINE:
            return job.job_id
        if job.input_stem_id is None:
            return None
        stem = session.get(Stem, job.input_stem_id)
        current = stem.job_id if stem is not None else None
    return None


def _rows_of_jobs(session: Session, job_ids: list[int]) -> list[tuple[Stem, StemType]]:
    if not job_ids:
        return []
    return [
        (s, t)
        for s, t in session.execute(
            select(Stem, StemType)
            .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
            .where(Stem.job_id.in_(job_ids))
        ).all()
    ]


def build_view(session: Session, job_id: int) -> StemView:
    """ジョブの stem と、それを分けた完了済み refine ジョブの子を、木の順で返す。"""
    rows = _rows_of_jobs(session, [job_id])
    refined_by: dict[int, SeparationJob] = {}
    frontier = [s.stem_id for s, _ in rows]
    seen_jobs = {job_id}
    while frontier:
        jobs = [
            j
            for j in session.scalars(
                select(SeparationJob)
                .where(
                    SeparationJob.job_kind == JOB_KIND_REFINE,
                    SeparationJob.status == DONE,
                    SeparationJob.input_stem_id.in_(frontier),
                )
                .order_by(SeparationJob.job_id)
            )
            if j.job_id not in seen_jobs
        ]
        if not jobs:
            break
        # 同じ stem を分けた完了済みジョブが複数あれば新しい方だけ（置き換えの途中の保険）
        latest: dict[int, SeparationJob] = {}
        for j in jobs:
            assert j.input_stem_id is not None
            latest[j.input_stem_id] = j
        seen_jobs.update(j.job_id for j in jobs)
        refined_by.update(latest)
        new_rows = _rows_of_jobs(session, [j.job_id for j in latest.values()])
        rows.extend(new_rows)
        frontier = [s.stem_id for s, _ in new_rows]
    return StemView(job_id=job_id, rows=_tree_order(rows), refined_by=refined_by)


def _tree_order(rows: list[tuple[Stem, StemType]]) -> list[tuple[Stem, StemType]]:
    ids = {s.stem_id for s, _ in rows}
    kids: dict[int | None, list[tuple[Stem, StemType]]] = {}
    for s, t in rows:
        parent = s.parent_stem_id if s.parent_stem_id in ids else None
        kids.setdefault(parent, []).append((s, t))
    for group in kids.values():
        group.sort(key=lambda r: (r[1].display_order, r[0].stem_id))
    out: list[tuple[Stem, StemType]] = []

    def walk(parent: int | None) -> None:
        for s, t in kids.get(parent, []):
            out.append((s, t))
            walk(s.stem_id)

    walk(None)
    return out
