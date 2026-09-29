"""T07 詳細分割のレビュー対応: 24bit でぴったり一致、残りの警告、書き出しとの整合、
ワーカー起動時の後始末、置き換えの途中の失敗、曲の削除、古い保存フォルダ。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stemapp.audio import read_audio
from stemapp.config import Settings
from stemapp.exports.service import (
    ExportInvalid,
    ExportRequest,
    create_export,
    export_dir,
    plan_export,
)
from stemapp.jobs import JobConflict, delete_job
from stemapp.jobs.queue import claim_next_job, recover_interrupted_jobs
from stemapp.jobs.worker import Worker
from stemapp.library import resolve_data_path
from stemapp.models import Export, SeparationJob, Stem, StemType
from stemapp.seed import DRUMSEP, MALE_FEMALE
from stemapp.separation import FakeSeparator
from stemapp.separation.refine import RefineConflict, enqueue_refine_job, quantize24
from stemapp.stem_view import build_view
from test_refine import (  # noqa: F401  fixture を使う
    _cfactory,
    _children,
    _refine_launcher,
    _run_refine,
    _run_worker,
    _stem_id,
    _stems,
    api_job,
    client,
    factory,
    full_job,
    seeded,
)


class Loud(FakeSeparator):
    """子をすべて -1.5 にする（子は -1 に丸まり、残りが 24bit の範囲を超える）。"""

    def separate(self, *a: Any, **k: Any) -> dict[str, np.ndarray]:
        out = super().separate(*a, **k)
        return {name: np.full_like(v, -1.5) for name, v in out.items()}


def test_quantize24_matches_flac(tmp_path: Path) -> None:
    import soundfile as sf

    rng = np.random.default_rng(1)
    x = rng.uniform(-1.2, 1.2, (5000, 2))
    path = tmp_path / "q.flac"
    sf.write(str(path), np.clip(x, -1, 1), 44100, subtype="PCM_24", format="FLAC")
    assert np.array_equal(read_audio(path).astype(np.float64), quantize24(x))


def test_rest_out_of_range_is_warned(
    settings: Settings, factory: sessionmaker[Session], full_job: int  # noqa: F811
) -> None:
    lead = _stem_id(factory, full_job, "lead_vocal")
    job_id = _run_refine(settings, factory, lead, MALE_FEMALE, separator=Loud())
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.status == "done"
        assert job.warning and "残り（メインボーカル）" in job.warning
        assert "一致しません" in job.warning


def _export_for(s: Session, job_id: int, codes: list[str], status: str = "queued") -> int:
    plan = plan_export(s, job_id, ExportRequest("mix", "wav", stems=[(c, 0.0) for c in codes]))
    exp = create_export(s, plan)
    exp.status = status
    s.commit()
    return exp.export_id


def _set_export(s: Session, export_id: int, status: str) -> None:
    e = s.get(Export, export_id)
    assert e is not None
    e.status = status
    s.commit()


def test_undo_with_exports(
    settings: Settings, factory: sessionmaker[Session], full_job: int  # noqa: F811
) -> None:
    drums = _stem_id(factory, full_job, "drums")
    refine_id = _run_refine(settings, factory, drums, DRUMSEP)
    with factory() as s:
        busy = _export_for(s, full_job, ["kick", "bass"], "running")
        other = _export_for(s, full_job, ["bass"], "done")
        # 作成中の書き出しが子を使っている → 戻せない（何も消えない）
        with pytest.raises(JobConflict, match="書き出し"):
            delete_job(s, settings, refine_id)
        assert s.get(SeparationJob, refine_id) is not None
        assert len(_children(s, drums)) == 7
        _set_export(s, busy, "done")
        export_dir(settings, busy).mkdir(parents=True, exist_ok=True)
        (export_dir(settings, busy) / "mix.wav").write_bytes(b"x")
        delete_job(s, settings, refine_id)
        s.expire_all()
        # 子を使った作成済みの書き出しはファイルごと消える。子を使わないものは残る
        assert s.get(Export, busy) is None
        assert not export_dir(settings, busy).exists()
        assert s.get(Export, other) is not None
        # 戻した後の書き出し: 子は指定できない、親は書き出せる
        with pytest.raises(ExportInvalid):
            plan_export(s, full_job, ExportRequest("mix", "wav", stems=[("kick", 0.0)]))
        plan = plan_export(s, full_job, ExportRequest("all", "wav"))
        assert "drums" in {i.stem.code for i in plan.items}
        assert "kick" not in {i.stem.code for i in plan.items}


def test_force_replace_with_exports(
    settings: Settings, factory: sessionmaker[Session], full_job: int  # noqa: F811
) -> None:
    drums = _stem_id(factory, full_job, "drums")
    first = _run_refine(settings, factory, drums, DRUMSEP)
    with factory() as s:
        busy = _export_for(s, full_job, ["kick"], "queued")
        # 作成待ちの書き出しが今の子を使っている → 置き換えの登録は 409
        with pytest.raises(RefineConflict, match="書き出し"):
            enqueue_refine_job(s, drums, DRUMSEP, force=True)
        _set_export(s, busy, "done")
        job_id = enqueue_refine_job(s, drums, DRUMSEP, force=True).job.job_id
        # 登録の後に書き出しが始まった → 置き換えずに失敗（古い子は残る）
        late = _export_for(s, full_job, ["snare"], "running")
    Worker(settings, factory, _refine_launcher(settings)).run_one()
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.status == "failed"
        assert "書き出し" in (job.error_message or "")
        assert s.get(SeparationJob, first) is not None
        assert all(k.job_id == first for k in _children(s, drums).values())
        _set_export(s, late, "done")
    second = _run_refine(settings, factory, drums, DRUMSEP, force=True)
    with factory() as s:
        assert s.get(SeparationJob, first) is None
        assert all(k.job_id == second for k in _children(s, drums).values())
        # 古い子を使った書き出しは消える
        assert s.get(Export, busy) is None and s.get(Export, late) is None


def test_force_replace_failure_keeps_old_children(
    settings: Settings, factory: sessionmaker[Session], full_job: int  # noqa: F811
) -> None:
    drums = _stem_id(factory, full_job, "drums")
    first = _run_refine(settings, factory, drums, DRUMSEP)
    with factory() as s:
        first_job = s.get(SeparationJob, first)
        assert first_job is not None and first_job.output_dir is not None
        old_dir = resolve_data_path(settings, first_job.output_dir)
        job_id = enqueue_refine_job(s, drums, DRUMSEP, force=True).job.job_id
    Worker(
        settings, factory, _refine_launcher(settings, FakeSeparator(fail_models={DRUMSEP}))
    ).run_one()
    with factory() as s:
        failed = s.get(SeparationJob, job_id)
        kept = s.get(SeparationJob, first)
        assert failed is not None and failed.status == "failed"
        assert kept is not None and kept.status == "done"
        assert len(_children(s, drums)) == 7
        assert build_view(s, full_job).refined_by[drums].job_id == first
    assert (old_dir / "kick.flac").is_file()


def test_recover_interrupted_refine(
    settings: Settings, factory: sessionmaker[Session], full_job: int  # noqa: F811
) -> None:
    """ワーカーの起動時: running のまま残った詳細分割は failed、作りかけの子とフォルダは消える。"""
    drums = _stem_id(factory, full_job, "drums")
    with factory() as s:
        job_id = enqueue_refine_job(s, drums, DRUMSEP).job.job_id
        assert claim_next_job(s) == job_id
        full = s.get(SeparationJob, full_job)
        job = s.get(SeparationJob, job_id)
        assert full is not None and job is not None
        job.output_dir = f"{full.output_dir}/drums"
        d = resolve_data_path(settings, job.output_dir)
        d.mkdir(parents=True)
        (d / "kick.flac").write_bytes(b"x")
        kick = s.scalars(select(StemType).where(StemType.code == "kick")).one()
        s.add(Stem(job_id=job_id, stem_type_id=kick.stem_type_id, parent_stem_id=drums))
        s.commit()
        assert recover_interrupted_jobs(s, settings) == [job_id]
        s.expire_all()
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.status == "failed" and job.output_dir is None
        assert _children(s, drums) == {}
    assert not d.exists()
    # 同じ stem をもう一度分けられる
    _run_refine(settings, factory, drums, DRUMSEP)


def test_orphan_refine_dirs_are_cleaned(
    settings: Settings, factory: sessionmaker[Session], full_job: int  # noqa: F811
) -> None:
    """置き換えの完了の後、古いフォルダを消す前に止められた跡を、起動時の片付けで消す。"""
    drums = _stem_id(factory, full_job, "drums")
    job_id = _run_refine(settings, factory, drums, DRUMSEP)
    with factory() as s:
        full = s.get(SeparationJob, full_job)
        refine = s.get(SeparationJob, job_id)
        assert full and refine and full.output_dir and refine.output_dir
        base = resolve_data_path(settings, full.output_dir)
        used = resolve_data_path(settings, refine.output_dir)
    orphan = base / "drums (5)"
    (orphan / "stream").mkdir(parents=True)
    (orphan / "kick.flac").write_bytes(b"x")
    (orphan / "stream" / "kick.webm").write_bytes(b"x")
    keep = base / "memo"  # 音声以外のものがあるフォルダは消さない
    keep.mkdir()
    (keep / "note.txt").write_text("user file", encoding="utf-8")
    with factory() as s:
        recover_interrupted_jobs(s, settings)
    assert not orphan.exists()
    assert keep.is_dir() and used.is_dir()
    assert (base / "stream").is_dir() and (base / "drums.flac").is_file()


def test_api_delete_track_with_children(
    client: TestClient, api_job: tuple[int, int]  # noqa: F811
) -> None:
    track_id, job_id = api_job
    stems = _stems(client, job_id)
    client.post(f"/api/stems/{stems['drums']['stem_id']}/refine", json={"model": DRUMSEP})
    _run_worker(client)
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _cfactory(client)() as s:
        full = s.get(SeparationJob, job_id)
        assert full is not None and full.output_dir is not None
        track_dir = resolve_data_path(settings, full.output_dir).parent
    assert (track_dir / "fast" / "drums" / "kick.flac").is_file()
    assert client.delete(f"/api/tracks/{track_id}").status_code == 200
    with _cfactory(client)() as s:
        assert s.scalars(select(SeparationJob)).all() == []
        assert s.scalars(select(Stem)).all() == []
    assert not track_dir.exists()


def test_api_refine_note_for_legacy_folder(
    client: TestClient, api_job: tuple[int, int]  # noqa: F811
) -> None:
    _, job_id = api_job
    assert client.get(f"/api/jobs/{job_id}/stems").json()["refine_note"] is None
    with _cfactory(client)() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None
        job.output_dir = None
        s.commit()
    body = client.get(f"/api/jobs/{job_id}/stems").json()
    assert "migrate-folders" in body["refine_note"]
    assert all(st["refine_methods"] == [] for st in body["stems"])
    drums = next(st for st in body["stems"] if st["code"] == "drums")["stem_id"]
    res = client.post(f"/api/stems/{drums}/refine", json={"model": DRUMSEP})
    assert res.status_code == 409 and "migrate-folders" in res.json()["detail"]
