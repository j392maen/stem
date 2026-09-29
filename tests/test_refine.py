"""T07 詳細分割（もっと分ける）: 合計一致、残りの型、木、配信用データ、キャンセル、再実行、削除。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from audio_helpers import synth_drums, synth_mix
from job_helpers import make_track, sync_launcher
from stemapp.audio import read_audio
from stemapp.config import Settings
from stemapp.db import make_session_factory
from stemapp.delivery import fake_encoder
from stemapp.jobs import delete_job, enqueue_full_job, request_cancel
from stemapp.jobs.child import EXIT_DONE, EXIT_FAILED, run_job
from stemapp.jobs.queue import claim_next_job
from stemapp.jobs.worker import ChildHandle, Worker, subprocess_launcher
from stemapp.library import resolve_data_path
from stemapp.models import (
    ListenPreset,
    ListenPresetItem,
    SeparationJob,
    Stem,
    StemRendition,
    StemType,
    Waveform,
)
from stemapp.peaks import DEFAULT_LEVELS
from stemapp.seed import ASPIRATION, DRUMSEP, HPSS, MALE_FEMALE, seed
from stemapp.separation import FakeSeparator
from stemapp.separation.fake import DEFAULT_REFINE_COEFS, fake_hpss
from stemapp.separation.hpss import (
    HARMONIC_KERNEL_FRAMES,
    HOP_LENGTH,
    MARGIN,
    N_FFT,
    PERCUSSIVE_KERNEL_BINS,
)
from stemapp.separation.refine import (
    RefineConflict,
    RefineInvalid,
    RefineMethod,
    TypeIndex,
    enqueue_refine_job,
    load_methods,
    method_applies,
    rest_code,
    run_refine,
)
from stemapp.stem_view import build_view
from test_api import _app

# 24bit の1段（2^-23）。保存した子の合計と親の差は、子の数 × 半段 以内
LSB24 = 2.0**-23


@pytest.fixture
def factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)


@pytest.fixture
def seeded(session: Session) -> Session:
    seed(session)
    return session


def _worker(settings: Settings, factory: sessionmaker[Session], **kw: Any) -> Worker:
    return Worker(settings, factory, sync_launcher(settings, **kw))


def _refine_launcher(settings: Settings, separator: Any = None) -> Callable[[int], ChildHandle]:
    """詳細分割も同期で実行する（HPSS はテスト用の fake_hpss）。"""
    from job_helpers import FinishedHandle

    def launch(job_id: int) -> ChildHandle:
        rc = run_job(
            settings, job_id, separator or FakeSeparator(), encoder=fake_encoder, hpss=fake_hpss
        )
        return FinishedHandle(rc)

    return launch


@pytest.fixture
def full_job(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], tmp_path: Path
) -> int:
    track_id = make_track(seeded, settings, tmp_path, seconds=1.0)
    job_id = enqueue_full_job(seeded, track_id, "fast").job.job_id
    assert _worker(settings, factory).run_one() == job_id
    return job_id


def _stem_id(factory: sessionmaker[Session], job_id: int, code: str) -> int:
    with factory() as s:
        view = build_view(s, job_id)
        return next(st.stem_id for st, t in view.rows if t.code == code)


def _run_refine(
    settings: Settings, factory: sessionmaker[Session], stem_id: int, model: str,
    *, force: bool = False, separator: Any = None,
) -> int:
    with factory() as s:
        res = enqueue_refine_job(s, stem_id, model, force=force)
        assert res.created, res.reason
        job_id = res.job.job_id
    worker = Worker(settings, factory, _refine_launcher(settings, separator))
    assert worker.run_one() == job_id
    return job_id


def _master(settings: Settings, s: Session, stem_id: int) -> np.ndarray:
    r = s.scalars(
        select(StemRendition).where(
            StemRendition.stem_id == stem_id, StemRendition.purpose == "master"
        )
    ).one()
    return read_audio(resolve_data_path(settings, r.file_path)).astype(np.float64)


def _children(s: Session, parent_id: int) -> dict[str, Stem]:
    rows = s.execute(
        select(Stem, StemType)
        .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
        .where(Stem.parent_stem_id == parent_id)
    ).all()
    return {t.code: st for st, t in rows}


# --- 型と方法 ------------------------------------------------------------------------------


def test_seed_rest_types_and_methods(seeded: Session) -> None:
    types = TypeIndex.load(seeded)
    for parent in ("drums", "other", "lead_vocal", "backing_vocal"):
        rest = types.by_code[rest_code(parent)]
        assert types.by_id[rest.parent_id].code == parent  # type: ignore[index]
        assert rest.tier == "detail" and rest.refine_model_id is None
        assert rest.display_name.startswith("残り（")
    for code, name in (("sustained", "持続音（パッド等）"), ("transient", "短い音（ヒット等）")):
        t = types.by_code[code]
        assert t.display_name == name and t.tier == "detail" and t.experimental
        assert types.by_id[t.parent_id].code == "other"  # type: ignore[index]

    methods = load_methods(seeded, types)
    assert set(methods) == {DRUMSEP, MALE_FEMALE, ASPIRATION, HPSS}
    assert methods[DRUMSEP].child_codes == ("kick", "snare", "toms", "hihat", "ride", "crash")
    assert methods[DRUMSEP].parent_code == "drums"
    assert methods[MALE_FEMALE].child_codes == ("male", "female")
    assert methods[ASPIRATION].child_codes == ("breath",)
    assert methods[HPSS].child_codes == ("sustained", "transient")
    assert methods[HPSS].is_hpss and not methods[HPSS].uses_gpu

    def applies(model: str, code: str) -> bool:
        return method_applies(methods[model], code, types)

    assert applies(DRUMSEP, "drums") and applies(HPSS, "other")
    assert applies(MALE_FEMALE, "lead_vocal") and applies(MALE_FEMALE, "backing_vocal")
    assert applies(ASPIRATION, "backing_vocal")
    # vocals は分割時に lead / backing に分かれている（残りの型も無い）。子や別の親には使えない
    assert not applies(MALE_FEMALE, "vocals")
    assert not applies(DRUMSEP, "kick") and not applies(DRUMSEP, "bass")
    assert not applies(HPSS, "drums") and not applies(DRUMSEP, "drums_rest")


def test_hpss_settings_follow_research() -> None:
    """R01 D-3: kernel 約1秒 × 約300Hz、margin=2（44.1kHz、窓 2048、送り 512）。"""
    sr = 44100
    assert abs(HARMONIC_KERNEL_FRAMES * HOP_LENGTH / sr - 1.0) < 0.05
    assert abs(PERCUSSIVE_KERNEL_BINS * sr / N_FFT - 300) < 30
    assert MARGIN == 2.0
    assert HARMONIC_KERNEL_FRAMES % 2 == 1 and PERCUSSIVE_KERNEL_BINS % 2 == 1


# --- 計算（合計一致） ---------------------------------------------------------------------------


def _method(seeded: Session, model: str) -> RefineMethod:
    return load_methods(seeded)[model]


@pytest.mark.parametrize(
    ("model", "parent"),
    [(DRUMSEP, "drums"), (MALE_FEMALE, "lead_vocal"), (ASPIRATION, "backing_vocal")],
)
def test_run_refine_children_sum_to_parent(
    seeded: Session, tmp_path: Path, model: str, parent: str
) -> None:
    x = synth_mix(0.5)
    m = _method(seeded, model)
    out = run_refine(x, m, parent, FakeSeparator(), workdir=tmp_path / "w", device="cpu")
    assert set(out.stems) == {*m.child_codes, rest_code(parent)}
    assert out.rest == rest_code(parent)
    total = sum(a.astype(np.float64) for a in out.stems.values())
    assert np.max(np.abs(total - x)) < 1e-5
    # 子は Fake の係数どおり（息のモデルの「息以外」は子に使わない）
    for code in m.child_codes:
        coef = DEFAULT_REFINE_COEFS[model][code]
        assert np.allclose(out.stems[code], x * coef, atol=1e-6)
    assert "no_breath" not in out.stems


def test_run_refine_real_hpss_sums_to_parent(seeded: Session, tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    # 伸びる音（和音）と短い音（ドラム）を混ぜた音
    x = (0.5 * synth_mix(2.0) + 0.5 * synth_drums([(0.0, 120.0)], 2.0)).astype(np.float32)
    m = _method(seeded, HPSS)
    stages: list[str] = []
    out = run_refine(
        x, m, "other", None, workdir=tmp_path / "w", progress=lambda p, s: stages.append(s)
    )
    assert set(out.stems) == {"sustained", "transient", "other_rest"}
    total = sum(a.astype(np.float64) for a in out.stems.values())
    assert np.max(np.abs(total - x)) < 1e-5
    energy = {k: float(np.sum(v.astype(np.float64) ** 2)) for k, v in out.stems.items()}
    assert energy["sustained"] > 0 and energy["transient"] > 0
    assert stages


def test_run_refine_clips_named_children_into_rest(seeded: Session, tmp_path: Path) -> None:
    """名前の付いた子が ±1 を超えたら丸め、はみ出た分は残りに入る（合計は親のまま）。"""
    x = np.full((1000, 2), 0.9, dtype=np.float32)

    class Loud(FakeSeparator):
        def separate(self, *a: Any, **k: Any) -> dict[str, np.ndarray]:
            out = super().separate(*a, **k)
            out["male"] = out["male"] * 4  # 0.45 × 0.9 × 4 = 1.62
            return out

    out = run_refine(x, _method(seeded, MALE_FEMALE), "lead_vocal", Loud(), workdir=tmp_path)
    assert out.clipped == {"male": 2000}
    assert np.max(np.abs(out.stems["male"])) <= 1.0
    total = sum(a.astype(np.float64) for a in out.stems.values())
    assert np.max(np.abs(total - x)) < 1e-5


# --- ジョブ（保存・木・配信用データ） ------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "parent"),
    [(DRUMSEP, "drums"), (MALE_FEMALE, "lead_vocal"), (ASPIRATION, "backing_vocal"),
     (HPSS, "other")],
)
def test_refine_job_saves_children(
    settings: Settings, factory: sessionmaker[Session], full_job: int, model: str, parent: str
) -> None:
    parent_id = _stem_id(factory, full_job, parent)
    job_id = _run_refine(settings, factory, parent_id, model)
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.status == "done", job.error_message if job else None
        assert job.job_kind == "refine" and job.input_stem_id == parent_id
        full = s.get(SeparationJob, full_job)
        assert full is not None
        # 保存先は親のジョブのフォルダの下の <親の code>
        assert job.output_dir == f"{full.output_dir}/{parent}"
        assert job.run_on == ("cpu" if model == HPSS else "gpu")
        kids = _children(s, parent_id)
        methods = load_methods(s)
        assert set(kids) == {*methods[model].child_codes, rest_code(parent)}
        assert [c for c, k in kids.items() if k.is_residual] == [rest_code(parent)]
        assert all(k.job_id == job_id for k in kids.values())
        # 保存した子（24bit）の合計＝親（24bit の量子化誤差の範囲）
        parent_audio = _master(settings, s, parent_id)
        total = sum(_master(settings, s, k.stem_id) for k in kids.values())
        assert np.max(np.abs(total - parent_audio)) <= len(kids) * LSB24
        # 配信用データ（stream と全解像度の peaks）
        for code, k in kids.items():
            rends = s.scalars(select(StemRendition).where(StemRendition.stem_id == k.stem_id)).all()
            assert {r.purpose for r in rends} == {"master", "stream"}, code
            paths = {r.purpose: r.file_path for r in rends}
            assert paths["master"] == f"{job.output_dir}/{code}.flac"
            assert paths["stream"].startswith(f"{job.output_dir}/stream/{code}.")
            waves = s.scalars(select(Waveform).where(Waveform.stem_id == k.stem_id)).all()
            assert {w.samples_per_px for w in waves} == set(DEFAULT_LEVELS)
            assert all(resolve_data_path(settings, w.peaks_path).is_file() for w in waves)


def test_view_orders_children_under_parent(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    drums = _stem_id(factory, full_job, "drums")
    job_id = _run_refine(settings, factory, drums, DRUMSEP)
    with factory() as s:
        view = build_view(s, full_job)
        codes = [t.code for _, t in view.rows]
        i = codes.index("drums")
        assert codes[i + 1 : i + 8] == [
            "kick", "snare", "toms", "hihat", "ride", "crash", "drums_rest",
        ]
        assert codes[:3] == ["vocals", "lead_vocal", "backing_vocal"]
        assert view.refined_by[drums].job_id == job_id
        assert len(codes) == len(set(codes))


def test_same_method_is_not_rerun_and_force_replaces(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    drums = _stem_id(factory, full_job, "drums")
    first = _run_refine(settings, factory, drums, DRUMSEP)
    with factory() as s:
        res = enqueue_refine_job(s, drums, DRUMSEP)
        assert not res.created and res.reason == "done" and res.job.job_id == first
        old_dir = s.get(SeparationJob, first).output_dir  # type: ignore[union-attr]
    assert old_dir is not None and resolve_data_path(settings, old_dir).is_dir()
    second = _run_refine(settings, factory, drums, DRUMSEP, force=True)
    with factory() as s:
        assert s.get(SeparationJob, first) is None  # 置き換えた古い結果は消える
        new = s.get(SeparationJob, second)
        assert new is not None and new.status == "done"
        assert new.output_dir == f"{old_dir} (2)"
        assert set(_children(s, drums)) == {
            "kick", "snare", "toms", "hihat", "ride", "crash", "drums_rest",
        }
        assert all(k.job_id == second for k in _children(s, drums).values())
    assert not resolve_data_path(settings, old_dir).exists()


def test_invalid_and_conflicting_requests(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    with factory() as s:
        with pytest.raises(RefineInvalid):  # 無い方法
            enqueue_refine_job(s, _stem_id(factory, full_job, "drums"), "nope.ckpt")
        with pytest.raises(RefineInvalid):  # 方法に合わない stem
            enqueue_refine_job(s, _stem_id(factory, full_job, "bass"), DRUMSEP)
        with pytest.raises(RefineInvalid):  # vocals は lead / backing に分かれている
            enqueue_refine_job(s, _stem_id(factory, full_job, "vocals"), MALE_FEMALE)
    lead = _stem_id(factory, full_job, "lead_vocal")
    backing = _stem_id(factory, full_job, "backing_vocal")
    _run_refine(settings, factory, lead, MALE_FEMALE)
    with factory() as s:
        # 男声・女声が木に2つできる分け方はできない
        with pytest.raises(RefineConflict, match="男声"):
            enqueue_refine_job(s, backing, MALE_FEMALE)
        # 別の方法で分けてある stem（force で置き換え）
        with pytest.raises(RefineConflict, match="別の方法"):
            enqueue_refine_job(s, lead, ASPIRATION)
        # 息は重ならないので backing に使える
        assert enqueue_refine_job(s, backing, ASPIRATION).created
        # 分割待ちのものと同じ stem に別の方法
        with pytest.raises(RefineConflict, match="分割待ち"):
            enqueue_refine_job(s, backing, MALE_FEMALE, force=True)
        # 子（refine の結果）はさらに分けられない（残りの型が無い）
        kick_like = _children(s, lead)["male"].stem_id
        with pytest.raises(RefineInvalid):
            enqueue_refine_job(s, kick_like, ASPIRATION)


def test_force_other_method_replaces(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    lead = _stem_id(factory, full_job, "lead_vocal")
    first = _run_refine(settings, factory, lead, MALE_FEMALE)
    _run_refine(settings, factory, lead, ASPIRATION, force=True)
    with factory() as s:
        assert s.get(SeparationJob, first) is None
        assert set(_children(s, lead)) == {"breath", "lead_vocal_rest"}


def test_refine_failure_cleans_up(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    drums = _stem_id(factory, full_job, "drums")
    with factory() as s:
        job_id = enqueue_refine_job(s, drums, DRUMSEP).job.job_id
    worker = Worker(
        settings, factory, _refine_launcher(settings, FakeSeparator(fail_models={DRUMSEP}))
    )
    assert worker.run_one() == job_id
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.status == "failed"
        assert "詳細分割に失敗しました" in (job.error_message or "")
        assert job.output_dir is None
        assert _children(s, drums) == {}
        full = s.get(SeparationJob, full_job)
        assert full is not None and full.output_dir is not None
        assert not (resolve_data_path(settings, full.output_dir) / "drums").exists()
        # 失敗したジョブの行は、次に登録したときに片付ける
        again = enqueue_refine_job(s, drums, DRUMSEP)
        assert again.created
        assert s.get(SeparationJob, job_id) is None


class CancelDuring(FakeSeparator):
    """分離の途中でキャンセルが依頼されたことにする。"""

    def __init__(self, factory: sessionmaker[Session], job_id: int) -> None:
        super().__init__()
        self.factory = factory
        self.job_id = job_id

    def separate(self, *args: Any, **kw: Any) -> dict[str, np.ndarray]:
        with self.factory() as s:
            request_cancel(s, self.job_id)
        return super().separate(*args, **kw)


def test_cancel_running_refine(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    drums = _stem_id(factory, full_job, "drums")
    with factory() as s:
        job_id = enqueue_refine_job(s, drums, DRUMSEP).job.job_id
        assert claim_next_job(s) == job_id
    rc = run_job(settings, job_id, CancelDuring(factory, job_id), encoder=fake_encoder)
    assert rc == EXIT_FAILED
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.status == "canceled"
        assert _children(s, drums) == {}
        full = s.get(SeparationJob, full_job)
        assert not (resolve_data_path(settings, full.output_dir) / "drums").exists()  # type: ignore[union-attr, arg-type]


def test_cancel_queued_refine(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    with factory() as s:
        job_id = enqueue_refine_job(s, _stem_id(factory, full_job, "drums"), DRUMSEP).job.job_id
        assert request_cancel(s, job_id).status == "canceled"


def _wait_for(cond: Callable[[], bool], timeout: float = 60.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("時間内に条件を満たしませんでした")


def test_cancel_kills_refine_child_process(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    """実際の子プロセス（--fake）で詳細分割を始め、キャンセルすると止まる。"""
    drums = _stem_id(factory, full_job, "drums")
    with factory() as s:
        job_id = enqueue_refine_job(s, drums, DRUMSEP).job.job_id
    launched: list[ChildHandle] = []
    real = subprocess_launcher(settings, ["--fake", "--fake-delay", "60"])

    def launch(jid: int) -> ChildHandle:
        child = real(jid)
        launched.append(child)
        return child

    worker = Worker(settings, factory, launch, cancel_check_interval=0.2)
    t = threading.Thread(target=worker.run_one)
    t.start()
    try:
        def stage() -> str:
            with factory() as s:
                return s.get(SeparationJob, job_id).stage or ""  # type: ignore[union-attr]

        _wait_for(lambda: stage().startswith("分離中"))
        with factory() as s:
            assert request_cancel(s, job_id).status == "running"
        t.join(30)
        assert not t.is_alive()
    finally:
        for c in launched:
            if c.poll() is None:
                c.kill()
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.status == "canceled"
        assert _children(s, drums) == {}


def test_delete_refine_and_full_job(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    drums = _stem_id(factory, full_job, "drums")
    other = _stem_id(factory, full_job, "other")
    drum_job = _run_refine(settings, factory, drums, DRUMSEP)
    other_job = _run_refine(settings, factory, other, HPSS)
    with factory() as s:
        drum_dir = resolve_data_path(settings, s.get(SeparationJob, drum_job).output_dir)  # type: ignore[union-attr, arg-type]
        full_dir = resolve_data_path(settings, s.get(SeparationJob, full_job).output_dir)  # type: ignore[union-attr, arg-type]
        # 子を消す（分ける前に戻す）
        delete_job(s, settings, drum_job)
    assert not drum_dir.exists() and full_dir.is_dir()
    with factory() as s:
        assert _children(s, drums) == {}
        view = build_view(s, full_job)
        assert "kick" not in {t.code for _, t in view.rows}
        assert "sustained" in {t.code for _, t in view.rows}
        # 分け方（full）を消すと、その stem を分けたジョブも消える
        delete_job(s, settings, full_job)
        assert s.get(SeparationJob, other_job) is None
    assert not full_dir.exists()


def test_enqueue_full_job_ignores_active_refine(
    seeded: Session, settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    with factory() as s:
        enqueue_refine_job(s, _stem_id(factory, full_job, "drums"), DRUMSEP)
        track_id = s.get(SeparationJob, full_job).track_id  # type: ignore[union-attr]
        res = enqueue_full_job(s, track_id, "standard")
        assert res.created and res.job.job_kind == "full"


def test_refine_job_on_worker_child_dispatch(
    settings: Settings, factory: sessionmaker[Session], full_job: int
) -> None:
    """run_job は refine のジョブを詳細分割として実行する（HPSS はテスト用の関数）。"""
    other = _stem_id(factory, full_job, "other")
    with factory() as s:
        job_id = enqueue_refine_job(s, other, HPSS).job.job_id
        assert claim_next_job(s) == job_id
    assert run_job(settings, job_id, FakeSeparator(), encoder=fake_encoder, hpss=fake_hpss) == (
        EXIT_DONE
    )


# --- API ------------------------------------------------------------------------------------------


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(_app(settings)) as c:
        yield c


def _cfactory(client: TestClient) -> sessionmaker[Session]:
    return client.app.state.session_factory  # type: ignore[attr-defined]


@pytest.fixture
def api_job(client: TestClient, tmp_path: Path) -> tuple[int, int]:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _cfactory(client)() as s:
        track_id = make_track(s, settings, tmp_path)
    job_id = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"}).json()["job"][
        "job_id"
    ]
    assert _worker(settings, _cfactory(client)).run_one() == job_id
    return track_id, job_id


def _stems(client: TestClient, job_id: int) -> dict[str, dict[str, Any]]:
    return {s["code"]: s for s in client.get(f"/api/jobs/{job_id}/stems").json()["stems"]}


def _run_worker(client: TestClient) -> None:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    Worker(settings, _cfactory(client), _refine_launcher(settings)).run_one()


def test_api_refine_flow(client: TestClient, api_job: tuple[int, int]) -> None:
    track_id, job_id = api_job
    stems = _stems(client, job_id)
    # 分けられる stem と方法
    assert [m["model"] for m in stems["drums"]["refine_methods"]] == [DRUMSEP]
    assert {m["model"] for m in stems["lead_vocal"]["refine_methods"]} == {MALE_FEMALE, ASPIRATION}
    assert [m["model"] for m in stems["other"]["refine_methods"]] == [HPSS]
    assert stems["vocals"]["refine_methods"] == [] and stems["bass"]["refine_methods"] == []
    drum_method = stems["drums"]["refine_methods"][0]
    assert drum_method["available"] is True and drum_method["gpu"] is True
    assert [c["code"] for c in drum_method["children"]][-1] == "drums_rest"
    assert drum_method["children"][-1]["display_name"] == "残り（ドラム）"

    drums_id = stems["drums"]["stem_id"]
    res = client.post(f"/api/stems/{drums_id}/refine", json={"model": DRUMSEP})
    assert res.status_code == 201, res.text
    refine_id = res.json()["job"]["job_id"]
    assert res.json()["job"]["job_kind"] == "refine"
    assert res.json()["job"]["input_stem_id"] == drums_id
    # 分割待ち: 同じ依頼は 200、画面には進み具合、一覧の状態には出さない
    again = client.post(f"/api/stems/{drums_id}/refine", json={"model": DRUMSEP})
    assert again.status_code == 200 and again.json()["reason"] == "active"
    body = client.get(f"/api/jobs/{job_id}/stems").json()
    assert [j["job_id"] for j in body["refine_jobs"]] == [refine_id]
    assert body["refine_jobs"][0]["input_code"] == "drums"
    assert body["refine_jobs"][0]["model"] == DRUMSEP
    assert _stems(client, job_id)["drums"]["refine_methods"][0]["available"] is False
    track = next(t for t in client.get("/api/tracks").json()["tracks"] if t["track_id"] == track_id)
    assert track["latest_job"]["job_id"] == job_id and track["active_job"] is None

    _run_worker(client)
    body = client.get(f"/api/jobs/{job_id}/stems").json()
    assert body["refine_jobs"] == [] and body["delivery_ready"] is True
    codes = [s["code"] for s in body["stems"]]
    i = codes.index("drums")
    assert codes[i + 1 : i + 8] == ["kick", "snare", "toms", "hihat", "ride", "crash", "drums_rest"]
    stems = {s["code"]: s for s in body["stems"]}
    assert stems["kick"]["parent_code"] == "drums" and stems["kick"]["job_id"] == refine_id
    assert stems["drums_rest"]["is_residual"] is True
    assert {r["purpose"] for r in stems["kick"]["renditions"]} == {"master", "stream"}
    assert len(stems["kick"]["peaks"]) == len(DEFAULT_LEVELS)
    assert stems["drums"]["refined_by"] == {
        "job_id": refine_id, "model": DRUMSEP, "display_name": "MDX23C ドラム分割",
    }
    assert stems["drums"]["refine_methods"] == []  # 子を持つ stem には出さない
    # 同じ方法は分け直さない、force なら登録する
    done = client.post(f"/api/stems/{drums_id}/refine", json={"model": DRUMSEP})
    assert done.status_code == 200 and done.json()["reason"] == "done"

    # 子の音は配信・書き出しできる
    kick_stream = next(r for r in stems["kick"]["renditions"] if r["purpose"] == "stream")
    assert client.get(kick_stream["url"]).status_code == 200

    # 子を削除（分ける前に戻す）
    assert client.delete(f"/api/jobs/{refine_id}").status_code == 200
    stems = _stems(client, job_id)
    assert "kick" not in stems and stems["drums"]["refined_by"] is None
    assert stems["drums"]["refine_methods"][0]["available"] is True


def test_api_refine_errors(client: TestClient, api_job: tuple[int, int]) -> None:
    _, job_id = api_job
    stems = _stems(client, job_id)
    assert client.post("/api/stems/99999/refine", json={"model": DRUMSEP}).status_code == 404
    bad = client.post(f"/api/stems/{stems['bass']['stem_id']}/refine", json={"model": DRUMSEP})
    assert bad.status_code == 400 and "分けられません" in bad.json()["detail"]
    nope = client.post(f"/api/stems/{stems['drums']['stem_id']}/refine", json={"model": "x"})
    assert nope.status_code == 400
    lead, backing = stems["lead_vocal"]["stem_id"], stems["backing_vocal"]["stem_id"]
    assert client.post(f"/api/stems/{lead}/refine", json={"model": MALE_FEMALE}).status_code == 201
    _run_worker(client)
    stems = _stems(client, job_id)
    mf = next(m for m in stems["backing_vocal"]["refine_methods"] if m["model"] == MALE_FEMALE)
    assert mf["available"] is False and "男声" in mf["reason"]
    conflict = client.post(f"/api/stems/{backing}/refine", json={"model": MALE_FEMALE})
    assert conflict.status_code == 409


def test_api_failed_refine_is_reported(client: TestClient, api_job: tuple[int, int]) -> None:
    _, job_id = api_job
    drums = _stems(client, job_id)["drums"]["stem_id"]
    client.post(f"/api/stems/{drums}/refine", json={"model": DRUMSEP})
    settings = client.app.state.settings  # type: ignore[attr-defined]
    Worker(
        settings, _cfactory(client),
        _refine_launcher(settings, FakeSeparator(fail_models={DRUMSEP})),
    ).run_one()
    body = client.get(f"/api/jobs/{job_id}/stems").json()
    assert [j["status"] for j in body["refine_jobs"]] == ["failed"]
    assert body["refine_jobs"][0]["error_message"]


def test_exports_use_refined_children(client: TestClient, api_job: tuple[int, int]) -> None:
    """書き出しは子を含む木を使う。子を指定した組み合わせは、子の無い曲では親として扱う。"""
    from stemapp.exports.service import ExportRequest, plan_export

    _, job_id = api_job
    stems = _stems(client, job_id)
    listed = client.get("/api/stem-types").json()["stem_types"]
    types = {t["code"]: t["stem_type_id"] for t in listed}
    with _cfactory(client)() as s:
        preset = ListenPreset(name="キックだけ", sort_order=999)
        s.add(preset)
        s.flush()
        s.add(ListenPresetItem(
            listen_preset_id=preset.listen_preset_id, stem_type_id=types["kick"], gain_db=-3.0,
        ))
        s.commit()
        preset_id = preset.listen_preset_id
        # 子の無い曲: kick → drums
        plan = plan_export(s, job_id, ExportRequest("mix", "wav", listen_preset_id=preset_id))
        assert [(i.stem.code, i.gain_db) for i in plan.items] == [("drums", -3.0)]
    client.post(f"/api/stems/{stems['drums']['stem_id']}/refine", json={"model": DRUMSEP})
    _run_worker(client)
    with _cfactory(client)() as s:
        plan = plan_export(s, job_id, ExportRequest("mix", "wav", listen_preset_id=preset_id))
        assert [i.stem.code for i in plan.items] == ["kick"]
        single = plan_export(s, job_id, ExportRequest("single", "wav", stem_code="snare"))
        assert single.items[0].stem.code == "snare"
        everything = plan_export(s, job_id, ExportRequest("all", "wav"))
        codes = {i.stem.code for i in everything.items}
        assert "kick" in codes and "drums" not in codes
