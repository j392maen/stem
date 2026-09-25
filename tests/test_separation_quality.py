"""T12: 分離品質の聴き比べ（残差の行き先、karaoke の入力、実験プリセット、同じ曲の複数ジョブ）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from audio_helpers import synth_mix
from job_helpers import make_track
from stemapp.config import Settings
from stemapp.jobs import JobConflict, JobNotFound, delete_job, enqueue_full_job
from stemapp.models import Model, PresetStep, SeparationJob, SeparationPreset, Stem
from stemapp.seed import (
    EXPERIMENTAL_PRESETS,
    KARAOKE_BS_ANVUEW,
    KARAOKE_BS_FRAZER,
    KIM_VOCALS,
    SW,
    seed,
)
from stemapp.separation import FakeSeparator
from stemapp.separation.pipeline import (
    PresetPlan,
    SeparationError,
    StepSpec,
    load_plan,
    run_plan,
    separate_track,
)

TOP = ["vocals", "drums", "bass", "guitar", "piano", "other"]
ATOL = 1e-5


@pytest.fixture
def seeded(session: Session) -> Session:
    seed(session)
    return session


@pytest.fixture
def mix() -> np.ndarray:
    return synth_mix(2.0)


def _plan(
    route: str | None, karaoke_input: str = "vocals", karaokes: int = 1, kim: bool | None = None
) -> PresetPlan:
    """kim が None なら、残差の行き先が other（既定）のときだけ Kim を入れる。"""
    if kim is None:
        kim = route in (None, "other")
    kara = [KARAOKE_BS_FRAZER, KARAOKE_BS_ANVUEW][:karaokes]
    steps = [StepSpec(1, SW, "SW", "mixture", "multistem")]
    if kim:
        steps.append(StepSpec(2, KIM_VOCALS, "Kim", "mixture", "vocals"))
    steps += [StepSpec(3 + i, m, m, karaoke_input, "karaoke") for i, m in enumerate(kara)]
    options = {"residual_to": route} if route else {}
    return PresetPlan("t", "テスト", steps, options=options)


def _check_sums(out, mix: np.ndarray) -> None:  # type: ignore[no-untyped-def]
    top_sum = sum(out.stems[k].astype(np.float64) for k in TOP)
    assert np.max(np.abs(top_sum - mix)) < ATOL
    lb = out.stems["lead_vocal"].astype(np.float64) + out.stems["backing_vocal"]
    assert np.max(np.abs(lb - out.stems["vocals"])) < ATOL


# Fake: SW の係数の合計 0.97（vocals 0.30, other 0.12）、Kim vocals 0.40、karaoke の lead 0.70。
# standard 相当（Kim あり）では vocals の平均 0.35 → 上位 stem の合計 1.02 → 残差 −0.02。
# SW だけなら残差 +0.03。
@pytest.mark.parametrize(
    ("route", "vocals", "other", "resid"),
    [
        (None, 0.35, 0.10, 0.02),  # 既定（other）＋ Kim: 今までどおり
        ("other", 0.35, 0.10, 0.02),
        ("vocals", 0.33, 0.12, 0.03),  # vocals = 元の曲 − 楽器 = 1 − 0.67
        # split（SW だけ）: ボーカルの平均による差は 0 なので other と同じ
        ("split", 0.30, 0.15, 0.03),
    ],
)
def test_residual_routes_keep_sums(
    tmp_path: Path, mix: np.ndarray, route: str | None, vocals: float, other: float, resid: float
) -> None:
    out = run_plan(mix, _plan(route), FakeSeparator(), workdir=tmp_path)
    _check_sums(out, mix)
    np.testing.assert_allclose(out.stems["vocals"], mix * vocals, atol=ATOL)
    np.testing.assert_allclose(out.stems["other"], mix * other, atol=ATOL)
    np.testing.assert_allclose(out.stems["drums"], mix * 0.20, atol=ATOL)
    # karaoke は残差を足した後の vocals にかかる
    np.testing.assert_allclose(out.stems["lead_vocal"], mix * vocals * 0.70, atol=ATOL)
    # 補正前の残差（行き先によらず同じ）
    assert out.residual_rms_db == pytest.approx(
        20 * np.log10(np.sqrt(np.mean((mix.astype(np.float64) * resid) ** 2))), abs=0.01
    )


@pytest.mark.parametrize("route", ["other", "vocals", "split"])
def test_karaoke_on_mixture(tmp_path: Path, mix: np.ndarray, route: str) -> None:
    sep = FakeSeparator(model_coefs={KARAOKE_BS_FRAZER: 0.20, KARAOKE_BS_ANVUEW: 0.30})
    out = run_plan(mix, _plan(route, "mixture", karaokes=2), sep, workdir=tmp_path)
    _check_sums(out, mix)
    # lead = 元の曲にかけた karaoke の平均、backing = vocals − lead
    np.testing.assert_allclose(out.stems["lead_vocal"], mix * 0.25, atol=ATOL)
    np.testing.assert_allclose(
        out.stems["backing_vocal"],
        out.stems["vocals"].astype(np.float64) - mix * 0.25,
        atol=ATOL,
    )
    # karaoke も元の曲の段階（vocals が決まる前）に動く
    kim = ["vocals"] if route == "other" else []
    assert [c.role for c in sep.calls] == ["multistem", *kim, "karaoke", "karaoke"]


@pytest.mark.parametrize("route", ["vocals", "split"])
def test_vocals_model_is_rejected_when_it_cannot_matter(
    tmp_path: Path, mix: np.ndarray, route: str
) -> None:
    """残差を vocals / split に戻すと Kim の出力は結果に効かない（数式で確かめてから弾く）。"""
    sep = FakeSeparator(model_coefs={KIM_VOCALS: 0.40})
    with pytest.raises(SeparationError, match="結果に効きません"):
        run_plan(mix, _plan(route, kim=True), sep, workdir=tmp_path)
    # Kim の係数を変えても、Kim なしと同じ結果になる（だから弾く）ことの確認
    from stemapp.separation.pipeline import validate_plan

    outs = []
    for coef in (0.40, 0.90):
        plan = _plan(route, kim=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("stemapp.separation.pipeline.validate_plan", lambda _p: None)
            outs.append(run_plan(mix, plan, FakeSeparator(model_coefs={KIM_VOCALS: coef}),
                                 workdir=tmp_path / str(coef)))
    for code in outs[0].stems:
        np.testing.assert_allclose(outs[0].stems[code], outs[1].stems[code], atol=ATOL)
    validate_plan(_plan(route))  # Kim なしなら通る


def test_plan_validation_rejects_bad_combinations(tmp_path: Path, mix: np.ndarray) -> None:
    bad_route = _plan("drums")
    with pytest.raises(SeparationError, match="residual_to"):
        run_plan(mix, bad_route, FakeSeparator(), workdir=tmp_path)
    typo = PresetPlan("t", "t", _plan(None).steps, options={"residual": "vocals"})
    with pytest.raises(SeparationError, match="residual"):
        run_plan(mix, typo, FakeSeparator(), workdir=tmp_path)
    mixed = PresetPlan(
        "t",
        "t",
        [
            StepSpec(1, SW, "SW", "mixture", "multistem"),
            StepSpec(2, KARAOKE_BS_FRAZER, "k1", "mixture", "karaoke"),
            StepSpec(3, KARAOKE_BS_ANVUEW, "k2", "vocals", "karaoke"),
        ],
    )
    with pytest.raises(SeparationError, match="混ぜられません"):
        run_plan(mix, mixed, FakeSeparator(), workdir=tmp_path)
    vocals_on_vocals = PresetPlan(
        "t",
        "t",
        [
            StepSpec(1, SW, "SW", "mixture", "multistem"),
            StepSpec(2, KIM_VOCALS, "Kim", "vocals", "vocals"),
            StepSpec(3, KARAOKE_BS_FRAZER, "k", "vocals", "karaoke"),
        ],
    )
    with pytest.raises(SeparationError, match="input"):
        run_plan(mix, vocals_on_vocals, FakeSeparator(), workdir=tmp_path)


def test_bad_preset_options_in_db_are_rejected(seeded: Session) -> None:
    p = seeded.scalars(select(SeparationPreset).where(SeparationPreset.code == "fast")).one()
    p.options_json = {"residual_to": "bass"}
    seeded.commit()
    with pytest.raises(SeparationError, match="residual_to"):
        load_plan(seeded, "fast")


@pytest.mark.parametrize("code", [d.code for d in EXPERIMENTAL_PRESETS])
def test_experimental_presets_run(
    seeded: Session, tmp_path: Path, mix: np.ndarray, code: str
) -> None:
    plan = load_plan(seeded, code)
    out = run_plan(mix, plan, FakeSeparator(), workdir=tmp_path)
    _check_sums(out, mix)


def test_experimental_preset_contents(seeded: Session) -> None:
    def desc(code: str) -> tuple[str, list[tuple[str, str, str]]]:
        plan = load_plan(seeded, code)
        return plan.residual_to, [(s.model_filename, s.input, s.role) for s in plan.steps]

    sw = [(SW, "mixture", "multistem")]
    base = [*sw, (KIM_VOCALS, "mixture", "vocals")]
    assert desc("exp_resid_vocals") == ("vocals", [*sw, (KARAOKE_BS_FRAZER, "vocals", "karaoke")])
    # 対照: SW だけ（Kim なし）
    assert desc("exp_resid_split") == ("other", [*sw, (KARAOKE_BS_FRAZER, "vocals", "karaoke")])
    assert desc("exp_kara_mix") == ("other", [*base, (KARAOKE_BS_FRAZER, "mixture", "karaoke")])
    assert desc("exp_kara_anvuew") == (
        "other",
        [*base, (KARAOKE_BS_ANVUEW, "vocals", "karaoke"), (KARAOKE_BS_FRAZER, "vocals", "karaoke")],
    )
    assert desc("exp_combo") == (
        "vocals",
        [
            *sw,
            (KARAOKE_BS_ANVUEW, "mixture", "karaoke"),
            (KARAOKE_BS_FRAZER, "mixture", "karaoke"),
        ],
    )
    assert desc("exp_combo_gabox")[1][1:] == [
        (KARAOKE_BS_ANVUEW, "mixture", "karaoke"),
        (KARAOKE_BS_FRAZER, "mixture", "karaoke"),
        ("mel_band_roformer_karaoke_gabox_v2.ckpt", "mixture", "karaoke"),
    ]
    # 既定は standard のまま
    assert load_plan(seeded).code == "standard"
    rows = seeded.scalars(select(SeparationPreset)).all()
    assert all(r.is_experimental == r.code.startswith("exp_") for r in rows)


# --- 同じ曲の複数ジョブ -------------------------------------------------------------------


def _separate(session: Session, settings: Settings, track_id: int, preset: str | None):  # type: ignore[no-untyped-def]
    return separate_track(session, settings, track_id, FakeSeparator(), preset_code=preset)


def test_separate_track_records_levels(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    track_id = make_track(seeded, settings, tmp_path)
    res = _separate(seeded, settings, track_id, "exp_resid_vocals")
    job = seeded.get(SeparationJob, res.job_id)
    assert job is not None and job.residual_rms_db is not None and job.mixture_rms_db is not None
    assert job.residual_rms_db < job.mixture_rms_db


def test_same_track_can_have_jobs_per_preset(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    track_id = make_track(seeded, settings, tmp_path)
    first = _separate(seeded, settings, track_id, "fast")
    # 別のプリセットなら force なしで分割する
    second = _separate(seeded, settings, track_id, "exp_kara_mix")
    assert second.skipped is False and second.job_id != first.job_id
    # 同じプリセットなら分割しない
    again = _separate(seeded, settings, track_id, "exp_kara_mix")
    assert again.skipped is True and again.job_id == second.job_id
    # プリセットを指定しないときは、どれかが分割済みなら分割しない
    default = _separate(seeded, settings, track_id, None)
    assert default.skipped is True


def test_enqueue_per_preset(seeded: Session, settings: Settings, tmp_path: Path) -> None:
    track_id = make_track(seeded, settings, tmp_path)
    a = enqueue_full_job(seeded, track_id, "fast")
    assert a.created
    # 別のプリセットは、分割待ちがあっても登録できる
    b = enqueue_full_job(seeded, track_id, "exp_combo")
    assert b.created and b.job.job_id != a.job.job_id
    # 同じプリセットは分割待ちのものを返す
    again = enqueue_full_job(seeded, track_id, "exp_combo", force=True)
    assert again.created is False and again.reason == "active"
    # プリセットを指定しないときは、どれかが分割待ちならそれを返す
    none = enqueue_full_job(seeded, track_id, None)
    assert none.created is False and none.reason == "active"

    # 分割済みになったら、同じプリセットは done、別のプリセットは登録できる
    for job in (a.job, b.job):
        job.status = "done"
    seeded.commit()
    done = enqueue_full_job(seeded, track_id, "exp_combo")
    assert done.created is False and done.reason == "done" and done.job.job_id == b.job.job_id
    assert enqueue_full_job(seeded, track_id, None).reason == "done"
    assert enqueue_full_job(seeded, track_id, "exp_kara_mix").created is True


def test_delete_job(seeded: Session, settings: Settings, tmp_path: Path) -> None:
    track_id = make_track(seeded, settings, tmp_path)
    keep = _separate(seeded, settings, track_id, "fast")
    gone = _separate(seeded, settings, track_id, "exp_resid_vocals")
    assert (settings.stems_dir / str(gone.job_id)).is_dir()

    deleted = delete_job(seeded, settings, gone.job_id)
    assert deleted.job_id == gone.job_id and deleted.track_id == track_id
    assert seeded.get(SeparationJob, gone.job_id) is None
    assert seeded.scalars(select(Stem).where(Stem.job_id == gone.job_id)).all() == []
    assert not (settings.stems_dir / str(gone.job_id)).exists()
    # ほかの分け方は残る
    assert seeded.get(SeparationJob, keep.job_id) is not None
    assert (settings.stems_dir / str(keep.job_id)).is_dir()
    assert seeded.scalars(select(Stem).where(Stem.job_id == keep.job_id)).all()

    with pytest.raises(JobNotFound):
        delete_job(seeded, settings, gone.job_id)
    # 分割待ち・配信用データの作成中は消せない
    queued = enqueue_full_job(seeded, track_id, "exp_combo").job
    with pytest.raises(JobConflict, match="キャンセル"):
        delete_job(seeded, settings, queued.job_id)
    job = seeded.get(SeparationJob, keep.job_id)
    assert job is not None
    job.postprocess_status = "running"
    seeded.commit()
    with pytest.raises(JobConflict, match="配信用データ"):
        delete_job(seeded, settings, keep.job_id)
    assert seeded.get(SeparationJob, keep.job_id) is not None


def test_reseed_updates_experimental_presets(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    """古い定義（Kim あり・古い名前）のまま作ったジョブも、seed し直すと新しい名前で表示される。"""
    track_id = make_track(seeded, settings, tmp_path)
    res = _separate(seeded, settings, track_id, "exp_combo")
    p = seeded.scalars(select(SeparationPreset).where(SeparationPreset.code == "exp_combo")).one()
    # T12 の最初の版の定義に戻す
    p.display_name = "残差ボーカル＋カラオケ2種を元の曲に"
    kim = seeded.scalars(select(Model).where(Model.filename == KIM_VOCALS)).one()
    steps = seeded.scalars(
        select(PresetStep).where(PresetStep.preset_id == p.preset_id)
    ).all()
    seeded.add(PresetStep(preset_id=p.preset_id, step_order=len(steps) + 1, model_id=kim.model_id,
                          input="mixture", role="vocals", ensemble_weight=1.0, options_json={}))
    seeded.commit()
    with pytest.raises(SeparationError):
        load_plan(seeded, "exp_combo")

    seed(seeded)
    seeded.expire_all()
    plan = load_plan(seeded, "exp_combo")
    assert plan.display_name == "ボーカル＝元の曲−楽器＋カラオケ2種を元の曲に"
    assert KIM_VOCALS not in [s.model_filename for s in plan.steps]
    # 既存ジョブは同じプリセット行を指すので、表示名も新しくなる
    from stemapp.api.common import job_to_dict, preset_codes

    job = seeded.get(SeparationJob, res.job_id)
    assert job is not None
    d = job_to_dict(job, preset_codes(seeded))
    assert d["preset"] == "exp_combo" and d["preset_experimental"] is True
    assert d["preset_name"] == "ボーカル＝元の曲−楽器＋カラオケ2種を元の曲に"
