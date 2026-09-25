from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import select
from sqlalchemy.orm import Session

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from stemapp.config import Settings
from stemapp.models import (
    SeparationJob,
    SeparationPreset,
    Stem,
    StemRendition,
    StemType,
    Track,
)
from stemapp.seed import (
    KARAOKE_BS_FRAZER,
    KARAOKE_MEL_BECRUILY,
    KIM_FT_UNWA,
    KIM_VOCALS,
    SW,
    seed,
)
from stemapp.separation import FakeSeparator, is_oom_error, pipeline
from stemapp.separation.audio_separator_backend import map_outputs, tta_combine
from stemapp.separation.base import DEVICE_CPU, DEVICE_CUDA, OPT_CHUNK_SCALE
from stemapp.separation.pipeline import (
    OomPolicy,
    PresetPlan,
    SeparationError,
    StepSpec,
    load_plan,
    resolve_data_path,
    run_plan,
    separate_file,
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


@pytest.fixture
def src(tmp_path: Path, mix: np.ndarray) -> Path:
    return write_source(tmp_path / "src" / "song.wav", mix)


def _run(
    session: Session,
    settings: Settings,
    src: Path,
    sep: FakeSeparator,
    **kw: object,
):
    return separate_file(session, settings, src, sep, ffmpeg_runner=fake_ffmpeg, **kw)  # type: ignore[arg-type]


# --- 計算部分 ----------------------------------------------------------------------


@pytest.mark.parametrize("code", ["fast", "standard", "best"])
def test_sums_match_for_each_preset(
    seeded: Session, tmp_path: Path, mix: np.ndarray, code: str
) -> None:
    plan = load_plan(seeded, code)
    out = run_plan(mix, plan, FakeSeparator(), workdir=tmp_path / "w")
    assert out.top_level == TOP
    assert set(out.stems) == set(TOP) | {"lead_vocal", "backing_vocal"}
    top_sum = sum(out.stems[k].astype(np.float64) for k in TOP)
    assert np.max(np.abs(top_sum - mix)) < ATOL
    lb = out.stems["lead_vocal"].astype(np.float64) + out.stems["backing_vocal"]
    assert np.max(np.abs(lb - out.stems["vocals"])) < ATOL
    for arr in out.stems.values():
        assert arr.dtype == np.float32 and arr.shape == mix.shape


def test_residual_goes_to_other(seeded: Session, tmp_path: Path, mix: np.ndarray) -> None:
    # Fake の multistem 係数の合計は 0.97。足りない 0.03 が other に足される。
    out = run_plan(mix, load_plan(seeded, "fast"), FakeSeparator(), workdir=tmp_path)
    np.testing.assert_allclose(out.stems["other"], mix * (0.12 + 0.03), atol=ATOL)
    np.testing.assert_allclose(out.stems["drums"], mix * 0.20, atol=ATOL)


def test_standard_averages_sw_and_kim(seeded: Session, tmp_path: Path, mix: np.ndarray) -> None:
    out = run_plan(mix, load_plan(seeded, "standard"), FakeSeparator(), workdir=tmp_path)
    # SW vocals 0.30 と Kim vocals 0.40 の等重み平均
    np.testing.assert_allclose(out.stems["vocals"], mix * 0.35, atol=ATOL)
    np.testing.assert_allclose(out.stems["lead_vocal"], mix * 0.35 * 0.70, atol=ATOL)


def test_weighted_ensemble(tmp_path: Path, mix: np.ndarray) -> None:
    plan = PresetPlan(
        "custom",
        "テスト",
        [
            StepSpec(1, SW, "SW", "mixture", "multistem", weight=1.0),
            StepSpec(2, KIM_VOCALS, "Kim", "mixture", "vocals", weight=3.0),
            StepSpec(3, KIM_FT_UNWA, "unwa", "mixture", "vocals", weight=4.0),
            # karaoke は vocals が決まってから動く（step_order が先でも）
            StepSpec(0, KARAOKE_MEL_BECRUILY, "k1", "vocals", "karaoke", weight=1.0),
            StepSpec(5, KARAOKE_BS_FRAZER, "k2", "vocals", "karaoke", weight=3.0),
        ],
    )
    sep = FakeSeparator(
        model_coefs={
            KIM_VOCALS: 0.40,
            KIM_FT_UNWA: 0.50,
            KARAOKE_MEL_BECRUILY: 0.60,
            KARAOKE_BS_FRAZER: 0.80,
        }
    )
    out = run_plan(mix, plan, sep, workdir=tmp_path)
    vocals_coef = (1 * 0.30 + 3 * 0.40 + 4 * 0.50) / 8
    lead_coef = (1 * 0.60 + 3 * 0.80) / 4
    np.testing.assert_allclose(out.stems["vocals"], mix * vocals_coef, atol=ATOL)
    np.testing.assert_allclose(out.stems["lead_vocal"], mix * vocals_coef * lead_coef, atol=ATOL)
    np.testing.assert_allclose(
        out.stems["backing_vocal"], mix * vocals_coef * (1 - lead_coef), atol=ATOL
    )
    assert [c.model_filename for c in sep.calls][-2:] == [KARAOKE_MEL_BECRUILY, KARAOKE_BS_FRAZER]
    assert [c.role for c in sep.calls] == [
        "multistem", "vocals", "vocals", "karaoke", "karaoke",
    ]


def test_steps_come_from_db(seeded: Session, tmp_path: Path, mix: np.ndarray) -> None:
    # PRESET_STEP を書き換えると手順が変わる（コードに直書きしていない）
    preset = seeded.scalars(select(SeparationPreset).where(SeparationPreset.code == "fast")).one()
    plan = load_plan(seeded, "fast")
    assert plan.preset_id == preset.preset_id
    assert [s.model_filename for s in plan.steps] == [SW, KARAOKE_MEL_BECRUILY]
    assert plan.steps[0].options == {"overlap": 2}
    sep = FakeSeparator()
    run_plan(mix, plan, sep, workdir=tmp_path)
    assert sep.calls[0].options == {"overlap": 2}
    assert load_plan(seeded).code == "standard"  # 既定
    with pytest.raises(SeparationError):
        load_plan(seeded, "nothing")


def test_plan_validation(tmp_path: Path, mix: np.ndarray) -> None:
    bad = PresetPlan("x", "x", [StepSpec(1, SW, "SW", "mixture", "multistem")])
    with pytest.raises(SeparationError, match="karaoke"):
        run_plan(mix, bad, FakeSeparator(), workdir=tmp_path)
    bad2 = PresetPlan(
        "x",
        "x",
        [
            StepSpec(1, SW, "SW", "mixture", "multistem"),
            StepSpec(2, KARAOKE_BS_FRAZER, "k", "stems", "karaoke"),
        ],
    )
    with pytest.raises(SeparationError, match="input"):
        run_plan(mix, bad2, FakeSeparator(), workdir=tmp_path)


# --- OOM ------------------------------------------------------------------------


def test_is_oom_error() -> None:
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))

    class OutOfMemoryError(RuntimeError):
        pass

    assert is_oom_error(OutOfMemoryError("x"))
    assert not is_oom_error(RuntimeError("other"))
    assert not is_oom_error(ValueError("out of memory"))


def test_oom_shrinks_chunk_then_succeeds(
    seeded: Session, tmp_path: Path, mix: np.ndarray
) -> None:
    sep = FakeSeparator(oom_times=1)
    out = run_plan(mix, load_plan(seeded, "fast"), sep, workdir=tmp_path)
    first = [c for c in sep.calls if c.model_filename == SW]
    assert [c.device for c in first] == [DEVICE_CUDA, DEVICE_CUDA]
    assert OPT_CHUNK_SCALE not in first[0].options
    assert first[1].options[OPT_CHUNK_SCALE] == 0.5
    assert out.steps[0].chunk_scale == 0.5 and out.steps[0].device == DEVICE_CUDA
    assert len(out.steps[0].attempts) == 1
    # 次のステップは元のチャンクで GPU
    assert sep.calls[-1].device == DEVICE_CUDA and OPT_CHUNK_SCALE not in sep.calls[-1].options


def test_oom_falls_back_to_cpu(
    seeded: Session, settings: Settings, src: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sep = FakeSeparator(oom_times=3)
    with caplog.at_level("WARNING"):
        res = _run(seeded, settings, src, sep, preset_code="fast", oom_policy=OomPolicy(2, 0.5))
    sw_calls = [(c.device, c.options.get(OPT_CHUNK_SCALE, 1.0)) for c in sep.calls[:4]]
    assert sw_calls == [
        (DEVICE_CUDA, 1.0),
        (DEVICE_CUDA, 0.5),
        (DEVICE_CUDA, 0.25),
        (DEVICE_CPU, 1.0),
    ]
    assert res.steps[0].device == DEVICE_CPU
    assert len(res.steps[0].attempts) == 3
    assert "CPU で実行します" in caplog.text
    job = seeded.get(SeparationJob, res.job_id)
    assert job is not None and job.status == "done" and job.run_on == "cpu"


def test_oom_on_cpu_is_not_retried(seeded: Session, tmp_path: Path, mix: np.ndarray) -> None:
    sep = FakeSeparator(oom_times=5, fail_models=())
    out = run_plan(mix, load_plan(seeded, "fast"), sep, workdir=tmp_path, device=DEVICE_CPU)
    assert all(c.device == DEVICE_CPU for c in sep.calls)
    assert len(sep.calls) == 2 and out.steps[0].attempts == []


def test_oom_without_cpu_fallback_raises(
    seeded: Session, tmp_path: Path, mix: np.ndarray
) -> None:
    sep = FakeSeparator(oom_times=10)
    with pytest.raises(RuntimeError, match="out of memory"):
        run_plan(
            mix,
            load_plan(seeded, "fast"),
            sep,
            workdir=tmp_path,
            oom_policy=OomPolicy(max_shrinks=1, cpu_fallback=False),
        )
    assert len(sep.calls) == 2


# --- DB とファイル -------------------------------------------------------------------


def test_separate_file_registers_everything(
    seeded: Session, settings: Settings, src: Path, mix: np.ndarray
) -> None:
    progress: list[tuple[float, str]] = []
    res = _run(
        seeded, settings, src, FakeSeparator(), preset_code="standard",
        progress=lambda p, s: progress.append((p, s)),
    )
    assert not res.skipped

    track = seeded.get(Track, res.track_id)
    assert track is not None and track.title == "song"
    assert track.duration_sec == pytest.approx(2.0)
    assert track.normalized_path == f"tracks/{track.track_id}/normalized.wav"
    normalized = resolve_data_path(settings, track.normalized_path)
    assert normalized.is_file()

    job = seeded.get(SeparationJob, res.job_id)
    assert job is not None
    assert (job.job_kind, job.status, job.run_on) == ("full", "done", "gpu")
    assert job.progress == 1.0 and job.stage == "完了"
    assert job.started_at is not None and job.finished_at is not None
    assert job.error_message is None
    preset = seeded.get(SeparationPreset, job.preset_id)
    assert preset is not None and preset.code == "standard"

    rows = seeded.execute(
        select(Stem, StemType).join(StemType).where(Stem.job_id == job.job_id)
    ).all()
    by_code = {t.code: s for s, t in rows}
    assert set(by_code) == set(TOP) | {"lead_vocal", "backing_vocal"}
    for code in TOP:
        assert by_code[code].parent_stem_id is None
        assert by_code[code].is_residual is False
    vocals_id = by_code["vocals"].stem_id
    assert by_code["lead_vocal"].parent_stem_id == vocals_id
    assert by_code["backing_vocal"].parent_stem_id == vocals_id
    assert by_code["lead_vocal"].is_residual is False
    assert by_code["backing_vocal"].is_residual is True

    audio: dict[str, np.ndarray] = {}
    for code, stem in by_code.items():
        rends = seeded.scalars(select(StemRendition).where(StemRendition.stem_id == stem.stem_id))
        (rend,) = list(rends)
        assert (rend.purpose, rend.codec) == ("master", "flac")
        assert rend.file_path == f"stems/{job.job_id}/{code}.flac"
        path = resolve_data_path(settings, rend.file_path)
        assert rend.bytes == path.stat().st_size
        info = sf.info(str(path))
        assert (info.subtype, info.samplerate, info.channels) == ("PCM_24", 44100, 2)
        audio[code] = sf.read(str(path), dtype="float64")[0]
        assert stem.rms_db is not None and stem.is_silent is False

    # 24bit の量子化誤差（1LSB ≒ 1.2e-7）の範囲で合計が一致
    top_sum = sum(audio[c] for c in TOP)
    assert np.max(np.abs(top_sum - mix)) < 2e-6
    lb = audio["lead_vocal"] + audio["backing_vocal"]
    assert np.max(np.abs(lb - audio["vocals"])) < 2e-6

    # 進捗は増える一方で 1.0 で終わる
    ps = [p for p, _ in progress]
    assert ps == sorted(ps) and ps[-1] == 1.0
    assert any("分離中" in s for _, s in progress)
    assert [i.code for i in res.stems][0] == "vocals"


def test_silent_stem_is_flagged(seeded: Session, settings: Settings, src: Path) -> None:
    res = _run(seeded, settings, src, FakeSeparator(silent_stems={"piano"}), preset_code="fast")
    info = {s.code: s for s in res.stems}
    assert info["piano"].is_silent is True
    assert info["piano"].rms_db is not None and info["piano"].rms_db < -60
    assert info["drums"].is_silent is False
    stem = seeded.scalars(
        select(Stem).join(StemType).where(Stem.job_id == res.job_id, StemType.code == "piano")
    ).one()
    assert stem.is_silent is True


def test_second_run_is_skipped_and_force_reseparates(
    seeded: Session, settings: Settings, src: Path, tmp_path: Path, mix: np.ndarray
) -> None:
    sep = FakeSeparator()
    first = _run(seeded, settings, src, sep, preset_code="fast")
    n_calls = len(sep.calls)

    # 同じ音（別名のファイル）でも分割しない
    again_src = write_source(tmp_path / "other_name.wav", mix)
    second = _run(seeded, settings, again_src, sep, preset_code="fast")
    assert second.skipped is True
    assert second.job_id == first.job_id and second.track_id == first.track_id
    assert len(sep.calls) == n_calls
    assert {s.code for s in second.stems} == {s.code for s in first.stems}

    forced = _run(seeded, settings, src, sep, preset_code="fast", force=True)
    assert forced.skipped is False
    assert forced.job_id != first.job_id and forced.track_id == first.track_id
    assert len(sep.calls) > n_calls
    assert len(seeded.scalars(select(Track)).all()) == 1
    assert len(seeded.scalars(select(SeparationJob)).all()) == 2


def test_failure_marks_job_failed(seeded: Session, settings: Settings, src: Path) -> None:
    sep = FakeSeparator(fail_models={KIM_VOCALS})
    with pytest.raises(SeparationError, match="分割に失敗しました"):
        _run(seeded, settings, src, sep, preset_code="standard")
    job = seeded.scalars(select(SeparationJob)).one()
    assert job.status == "failed"
    assert job.finished_at is not None
    assert job.error_message is not None and "分割に失敗しました" in job.error_message
    assert "fake failure" in job.error_message
    assert seeded.scalars(select(Stem)).all() == []
    assert not (settings.stems_dir / str(job.job_id)).exists()

    # 失敗した曲は次に分割できる（完了済みではないので skip しない）
    res = _run(seeded, settings, src, FakeSeparator(), preset_code="standard")
    assert res.skipped is False and res.job_id != job.job_id


def test_loud_stem_is_limited_not_clipped(
    seeded: Session, settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """stem が ±1 を超えるときは全体に倍率をかけるので、FLAC 保存で切り詰めが起きない。"""
    loud = synth_mix(1.0, amp=0.95)
    src = write_source(tmp_path / "loud.wav", loud)
    sep = FakeSeparator(multistem_coefs={**dict.fromkeys(TOP, 0.0), "vocals": 1.5, "other": 0.0})
    with caplog.at_level("WARNING"):
        res = _run(seeded, settings, src, sep, preset_code="fast")
    assert "クリップ" not in caplog.text
    job = seeded.get(SeparationJob, res.job_id)
    assert job is not None and job.output_gain_db < 0.0


def test_map_outputs() -> None:
    a = np.zeros((4, 2), dtype=np.float32)
    assert set(map_outputs("karaoke", {"Vocals": a, "Instrumental": a})) == {
        "lead_vocal", "backing_vocal",
    }
    assert set(map_outputs("vocals", {"vocals": a, "other": a})) == {"vocals", "instrumental"}
    assert set(map_outputs("multistem", {k: a for k in TOP})) == set(TOP)
    with pytest.raises(RuntimeError, match="対応づけられません"):
        map_outputs("multistem", {"kazoo": a})
    # yaml で確認できていない名前は受け付けない
    with pytest.raises(RuntimeError, match="対応づけられません"):
        map_outputs("karaoke", {"Karaoke": a, "Instrumental": a})


def test_tta_combine_undoes_inversion_and_swap(mix: np.ndarray) -> None:
    # 左右で係数が違う偽の demix。左右入れ替えを戻し忘れると結果が変わる。
    cl, cr = 0.2, 0.7
    calls: list[np.ndarray] = []

    def demix(x: np.ndarray) -> dict[str, np.ndarray]:
        calls.append(x)
        coef = np.array([cl, cr], dtype=np.float32)
        return {"a": x * coef, "b": x * coef[::-1]}

    out = tta_combine(demix, mix)
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[1], -mix)
    np.testing.assert_array_equal(calls[2], mix[:, ::-1])
    # 元: (L*cl, R*cr)、反転を戻す: 同じ、入替を戻す: (L*cr, R*cl) → 平均
    expect_a = mix * np.array([(2 * cl + cr) / 3, (2 * cr + cl) / 3])
    expect_b = mix * np.array([(2 * cr + cl) / 3, (2 * cl + cr) / 3])
    np.testing.assert_allclose(out["a"], expect_a, atol=1e-6)
    np.testing.assert_allclose(out["b"], expect_b, atol=1e-6)
    assert out["a"].dtype == np.float32

    # 左右対称な demix なら TTA しても結果は同じ
    same = tta_combine(lambda x: {"a": x * 0.5}, mix)
    np.testing.assert_allclose(same["a"], mix * 0.5, atol=1e-6)


def test_residual_before_correction_is_recorded(
    seeded: Session, tmp_path: Path, mix: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    from stemapp.audio import rms_db

    with caplog.at_level("INFO"):
        out = run_plan(mix, load_plan(seeded, "fast"), FakeSeparator(), workdir=tmp_path)
    # Fake の multistem 係数の合計は 0.97 → 残差は mixture の 0.03 倍
    assert out.mixture_rms_db == pytest.approx(rms_db(mix), abs=1e-4)
    assert out.residual_rms_db == pytest.approx(
        out.mixture_rms_db + 20 * np.log10(0.03), abs=1e-3
    )
    assert "補正前の残差" in caplog.text


def test_oom_frees_gpu_memory_before_retry(
    seeded: Session, tmp_path: Path, mix: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    freed: list[int] = []
    sep = FakeSeparator(oom_times=3)
    # 解放は例外ブロックを抜けたあと・次の呼び出しの前に行う
    monkeypatch.setattr(pipeline, "free_gpu_memory", lambda: freed.append(len(sep.calls)))
    run_plan(mix, load_plan(seeded, "fast"), sep, workdir=tmp_path)
    assert freed == [1, 2, 3]


def test_free_gpu_memory_runs() -> None:
    pipeline.free_gpu_memory()  # torch の有無・GPU の有無にかかわらず例外を出さない
