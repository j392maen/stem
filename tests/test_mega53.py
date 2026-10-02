"""T07b Mega 53（MSST 形式の BS-Roformer）での詳細分割。GPU・重みは使わない。

- 無音の子を作らない（残りに入れる）こと、子の合計＝親、木の形（Fake の分離器）
- チャンク分割と重ね合わせ（偽の forward）
- 取り込んだ MSST のコード: 選んだ stem のマスク推定器だけを残したモデルが、元のモデルの
  同じ stem の出力と一致すること（小さな乱数のモデルを CPU で）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from audio_helpers import synth_mix
from stemapp.audio import rms_db
from stemapp.config import Settings
from stemapp.models import SeparationJob, StemRendition
from stemapp.seed import MEGA53, MEGA53_CHILDREN
from stemapp.separation import FakeSeparator
from stemapp.separation.fake import DEFAULT_REFINE_COEFS
from stemapp.separation.msst import runner
from stemapp.separation.pipeline import SILENT_THRESHOLD_DB
from stemapp.separation.refine import load_methods, run_refine
from stemapp.stem_view import build_view
from test_refine import (  # noqa: F401  fixture を使う
    _children,
    _master,
    _run_refine,
    _stem_id,
    factory,
    full_job,
    seeded,
)

KEPT = ["brass", "strings", "synth", "percussion"]  # Fake は woodwind を無音にしている


# --- 計算（無音の子・合計） -------------------------------------------------------------------


def test_fake_mega53_has_a_silent_output() -> None:
    assert DEFAULT_REFINE_COEFS[MEGA53]["woodwind"] == 0.0
    assert set(DEFAULT_REFINE_COEFS[MEGA53]) == set(MEGA53_CHILDREN)


def test_run_refine_drops_silent_children(seeded: Session, tmp_path: Path) -> None:  # noqa: F811
    x = synth_mix(0.5)
    m = load_methods(seeded)[MEGA53]
    out = run_refine(x, m, "other", FakeSeparator(), workdir=tmp_path / "w", device="cpu")
    assert set(out.stems) == {*KEPT, "other_rest"}
    assert out.dropped == ["woodwind"]
    total = sum(a.astype(np.float64) for a in out.stems.values())
    assert np.max(np.abs(total - x)) < 1e-5
    for code in KEPT:
        assert np.allclose(out.stems[code], x * DEFAULT_REFINE_COEFS[MEGA53][code], atol=1e-6)


def test_quiet_child_goes_to_rest_and_audible_child_is_kept(
    seeded: Session, tmp_path: Path  # noqa: F811
) -> None:
    """-60dB 未満の子は作らず残りに入れる。それより大きい子（-50dB 程度）は作る。"""
    x = synth_mix(0.5, amp=0.5)
    quiet = (x * np.float32(1e-3)).astype(np.float32)  # 親より 60dB 小さい
    audible = (x * np.float32(1e-1)).astype(np.float32)
    assert rms_db(quiet) < SILENT_THRESHOLD_DB < rms_db(audible)

    class Sep(FakeSeparator):
        def separate(self, *a: Any, **k: Any) -> dict[str, np.ndarray]:
            out = super().separate(*a, **k)
            out["brass"] = quiet
            out["woodwind"] = audible
            return out

    m = load_methods(seeded)[MEGA53]
    out = run_refine(x, m, "other", Sep(), workdir=tmp_path, device="cpu")
    assert out.dropped == ["brass"]
    assert "brass" not in out.stems and "woodwind" in out.stems
    named = sum(out.stems[c].astype(np.float64) for c in out.stems if c != "other_rest")
    # 残りに brass の分が入っている
    assert np.allclose(out.stems["other_rest"], x.astype(np.float64) - named, atol=1e-6)
    total = sum(a.astype(np.float64) for a in out.stems.values())
    assert np.max(np.abs(total - x)) < 1e-5


def test_all_children_silent_leaves_only_rest(
    seeded: Session, tmp_path: Path  # noqa: F811
) -> None:
    x = synth_mix(0.5)
    sep = FakeSeparator(silent_stems=MEGA53_CHILDREN)
    out = run_refine(x, load_methods(seeded)[MEGA53], "other", sep, workdir=tmp_path)
    assert set(out.stems) == {"other_rest"}
    assert sorted(out.dropped) == sorted(MEGA53_CHILDREN)
    assert np.max(np.abs(out.stems["other_rest"].astype(np.float64) - x)) < 1e-5


# --- ジョブ（保存・木） -----------------------------------------------------------------------


def test_mega53_job_saves_only_audible_children(
    settings: Settings, factory: sessionmaker[Session], full_job: int  # noqa: F811
) -> None:
    other = _stem_id(factory, full_job, "other")
    job_id = _run_refine(settings, factory, other, MEGA53)
    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.status == "done", job.error_message if job else None
        assert job.run_on == "gpu" and job.warning is None
        kids = _children(s, other)
        assert set(kids) == {*KEPT, "other_rest"}
        assert [c for c, k in kids.items() if k.is_residual] == ["other_rest"]
        assert not any(k.is_silent for k in kids.values())
        parent_audio = _master(settings, s, other)
        total = sum(_master(settings, s, k.stem_id) for k in kids.values())
        assert np.array_equal(total, parent_audio)
        for code, k in kids.items():
            rends = s.scalars(select(StemRendition).where(StemRendition.stem_id == k.stem_id))
            assert {r.purpose for r in rends} == {"master", "stream"}, code
        # 木: other の直後に子（表示順）、作らなかった woodwind は無い
        view = build_view(s, full_job)
        codes = [t.code for _, t in view.rows]
        i = codes.index("other")
        assert codes[i + 1 : i + 6] == ["brass", "strings", "synth", "percussion", "other_rest"]
        assert "woodwind" not in codes
        assert view.refined_by[other].job_id == job_id


# --- チャンク分割と重ね合わせ -------------------------------------------------------------------


def _scale_forward(coefs: list[float]):  # type: ignore[no-untyped-def]
    def forward(part: np.ndarray) -> np.ndarray:
        assert part.dtype == np.float32
        return np.stack([part * c for c in coefs])

    return forward


@pytest.mark.parametrize("n", [500, 4000, 10_000, 33_333])
def test_demix_reconstructs_linear_model(n: int) -> None:
    """出力が入力に比例するモデルなら、チャンクの継ぎ目でも入力×係数に戻る。"""
    rng = np.random.default_rng(1)
    mix = rng.standard_normal((2, n)).astype(np.float32)
    calls: list[float] = []
    y = runner.demix(_scale_forward([0.5, 2.0]), mix, 2, 2048, 2, progress=calls.append)
    assert y.shape == (2, 2, n)
    assert np.allclose(y[0], mix * 0.5, atol=1e-5)
    assert np.allclose(y[1], mix * 2.0, atol=1e-5)
    assert calls and calls[-1] == 1.0


def test_load_config_reads_python_tuple(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  bands: !!python/tuple\n    - 2\n    - 3\n", encoding="utf-8")
    assert runner.load_config(p) == {"model": {"bands": (2, 3)}}
    bad = tmp_path / "bad.yaml"
    bad.write_text("x: !!python/object/apply:os.system ['echo hi']\n", encoding="utf-8")
    with pytest.raises(Exception, match="python/object"):
        runner.load_config(bad)


# --- 取り込んだ MSST のコード（小さな乱数のモデル、CPU） ------------------------------------------

TINY_MODEL = {
    "dim": 16,
    "depth": 1,
    "stereo": True,
    "num_stems": 3,
    "time_transformer_depth": 1,
    "freq_transformer_depth": 1,
    "freqs_per_bands": (16, 17),
    "dim_head": 8,
    "heads": 2,
    "flash_attn": True,
    "stft_n_fft": 64,
    "stft_hop_length": 16,
    "stft_win_length": 64,
    "mask_estimator_depth": 2,
    "mlp_expansion_factor": 1,
}


def _tiny(tmp_path: Path) -> tuple[Path, Path, Any]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    pytest.importorskip("beartype")
    pytest.importorskip("rotary_embedding_torch")
    import yaml

    from stemapp.separation.msst.bs_roformer import BSRoformer

    torch.manual_seed(0)
    model = BSRoformer(**TINY_MODEL)
    ckpt = tmp_path / "tiny.ckpt"
    torch.save({k: v.half() for k, v in model.state_dict().items()}, ckpt)
    cfg = {
        "audio": {"chunk_size": 44100, "sample_rate": 44100},
        "model": {**TINY_MODEL, "freqs_per_bands": list(TINY_MODEL["freqs_per_bands"])},
        "training": {"instruments": ["a", "b", "c"]},
        "inference": {"chunk_size": 44100, "num_overlap": 2},
    }
    text = yaml.safe_dump(cfg).replace(
        "freqs_per_bands:\n", "freqs_per_bands: !!python/tuple\n"
    )
    config = tmp_path / "tiny.yaml"
    config.write_text(text, encoding="utf-8")
    return ckpt, config, torch


def test_selected_mask_estimators_match_full_model(tmp_path: Path) -> None:
    ckpt, config, torch = _tiny(tmp_path)
    full = runner.load_model(ckpt, config, ["a", "b", "c"], "cpu")
    sub = runner.load_model(ckpt, config, ["c", "a"], "cpu")
    assert len(sub.model.mask_estimators) == 2 and sub.model.num_stems == 2
    x = torch.from_numpy(synth_mix(0.2).T.copy()).unsqueeze(0)
    with torch.inference_mode():
        y_full = full.model(x)
        y_sub = sub.model(x)
    assert y_sub.shape == (1, 2, 2, x.shape[-1])
    assert torch.allclose(y_sub[0, 0], y_full[0, 2], atol=1e-5)
    assert torch.allclose(y_sub[0, 1], y_full[0, 0], atol=1e-5)
    with pytest.raises(ValueError, match="モデルに無い stem"):
        runner.load_model(ckpt, config, ["z"], "cpu")


def test_separate_returns_named_stems(tmp_path: Path) -> None:
    ckpt, config, _ = _tiny(tmp_path)
    loaded = runner.load_model(ckpt, config, ["b"], "cpu")
    x = synth_mix(1.5)
    out = runner.separate(loaded, x, use_fp16=True)  # CPU では fp16 を使わない
    assert set(out) == {"b"}
    assert out["b"].shape == x.shape and out["b"].dtype == np.float32
    assert np.isfinite(out["b"]).all()


def test_backend_name_table_matches_seed() -> None:
    from stemapp.separation.audio_separator_backend import (
        MSST_MODELS,
        REFINE_OUTPUT_NAME_MAP,
        map_outputs,
    )

    table = REFINE_OUTPUT_NAME_MAP[MEGA53]
    assert sorted(table.values()) == sorted(MEGA53_CHILDREN)
    assert MEGA53 in MSST_MODELS and MSST_MODELS[MEGA53].url.endswith(MEGA53)
    z = np.zeros((4, 2), dtype=np.float32)
    assert set(map_outputs("refine", dict.fromkeys(table, z), MEGA53)) == set(MEGA53_CHILDREN)
