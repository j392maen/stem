"""実 GPU・実モデルで詳細分割するテスト（T07）。`uv run pytest -m gpu` で実行する。

モデルは設定（.env）の models_dir にあるものを使う（無ければ audio-separator がダウンロードする）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from audio_helpers import synth_drums, synth_mix
from stemapp.config import Settings
from stemapp.seed import ASPIRATION, DRUMSEP, MALE_FEMALE, MEGA53, seed
from stemapp.separation.refine import load_methods, rest_code, run_refine

pytestmark = pytest.mark.gpu


@pytest.fixture
def backend(settings: Settings):  # type: ignore[no-untyped-def]
    from stemapp.separation.audio_separator_backend import AudioSeparatorBackend

    return AudioSeparatorBackend(
        models_dir=Settings().models_dir, work_dir=settings.cache_dir / "audio-separator"
    )


@pytest.mark.parametrize(
    ("model", "parent", "make"),
    [
        (DRUMSEP, "drums", lambda: synth_drums([(0.0, 120.0)], 8.0)),
        (MALE_FEMALE, "lead_vocal", lambda: synth_mix(8.0, amp=0.5)),
        (ASPIRATION, "backing_vocal", lambda: synth_mix(8.0, amp=0.5)),
    ],
)
def test_refine_on_gpu_sums_to_parent(
    session: Session, backend: object, tmp_path: Path, model: str, parent: str, make: object
) -> None:
    seed(session)
    method = load_methods(session)[model]
    x = make()  # type: ignore[operator]
    out = run_refine(x, method, parent, backend, workdir=tmp_path)  # type: ignore[arg-type]
    assert set(out.stems) == {*method.child_codes, rest_code(parent)}
    assert all(r.device == "cuda" for r in out.steps)
    total = sum(a.astype(np.float64) for a in out.stems.values())
    assert np.max(np.abs(total - x)) < 1e-5
    # 名前の付いた子が音を持っている（全部が残りに入っていない）
    named = sum(float(np.sum(out.stems[c].astype(np.float64) ** 2)) for c in method.child_codes)
    assert named > 0
    peak = out.peak_memory_mb
    print(f"{model}: {out.seconds:.1f} 秒, GPU 最大 {peak:.0f} MB, 丸め {out.clipped}")
    assert peak is not None and peak > 0


def test_mega53_on_gpu_sums_to_parent(session: Session, backend: object, tmp_path: Path) -> None:
    """Mega 53（T07b）: 使う stem のマスク推定器だけを動かし、8GB に収まる。無音の子は作らない。"""
    seed(session)
    method = load_methods(session)[MEGA53]
    x = synth_mix(8.0, amp=0.5)
    out = run_refine(x, method, "other", backend, workdir=tmp_path)  # type: ignore[arg-type]
    assert all(r.device == "cuda" and r.chunk_scale == 1.0 for r in out.steps)
    kept = [c for c in out.stems if c != out.rest]
    assert out.rest == "other_rest" and out.rest in out.stems
    assert sorted(kept + out.dropped) == sorted(method.child_codes)
    total = sum(a.astype(np.float64) for a in out.stems.values())
    assert np.max(np.abs(total - x)) < 1e-5
    peak = out.peak_memory_mb
    print(f"{MEGA53}: {out.seconds:.1f} 秒, GPU 最大 {peak:.0f} MB, 作らなかった子 {out.dropped}")
    assert peak is not None and 0 < peak < 4096


def test_mega53_selected_stems_match_full_model() -> None:
    """実際の重みで: 使う 5 stem だけを読み込んだモデルの出力が、53 stem すべてを読み込んだ
    モデルの同じ stem の出力と一致する（state_dict の番号の付け直しで順を取り違えない）。"""
    import time

    import torch

    from stemapp.separation.audio_separator_backend import (
        MSST_MODELS,
        REFINE_OUTPUT_NAME_MAP,
    )
    from stemapp.separation.msst import runner

    models = Settings().models_dir
    files = MSST_MODELS[MEGA53]
    ckpt, config = models / MEGA53, models / files.config
    if not (ckpt.is_file() and config.is_file()):
        pytest.skip("Mega 53 の重みがありません")
    instruments = list(runner.load_config(config)["training"]["instruments"])
    stems = list(reversed(REFINE_OUTPUT_NAME_MAP[MEGA53]))  # 元の順と違う順で
    # CPU で比べる（実測で差 0。GPU は演算の選び方で 1e-4 程度ずれ（実測 8.6e-5）、静かな stem
    # どうしの差（2e-4 程度）と区別しにくい。重みが要るので -m gpu の側に置く）
    x = torch.from_numpy(synth_mix(2.0, amp=0.5).T.copy()).unsqueeze(0)
    t0 = time.perf_counter()
    sub = runner.load_model(ckpt, config, stems, "cpu")
    t_sub = time.perf_counter() - t0
    with torch.inference_mode():
        y_sub = sub.model(x)[0].float().cpu()
    del sub
    t0 = time.perf_counter()
    full = runner.load_model(ckpt, config, instruments, "cpu")
    t_full = time.perf_counter() - t0
    with torch.inference_mode():
        y_full = full.model(x)[0].float().cpu()
    del full
    print(f"読み込み: 5 stem {t_sub:.1f} 秒, 53 stem {t_full:.1f} 秒")
    assert y_full.shape[0] == 53 and y_sub.shape[0] == len(stems)
    # 順を取り違えれば別の楽器の音になり差は桁違いに大きいので、それも確かめる
    scale = float(y_full.abs().max())
    for j, name in enumerate(stems):
        i = instruments.index(name)
        diff = float((y_sub[j] - y_full[i]).abs().max())
        print(f"{name}: 差 {diff:.2e}（出力の最大 {scale:.2f}）")
        assert diff <= 1e-5 * max(scale, 1e-3), (name, diff)
        others = [
            float((y_sub[j] - y_full[instruments.index(o)]).abs().max())
            for o in stems
            if o != name
        ]
        assert diff * 20 < min(others), (name, diff, others)
