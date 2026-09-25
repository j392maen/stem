"""実 GPU・実モデルで分割するテスト。`uv run pytest -m gpu` で実行する。

モデルは設定（.env）の models_dir にあるものを使う（無ければ audio-separator がダウンロードする）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy.orm import Session

from audio_helpers import synth_mix, write_source
from stemapp.config import Settings
from stemapp.seed import seed
from stemapp.separation.pipeline import resolve_data_path, separate_file

pytestmark = pytest.mark.gpu

TOP = ["vocals", "drums", "bass", "guitar", "piano", "other"]


def test_fast_preset_on_gpu(session: Session, settings: Settings, tmp_path: Path) -> None:
    from stemapp.separation.audio_separator_backend import AudioSeparatorBackend

    real = Settings()  # .env の設定（モデルの置き場）
    backend = AudioSeparatorBackend(
        models_dir=real.models_dir, work_dir=settings.cache_dir / "audio-separator"
    )
    seed(session)
    mix = synth_mix(12.0, amp=0.5)
    src = write_source(tmp_path / "synth.wav", mix)

    res = separate_file(session, settings, src, backend, preset_code="fast")

    assert not res.skipped
    assert {s.code for s in res.stems} == set(TOP) | {"lead_vocal", "backing_vocal"}
    assert all(r.device == "cuda" for r in res.steps)
    audio = {
        s.code: sf.read(str(resolve_data_path(settings, str(s.file_path))), dtype="float64")[0]
        for s in res.stems
    }
    top_sum = sum(audio[c] for c in TOP)
    assert np.max(np.abs(top_sum - mix)) < 1e-4
    lb = audio["lead_vocal"] + audio["backing_vocal"]
    assert np.max(np.abs(lb - audio["vocals"])) < 1e-4
    for r in res.steps:
        assert r.peak_memory_mb is not None and r.peak_memory_mb > 0


@pytest.fixture
def backend(settings: Settings):  # type: ignore[no-untyped-def]
    from stemapp.separation.audio_separator_backend import AudioSeparatorBackend

    return AudioSeparatorBackend(
        models_dir=Settings().models_dir, work_dir=settings.cache_dir / "audio-separator"
    )


def test_residual_before_correction_is_small(
    session: Session, backend: object, tmp_path: Path
) -> None:
    from stemapp.separation.pipeline import load_plan, run_plan

    seed(session)
    mix = synth_mix(12.0, amp=0.5)
    out = run_plan(mix, load_plan(session, "fast"), backend, workdir=tmp_path)  # type: ignore[arg-type]
    print(f"残差 {out.residual_rms_db:.1f} dBFS / mixture {out.mixture_rms_db:.1f} dBFS")
    assert out.residual_rms_db <= out.mixture_rms_db - 15.0


def test_silent_right_channel_stays_silent(
    session: Session, backend: object, tmp_path: Path
) -> None:
    from stemapp.audio import rms_db
    from stemapp.separation.pipeline import load_plan, run_plan

    seed(session)
    mix = synth_mix(12.0, amp=0.5)
    mix[:, 1] = 0.0
    out = run_plan(mix, load_plan(session, "fast"), backend, workdir=tmp_path)  # type: ignore[arg-type]
    for code, arr in out.stems.items():
        right = np.abs(arr[:, 1]).max()
        print(f"{code}: 右 max {right:.2e} / 左 {rms_db(arr[:, 0]):.1f} dBFS")
        assert right < 1e-4, f"{code} の右チャンネルに音が漏れている（max {right:.2e}）"
