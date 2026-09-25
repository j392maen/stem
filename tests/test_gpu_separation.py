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
