"""音量の上限（mixture や stem が ±1 を超えるとき、全 stem に同じ倍率をかける）。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy.orm import Session

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from job_helpers import no_tags
from stemapp.config import Settings
from stemapp.models import SeparationJob
from stemapp.seed import seed
from stemapp.separation import FakeSeparator
from stemapp.separation.pipeline import (
    LIMIT_PEAK,
    SeparationOutput,
    limit_output,
    resolve_data_path,
    separate_file,
)

TOP = ["vocals", "drums", "bass", "guitar", "piano", "other"]


def _read(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), dtype="float64", always_2d=True)
    return data


def test_loud_mixture_is_scaled_and_sums_match(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    seed(session)
    mix = synth_mix(1.0, amp=1.3)  # 元の曲自体が ±1 を超える（mp3 のデコード結果など）
    src = write_source(tmp_path / "loud.wav", mix)
    res = separate_file(
        session, settings, src, FakeSeparator(), preset_code="fast",
        ffmpeg_runner=fake_ffmpeg, tag_reader=no_tags,
    )
    job = session.get(SeparationJob, res.job_id)
    assert job is not None

    # FakeSeparator の既定（fast）で最大になるのは mixture 自体
    peak = float(np.max(np.abs(mix)))
    gain = LIMIT_PEAK / peak
    assert job.output_gain_db == pytest.approx(20 * math.log10(gain), abs=1e-3)
    assert job.output_gain_db < 0

    stems = {st.code: _read(resolve_data_path(settings, str(st.file_path))) for st in res.stems}
    for arr in stems.values():
        assert np.max(np.abs(arr)) <= 1.0
    total = sum(stems[c] for c in TOP)
    np.testing.assert_allclose(total, mix.astype(np.float64) * gain, atol=1e-4)
    np.testing.assert_allclose(
        stems["lead_vocal"] + stems["backing_vocal"], stems["vocals"], atol=1e-4
    )


def test_quiet_mixture_is_unchanged(session: Session, settings: Settings, tmp_path: Path) -> None:
    seed(session)
    src = write_source(tmp_path / "quiet.wav", synth_mix(1.0, amp=0.5))
    res = separate_file(
        session, settings, src, FakeSeparator(), preset_code="fast",
        ffmpeg_runner=fake_ffmpeg, tag_reader=no_tags,
    )
    job = session.get(SeparationJob, res.job_id)
    assert job is not None and job.output_gain_db == 0.0


def test_limit_output_uses_largest_stem() -> None:
    mix = np.full((10, 2), 0.5, dtype=np.float32)
    stems = {"a": np.full((10, 2), 2.0, dtype=np.float32), "b": -mix * 3}
    out = SeparationOutput(stems=dict(stems), top_level=["a", "b"], steps=[], seconds=0.0)
    gain = limit_output(out, mix)
    assert gain == pytest.approx(LIMIT_PEAK / 2.0)
    np.testing.assert_allclose(out.stems["a"], 2.0 * gain, rtol=1e-6)
    np.testing.assert_allclose(out.stems["b"], -1.5 * gain, rtol=1e-6)
    assert limit_output(out, mix) == 1.0  # もう超えていない
