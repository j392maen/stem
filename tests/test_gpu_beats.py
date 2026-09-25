"""beat_this を実 GPU で動かすテスト。`uv run pytest -m gpu` で実行する。

重みは設定（.env）の models_dir/beat_this に置く（無ければダウンロードする。約 78MB）。
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_helpers import synth_drums
from stemapp.beats.tempo import estimate_time_signature, tempo_segments
from stemapp.config import Settings

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def analyzer():
    import torch

    from stemapp.beats.beat_this_backend import BeatThisAnalyzer

    if not torch.cuda.is_available():
        pytest.skip("CUDA が使えません")
    return BeatThisAnalyzer(Settings().models_dir, device="cuda")


def test_constant_120_bpm(analyzer) -> None:
    res = analyzer.analyze(synth_drums([(0, 120)], 30.0))
    assert res.device == "cuda" and res.analyzer.startswith("beat_this ")
    segs = tempo_segments(res.beats)
    assert len(segs) == 1, [s.as_dict() for s in segs]
    assert segs[0].bpm == pytest.approx(120, abs=1)
    assert estimate_time_signature(res.beats, res.downbeats) == 4
    assert len(res.downbeats) >= 10


def test_tempo_change_120_to_150(analyzer) -> None:
    res = analyzer.analyze(synth_drums([(0, 120), (30, 150)], 60.0))
    segs = tempo_segments(res.beats)
    assert len(segs) == 2, [s.as_dict() for s in segs]
    assert segs[0].bpm == pytest.approx(120, abs=1)
    assert segs[1].bpm == pytest.approx(150, abs=1)
    assert segs[0].end_sec == pytest.approx(30, abs=1.0)
    assert np.all(np.diff(res.beats) > 0)
