"""テスト用の音源と、ffmpeg を使わない正規化の代わり。音源ファイルはテストの中で作る。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from stemapp.audio import SAMPLE_RATE, as_stereo


def synth_mix(seconds: float = 2.0, sr: int = SAMPLE_RATE, amp: float = 0.25) -> np.ndarray:
    """複数の正弦波と雑音を重ねたステレオ音（(samples, 2) float32）。"""
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(0)
    left = (
        np.sin(2 * np.pi * 110 * t)
        + 0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 1760 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
        + 0.1 * rng.standard_normal(t.size)
    )
    right = (
        0.8 * np.sin(2 * np.pi * 110 * t + 0.3)
        + 0.6 * np.sin(2 * np.pi * 660 * t)
        + 0.1 * rng.standard_normal(t.size)
    )
    x = np.stack([left, right], axis=1)
    return (amp * x / np.abs(x).max()).astype(np.float32)


def write_source(path: Path, data: np.ndarray, subtype: str = "FLOAT") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, SAMPLE_RATE, subtype=subtype)
    return path


def fake_ffmpeg(args: Sequence[str]) -> None:
    """ffmpeg の代わり: 44.1kHz の WAV/FLAC を float32 ステレオ WAV に書き直す。"""
    src = Path(args[list(args).index("-i") + 1])
    dst = Path(args[-1])
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE, "fake_ffmpeg はリサンプルしない"
    sf.write(str(dst), as_stereo(data), SAMPLE_RATE, subtype="FLOAT", format="WAV")
