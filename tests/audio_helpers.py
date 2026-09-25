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


def synth_drums(
    sections: Sequence[tuple[float, float]], seconds: float, sr: int = SAMPLE_RATE,
    beats_per_bar: int = 4,
) -> np.ndarray:
    """ドラム風の合成音（(samples, 2) float32）。[(開始秒, BPM)] で途中のテンポを変えられる。

    キック（小節の頭と3拍目）、スネア（2・4拍目）、ハイハット（8分音符）、小節の頭にベースの音。
    """
    n = int(seconds * sr)
    out = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(0)

    def add(at: float, sig: np.ndarray) -> None:
        i = int(at * sr)
        if i >= n:
            return
        m = min(len(sig), n - i)
        out[i : i + m] += sig[:m]

    tk = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 12)
    ts = np.arange(int(0.18 * sr)) / sr
    snare = (0.6 * rng.standard_normal(ts.size) + 0.4 * np.sin(2 * np.pi * 190 * ts)) * np.exp(
        -ts * 22
    )
    th = np.arange(int(0.05 * sr)) / sr
    hat = 0.25 * rng.standard_normal(th.size) * np.exp(-th * 80)
    tb = np.arange(int(0.4 * sr)) / sr
    bass = 0.5 * np.sin(2 * np.pi * 55 * tb) * np.exp(-tb * 4)

    ordered = sorted(sections)

    def bpm_at(t: float) -> float:
        bpm = ordered[0][1]
        for start, b in ordered:
            if start <= t + 1e-9:
                bpm = b
        return bpm

    t, k = 0.0, 0
    while t < seconds:
        step = 60.0 / bpm_at(t)
        pos = k % beats_per_bar
        if pos in (0, 2):
            add(t, kick)
        if pos in (1, 3):
            add(t, snare)
        if pos == 0:
            add(t, bass)
        add(t, hat)
        add(t + step / 2, hat * 0.6)
        t += step
        k += 1
    out = 0.5 * out / np.abs(out).max()
    return np.stack([out, out], axis=1).astype(np.float32)
