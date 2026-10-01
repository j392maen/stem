"""HPSS（Harmonic–Percussive Source Separation）で「持続音」と「短い音」を分ける。

時間方向に長く続く成分（パッド・伸ばした和音など）と、周波数方向に広がる短い成分
（ヒット・打撃の頭）を、スペクトログラムのメディアンフィルタで分ける（librosa の
`decompose.hpss`）。margin を 1 より大きくすると、どちらとも言えない成分がどちらのマスクにも
入らず「残り」になる（Driedger らの拡張）。残りは呼び出し側が「親 − 持続音 − 短い音」で作る。

設定は docs/research/R01-bpm-tempo-quality.md D-3 の実測（other に kernel 約1秒×約300Hz、
margin=2 で、エネルギーの約 40% が持続、11% が短い音、13% が残り）から始める。
GPU は使わない（CPU のみ）。librosa は audio-separator の依存として入っている。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from stemapp.audio import SAMPLE_RATE

# STFT の窓（2048 サンプル = 約 46ms、周波数の刻み 44100/2048 ≈ 21.5Hz）と送り幅（約 11.6ms）
N_FFT = 2048
HOP_LENGTH = 512
# 持続音のメディアンフィルタの長さ（時間方向）: 約1秒。
# 1秒 × 44100 / 512 ≈ 86 フレーム → 奇数にそろえて 87（中央がちょうど1つになるように）
HARMONIC_KERNEL_FRAMES = 87
# 短い音のメディアンフィルタの長さ（周波数方向）: 約300Hz。300 / 21.5 ≈ 14 ビン → 奇数で 15
PERCUSSIVE_KERNEL_BINS = 15
# マスクの余裕（>1 で「どちらでもない」成分を残りに回す）。R01 D-3 の実測で使った値
MARGIN = 2.0

SUSTAINED = "sustained"
TRANSIENT = "transient"

HpssProgress = Callable[[float, str], None]


def split_sustained_transient(
    x: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    progress: HpssProgress | None = None,
) -> dict[str, np.ndarray]:
    """(samples, 2) の音を {"sustained": …, "transient": …} に分ける（float32、同じ長さ）。

    マスクは左右の振幅の平均から1つだけ作り、左右に同じものをかける（左右で分け方がずれて
    定位が揺れないように。計算も半分で済む）。sample_rate は 44.1kHz を前提にした
    フィルタの長さを、ほかのサンプリング周波数でも同じ秒数・Hz になるよう直すのに使う。
    """
    import librosa

    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        return {SUSTAINED: np.zeros_like(x), TRANSIENT: np.zeros_like(x)}
    scale = sample_rate / SAMPLE_RATE
    kernel = (
        _odd(HARMONIC_KERNEL_FRAMES * scale),
        _odd(PERCUSSIVE_KERNEL_BINS / scale),
    )

    def report(p: float, stage: str) -> None:
        if progress is not None:
            progress(p, stage)

    report(0.0, "スペクトログラムを計算中")
    specs = [
        librosa.stft(np.ascontiguousarray(x[:, ch]), n_fft=N_FFT, hop_length=HOP_LENGTH)
        for ch in range(x.shape[1])
    ]
    mag = np.mean([np.abs(s) for s in specs], axis=0)
    report(0.2, "持続音と短い音を分けています")
    mask_h, mask_p = librosa.decompose.hpss(mag, kernel_size=kernel, margin=MARGIN, mask=True)
    del mag
    report(0.8, "音に戻しています")
    out = {SUSTAINED: np.zeros_like(x), TRANSIENT: np.zeros_like(x)}
    for ch, spec in enumerate(specs):
        for name, mask in ((SUSTAINED, mask_h), (TRANSIENT, mask_p)):
            y = librosa.istft(spec * mask, hop_length=HOP_LENGTH, n_fft=N_FFT, length=n)
            out[name][:, ch] = y.astype(np.float32)
    report(1.0, "完了")
    return out


def _odd(v: float) -> int:
    k = max(1, int(round(v)))
    return k if k % 2 == 1 else k + 1
