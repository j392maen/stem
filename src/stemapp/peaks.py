"""波形 peaks（波形表示用の事前計算データ）。形式は docs/PEAKS.md。

1ピクセル（samples_per_px サンプル）ごとに、左右を平均したモノラルの最小値・最大値を
int8（-127〜127）で持つ。最小値は切り下げ、最大値は切り上げで量子化するので、
量子化後の [min, max] は元の波形を必ず含む。
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"STPK"
VERSION = 1
# マジック(4) 版(u16) 予約(u16) samples_per_px(u32) サンプルレート(u32) 点の数(u32)
HEADER = struct.Struct("<4sHHIII")
HEADER_SIZE = HEADER.size  # 20 バイト
SCALE = 127
DEFAULT_LEVELS: tuple[int, ...] = (256, 1024, 4096, 16384)


class PeaksError(ValueError):
    """peaks ファイルの形式が正しくない。"""


@dataclass(frozen=True)
class Peaks:
    samples_per_px: int
    sample_rate: int
    mins: np.ndarray  # int8
    maxs: np.ndarray  # int8

    @property
    def points(self) -> int:
        return int(self.mins.shape[0])


def _to_mono(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1, dtype=np.float32)
    return arr


def _quantize(mins: np.ndarray, maxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qmin = np.clip(np.floor(mins.astype(np.float64) * SCALE), -SCALE, SCALE).astype(np.int8)
    qmax = np.clip(np.ceil(maxs.astype(np.float64) * SCALE), -SCALE, SCALE).astype(np.int8)
    return qmin, qmax


def compute_peaks(
    data: np.ndarray, sample_rate: int, levels: Iterable[int] = DEFAULT_LEVELS
) -> dict[int, Peaks]:
    """音声（(samples,) か (samples, 2)）から各解像度の peaks を作る。

    levels は小さい順に並べ直す。大きい解像度が小さい解像度の整数倍なら、小さい解像度の
    結果をまとめて計算する（結果は直接計算したものと同じ）。
    """
    mono = _to_mono(data)
    n = mono.shape[0]
    out: dict[int, Peaks] = {}
    prev_spp: int | None = None
    prev_min = prev_max = np.zeros(0, dtype=np.float32)
    for spp in sorted(set(int(v) for v in levels)):
        if spp <= 0:
            raise ValueError(f"samples_per_px は正の整数にしてください: {spp}")
        if n == 0:
            mins = maxs = np.zeros(0, dtype=np.float32)
        elif prev_spp is not None and spp % prev_spp == 0:
            step = spp // prev_spp
            idx = np.arange(0, prev_min.shape[0], step)
            mins = np.minimum.reduceat(prev_min, idx)
            maxs = np.maximum.reduceat(prev_max, idx)
        else:
            idx = np.arange(0, n, spp)
            mins = np.minimum.reduceat(mono, idx)
            maxs = np.maximum.reduceat(mono, idx)
        prev_spp, prev_min, prev_max = spp, mins, maxs
        qmin, qmax = _quantize(mins, maxs)
        out[spp] = Peaks(spp, sample_rate, qmin, qmax)
    return out


def encode_peaks(peaks: Peaks) -> bytes:
    body = np.empty(peaks.points * 2, dtype=np.int8)
    body[0::2] = peaks.mins
    body[1::2] = peaks.maxs
    header = HEADER.pack(MAGIC, VERSION, 0, peaks.samples_per_px, peaks.sample_rate, peaks.points)
    return header + body.tobytes()


def decode_peaks(blob: bytes) -> Peaks:
    if len(blob) < HEADER_SIZE:
        raise PeaksError("peaks ファイルが短すぎます。")
    magic, version, _reserved, spp, sr, points = HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise PeaksError("peaks ファイルではありません（マジックが違います）。")
    if version != VERSION:
        raise PeaksError(f"未対応の peaks の版です: {version}")
    if len(blob) != HEADER_SIZE + points * 2:
        raise PeaksError("peaks ファイルの長さがヘッダの点の数と合いません。")
    body = np.frombuffer(blob, dtype=np.int8, offset=HEADER_SIZE)
    return Peaks(spp, sr, body[0::2].copy(), body[1::2].copy())


def write_peaks(path: Path, peaks: Peaks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_peaks(peaks))


def read_peaks(path: Path) -> Peaks:
    return decode_peaks(path.read_bytes())
