"""拍の解析器の共通インターフェース。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# 解析器に渡す音声: 44.1kHz ステレオ float32 の配列（(samples, 2)）、または normalized.wav のパス
AudioInput = np.ndarray | Path


@dataclass(frozen=True)
class BeatResult:
    beats: list[float]  # 拍の時刻（秒、昇順）
    downbeats: list[float]  # 小節の頭の時刻（秒、昇順。拍の一部）
    analyzer: str  # 解析器の名前と版（BEAT_GRID.analyzer）
    device: str = "cpu"  # 実際に使った装置（cuda / cpu）
    seconds: float = 0.0  # 解析にかかった時間
    extra: dict[str, object] = field(default_factory=dict)


class BeatAnalysisError(RuntimeError):
    """拍を解析できなかった（メッセージは日本語）。"""


@runtime_checkable
class BeatAnalyzer(Protocol):
    """曲の音声から拍と小節の頭の時刻を求める。"""

    @property
    def name(self) -> str:
        """解析器の名前と版（例 "beat_this 1.1.0 final0"）。"""
        ...

    def analyze(self, audio: AudioInput) -> BeatResult:
        ...
