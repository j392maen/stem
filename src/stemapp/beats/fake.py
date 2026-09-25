"""テスト用の拍の解析器（GPU・beat_this 不要）。指定した BPM と拍子で規則的な拍を返す。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from stemapp.audio import SAMPLE_RATE
from stemapp.beats.base import AudioInput, BeatAnalysisError, BeatResult


def _duration_sec(audio: AudioInput) -> float:
    if isinstance(audio, Path):
        return float(sf.info(str(audio)).duration)
    return float(np.asarray(audio).shape[0]) / SAMPLE_RATE


class FakeBeatAnalyzer:
    """規則的な拍を返す。

    tempo: 一定の BPM、または [(開始秒, BPM), ...]（区間ごとに変える。開始秒の昇順）。
    beats_per_bar: 拍子（1小節の拍数）。offset: 最初の拍（＝1小節目の頭）の時刻。
    fail=True なら BeatAnalysisError を出す。
    """

    def __init__(
        self,
        tempo: float | Sequence[tuple[float, float]] = 120.0,
        *,
        beats_per_bar: int = 4,
        offset: float = 0.0,
        fail: bool = False,
    ) -> None:
        if isinstance(tempo, int | float):
            self.sections: list[tuple[float, float]] = [(0.0, float(tempo))]
        else:
            self.sections = sorted((float(s), float(b)) for s, b in tempo)
        self.beats_per_bar = beats_per_bar
        self.offset = offset
        self.fail = fail
        self.calls: list[AudioInput] = []

    @property
    def name(self) -> str:
        return "fake 1"

    def _bpm_at(self, t: float) -> float:
        bpm = self.sections[0][1]
        for start, b in self.sections:
            if start <= t + 1e-9:
                bpm = b
        return bpm

    def analyze(self, audio: AudioInput) -> BeatResult:
        self.calls.append(audio)
        if self.fail:
            raise BeatAnalysisError("拍を解析できませんでした（テスト用の失敗）。")
        duration = _duration_sec(audio)
        beats: list[float] = []
        downbeats: list[float] = []
        t = self.offset
        while t < duration:
            if len(beats) % self.beats_per_bar == 0:
                downbeats.append(round(t, 6))
            beats.append(round(t, 6))
            t += 60.0 / self._bpm_at(t)
        return BeatResult(beats=beats, downbeats=downbeats, analyzer=self.name)
