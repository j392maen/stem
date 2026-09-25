"""ジョブ・API のテストで共通に使うもの。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from stemapp.beats.base import BeatAnalyzer
from stemapp.beats.fake import FakeBeatAnalyzer
from stemapp.config import Settings
from stemapp.delivery import fake_encoder
from stemapp.ingest import import_file
from stemapp.jobs.child import run_job
from stemapp.jobs.worker import ChildHandle, ChildLauncher
from stemapp.separation import FakeSeparator
from stemapp.separation.base import Separator


def no_tags(_path: Path) -> Mapping[str, str]:
    return {}


def make_track(
    session: Session, settings: Settings, tmp_path: Path, name: str = "song",
    seconds: float = 1.0, seed_offset: float = 0.0,
) -> int:
    """合成音の曲を取り込んで track_id を返す（seed_offset で音を変える）。"""
    data = synth_mix(seconds)
    if seed_offset:
        data = (data * np.float32(1.0 - seed_offset)).astype(np.float32)
    src = write_source(tmp_path / "src" / f"{name}.wav", data)
    res = import_file(session, settings, src, ffmpeg_runner=fake_ffmpeg, tag_reader=no_tags)
    return res.track_id


class FinishedHandle:
    """終了済みの子プロセスの代わり。"""

    def __init__(self, rc: int) -> None:
        self.rc = rc
        self.killed = False

    def poll(self) -> int | None:
        return self.rc

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        return self.rc


def sync_launcher(
    settings: Settings,
    separator_factory: Callable[[], Separator] = FakeSeparator,
    launched: list[int] | None = None,
    beat_analyzer: BeatAnalyzer | None = None,
) -> ChildLauncher:
    """子プロセスを使わず、その場で最後まで実行する launcher（同期実行モード）。

    拍は beat_analyzer（省略時は 120 BPM の FakeBeatAnalyzer）で作る。
    """
    analyzer = beat_analyzer if beat_analyzer is not None else FakeBeatAnalyzer()

    def launch(job_id: int) -> ChildHandle:
        if launched is not None:
            launched.append(job_id)
        rc = run_job(
            settings, job_id, separator_factory(), encoder=fake_encoder, beat_analyzer=analyzer
        )
        return FinishedHandle(rc)

    return launch
