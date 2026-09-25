"""拍・小節の頭の解析（T10）。

beat_this は `stemapp.beats.beat_this_backend` の中でだけ import する。
"""

from stemapp.beats.base import BeatAnalysisError, BeatAnalyzer, BeatResult
from stemapp.beats.fake import FakeBeatAnalyzer
from stemapp.beats.service import (
    STAGE_BEATS,
    BeatsOutcome,
    analyze_job_beats,
    analyze_track,
    beats_payload,
    get_grid,
)
from stemapp.beats.tempo import (
    TempoSegment,
    clean_beats,
    estimate_time_signature,
    segment_at,
    tempo_segments,
)

__all__ = [
    "STAGE_BEATS",
    "BeatAnalysisError",
    "BeatAnalyzer",
    "BeatResult",
    "BeatsOutcome",
    "FakeBeatAnalyzer",
    "TempoSegment",
    "analyze_job_beats",
    "analyze_track",
    "beats_payload",
    "clean_beats",
    "estimate_time_signature",
    "get_grid",
    "segment_at",
    "tempo_segments",
]
