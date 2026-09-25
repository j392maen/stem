"""プリセットごとの処理時間と GPU メモリの計測（`stemapp bench`）。

一時フォルダで分割して結果は捨てる。DB にはプリセットの読み出し以外で触らない。
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from stemapp.audio import FfmpegRunner, normalize_audio
from stemapp.config import Settings
from stemapp.separation.base import DEVICE_CUDA, Separator
from stemapp.separation.pipeline import OomPolicy, StepResult, load_plan, run_plan


@dataclass
class PresetBench:
    preset: str
    seconds: float
    peak_memory_mb: float | None
    steps: list[StepResult] = field(default_factory=list)


@dataclass
class BenchReport:
    source: str
    duration_sec: float
    device: str
    created_at: str
    presets: list[PresetBench] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def run_bench(
    session: Session,
    settings: Settings,
    src: Path,
    separator: Separator,
    presets: list[str],
    *,
    device: str = DEVICE_CUDA,
    oom_policy: OomPolicy | None = None,
    ffmpeg_runner: FfmpegRunner | None = None,
) -> BenchReport:
    plans = [load_plan(session, code) for code in presets]
    with tempfile.TemporaryDirectory(dir=settings.cache_dir, prefix="bench-") as tmp:
        tmp_dir = Path(tmp)
        norm = normalize_audio(src, tmp_dir / "normalized.wav", runner=ffmpeg_runner)
        report = BenchReport(
            source=str(src),
            duration_sec=round(norm.duration_sec, 2),
            device=device,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        for plan in plans:
            out = run_plan(
                norm.data,
                plan,
                separator,
                workdir=tmp_dir / plan.code,
                device=device,
                mix_path=norm.path,
                oom_policy=oom_policy,
            )
            report.presets.append(
                PresetBench(
                    preset=plan.code,
                    seconds=out.seconds,
                    peak_memory_mb=out.peak_memory_mb,
                    steps=out.steps,
                )
            )
            del out
    return report


def save_report(settings: Settings, report: BenchReport, now: datetime | None = None) -> Path:
    """data/cache/bench/<日時>.json に保存する。"""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = settings.cache_dir / "bench" / f"{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    return path
