from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from stemapp import audio, cli
from stemapp.config import Settings
from stemapp.db import make_engine, make_session_factory
from stemapp.models import SeparationJob, Stem, Track
from stemapp.separation import FakeSeparator

runner = CliRunner()


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> FakeSeparator:
    sep = FakeSeparator(peak_mb=123.0)
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "make_separator", lambda _s: sep)
    monkeypatch.setattr(audio, "run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)
    return sep


def _count(settings: Settings, model: type) -> int:
    engine = make_engine(settings.db_path)
    try:
        with make_session_factory(engine)() as s:
            return int(s.scalar(select(func.count()).select_from(model)) or 0)
    finally:
        engine.dispose()


def test_separate_command(cli_env: FakeSeparator, settings: Settings, tmp_path: Path) -> None:
    src = write_source(tmp_path / "曲.wav", synth_mix(1.0))
    res = runner.invoke(cli.app, ["separate", str(src), "--preset", "fast"])
    assert res.exit_code == 0, res.output
    assert "lead_vocal" in res.output and "所要時間" in res.output
    assert _count(settings, Stem) == 8

    again = runner.invoke(cli.app, ["separate", str(src)])
    assert again.exit_code == 0, again.output
    assert "分割済み" in again.output
    assert _count(settings, SeparationJob) == 1

    forced = runner.invoke(cli.app, ["separate", str(src), "--force", "--cpu"])
    assert forced.exit_code == 0, forced.output
    assert _count(settings, SeparationJob) == 2
    assert cli_env.calls[-1].device == "cpu"


def test_separate_command_failure(
    cli_env: FakeSeparator, settings: Settings, tmp_path: Path
) -> None:
    cli_env.fail_models = {"BS-Roformer-SW.ckpt"}
    src = write_source(tmp_path / "a.wav", synth_mix(1.0))
    res = runner.invoke(cli.app, ["separate", str(src)])
    assert res.exit_code == 1
    assert "分割に失敗しました" in res.output


def test_bench_command(cli_env: FakeSeparator, settings: Settings, tmp_path: Path) -> None:
    src = write_source(tmp_path / "a.wav", synth_mix(1.0))
    res = runner.invoke(cli.app, ["bench", str(src), "--presets", "fast,standard"])
    assert res.exit_code == 0, res.output
    files = list((settings.cache_dir / "bench").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert [p["preset"] for p in data["presets"]] == ["fast", "standard"]
    assert len(data["presets"][1]["steps"]) == 3
    assert data["presets"][0]["peak_memory_mb"] == 123.0
    assert data["duration_sec"] == pytest.approx(1.0)
    # bench は DB に曲を登録しない・一時フォルダも残さない
    assert _count(settings, Track) == 0
    assert _count(settings, SeparationJob) == 0
    assert not list(settings.cache_dir.glob("bench-*"))
    assert np.isfinite(data["presets"][0]["seconds"])
