from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from stemapp import audio, cli
from stemapp.config import Settings
from stemapp.db import make_engine, make_session_factory
from stemapp.ingest import service as ingest_service
from stemapp.ingest import url as url_mod
from stemapp.ingest.url import UpdateResult
from stemapp.models import InputSource, SeparationJob, Track
from stemapp.separation import FakeSeparator
from test_url import FakeYtDlp

runner = CliRunner()
URL = "https://example.com/watch?v=abc"


@pytest.fixture
def ytdlp() -> FakeYtDlp:
    return FakeYtDlp(synth_mix(1.0, amp=0.3))


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, settings: Settings, ytdlp: FakeYtDlp) -> None:
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "make_separator", lambda _s: FakeSeparator())
    monkeypatch.setattr(cli, "make_ytdlp_runner", lambda _s: ytdlp)
    monkeypatch.setattr(audio, "run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(ingest_service, "read_tags_ffprobe", lambda _p: {})
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)


def _count(settings: Settings, model: type) -> int:
    engine = make_engine(settings.db_path)
    try:
        with make_session_factory(engine)() as s:
            return int(s.scalar(select(func.count()).select_from(model)) or 0)
    finally:
        engine.dispose()


def _sources(settings: Settings) -> list[InputSource]:
    engine = make_engine(settings.db_path)
    try:
        with make_session_factory(engine)() as s:
            return list(s.scalars(select(InputSource).order_by(InputSource.source_id)))
    finally:
        engine.dispose()


def test_import_file_command(cli_env: None, settings: Settings, tmp_path: Path) -> None:
    src = write_source(tmp_path / "私の曲.wav", synth_mix(1.0))
    res = runner.invoke(cli.app, ["import", str(src)])
    assert res.exit_code == 0, res.output
    assert "track_id: 1" in res.output and "新規" in res.output and "私の曲" in res.output

    again = runner.invoke(cli.app, ["import", str(src)])
    assert again.exit_code == 0, again.output
    assert "既存" in again.output
    assert _count(settings, Track) == 1 and _count(settings, InputSource) == 2


def test_import_url_command(cli_env: None, settings: Settings) -> None:
    res = runner.invoke(cli.app, ["import", URL])
    assert res.exit_code == 0, res.output
    assert "新規" in res.output and "テスト動画" in res.output
    (source,) = _sources(settings)
    assert (source.source_type, source.url, source.fetch_status) == ("url", URL, "done")


def test_import_url_failure(cli_env: None, settings: Settings, ytdlp: FakeYtDlp) -> None:
    ytdlp.returncode = 1
    ytdlp.stderr = "ERROR: Unsupported URL: https://example.com/watch?v=abc\n"
    res = runner.invoke(cli.app, ["import", URL])
    assert res.exit_code == 1
    assert "対応していません" in res.output and "unsupported_site" in res.output
    (source,) = _sources(settings)
    assert (source.fetch_status, source.error_code, source.track_id) == (
        "failed", "unsupported_site", None
    )
    assert _count(settings, Track) == 0


def test_import_missing_file(cli_env: None, tmp_path: Path) -> None:
    res = runner.invoke(cli.app, ["import", str(tmp_path / "none.mp3")])
    assert res.exit_code == 1
    assert "取り込みに失敗しました" in res.output


def test_separate_url(cli_env: None, settings: Settings) -> None:
    res = runner.invoke(cli.app, ["separate", URL, "--preset", "fast"])
    assert res.exit_code == 0, res.output
    assert "lead_vocal" in res.output
    assert _count(settings, SeparationJob) == 1

    again = runner.invoke(cli.app, ["separate", URL])
    assert again.exit_code == 0, again.output
    assert "分割済み" in again.output
    assert _count(settings, SeparationJob) == 1
    assert _count(settings, Track) == 1


def test_separate_bad_preset_does_not_import(cli_env: None, settings: Settings,
                                             ytdlp: FakeYtDlp) -> None:
    res = runner.invoke(cli.app, ["separate", URL, "--preset", "nope"])
    assert res.exit_code == 1
    assert ytdlp.calls == []


@pytest.mark.parametrize(
    ("status", "label", "code"),
    [("updated", "更新しました", 0), ("latest", "最新です", 0), ("failed", "失敗", 1)],
)
def test_ytdlp_update_command(
    cli_env: None, monkeypatch: pytest.MonkeyPatch, status: str, label: str, code: int
) -> None:
    monkeypatch.setattr(url_mod, "update_ytdlp", lambda _s: UpdateResult(status, "detail"))
    res = runner.invoke(cli.app, ["ytdlp-update"])
    assert res.exit_code == code, res.output
    assert label in res.output and "detail" in res.output


def test_error_message_is_not_wrapped(cli_env: None, ytdlp: FakeYtDlp) -> None:
    from stemapp.ingest.url import ERROR_MESSAGES

    ytdlp.returncode = 1
    ytdlp.stderr = "ERROR: [youtube] xxxxxxxxxxx: Video unavailable\n"
    res = runner.invoke(cli.app, ["import", URL], terminal_width=40)
    assert res.exit_code == 1
    line = f"{ERROR_MESSAGES['private_or_removed']}（理由コード: private_or_removed）"
    assert line in res.output.splitlines()
