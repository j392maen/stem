from __future__ import annotations

from pathlib import Path

import pytest

from stemapp.config import REPO_ROOT, Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.data_dir == REPO_ROOT / "data"
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.passcode is None
    assert s.ytdlp_path == Path(r"C:\mine\yt-dlp.exe")


def test_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEMAPP_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("STEMAPP_PORT", "9123")
    monkeypatch.setenv("STEMAPP_HOST", "0.0.0.0")
    monkeypatch.setenv("STEMAPP_PASSCODE", "secret")
    monkeypatch.setenv("STEMAPP_YTDLP_PATH", str(tmp_path / "yt.exe"))
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.data_dir == tmp_path / "d"
    assert s.port == 9123
    assert s.host == "0.0.0.0"
    assert s.passcode == "secret"
    assert s.ytdlp_path == tmp_path / "yt.exe"


def test_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    content = f"STEMAPP_DATA_DIR={tmp_path / 'fromfile'}\nSTEMAPP_PORT=8111\n"
    env.write_text(content, encoding="utf-8")
    s = Settings(_env_file=env)  # type: ignore[call-arg]
    assert s.data_dir == tmp_path / "fromfile"
    assert s.port == 8111


def test_dirs_created_on_access(settings: Settings) -> None:
    assert not settings.data_dir.exists()
    assert settings.db_path == settings.data_dir / "stemapp.db"
    assert settings.data_dir.is_dir()
    for d, name in [
        (settings.tracks_dir, "tracks"),
        (settings.stems_dir, "stems"),
        (settings.cache_dir, "cache"),
        (settings.models_dir, "models"),
    ]:
        assert d == settings.data_dir / name
        assert d.is_dir()
