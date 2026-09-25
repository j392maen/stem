"""アプリ設定（環境変数 STEMAPP_* と .env から読む）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/stemapp/config.py → リポジトリ直下
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_YTDLP_PATH: Path = Path(r"C:\mine\yt-dlp.exe")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class Settings(BaseSettings):
    """stemapp の設定。

    ディレクトリ系のプロパティは、初めて参照したときにフォルダを作る。
    """

    model_config = SettingsConfigDict(
        env_prefix="STEMAPP_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = REPO_ROOT / "data"
    host: str = "127.0.0.1"
    port: int = 8000
    passcode: str | None = None
    ytdlp_path: Path = DEFAULT_YTDLP_PATH
    # アップロードできるファイルの大きさの上限（MB）
    max_upload_mb: int = 1024

    @property
    def data_root(self) -> Path:
        """データフォルダ（作成済みを保証）。"""
        return _ensure_dir(self.data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_root / "stemapp.db"

    @property
    def tracks_dir(self) -> Path:
        return _ensure_dir(self.data_dir / "tracks")

    @property
    def stems_dir(self) -> Path:
        return _ensure_dir(self.data_dir / "stems")

    @property
    def cache_dir(self) -> Path:
        return _ensure_dir(self.data_dir / "cache")

    @property
    def models_dir(self) -> Path:
        return _ensure_dir(self.data_dir / "models")


@lru_cache
def get_settings() -> Settings:
    """プロセス全体で共有する設定。"""
    return Settings()
