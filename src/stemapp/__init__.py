"""stemapp: 個人用の楽曲 stem 分割・再生アプリ。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stemapp")
except PackageNotFoundError:  # pragma: no cover - 未インストールで直接読み込んだとき
    __version__ = "0.0.0"

__all__ = ["__version__"]
