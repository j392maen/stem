"""stem 分離（分離器の抽象・実装とパイプライン）。

`AudioSeparatorBackend` は GPU 依存（audio-separator, torch）があるので、ここでは import しない。
使うときは `stemapp.separation.audio_separator_backend` から import する。
"""

from stemapp.separation.base import Separator, is_oom_error
from stemapp.separation.fake import FakeSeparator

__all__ = ["FakeSeparator", "Separator", "is_oom_error"]
