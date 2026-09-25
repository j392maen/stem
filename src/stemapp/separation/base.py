"""分離器の共通インターフェース。

分離器は「1つのモデルを入力 WAV に適用し、{出力名: 配列} を返す」だけを担う。
手順の組み立て・平均・残差補正はパイプライン（pipeline.py）が行う。

出力名は stem の名前にそろえる:
- role=multistem: STEM_TYPE の code（vocals, drums, bass, guitar, piano, other）
- role=vocals:    vocals, instrumental
- role=karaoke:   lead_vocal, backing_vocal
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

# options のキー（PRESET_STEP.options_json とパイプラインが足すもの）
OPT_OVERLAP = "overlap"  # 重ねる窓の数（大きいほど高品質・遅い）
OPT_TTA = "tta"  # 反転・左右入れ替えの平均（約3倍の時間）
OPT_SEGMENT_SIZE = "segment_size"  # チャンク長（dim_t）。省略時はモデルの既定値
OPT_CHUNK_SCALE = "chunk_scale"  # パイプラインが OOM 時に足す縮小率（1.0, 0.5, 0.25, ...）

DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"


@runtime_checkable
class Separator(Protocol):
    """1つのモデルを入力 WAV に適用する分離器。"""

    def separate(
        self,
        wav_path: Path,
        model_filename: str,
        options: Mapping[str, Any],
        device: str,
        role: str,
    ) -> dict[str, np.ndarray]:
        """wav_path（44.1kHz・ステレオ）を分け、{出力名: (samples, 2) float32} を返す。

        role はモデルの出力名を stem 名に対応づけるために使う（multistem / vocals / karaoke）。
        """
        ...

    def reset_peak_memory(self) -> None:
        """GPU メモリ最大使用量の計測をリセットする（GPU が無ければ何もしない）。"""
        ...

    def peak_memory_mb(self) -> float | None:
        """直近のリセット以降の GPU メモリ最大使用量（MB）。測れなければ None。"""
        ...


def is_oom_error(exc: BaseException) -> bool:
    """GPU のメモリ不足か（torch.cuda.OutOfMemoryError / "out of memory" を含む RuntimeError）。"""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
