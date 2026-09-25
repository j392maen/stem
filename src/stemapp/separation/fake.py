"""テスト用の偽の分離器（GPU・モデル不要）。

入力に固定の係数を掛けて出力を作る。multistem の係数の合計はわざと 1 からずらしてあり、
パイプラインの残差補正（mixture − Σstems を other に足す）が効くことを確かめられる。
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stemapp.audio import read_audio
from stemapp.separation.base import DEVICE_CUDA, OPT_CHUNK_SCALE

# 合計 0.97（1 にならない）
DEFAULT_MULTISTEM_COEFS: dict[str, float] = {
    "vocals": 0.30,
    "drums": 0.20,
    "bass": 0.15,
    "guitar": 0.10,
    "piano": 0.10,
    "other": 0.12,
}
DEFAULT_VOCALS_COEF = 0.40  # vocals モデルの vocals（instrumental は 1 − これ）
DEFAULT_KARAOKE_COEF = 0.70  # karaoke の lead（backing は 0.25 にして合計をずらす）
KARAOKE_BACKING_COEF = 0.25


@dataclass(frozen=True)
class FakeCall:
    model_filename: str
    role: str
    device: str
    options: dict[str, Any]


class FakeSeparator:
    """決まった計算で出力を作る分離器。

    - model_coefs: モデルごとの主出力の係数（vocals / karaoke 用）。無ければ既定値。
    - silent_stems: 0 を返す出力名（無音判定のテスト用）。
    - oom_times: device=cuda の呼び出しで、最初の N 回だけメモリ不足を起こす。
    - fail_models: この名前のモデルを呼ぶと例外を出す。
    - peak_mb: peak_memory_mb が返す値。
    - delay_sec: 1回の呼び出しごとに待つ秒数（キャンセルのテスト用）。
    """

    def __init__(
        self,
        *,
        multistem_coefs: Mapping[str, float] | None = None,
        model_coefs: Mapping[str, float] | None = None,
        silent_stems: Iterable[str] = (),
        oom_times: int = 0,
        fail_models: Iterable[str] = (),
        peak_mb: float | None = None,
        delay_sec: float = 0.0,
    ) -> None:
        self.multistem_coefs = dict(multistem_coefs or DEFAULT_MULTISTEM_COEFS)
        self.model_coefs = dict(model_coefs or {})
        self.silent_stems = set(silent_stems)
        self.oom_times = oom_times
        self.fail_models = set(fail_models)
        self.peak_mb = peak_mb
        self.delay_sec = delay_sec
        self.calls: list[FakeCall] = []

    def separate(
        self,
        wav_path: Path,
        model_filename: str,
        options: Mapping[str, Any],
        device: str,
        role: str,
    ) -> dict[str, np.ndarray]:
        self.calls.append(FakeCall(model_filename, role, device, dict(options)))
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        if model_filename in self.fail_models:
            raise RuntimeError(f"fake failure in {model_filename}")
        if device == DEVICE_CUDA and self.oom_times > 0:
            self.oom_times -= 1
            scale = options.get(OPT_CHUNK_SCALE, 1.0)
            raise RuntimeError(f"CUDA out of memory (fake, chunk_scale={scale})")

        x = read_audio(wav_path)
        if role == "multistem":
            coefs = self.multistem_coefs
        elif role == "vocals":
            c = self.model_coefs.get(model_filename, DEFAULT_VOCALS_COEF)
            coefs = {"vocals": c, "instrumental": 1.0 - c}
        elif role == "karaoke":
            c = self.model_coefs.get(model_filename, DEFAULT_KARAOKE_COEF)
            coefs = {"lead_vocal": c, "backing_vocal": KARAOKE_BACKING_COEF}
        else:
            raise ValueError(f"未知の role: {role}")
        out: dict[str, np.ndarray] = {}
        for name, c in coefs.items():
            if name in self.silent_stems:
                out[name] = np.zeros_like(x)
            else:
                out[name] = (x * np.float32(c)).astype(np.float32)
        return out

    def reset_peak_memory(self) -> None:
        return None

    def peak_memory_mb(self) -> float | None:
        return self.peak_mb
