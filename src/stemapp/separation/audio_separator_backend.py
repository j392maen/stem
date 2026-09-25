"""python-audio-separator を使う分離器。

`audio_separator` と `torch` を import するのはこのモジュールの中だけ（GPU 依存の無い環境でも
他のコードとテストが動くように、import は関数の中で行う）。

audio-separator の `Separator.separate()` は入力と各出力を個別に正規化（ピーク 0.9 に縮める）
してファイルに書くため、stem の合計が元の曲と一致しなくなる。そこでモデルのロードだけ
audio-separator に任せ、推論はロード済みインスタンスの `demix()` を直接呼んで、
正規化せずに配列のまま受け取る。
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from stemapp.audio import read_audio
from stemapp.separation.base import (
    DEVICE_CPU,
    DEVICE_CUDA,
    OPT_CHUNK_SCALE,
    OPT_OVERLAP,
    OPT_SEGMENT_SIZE,
    OPT_TTA,
)

log = logging.getLogger(__name__)

# audio-separator の出力名（ファイル名の "(Vocals)" の部分。小文字で比較）→ stem 名。
# role ごとに持つ。karaoke は "(Vocals)"=lead、"(Instrumental)"=それ以外。
OUTPUT_NAME_MAP: dict[str, dict[str, str]] = {
    "multistem": {
        "vocals": "vocals",
        "drums": "drums",
        "bass": "bass",
        "guitar": "guitar",
        "piano": "piano",
        "other": "other",
    },
    "vocals": {
        "vocals": "vocals",
        "instrumental": "instrumental",
        "other": "instrumental",
        "no vocals": "instrumental",
    },
    "karaoke": {
        "vocals": "lead_vocal",
        "lead vocals": "lead_vocal",
        "karaoke": "lead_vocal",
        "instrumental": "backing_vocal",
        "other": "backing_vocal",
        "no vocals": "backing_vocal",
    },
}

MIN_SEGMENT = 32  # チャンクを縮めるときの下限（dim_t）


def map_outputs(role: str, outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """audio-separator の出力名を stem 名に置き換える。表に無い名前はエラー。"""
    table = OUTPUT_NAME_MAP.get(role)
    if table is None:
        raise ValueError(f"未知の role: {role}")
    mapped: dict[str, np.ndarray] = {}
    for name, arr in outputs.items():
        key = table.get(name.strip().lower())
        if key is None:
            raise RuntimeError(
                f"モデルの出力「{name}」を stem に対応づけられません（role={role}）。"
                "OUTPUT_NAME_MAP に追加してください。"
            )
        if key in mapped:
            raise RuntimeError(f"出力「{name}」が {key} に重複して対応づけられました。")
        mapped[key] = arr
    return mapped


class AudioSeparatorBackend:
    """audio-separator でモデルを1つずつロードし、使い終わったら解放する分離器。"""

    def __init__(self, models_dir: Path, work_dir: Path, use_fp16: bool = True) -> None:
        self.models_dir = Path(models_dir)
        self.work_dir = Path(work_dir)
        self.use_fp16 = use_fp16

    # --- GPU メモリ計測 -------------------------------------------------------------

    def reset_peak_memory(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self) -> float | None:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --- 分離 ----------------------------------------------------------------------

    def _load(self, model_filename: str, device: str) -> tuple[Any, Any]:
        """audio-separator の Separator を作り、モデルをロードして (separator, instance) を返す。"""
        import torch
        from audio_separator.separator import Separator as AsSeparator

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if device == DEVICE_CUDA and not torch.cuda.is_available():
            raise RuntimeError("CUDA が使えません（--cpu で CPU 実行できます）。")
        sep = AsSeparator(
            log_level=logging.WARNING,
            model_file_dir=str(self.models_dir),
            output_dir=str(self.work_dir),
            # 推論は demix() を直接呼び、autocast は自分でかける
            use_autocast=False,
        )
        if device == DEVICE_CPU:
            sep.torch_device = sep.torch_device_cpu
        try:
            sep.load_model(model_filename)  # 無ければ models_dir に自動でダウンロードされる
        except SystemExit as e:  # audio-separator はロード失敗時に sys.exit(1) する
            raise RuntimeError(
                f"モデル {model_filename} のロードに失敗しました（ファイル破損の可能性）。"
            ) from e
        instance = sep.model_instance
        if instance is None or not hasattr(instance, "demix"):
            raise RuntimeError(f"モデル {model_filename} は demix に対応していません。")
        return sep, instance

    @staticmethod
    def _release(sep: Any, instance: Any) -> None:
        import torch

        try:
            if sep is not None:
                sep.model_instance = None
            if instance is not None and hasattr(instance, "model_run"):
                instance.model_run = None
        finally:
            del sep, instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def separate(
        self,
        wav_path: Path,
        model_filename: str,
        options: Mapping[str, Any],
        device: str,
        role: str,
    ) -> dict[str, np.ndarray]:
        import torch

        mix = read_audio(wav_path)  # (samples, 2)
        n = mix.shape[0]
        sep = instance = None
        try:
            sep, instance = self._load(model_filename, device)
            cfg = instance.model_data_cfgdict
            base_segment = int(options.get(OPT_SEGMENT_SIZE) or cfg.inference.dim_t)
            scale = float(options.get(OPT_CHUNK_SCALE, 1.0))
            segment = max(MIN_SEGMENT, int(base_segment * scale))
            instance.segment_size = segment
            if options.get(OPT_OVERLAP):
                instance.overlap = int(options[OPT_OVERLAP])
            log.info(
                "%s: device=%s segment=%d overlap=%s tta=%s",
                model_filename, device, segment, instance.overlap, bool(options.get(OPT_TTA)),
            )

            use_autocast = self.use_fp16 and device == DEVICE_CUDA
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_autocast
                else nullcontext()
            )

            def demix(x: np.ndarray) -> dict[str, np.ndarray]:
                # demix は (2, samples) を受け取り {名前: (2, samples)} を返す
                with ctx:
                    src = instance.demix(
                        mix=np.ascontiguousarray(x.T), override_model_segment_size=True
                    )
                if not isinstance(src, dict):
                    target = cfg.training.target_instrument or "output"
                    src = {target: src}
                return {k: np.asarray(v, dtype=np.float32).T for k, v in src.items()}

            outputs = demix(mix)
            if options.get(OPT_TTA):
                # TTA: 位相反転と左右入れ替えでも分け、元に戻して平均する
                inv = demix(-mix)
                swp = demix(mix[:, ::-1])
                outputs = {
                    k: (v - inv[k] + swp[k][:, ::-1]) / 3.0 for k, v in outputs.items()
                }
            mapped = map_outputs(role, outputs)
            return {k: _fit(v, n) for k, v in mapped.items()}
        finally:
            self._release(sep, instance)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape[0] >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - x.shape[0], x.shape[1]), dtype=np.float32)])
