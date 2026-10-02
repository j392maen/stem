"""MSST 形式の BS-Roformer（多 stem）を、必要な stem だけ動かして推論する（T07b）。

MVSep Mega 53 stems（53 stem）は 1 本の共通部分（Transformer）と stem ごとの「マスク推定器」から
なる。作者は VRAM 16GB 以上を勧めているが、メモリの大半は 53 個のマスク推定器の中間値と出力なので、
使う stem のマスク推定器だけを GPU に載せて動かせば 8GB に収まる（docs/research/R02-mega53.md）。

チャンク分割と重ね合わせは MSST の `utils/model_utils.py` の `demix`（generic 方式）と同じ考え方:
両端を反射で埋め、チャンクを `chunk_size / num_overlap` ずつずらし、両端を線形にフェードする窓で
重ねて足し、窓の合計で割る。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stemapp.audio import SAMPLE_RATE

log = logging.getLogger(__name__)

Progress = Callable[[float], None]


def load_config(path: Path) -> dict[str, Any]:
    """MSST の yaml を読む（`!!python/tuple` を tuple として読む。ほかの python タグは読まない）"""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple[Any, ...]:
        return tuple(loader.construct_sequence(node))  # type: ignore[arg-type]

    _Loader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple)
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)  # SafeLoader の派生（任意のオブジェクトは作らない）


_MASK_PREFIX = "mask_estimators."


def select_stems_state(state: Mapping[str, Any], keep: Sequence[int]) -> dict[str, Any]:
    """state_dict から keep の順のマスク推定器だけを残し、番号を 0, 1, ... に付け直す

    `mask_estimators.<i>.…` のうち i が keep に無いものは捨て、keep[j] のものを
    `mask_estimators.<j>.…` にする（出力の順 = keep の順）。ほかのキーはそのまま。
    """
    new_index = {old: new for new, old in enumerate(keep)}
    if len(new_index) != len(keep):
        raise ValueError("同じ stem が 2 回指定されています。")
    out: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith(_MASK_PREFIX):
            idx, _, rest = key[len(_MASK_PREFIX) :].partition(".")
            j = new_index.get(int(idx))
            if j is None:
                continue
            key = f"{_MASK_PREFIX}{j}.{rest}"
        out[key] = value
    return out


@dataclass
class LoadedModel:
    model: Any  # BSRoformer（選んだ stem のマスク推定器だけを持つ）
    stems: list[str]  # 出力の順
    sample_rate: int
    chunk_size: int
    num_overlap: int
    device: str


def load_model(
    ckpt_path: Path,
    config_path: Path,
    stems: Sequence[str],
    device: str,
) -> LoadedModel:
    """重みを読み、stems のマスク推定器だけを残したモデルを device に載せる。"""
    import torch

    from stemapp.separation.msst.bs_roformer import BSRoformer

    cfg = load_config(config_path)
    instruments: list[str] = list(cfg["training"]["instruments"])
    unknown = [s for s in stems if s not in instruments]
    if unknown:
        raise ValueError(f"モデルに無い stem です: {', '.join(unknown)}")
    if not stems:
        raise ValueError("stem を 1 つ以上指定してください。")
    sample_rate = int(cfg["audio"].get("sample_rate", 44100))
    if sample_rate != SAMPLE_RATE:
        # stemapp の音声はすべて SAMPLE_RATE（read_audio が確かめる）。
        # 違うモデルは変換しないと使えない
        raise ValueError(
            f"モデルのサンプルレート {sample_rate} Hz が stemapp の {SAMPLE_RATE} Hz と違います。"
        )
    keep = [instruments.index(s) for s in stems]
    # 使う stem のマスク推定器だけを持つモデルを組み立て、その分の重みだけを読み込む
    # （53 個すべてを fp32 で作ってから捨てるより、CPU メモリと時間が少なくて済む）。
    # mmap: ファイルを丸ごとメモリに読まず、使う重みだけを読む
    model = BSRoformer(**{**cfg["model"], "num_stems": len(keep)})
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:  # mmap できない古い保存形式
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(select_stems_state(state, keep))
    del state
    model.eval()
    model.to(device)
    inf = cfg.get("inference", {})
    return LoadedModel(
        model=model,
        stems=list(stems),
        sample_rate=sample_rate,
        chunk_size=int(inf.get("chunk_size", cfg["audio"]["chunk_size"])),
        num_overlap=int(inf.get("num_overlap", 2)),
        device=device,
    )


def fade_window(size: int, fade: int) -> np.ndarray:
    """両端を fade サンプルずつ線形にフェードする窓（MSST の _getWindowingArray と同じ形）。"""
    w = np.ones(size, dtype=np.float32)
    if fade > 0:
        w[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        w[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return w


def demix(
    forward: Callable[[np.ndarray], np.ndarray],
    mix: np.ndarray,
    num_stems: int,
    chunk_size: int,
    num_overlap: int,
    progress: Progress | None = None,
) -> np.ndarray:
    """mix (channels, samples) をチャンクに分けて forward にかけ、(stems, channels, samples) で返す

    forward は (channels, chunk_size) を受け取り (stems, channels, chunk_size) を返す関数
    （GPU を使わないテストでは偽の関数を渡す）。
    """
    ch, n = mix.shape
    step = max(1, chunk_size // num_overlap)
    fade = chunk_size // 10
    border = chunk_size - step
    padded = n > 2 * border and border > 0
    x = np.pad(mix, ((0, 0), (border, border)), mode="reflect") if padded else mix
    total = x.shape[1]
    window = fade_window(chunk_size, fade)
    out = np.zeros((num_stems, ch, total), dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    starts = list(range(0, max(total - chunk_size, 0) + step, step))
    for k, i in enumerate(starts):
        part = x[:, i : i + chunk_size]
        length = part.shape[1]
        if length < chunk_size:
            # 最後のチャンクの足りない分（と短い曲）は、反射できれば反射、無理なら 0 で埋める
            mode = "reflect" if length > chunk_size // 2 + 1 else "constant"
            part = np.pad(part, ((0, 0), (0, chunk_size - length)), mode=mode)
        y = forward(np.ascontiguousarray(part, dtype=np.float32))
        w = window.copy()
        if i == 0:
            w[:fade] = 1.0
        if i + step >= total:
            w[-fade:] = 1.0
        out[:, :, i : i + length] += y[:, :, :length] * w[:length]
        weight[i : i + length] += w[:length]
        if progress is not None:
            progress((k + 1) / len(starts))
    out /= np.maximum(weight, 1e-8)
    if padded:
        out = out[:, :, border : border + n]
    return out


def separate(
    loaded: LoadedModel,
    mix: np.ndarray,
    *,
    chunk_scale: float = 1.0,
    use_fp16: bool = True,
    progress: Progress | None = None,
) -> dict[str, np.ndarray]:
    """mix (samples, 2) float32 を分け、{stem 名: (samples, 2) float32} を返す。"""
    import torch

    chunk = max(loaded.sample_rate, int(loaded.chunk_size * chunk_scale))
    use_amp = use_fp16 and loaded.device.startswith("cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    model = loaded.model

    def forward(part: np.ndarray) -> np.ndarray:
        with torch.inference_mode(), ctx:
            t = torch.from_numpy(part).unsqueeze(0).to(loaded.device)
            y = model(t)  # (1, stems, channels, samples)
            return y[0].float().cpu().numpy()

    log.info(
        "MSST: stems=%d chunk=%d overlap=%d device=%s fp16=%s",
        len(loaded.stems), chunk, loaded.num_overlap, loaded.device, use_amp,
    )
    y = demix(
        forward, np.ascontiguousarray(mix.T, dtype=np.float32), len(loaded.stems), chunk,
        loaded.num_overlap, progress,
    )
    return {name: np.ascontiguousarray(y[i].T) for i, name in enumerate(loaded.stems)}
