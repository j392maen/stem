"""beat_this（CPJKU、MIT）で拍と小節の頭を求める解析器。

beat_this・torch はこのモジュールの `analyze` の中でだけ import する（GPU 依存が無い環境でも
他の部分が動くように）。

- 入力は元の曲（normalized.wav、44.1kHz ステレオ）。beat_this 側でモノラル化・22.05kHz に
  変換する。音声は soundfile で読む（torchaudio の読み込み機能は使わない）。
- テンポ・拍子を固定する後処理（DBN）は使わない（テンポが途中で変わる曲に対応するため）。
- 重みは `<models_dir>/beat_this/beat_this-<checkpoint>.ckpt` に置く。無ければ初回に
  ダウンロードする（beat_this 既定のキャッシュ先＝ホームの torch hub は使わない）。
- 使い終わったら GPU メモリを解放する。
"""

from __future__ import annotations

import logging
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import soundfile as sf

from stemapp.audio import SAMPLE_RATE, as_stereo
from stemapp.beats.base import AudioInput, BeatAnalysisError, BeatResult

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "final0"
# beat_this の inference.CHECKPOINT_URL と同じ（import せずに重みを落とせるよう写しておく）
CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"


def _beat_this_version() -> str:
    try:
        return metadata.version("beat-this")
    except metadata.PackageNotFoundError:
        return "?"


class BeatThisAnalyzer:
    def __init__(
        self,
        models_dir: Path,
        *,
        device: str | None = None,
        checkpoint: str = DEFAULT_CHECKPOINT,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.device = device  # None なら GPU があれば cuda、無ければ cpu
        self.checkpoint = checkpoint

    @property
    def name(self) -> str:
        return f"beat_this {_beat_this_version()} {self.checkpoint}"

    @property
    def weights_path(self) -> Path:
        return self.models_dir / "beat_this" / f"beat_this-{self.checkpoint}.ckpt"

    def ensure_weights(self) -> Path:
        """重みが無ければダウンロードする（途中は .part に書き、終わってから名前を変える）。"""
        path = self.weights_path
        if path.is_file():
            return path
        from torch.hub import download_url_to_file

        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{CHECKPOINT_URL}/{self.checkpoint}.ckpt"
        tmp = path.with_suffix(".ckpt.part")
        log.info("beat_this の重みをダウンロードします: %s → %s", url, path)
        try:
            download_url_to_file(url, str(tmp), progress=False)
            tmp.replace(path)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise BeatAnalysisError(f"beat_this の重みをダウンロードできませんでした: {e}") from e
        return path

    def analyze(self, audio: AudioInput) -> BeatResult:
        import torch
        from beat_this.inference import Audio2Beats

        if isinstance(audio, Path):
            data, sr = sf.read(str(audio), dtype="float32", always_2d=True)
        else:
            data, sr = np.asarray(audio, dtype=np.float32), SAMPLE_RATE
        data = as_stereo(data)
        if data.shape[0] == 0:
            raise BeatAnalysisError("音声が空です。")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = self.ensure_weights()
        t0 = time.perf_counter()
        model = Audio2Beats(checkpoint_path=str(weights), device=device, dbn=False)
        try:
            beats, downbeats = model(data, sr)
        finally:
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        seconds = time.perf_counter() - t0
        log.info(
            "beat_this: 拍 %d・小節の頭 %d（%s、%.1f 秒）",
            len(beats), len(downbeats), device, seconds,
        )
        return BeatResult(
            beats=[float(x) for x in beats],
            downbeats=[float(x) for x in downbeats],
            analyzer=self.name,
            device=device,
            seconds=seconds,
        )
