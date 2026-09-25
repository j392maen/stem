"""音声の正規化と読み書き。

- 正規化: ffmpeg で 44.1kHz・ステレオ・32bit float の WAV にする。
  モノラルは同じ値のまま2チャンネルに複製する。
- 配列はいつも形 `(samples, 2)`、dtype float32。
- `audio_hash` は正規化後の PCM サンプル列（リトルエンディアン float32）の SHA-256。
  ファイルのヘッダは含めないので、同じ音なら書き出し方が違っても同じ値になる。
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
CHANNELS = 2
SILENCE_FLOOR_DB = -120.0

# ffmpeg の引数（先頭の "ffmpeg" を除く）を受け取って実行する関数。テストで差し替えられる。
FfmpegRunner = Callable[[Sequence[str]], None]


class AudioError(RuntimeError):
    """音声の読み込み・変換の失敗（メッセージは日本語）。"""


def run_ffmpeg(args: Sequence[str]) -> None:
    """PATH 上の ffmpeg を実行する。失敗したら AudioError。"""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioError("ffmpeg が見つかりません。インストールして PATH に追加してください。")
    proc = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        tail = detail[-1] if detail else f"終了コード {proc.returncode}"
        raise AudioError(f"ffmpeg での変換に失敗しました: {tail}")


def ffmpeg_normalize_args(src: Path, dst: Path) -> list[str]:
    """入力を 44.1kHz・float32 WAV にする ffmpeg 引数。

    チャンネルはモノラルかステレオにそろえる（3ch 以上はステレオにダウンミックス）。
    ffmpeg の `-ac 2` はモノラルを -3dB で左右に振り分けるため使わず、モノラルは
    `normalize_audio` が同じ値のまま2チャンネルに複製する。
    """
    return [
        "-y",
        "-i", str(src),
        "-vn",  # 動画・カバー画像を捨てる
        "-map", "0:a:0",
        "-af", "aformat=channel_layouts=mono|stereo",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_f32le",
        "-f", "wav",
        str(dst),
    ]


def as_stereo(data: np.ndarray) -> np.ndarray:
    """(samples,) / (samples, 1) / (samples, 2) を (samples, 2) float32 にする。"""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise AudioError(f"音声配列の形が不正です: {arr.shape}")
    if arr.shape[1] == 1:
        arr = np.repeat(arr, CHANNELS, axis=1)
    elif arr.shape[1] != CHANNELS:
        raise AudioError(f"チャンネル数が {arr.shape[1]} です（1 か 2 のみ対応）。")
    return np.ascontiguousarray(arr)


def read_audio(path: Path) -> np.ndarray:
    """WAV/FLAC を (samples, 2) float32 で読む。サンプルレートは 44.1kHz 前提。"""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise AudioError(f"サンプルレートが {sr}Hz です（{SAMPLE_RATE}Hz が必要）: {path}")
    return as_stereo(data)


def write_wav_float(path: Path, data: np.ndarray) -> None:
    """32bit float の WAV で書く（値は丸めない）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), as_stereo(data), SAMPLE_RATE, subtype="FLOAT", format="WAV")


def write_flac24(path: Path, data: np.ndarray) -> None:
    """24bit FLAC で書く。±1 を超える値は書く前に ±1 に切り詰める（件数は呼び出し側で記録）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(as_stereo(data), -1.0, 1.0)
    sf.write(str(path), clipped, SAMPLE_RATE, subtype="PCM_24", format="FLAC")


def audio_hash(data: np.ndarray) -> str:
    """正規化後 PCM（float32 リトルエンディアン、(samples, 2) をインターリーブ）の SHA-256。"""
    pcm = np.ascontiguousarray(as_stereo(data), dtype="<f4")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def rms_db(data: np.ndarray) -> float:
    """全サンプルの RMS を dBFS で返す（無音は SILENCE_FLOOR_DB）。"""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return SILENCE_FLOOR_DB
    rms = float(np.sqrt(np.mean(np.square(arr))))
    if rms <= 0.0:
        return SILENCE_FLOOR_DB
    return max(SILENCE_FLOOR_DB, 20.0 * float(np.log10(rms)))


def count_clipped(data: np.ndarray) -> int:
    """絶対値が 1 を超えるサンプル数。"""
    return int(np.count_nonzero(np.abs(data) > 1.0))


@dataclass(frozen=True)
class NormalizedAudio:
    path: Path
    data: np.ndarray  # (samples, 2) float32
    audio_hash: str

    @property
    def duration_sec(self) -> float:
        return self.data.shape[0] / SAMPLE_RATE


def normalize_audio(src: Path, dst: Path, runner: FfmpegRunner | None = None) -> NormalizedAudio:
    """src を正規化して dst（WAV）に書き、読み戻した配列と hash を返す。"""
    if not src.is_file():
        raise AudioError(f"入力ファイルが見つかりません: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    (runner or run_ffmpeg)(ffmpeg_normalize_args(src, dst))
    if not dst.is_file():
        raise AudioError(f"正規化した音声ができていません: {dst}")
    try:
        channels = sf.info(str(dst)).channels
        data = read_audio(dst)
    except (sf.LibsndfileError, RuntimeError) as e:
        raise AudioError(f"正規化した音声を読めません: {e}") from e
    if channels != CHANNELS:
        write_wav_float(dst, data)  # モノラルを複製したステレオで書き直す
    if data.shape[0] == 0:
        raise AudioError("音声が空です（長さ 0）。")
    return NormalizedAudio(path=dst, data=data, audio_hash=audio_hash(data))
