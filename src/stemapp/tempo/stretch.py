"""音声の伸縮（ピッチを保ったまま速度を変える）。

- `FfmpegStretcher`: ffmpeg の rubberband フィルタで伸縮し、配信用の形式
  （stream と同じ Opus 128kbps / WebM）で書く。
  長さは `apad` と `atrim` で必ず frames サンプル（44.1kHz）にそろえる
  （rubberband の出力の長さが stem ごとに数サンプル違っても、全 stem が同じ長さになる）。
  ffmpeg は `stemapp.proc.popen_bound` で起動する（ワーカーが終われば一緒に終わる）。
- `FakeStretcher`: テスト用。線形補間で長さだけを変えた WAV を書く（ffmpeg を使わない）。

どちらも `should_stop()` が True になったら処理をやめて `StretchCanceled` を出す。
"""

from __future__ import annotations

import collections
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from stemapp.audio import SAMPLE_RATE, AudioError, as_stereo
from stemapp.delivery import DEFAULT_STREAM_FORMAT, StreamFormat
from stemapp.proc import popen_bound

POLL_SEC = 0.2
KILL_WAIT_SEC = 10.0

ProgressFn = Callable[[float], None]
StopFn = Callable[[], bool]


class StretchCanceled(RuntimeError):
    """止める指示（キャンセル・ワーカーの停止）で伸縮をやめた。"""


class Stretcher(Protocol):
    def __call__(
        self, src: Path, dst: Path, ratio: float, frames: int,
        progress: ProgressFn, should_stop: StopFn,
    ) -> None:
        """src（master の FLAC）を速度 ratio 倍に伸縮し、長さ frames（44.1kHz）で dst に書く。"""
        ...


def rubberband_filter(ratio: float, frames: int, fmt: StreamFormat = DEFAULT_STREAM_FORMAT) -> str:
    """-af のフィルタ。伸縮 → 足りなければ無音を足す → frames で切る →（必要なら）リサンプル。"""
    chain = [f"rubberband=tempo={ratio:.6f}", "apad", f"atrim=end_sample={int(frames)}"]
    if fmt.sample_rate is not None:
        chain.append(f"aresample={fmt.sample_rate}")
    return ",".join(chain)


def stretch_args(
    src: Path, dst: Path, ratio: float, frames: int, fmt: StreamFormat = DEFAULT_STREAM_FORMAT
) -> list[str]:
    """ffmpeg の引数（先頭の実行ファイルを除く）。進み具合を標準出力に出す。"""
    return [
        "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
        "-y", "-i", str(src), "-vn",
        "-af", rubberband_filter(ratio, frames, fmt),
        "-c:a", fmt.encoder, "-b:a", f"{fmt.bitrate_kbps}k",
        "-f", fmt.container, str(dst),
    ]


class FfmpegStretcher:
    """ffmpeg の rubberband フィルタ（Rubber Band ライブラリ。ffmpeg 8 では R2 エンジン）。"""

    def __init__(self, fmt: StreamFormat = DEFAULT_STREAM_FORMAT, exe: str | None = None) -> None:
        self.fmt = fmt
        self.exe = exe

    def __call__(
        self, src: Path, dst: Path, ratio: float, frames: int,
        progress: ProgressFn, should_stop: StopFn,
    ) -> None:
        exe = self.exe or shutil.which("ffmpeg")
        if exe is None:
            raise AudioError("ffmpeg が見つかりません。インストールして PATH に追加してください。")
        dst.parent.mkdir(parents=True, exist_ok=True)
        total_us = frames / SAMPLE_RATE * 1e6
        tail: collections.deque[str] = collections.deque(maxlen=20)
        proc = popen_bound(
            [exe, *stretch_args(src, dst, ratio, frames, self.fmt)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )

        def read() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                key, _, value = line.partition("=")
                if key == "out_time_us" and value.lstrip("-").isdigit() and total_us > 0:
                    progress(min(1.0, max(0.0, int(value) / total_us)))
                elif key not in {
                    "out_time_us", "out_time_ms", "out_time", "bitrate", "total_size", "speed",
                    "progress", "dup_frames", "drop_frames", "frame", "fps",
                } and line and not key.startswith("stream_"):
                    tail.append(line)

        reader = threading.Thread(target=read, name="stretch-progress", daemon=True)
        reader.start()
        try:
            while proc.poll() is None:
                if should_stop():
                    proc.kill()
                    proc.wait(timeout=KILL_WAIT_SEC)
                    raise StretchCanceled("伸縮をやめました。")
                time.sleep(POLL_SEC)
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            raise
        finally:
            reader.join(timeout=5)
        if proc.returncode != 0:
            detail = tail[-1] if tail else f"終了コード {proc.returncode}"
            raise AudioError(f"ffmpeg での伸縮に失敗しました: {detail}")
        if not dst.is_file():
            raise AudioError(f"伸縮した音声ができていません: {dst}")
        progress(1.0)


class FakeStretcher:
    """テスト用: 線形補間で長さを frames にした 16bit WAV を書く（ピッチも変わる。音質は問わない）。

    delay_sec をかけて少しずつ進む（進み具合・キャンセルのテスト用）。fail=True なら失敗する。
    """

    def __init__(self, delay_sec: float = 0.0, fail: bool = False, steps: int = 10) -> None:
        self.delay_sec = delay_sec
        self.fail = fail
        self.steps = steps
        self.calls: list[tuple[Path, float, int]] = []
        self._lock = threading.Lock()

    def __call__(
        self, src: Path, dst: Path, ratio: float, frames: int,
        progress: ProgressFn, should_stop: StopFn,
    ) -> None:
        with self._lock:
            self.calls.append((src, ratio, frames))
        for i in range(self.steps):
            if should_stop():
                raise StretchCanceled("伸縮をやめました。")
            if self.delay_sec:
                time.sleep(self.delay_sec / self.steps)
            progress((i + 1) / self.steps * 0.9)
        if self.fail:
            raise AudioError("伸縮に失敗しました（テスト）。")
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        data = as_stereo(data)
        n = data.shape[0]
        x = np.linspace(0.0, max(0, n - 1), num=int(frames)) if frames > 0 else np.zeros(0)
        out = np.stack(
            [np.interp(x, np.arange(n), data[:, ch]) for ch in range(data.shape[1])], axis=1
        ).astype(np.float32)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 拡張子（.webm）に関係なく WAV で書く（ブラウザの decodeAudioData は中身で判定する）
        sf.write(str(dst), out, sr, subtype="PCM_16", format="WAV")
        progress(1.0)
