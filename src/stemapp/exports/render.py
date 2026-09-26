"""書き出しの音声ファイルを作る（DB には触れない）。

- 元の音源は各 stem の master（24bit FLAC）。呼び出し側が DB（STEM_RENDITION.file_path）から
  引いたパスを渡す（ここではパスを組み立てない）。
- 形式: WAV（24bit）/ FLAC（24bit）は soundfile で書く。MP3（320kbps）は一時的な
  32bit float の WAV を作り、ffmpeg（libmp3lame）で変換する。ffmpeg は `stemapp.audio.run_ffmpeg`
  （`stemapp.proc.run_bound` で起動）か、テスト用に差し替えた関数で呼ぶ。
- 音量: stem は分割時の倍率（SEPARATION_JOB.output_gain_db）がかかったまま書き出す（戻さない）。
  mix は gain_db をかけて足し、合計が ±1 を超えたら全体を同じ倍率で下げる（クリップさせない）。
"""

from __future__ import annotations

import math
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from stemapp.audio import SAMPLE_RATE, AudioError, FfmpegRunner, read_audio, run_ffmpeg

FORMAT_WAV = "wav"
FORMAT_FLAC = "flac"
FORMAT_MP3 = "mp3"
FORMATS: tuple[str, ...] = (FORMAT_WAV, FORMAT_FLAC, FORMAT_MP3)
MP3_BITRATE_KBPS = 320

TYPE_SINGLE = "single"
TYPE_ALL = "all"
TYPE_MIX = "mix"
EXPORT_TYPES: tuple[str, ...] = (TYPE_SINGLE, TYPE_ALL, TYPE_MIX)

# 形式ごとの画面表示
FORMAT_LABELS: dict[str, str] = {
    FORMAT_WAV: "WAV 24bit",
    FORMAT_FLAC: "FLAC 24bit",
    FORMAT_MP3: f"MP3 {MP3_BITRATE_KBPS}kbps",
}

# (0〜1 の進み具合, 画面に出す段階の説明)
ProgressFn = Callable[[float, str], None]


class ExportError(RuntimeError):
    """書き出しの失敗（メッセージは日本語）。"""


@dataclass(frozen=True)
class SourceStem:
    """書き出す stem 1つ。path は master の FLAC、name は ZIP の中のファイル名など。"""

    code: str
    display_name: str
    path: Path
    gain_db: float = 0.0


def _noop(_p: float, _s: str) -> None:
    return None


def _read(src: SourceStem) -> np.ndarray:
    if not src.path.is_file():
        raise ExportError(f"「{src.display_name}」の音声ファイルが見つかりません。")
    try:
        return read_audio(src.path)
    except (AudioError, sf.LibsndfileError, RuntimeError) as e:
        raise ExportError(f"「{src.display_name}」の音声を読めません: {e}") from e


def mp3_encode_args(src: Path, dst: Path) -> list[str]:
    return [
        "-y", "-i", str(src), "-vn",
        "-c:a", "libmp3lame", "-b:a", f"{MP3_BITRATE_KBPS}k",
        "-ar", str(SAMPLE_RATE),
        "-f", "mp3", str(dst),
    ]


def write_audio(
    dst: Path, data: np.ndarray, fmt: str, runner: FfmpegRunner | None = None
) -> Path:
    """data（(samples, 2) float、±1 以内）を fmt で dst に書く。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 整数の PCM に変換するとき ±1 を超えた値が折り返さないよう、念のため切り詰める
    clipped = np.clip(data, -1.0, 1.0)
    if fmt == FORMAT_WAV:
        sf.write(str(dst), clipped, SAMPLE_RATE, subtype="PCM_24", format="WAV")
    elif fmt == FORMAT_FLAC:
        sf.write(str(dst), clipped, SAMPLE_RATE, subtype="PCM_24", format="FLAC")
    elif fmt == FORMAT_MP3:
        tmp = dst.with_name(dst.name + ".tmp.wav")
        try:
            sf.write(str(tmp), clipped.astype(np.float32), SAMPLE_RATE, subtype="FLOAT",
                     format="WAV")
            try:
                (runner or run_ffmpeg)(mp3_encode_args(tmp, dst))
            except AudioError as e:
                raise ExportError(f"MP3 に変換できませんでした: {e}") from e
        finally:
            tmp.unlink(missing_ok=True)
        if not dst.is_file():
            raise ExportError("MP3 ができていません。")
    else:
        raise ExportError(f"対応していない形式です: {fmt}")
    return dst


def mix_stems(
    stems: Sequence[SourceStem], progress: ProgressFn = _noop
) -> tuple[np.ndarray, float]:
    """gain_db をかけて足す。合計が ±1 を超えたら全体を下げる。

    戻り値: (混ぜた音 (samples, 2) float64, 下げた量 dB（下げなければ 0.0）)。
    長さが違う stem は長いほうに合わせる（足りない部分は無音）。
    """
    if not stems:
        raise ExportError("ミックスする stem がありません。")
    total: np.ndarray | None = None
    for i, src in enumerate(stems):
        progress(i / len(stems), f"読み込み中（{i + 1}/{len(stems)}）: {src.display_name}")
        data = _read(src).astype(np.float64)
        if src.gain_db:
            data *= 10.0 ** (src.gain_db / 20.0)
        if total is None:
            total = data
        else:
            if data.shape[0] > total.shape[0]:
                total, data = data, total
            total[: data.shape[0]] += data
        del data
    assert total is not None
    peak = float(np.max(np.abs(total))) if total.size else 0.0
    gain_db = 0.0
    if peak > 1.0:
        total /= peak
        gain_db = -20.0 * math.log10(peak)
    return total, gain_db


@dataclass(frozen=True)
class RenderResult:
    path: Path
    bytes: int
    mix_gain_db: float = 0.0


def render_single(
    src: SourceStem, dst: Path, fmt: str, runner: FfmpegRunner | None = None,
    progress: ProgressFn = _noop,
) -> RenderResult:
    progress(0.1, f"読み込み中: {src.display_name}")
    data = _read(src)
    progress(0.5, f"{FORMAT_LABELS[fmt]} に変換中")
    write_audio(dst, data, fmt, runner)
    return RenderResult(dst, dst.stat().st_size)


def render_mix(
    stems: Sequence[SourceStem], dst: Path, fmt: str, runner: FfmpegRunner | None = None,
    progress: ProgressFn = _noop,
) -> RenderResult:
    data, gain_db = mix_stems(stems, lambda p, s: progress(p * 0.7, s))
    progress(0.75, f"{FORMAT_LABELS[fmt]} に変換中")
    write_audio(dst, data, fmt, runner)
    return RenderResult(dst, dst.stat().st_size, gain_db)


def _unique(name: str, used: set[str]) -> str:
    if name.lower() not in used:
        used.add(name.lower())
        return name
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    n = 2
    while True:
        cand = f"{stem} ({n}){'.' + ext if ext else ''}"
        if cand.lower() not in used:
            used.add(cand.lower())
            return cand
        n += 1


def render_zip(
    items: Sequence[tuple[SourceStem, str]], dst: Path, fmt: str,
    runner: FfmpegRunner | None = None, progress: ProgressFn = _noop,
) -> RenderResult:
    """items は (stem, ZIP の中のファイル名)。音声は圧縮済みなので ZIP では圧縮しない。"""
    if not items:
        raise ExportError("書き出す stem がありません。")
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = dst.parent / "work"
    used: set[str] = set()
    try:
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for i, (src, name) in enumerate(items):
                progress(i / len(items), f"変換中（{i + 1}/{len(items)}）: {src.display_name}")
                data = _read(src)
                part = write_audio(work / f"{i}.{fmt}", data, fmt, runner)
                del data
                zf.write(part, arcname=_unique(name, used))
                part.unlink(missing_ok=True)
    finally:
        if work.is_dir():
            for p in work.iterdir():
                p.unlink(missing_ok=True)
            work.rmdir()
    return RenderResult(dst, dst.stat().st_size)
