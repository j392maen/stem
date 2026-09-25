"""分割後の配信用データ（stream rendition と波形 peaks）を作る。

- stream rendition: 各 stem の master（FLAC）を ffmpeg で圧縮し、
  `data/stems/<job_id>/stream/<code>.<拡張子>` に置いて STEM_RENDITION（purpose=stream）に登録する。
  形式は `StreamFormat`（今は Opus 128kbps / WebM。iPhone 用に AAC も選べる）。
- peaks: `data/stems/<job_id>/peaks/<code>_<samples_per_px>.stpk` に置いて WAVEFORM に登録する。

DB へは flush まで（commit は呼び出し側）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.audio import SAMPLE_RATE, FfmpegRunner, read_audio, run_ffmpeg
from stemapp.config import Settings
from stemapp.library import data_relative, resolve_data_path
from stemapp.models import Stem, StemRendition, StemType, Waveform
from stemapp.peaks import DEFAULT_LEVELS, compute_peaks, write_peaks

log = logging.getLogger(__name__)

PURPOSE_MASTER = "master"
PURPOSE_STREAM = "stream"


@dataclass(frozen=True)
class StreamFormat:
    codec: str  # STEM_RENDITION.codec
    encoder: str  # ffmpeg のエンコーダ名
    container: str  # ffmpeg の -f
    extension: str
    bitrate_kbps: int
    media_type: str  # HTTP の Content-Type
    sample_rate: int | None = None  # 指定するとリサンプルする


OPUS_128 = StreamFormat(
    codec="opus",
    encoder="libopus",
    container="webm",
    extension="webm",
    bitrate_kbps=128,
    media_type="audio/webm",
    sample_rate=48000,  # Opus は 48kHz で符号化する
)
AAC_192 = StreamFormat(
    codec="aac",
    encoder="aac",
    container="ipod",
    extension="m4a",
    bitrate_kbps=192,
    media_type="audio/mp4",
)
DEFAULT_STREAM_FORMAT = OPUS_128

# 拡張子 → Content-Type（ファイル配信で使う）
MEDIA_TYPES: dict[str, str] = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".m4a": "audio/mp4",
    ".stpk": "application/octet-stream",
}


def stream_encode_args(src: Path, dst: Path, fmt: StreamFormat) -> list[str]:
    args = ["-y", "-i", str(src), "-vn", "-c:a", fmt.encoder, "-b:a", f"{fmt.bitrate_kbps}k"]
    if fmt.sample_rate is not None:
        args += ["-ar", str(fmt.sample_rate)]
    args += ["-f", fmt.container, str(dst)]
    return args


def encode_stream(
    src: Path, dst: Path, fmt: StreamFormat = DEFAULT_STREAM_FORMAT,
    runner: FfmpegRunner | None = None,
) -> Path:
    """src（FLAC など）を配信用の形式に変換する。失敗したら AudioError。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    (runner or run_ffmpeg)(stream_encode_args(src, dst, fmt))
    if not dst.is_file():
        raise RuntimeError(f"配信用の音声ができていません: {dst}")
    return dst


def fake_encoder(args: Sequence[str]) -> None:
    """テスト用の ffmpeg の代わり: 出力先に小さなダミーを書く（中身は音声ではない）。"""
    dst = Path(args[-1])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"FAKE-STREAM" * 100)


ProgressCallback = Callable[[float, str], None]


def create_delivery_files(
    session: Session,
    settings: Settings,
    job_id: int,
    *,
    stream_format: StreamFormat = DEFAULT_STREAM_FORMAT,
    levels: Iterable[int] = DEFAULT_LEVELS,
    encoder: FfmpegRunner | None = None,
    progress: ProgressCallback | None = None,
) -> None:
    """ジョブの全 stem について stream rendition と peaks を作り、DB に登録する（flush まで）。"""
    rows = session.execute(
        select(Stem, StemType, StemRendition)
        .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
        .join(StemRendition, StemRendition.stem_id == Stem.stem_id)
        .where(Stem.job_id == job_id, StemRendition.purpose == PURPOSE_MASTER)
        .order_by(StemType.display_order)
    ).all()
    if not rows:
        raise RuntimeError(f"job {job_id} に master の stem がありません。")
    out_dir = settings.stems_dir / str(job_id)
    levels = tuple(levels)
    for i, (stem, stype, master) in enumerate(rows):
        if progress is not None:
            progress(i / len(rows), f"配信用データを作成中（{i + 1}/{len(rows)}）")
        src = resolve_data_path(settings, master.file_path)

        data = read_audio(src)
        for spp, pk in compute_peaks(data, SAMPLE_RATE, levels).items():
            path = out_dir / "peaks" / f"{stype.code}_{spp}.stpk"
            write_peaks(path, pk)
            session.merge(
                Waveform(
                    stem_id=stem.stem_id,
                    samples_per_px=spp,
                    peaks_path=data_relative(settings, path),
                )
            )
        del data

        dst = out_dir / PURPOSE_STREAM / f"{stype.code}.{stream_format.extension}"
        encode_stream(src, dst, stream_format, runner=encoder)
        session.add(
            StemRendition(
                stem_id=stem.stem_id,
                purpose=PURPOSE_STREAM,
                codec=stream_format.codec,
                bitrate_kbps=stream_format.bitrate_kbps,
                file_path=data_relative(settings, dst),
                bytes=dst.stat().st_size,
            )
        )
        session.flush()
    if progress is not None:
        progress(1.0, "配信用データを作成中")
    log.info("job %d: 配信用データを作成しました（stem %d 個）。", job_id, len(rows))
