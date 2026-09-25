"""分割後の配信用データ（stream rendition と波形 peaks）を作る。

- stream rendition: 各 stem の master（FLAC）を ffmpeg で圧縮し、
  `data/stems/<job_id>/stream/<code>.<拡張子>` に置いて STEM_RENDITION（purpose=stream）に登録する。
  形式は `StreamFormat`（今は Opus 128kbps / WebM。iPhone 用に AAC も選べる）。
- peaks: `data/stems/<job_id>/peaks/<code>_<samples_per_px>.stpk` に置いて WAVEFORM に登録する。

DB へは flush まで（commit は呼び出し側）。
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
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


def missing_delivery(
    session: Session, job_id: int, levels: Iterable[int] = DEFAULT_LEVELS
) -> list[str]:
    """配信用データ（stream rendition・全解像度の peaks）が欠けている stem の code。

    stem が1つも無いジョブも「欠けている」とみなし、["*"] を返す。
    """
    rows = session.execute(
        select(Stem.stem_id, StemType.code)
        .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
        .where(Stem.job_id == job_id)
        .order_by(StemType.display_order)
    ).all()
    if not rows:
        return ["*"]
    ids = [r.stem_id for r in rows]
    streams = set(
        session.scalars(
            select(StemRendition.stem_id).where(
                StemRendition.stem_id.in_(ids), StemRendition.purpose == PURPOSE_STREAM
            )
        )
    )
    waves: dict[int, set[int]] = {}
    for stem_id, spp in session.execute(
        select(Waveform.stem_id, Waveform.samples_per_px).where(Waveform.stem_id.in_(ids))
    ):
        waves.setdefault(stem_id, set()).add(spp)
    want = set(levels)
    return [
        r.code
        for r in rows
        if r.stem_id not in streams or not want <= waves.get(r.stem_id, set())
    ]


def _drop_delivery(session: Session, settings: Settings, job_id: int) -> None:
    """ジョブの stream rendition・peaks（DB の行とファイル）を消す（commit まで）。"""
    stem_ids = select(Stem.stem_id).where(Stem.job_id == job_id)
    session.execute(
        delete(StemRendition).where(
            StemRendition.stem_id.in_(stem_ids), StemRendition.purpose == PURPOSE_STREAM
        )
    )
    session.execute(delete(Waveform).where(Waveform.stem_id.in_(stem_ids)))
    session.commit()
    out_dir = settings.stems_dir / str(job_id)
    shutil.rmtree(out_dir / PURPOSE_STREAM, ignore_errors=True)
    shutil.rmtree(out_dir / "peaks", ignore_errors=True)


def rebuild_delivery_files(
    session: Session,
    settings: Settings,
    job_id: int,
    *,
    encoder: FfmpegRunner | None = None,
    progress: ProgressCallback | None = None,
) -> None:
    """配信用データを作り直す（既存の stream rendition・peaks を消してから作る。commit まで）。

    失敗したら作りかけ（DB の行とファイル）を消して例外を出す（master には触れない）。
    """
    _drop_delivery(session, settings, job_id)
    try:
        create_delivery_files(session, settings, job_id, encoder=encoder, progress=progress)
        session.commit()
    except BaseException:
        session.rollback()
        _drop_delivery(session, settings, job_id)
        raise
