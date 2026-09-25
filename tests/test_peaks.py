from __future__ import annotations

import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from job_helpers import make_track
from stemapp.config import Settings
from stemapp.delivery import OPUS_128, create_delivery_files, stream_encode_args
from stemapp.library import resolve_data_path
from stemapp.models import Stem, StemRendition, Waveform
from stemapp.peaks import (
    DEFAULT_LEVELS,
    HEADER_SIZE,
    MAGIC,
    Peaks,
    PeaksError,
    compute_peaks,
    decode_peaks,
    encode_peaks,
    read_peaks,
)
from stemapp.seed import seed
from stemapp.separation import FakeSeparator
from stemapp.separation.pipeline import separate_track


def _direct(mono: np.ndarray, spp: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(0, mono.shape[0], spp)
    return np.minimum.reduceat(mono, idx), np.maximum.reduceat(mono, idx)


def test_compute_peaks_matches_direct_calculation() -> None:
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((50_000, 2)) * 0.3).astype(np.float32)
    mono = data.mean(axis=1, dtype=np.float32)
    out = compute_peaks(data, 44100)
    assert sorted(out) == list(DEFAULT_LEVELS)
    for spp, pk in out.items():
        assert pk.points == -(-50_000 // spp)  # 切り上げ
        assert pk.mins.dtype == np.int8 and pk.maxs.dtype == np.int8
        assert np.all(pk.mins <= pk.maxs)
        mn, mx = _direct(mono, spp)
        # 量子化後の範囲は元の波形を含む
        assert np.all(pk.mins / 127 <= mn + 1e-6)
        assert np.all(pk.maxs / 127 >= mx - 1e-6)
        assert np.all(np.abs(pk.mins / 127 - mn) <= 1 / 127 + 1e-6)


def test_peaks_clip_and_silence() -> None:
    loud = np.full((1000, 2), 1.5, dtype=np.float32)
    pk = compute_peaks(loud, 44100, [256])[256]
    assert pk.maxs.max() == 127 and pk.mins.min() >= -127
    silent = compute_peaks(np.zeros((1000, 2), dtype=np.float32), 44100, [256])[256]
    assert np.all(silent.mins == 0) and np.all(silent.maxs == 0)
    empty = compute_peaks(np.zeros((0, 2), dtype=np.float32), 44100, [256])[256]
    assert empty.points == 0


def test_encode_decode_roundtrip() -> None:
    pk = Peaks(1024, 44100, np.array([-3, -127], np.int8), np.array([5, 127], np.int8))
    blob = encode_peaks(pk)
    assert blob[:4] == MAGIC
    assert len(blob) == HEADER_SIZE + 4
    _magic, version, _r, spp, sr, n = struct.unpack_from("<4sHHIII", blob)
    assert (version, spp, sr, n) == (1, 1024, 44100, 2)
    back = decode_peaks(blob)
    assert back.samples_per_px == 1024 and back.sample_rate == 44100
    assert back.mins.tolist() == [-3, -127] and back.maxs.tolist() == [5, 127]
    with pytest.raises(PeaksError):
        decode_peaks(b"XXXX" + blob[4:])
    with pytest.raises(PeaksError):
        decode_peaks(blob[:-1])


def test_stream_encode_args() -> None:
    args = stream_encode_args(Path("in.flac"), Path("out.webm"), OPUS_128)
    assert args[args.index("-c:a") + 1] == "libopus"
    assert args[args.index("-b:a") + 1] == "128k"
    assert args[args.index("-f") + 1] == "webm"
    assert args[-1] == "out.webm"


@pytest.mark.ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません")
def test_delivery_files_with_real_ffmpeg(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    seed(session)
    track_id = make_track(session, settings, tmp_path, seconds=3.0)
    res = separate_track(
        session, settings, track_id, FakeSeparator(), preset_code="fast",
        postprocess=lambda s, st, jid, p: create_delivery_files(s, st, jid, progress=p),
    )
    stems = session.scalars(select(Stem).where(Stem.job_id == res.job_id)).all()
    assert len(stems) == 8
    for st in stems:
        rend = session.scalars(
            select(StemRendition).where(
                StemRendition.stem_id == st.stem_id, StemRendition.purpose == "stream"
            )
        ).one()
        assert (rend.codec, rend.bitrate_kbps) == ("opus", 128)
        path = resolve_data_path(settings, rend.file_path)
        assert path.suffix == ".webm" and path.is_file()
        assert rend.bytes == path.stat().st_size
        assert path.read_bytes()[:4] == b"\x1a\x45\xdf\xa3"  # WebM（EBML）

        waves = session.scalars(select(Waveform).where(Waveform.stem_id == st.stem_id)).all()
        assert sorted(w.samples_per_px for w in waves) == list(DEFAULT_LEVELS)
        for w in waves:
            pk = read_peaks(resolve_data_path(settings, w.peaks_path))
            assert pk.samples_per_px == w.samples_per_px
            assert pk.sample_rate == 44100
            assert pk.points == -(-3 * 44100 // w.samples_per_px)
            assert np.all(pk.mins <= pk.maxs)
