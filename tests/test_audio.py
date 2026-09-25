from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from stemapp.audio import (
    SAMPLE_RATE,
    SILENCE_FLOOR_DB,
    AudioError,
    as_stereo,
    audio_hash,
    count_clipped,
    ffmpeg_normalize_args,
    normalize_audio,
    rms_db,
    write_flac24,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません")


def test_as_stereo_duplicates_mono() -> None:
    mono = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    st = as_stereo(mono)
    assert st.shape == (3, 2) and st.dtype == np.float32
    assert np.array_equal(st[:, 0], st[:, 1])
    with pytest.raises(AudioError):
        as_stereo(np.zeros((4, 3)))


def test_audio_hash_depends_only_on_samples() -> None:
    x = synth_mix(0.5)
    assert audio_hash(x) == audio_hash(x.copy())
    assert audio_hash(x) == audio_hash(x.astype(np.float64))
    y = x.copy()
    y[0, 0] += 1e-3
    assert audio_hash(x) != audio_hash(y)


def test_rms_db_and_clip() -> None:
    assert rms_db(np.zeros((100, 2))) == SILENCE_FLOOR_DB
    full = np.ones((100, 2), dtype=np.float32)
    assert rms_db(full) == pytest.approx(0.0)
    assert rms_db(full * 0.001) == pytest.approx(-60.0, abs=1e-4)
    assert count_clipped(np.array([[1.0, -1.5], [2.0, 0.1]])) == 2


def test_flac24_roundtrip(tmp_path: Path) -> None:
    x = synth_mix(0.5)
    write_flac24(tmp_path / "a.flac", x)
    info = sf.info(str(tmp_path / "a.flac"))
    assert info.subtype == "PCM_24" and info.samplerate == SAMPLE_RATE and info.channels == 2
    back, _ = sf.read(str(tmp_path / "a.flac"), dtype="float32")
    assert np.max(np.abs(back - x)) < 1e-6


def test_normalize_args_are_44k_stereo_float() -> None:
    args = ffmpeg_normalize_args(Path("in.mp3"), Path("out.wav"))
    joined = " ".join(args)
    assert "-ar 44100" in joined and "pcm_f32le" in joined
    assert "channel_layouts=mono|stereo" in joined
    assert args[-1] == "out.wav"


def test_normalize_with_injected_runner(tmp_path: Path) -> None:
    x = synth_mix(0.5)
    src = write_source(tmp_path / "in.wav", x)
    norm = normalize_audio(src, tmp_path / "out" / "n.wav", runner=fake_ffmpeg)
    assert norm.data.shape == x.shape
    assert norm.audio_hash == audio_hash(x)
    assert norm.duration_sec == pytest.approx(0.5)


def test_normalize_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        normalize_audio(tmp_path / "nothing.mp3", tmp_path / "n.wav", runner=fake_ffmpeg)


@pytest.mark.ffmpeg
@needs_ffmpeg
def test_ffmpeg_resamples_to_44k_stereo(tmp_path: Path) -> None:
    sr = 22050
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)], 1) * 0.3
    src = tmp_path / "in.wav"
    sf.write(str(src), stereo, sr, subtype="PCM_16")
    norm = normalize_audio(src, tmp_path / "n.wav")
    info = sf.info(str(norm.path))
    assert info.samplerate == SAMPLE_RATE and info.channels == 2 and info.subtype == "FLOAT"
    assert norm.data.shape[1] == 2
    assert abs(norm.data.shape[0] - SAMPLE_RATE) <= 64


@pytest.mark.ffmpeg
@needs_ffmpeg
def test_ffmpeg_duplicates_mono(tmp_path: Path) -> None:
    mono = synth_mix(0.5)[:, 0]
    src = tmp_path / "mono.wav"
    sf.write(str(src), mono, SAMPLE_RATE, subtype="PCM_16")
    norm = normalize_audio(src, tmp_path / "n.wav")
    assert norm.data.shape == (mono.shape[0], 2)
    assert np.array_equal(norm.data[:, 0], norm.data[:, 1])
    # 音量を変えずに複製する（ffmpeg の -ac 2 は -3dB になる）
    assert np.max(np.abs(norm.data[:, 0] - mono)) < 1e-4
    assert sf.info(str(norm.path)).channels == 2


@pytest.mark.ffmpeg
@needs_ffmpeg
def test_ffmpeg_downmixes_multichannel(tmp_path: Path) -> None:
    x = synth_mix(0.5)[:, 0]
    src = tmp_path / "six.wav"
    sf.write(str(src), np.stack([x] * 6, axis=1) * 0.2, SAMPLE_RATE, subtype="PCM_16")
    norm = normalize_audio(src, tmp_path / "n.wav")
    assert norm.data.shape == (x.shape[0], 2)


@pytest.mark.ffmpeg
@needs_ffmpeg
def test_ffmpeg_same_sound_same_hash(tmp_path: Path) -> None:
    x = synth_mix(0.5)
    a = write_source(tmp_path / "a.wav", x, subtype="PCM_16")
    b = tmp_path / "b.flac"
    sf.write(str(b), sf.read(str(a), dtype="int16")[0], SAMPLE_RATE, subtype="PCM_16")
    na = normalize_audio(a, tmp_path / "na.wav")
    nb = normalize_audio(b, tmp_path / "nb.wav")
    assert na.audio_hash == nb.audio_hash
    y = x.copy()
    y[100:200] = 0
    c = write_source(tmp_path / "c.wav", y, subtype="PCM_16")
    assert normalize_audio(c, tmp_path / "nc.wav").audio_hash != na.audio_hash
