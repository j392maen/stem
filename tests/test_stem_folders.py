"""T13: stem の保存フォルダ名（data/stems/<元のファイル名>/<分け方>/）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from job_helpers import make_track, no_tags
from stemapp.config import Settings
from stemapp.delivery import fake_encoder, rebuild_delivery_files
from stemapp.ingest import import_file
from stemapp.jobs import delete_job
from stemapp.models import InputSource, SeparationJob, Stem, StemRendition, Track, Waveform
from stemapp.seed import seed
from stemapp.separation import FakeSeparator
from stemapp.separation.pipeline import separate_track
from stemapp.stem_folders import (
    job_dir,
    safe_folder_name,
    strip_extension,
    track_source_name,
)


@pytest.fixture
def seeded(session: Session) -> Session:
    seed(session)
    return session


# --- 名前の整形 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("My Song", "My Song"),
        ('a\\b/c:d*e?f"g<h>i|j', "abcdefghij"),
        ("tab\there\x00\x1fend", "tabhereend"),
        ("  前後の空白  ", "前後の空白"),
        ("末尾のピリオド...", "末尾のピリオド"),
        ("末尾 . . ", "末尾"),
        ("曲名：サブタイトル？", "曲名：サブタイトル？"),  # 全角の記号はそのまま
        ("日本語の曲 (Live)", "日本語の曲 (Live)"),
        ("Artist - Title [Official Video]", "Artist - Title [Official Video]"),
        ("CON", "CON_"),
        ("con", "con_"),
        ("Nul.txt", "Nul.txt_"),
        ("com1", "com1_"),
        ("LPT9.tar.gz", "LPT9.tar.gz_"),
        ("COM10", "COM10"),  # 予約名ではない
        ("CONSOLE", "CONSOLE"),
        ("<>:?", "track_7"),
        ("", "track_7"),
        (None, "track_7"),
        (" . ", "track_7"),
    ],
)
def test_safe_folder_name(raw: str | None, expected: str) -> None:
    assert safe_folder_name(raw, 7) == expected


def test_safe_folder_name_length() -> None:
    long = "あ" * 150
    assert safe_folder_name(long, 1) == "あ" * 100
    # 切った後に末尾が空白・ピリオドになっても取り除く
    assert safe_folder_name("a" * 99 + " b", 1) == "a" * 99
    assert len(safe_folder_name("x" * 300, 1)) == 100


def test_strip_extension() -> None:
    assert strip_extension("song.mp3") == "song"
    assert strip_extension("my.song.flac") == "my.song"
    assert strip_extension("Mr. Smith") == "Mr. Smith"  # 拡張子ではない
    assert strip_extension("noext") == "noext"


# --- 新しい分割のフォルダ構成 ---------------------------------------------------------


def _track(
    session: Session, settings: Settings, tmp_path: Path, name: str, offset: float = 0.0,
    source_type: str = "file",
) -> int:
    """original_name を指定して取り込む（Windows のファイル名に使えない文字も試せる）。"""
    data = synth_mix(1.0)
    if offset:
        data = (data * np.float32(1.0 - offset)).astype(np.float32)
    src = write_source(tmp_path / "src" / f"x{offset}.wav", data)
    res = import_file(
        session, settings, src, original_name=name, source_type=source_type,
        ffmpeg_runner=fake_ffmpeg, tag_reader=no_tags,
    )
    return res.track_id


def _split(session: Session, settings: Settings, track_id: int, preset: str, force: bool = False):  # type: ignore[no-untyped-def]
    return separate_track(
        session, settings, track_id, FakeSeparator(), preset_code=preset, force=force
    )


def _master_paths(session: Session, job_id: int) -> list[str]:
    return list(
        session.scalars(
            select(StemRendition.file_path)
            .join(Stem, Stem.stem_id == StemRendition.stem_id)
            .where(Stem.job_id == job_id, StemRendition.purpose == "master")
        )
    )


def test_new_split_uses_original_name(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    track_id = _track(seeded, settings, tmp_path, '日本語の曲: "Live"?.mp3')
    res = _split(seeded, settings, track_id, "fast")
    job = seeded.get(SeparationJob, res.job_id)
    assert job is not None and job.output_dir == "stems/日本語の曲 Live/fast"
    folder = settings.stems_dir / "日本語の曲 Live" / "fast"
    assert job_dir(settings, job.job_id, job.output_dir) == folder
    assert (folder / "vocals.flac").is_file()
    paths = _master_paths(seeded, res.job_id)
    assert paths and all(p.startswith("stems/日本語の曲 Live/fast/") for p in paths)
    assert not (settings.stems_dir / str(res.job_id)).exists()

    # 配信用データもその中の stream/ と peaks/
    rebuild_delivery_files(seeded, settings, res.job_id, encoder=fake_encoder)
    streams = seeded.scalars(
        select(StemRendition.file_path)
        .join(Stem, Stem.stem_id == StemRendition.stem_id)
        .where(Stem.job_id == res.job_id, StemRendition.purpose == "stream")
    ).all()
    assert streams and all(p.startswith("stems/日本語の曲 Live/fast/stream/") for p in streams)
    peaks = seeded.scalars(
        select(Waveform.peaks_path)
        .join(Stem, Stem.stem_id == Waveform.stem_id)
        .where(Stem.job_id == res.job_id)
    ).all()
    assert peaks and all(p.startswith("stems/日本語の曲 Live/fast/peaks/") for p in peaks)


def test_url_title_is_used_as_is(seeded: Session, settings: Settings, tmp_path: Path) -> None:
    # URL の曲はタイトルをそのまま使う（"Vol. 2" の ". 2" を拡張子として削らない）
    track_id = _track(seeded, settings, tmp_path, "Best Hits Vol.2", source_type="url")
    assert track_source_name(seeded, track_id) == "Best Hits Vol.2"
    res = _split(seeded, settings, track_id, "fast")
    assert (settings.stems_dir / "Best Hits Vol.2" / "fast" / "vocals.flac").is_file()
    assert res.job_id


def test_empty_name_falls_back_to_track_id(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    track_id = _track(seeded, settings, tmp_path, "???.wav")
    res = _split(seeded, settings, track_id, "fast")
    job = seeded.get(SeparationJob, res.job_id)
    assert job is not None and job.output_dir == f"stems/track_{track_id}/fast"


def test_no_source_falls_back_to_title(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    # 古いデータなどで INPUT_SOURCE が無い曲は、曲名（TRACK.title）を使う
    track_id = _track(seeded, settings, tmp_path, "whatever.wav")
    for src in seeded.scalars(select(InputSource).where(InputSource.track_id == track_id)):
        seeded.delete(src)
    track = seeded.get(Track, track_id)
    assert track is not None
    track.title = "synth: 4min"
    seeded.commit()
    res = _split(seeded, settings, track_id, "fast")
    job = seeded.get(SeparationJob, res.job_id)
    assert job is not None and job.output_dir == "stems/synth 4min/fast"


def test_multiple_jobs_and_force(seeded: Session, settings: Settings, tmp_path: Path) -> None:
    track_id = make_track(seeded, settings, tmp_path, name="song")
    fast = _split(seeded, settings, track_id, "fast")
    combo = _split(seeded, settings, track_id, "exp_combo")
    again = _split(seeded, settings, track_id, "fast", force=True)
    third = _split(seeded, settings, track_id, "fast", force=True)
    dirs = {
        j.job_id: j.output_dir
        for j in seeded.scalars(select(SeparationJob).where(SeparationJob.track_id == track_id))
    }
    assert dirs == {
        fast.job_id: "stems/song/fast",
        combo.job_id: "stems/song/exp_combo",
        again.job_id: "stems/song/fast (2)",
        third.job_id: "stems/song/fast (3)",
    }
    for rel in dirs.values():
        assert (settings.data_dir / rel / "vocals.flac").is_file()

    # 後から INPUT_SOURCE が増えてもフォルダ名は変えない
    seeded.add(
        InputSource(track_id=track_id, source_type="file", original_name="別名.wav",
                    fetch_status="done")
    )
    seeded.commit()
    more = _split(seeded, settings, track_id, "exp_resid_vocals")
    job = seeded.get(SeparationJob, more.job_id)
    assert job is not None and job.output_dir == "stems/song/exp_resid_vocals"

    # 消した後に同じ分け方を分けると、空いた名前を使う
    delete_job(seeded, settings, fast.job_id)
    assert not (settings.stems_dir / "song" / "fast").exists()
    assert (settings.stems_dir / "song" / "fast (2)").is_dir()
    redo = _split(seeded, settings, track_id, "fast", force=True)
    job = seeded.get(SeparationJob, redo.job_id)
    assert job is not None and job.output_dir == "stems/song/fast"


def test_same_name_different_tracks(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    a = _track(seeded, settings, tmp_path, "Song.mp3", offset=0.0)
    b = _track(seeded, settings, tmp_path, "song.wav", offset=0.1)  # 大文字小文字だけ違う
    c = _track(seeded, settings, tmp_path, "Song.flac", offset=0.2)
    assert len({a, b, c}) == 3
    ja = _split(seeded, settings, a, "fast")
    jb = _split(seeded, settings, b, "fast")
    jc = _split(seeded, settings, c, "fast")
    out = {
        j: seeded.get(SeparationJob, j).output_dir  # type: ignore[union-attr]
        for j in (ja.job_id, jb.job_id, jc.job_id)
    }
    assert out == {
        ja.job_id: "stems/Song/fast",
        jb.job_id: "stems/song (2)/fast",
        jc.job_id: "stems/Song (3)/fast",
    }
    # 同じ曲の2つ目のジョブは同じ曲フォルダへ
    jb2 = _split(seeded, settings, b, "exp_combo")
    job = seeded.get(SeparationJob, jb2.job_id)
    assert job is not None and job.output_dir == "stems/song (2)/exp_combo"


def test_existing_folder_is_not_reused(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    # DB に無いフォルダ（手で置いたもの・古い stems/<job_id> など）とも重ならない
    (settings.stems_dir / "song" / "keep.txt").parent.mkdir(parents=True)
    (settings.stems_dir / "song" / "keep.txt").write_text("x", encoding="utf-8")
    track_id = make_track(seeded, settings, tmp_path, name="song")
    res = _split(seeded, settings, track_id, "fast")
    job = seeded.get(SeparationJob, res.job_id)
    assert job is not None and job.output_dir == "stems/song (2)/fast"
    assert (settings.stems_dir / "song" / "keep.txt").is_file()


def test_delete_job_removes_empty_track_folder(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    track_id = make_track(seeded, settings, tmp_path, name="song")
    a = _split(seeded, settings, track_id, "fast")
    b = _split(seeded, settings, track_id, "exp_combo")
    delete_job(seeded, settings, a.job_id)
    assert not (settings.stems_dir / "song" / "fast").exists()
    assert (settings.stems_dir / "song" / "exp_combo").is_dir()
    delete_job(seeded, settings, b.job_id)
    assert not (settings.stems_dir / "song").exists()
    assert settings.stems_dir.is_dir()  # stems フォルダ自体は残す
