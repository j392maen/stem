from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from stemapp.audio import AudioError
from stemapp.config import Settings
from stemapp.ingest import import_file
from stemapp.ingest.service import decide_title, parse_ffprobe_tags, read_tags_ffprobe
from stemapp.library import resolve_data_path
from stemapp.models import InputSource, SeparationJob, Track


def no_tags(_path: Path) -> Mapping[str, str]:
    return {}


def _import(session: Session, settings: Settings, path: Path, **kw: object):
    kw.setdefault("tag_reader", no_tags)
    return import_file(session, settings, path, ffmpeg_runner=fake_ffmpeg, **kw)  # type: ignore[arg-type]


@pytest.fixture
def mix() -> np.ndarray:
    # 16bit で書いても値が変わらないよう、16bit に丸めた音にする
    return np.round(synth_mix(1.0) * 32768.0) / 32768.0


def test_new_track_is_registered(session: Session, settings: Settings, tmp_path: Path,
                                 mix: np.ndarray) -> None:
    src = write_source(tmp_path / "in" / "曲名.wav", mix)
    res = _import(session, settings, src)
    assert res.is_new is True and res.has_done_full_job is False
    assert res.title == "曲名"

    track = session.get(Track, res.track_id)
    assert track is not None
    assert track.title == "曲名" and track.artist is None
    assert track.duration_sec == pytest.approx(1.0)
    assert track.normalized_path == f"tracks/{track.track_id}/normalized.wav"
    assert resolve_data_path(settings, track.normalized_path).is_file()
    assert len(track.audio_hash) == 64

    (source,) = session.scalars(select(InputSource)).all()
    assert source.source_id == res.source_id
    assert source.track_id == track.track_id
    assert (source.source_type, source.fetch_status) == ("file", "done")
    assert source.original_name == "曲名.wav"
    assert source.url is None and source.error_code is None
    assert source.fetched_at is not None

    # 元のファイルは残す・一時フォルダは残さない
    assert src.is_file()
    assert not list((settings.cache_dir / "tmp").glob("*"))


def test_same_audio_twice_is_existing(session: Session, settings: Settings, tmp_path: Path,
                                      mix: np.ndarray) -> None:
    src = write_source(tmp_path / "a.wav", mix)
    first = _import(session, settings, src)
    second = _import(session, settings, src)
    assert second.is_new is False
    assert second.track_id == first.track_id
    assert second.has_done_full_job is False
    assert len(session.scalars(select(Track)).all()) == 1
    sources = session.scalars(select(InputSource).order_by(InputSource.source_id)).all()
    assert [s.track_id for s in sources] == [first.track_id, first.track_id]
    # 正規化ファイルは1つだけ（2回目のものは捨てる）
    assert [p.name for p in settings.tracks_dir.rglob("*") if p.is_file()] == ["normalized.wav"]
    assert not list((settings.cache_dir / "tmp").glob("*"))


def test_same_audio_other_name_and_format(session: Session, settings: Settings, tmp_path: Path,
                                          mix: np.ndarray) -> None:
    wav = write_source(tmp_path / "one.wav", mix, subtype="PCM_16")
    flac = write_source(tmp_path / "別の名前.flac", mix, subtype="PCM_16")
    first = _import(session, settings, wav)
    second = _import(session, settings, flac)
    assert second.is_new is False and second.track_id == first.track_id
    assert second.title == "one"  # 既存の曲のタイトルは変えない
    names = [s.original_name for s in session.scalars(select(InputSource))]
    assert sorted(names) == sorted(["one.wav", "別の名前.flac"])


def test_different_audio_is_new(session: Session, settings: Settings, tmp_path: Path,
                                mix: np.ndarray) -> None:
    a = _import(session, settings, write_source(tmp_path / "a.wav", mix))
    b = _import(session, settings, write_source(tmp_path / "b.wav", mix * 0.5))
    assert b.is_new is True and b.track_id != a.track_id


def test_existing_with_done_job(session: Session, settings: Settings, tmp_path: Path,
                                mix: np.ndarray) -> None:
    src = write_source(tmp_path / "a.wav", mix)
    first = _import(session, settings, src)
    session.add(SeparationJob(track_id=first.track_id, job_kind="full", status="done"))
    session.commit()
    again = _import(session, settings, src)
    assert again.is_new is False and again.has_done_full_job is True


def test_title_from_tags(session: Session, settings: Settings, tmp_path: Path,
                         mix: np.ndarray) -> None:
    src = write_source(tmp_path / "file_name.wav", mix)
    res = _import(
        session, settings, src,
        tag_reader=lambda _p: {"title": "タグの題名", "artist": "歌手"},
    )
    track = session.get(Track, res.track_id)
    assert track is not None
    assert (track.title, track.artist) == ("タグの題名", "歌手")


def test_title_hint_beats_tags(session: Session, settings: Settings, tmp_path: Path,
                               mix: np.ndarray) -> None:
    src = write_source(tmp_path / "audio.webm.wav", mix)
    res = _import(
        session, settings, src, title="ページの題名", artist="投稿者",
        tag_reader=lambda _p: {"title": "タグ", "artist": "タグの歌手"},
    )
    track = session.get(Track, res.track_id)
    assert track is not None
    assert (track.title, track.artist) == ("ページの題名", "投稿者")


def test_original_name_is_used_for_title(session: Session, settings: Settings, tmp_path: Path,
                                         mix: np.ndarray) -> None:
    src = write_source(tmp_path / "upload_tmp.wav", mix)
    res = _import(session, settings, src, original_name="元の名前.mp3")
    assert res.title == "元の名前"
    source = session.scalars(select(InputSource)).one()
    assert source.original_name == "元の名前.mp3"


def test_decide_title() -> None:
    assert decide_title({}, "a b.mp3") == "a b"
    assert decide_title({"title": "  "}, "x.flac") == "x"
    assert decide_title({"title": "T"}, "x.flac") == "T"
    assert decide_title({"title": "T"}, "x.flac", "H") == "H"


def test_parse_ffprobe_tags() -> None:
    out = (
        '{"programs": [], "streams": [{"tags": {"TITLE": "stream title", "ARTIST": "s"}}],'
        ' "format": {"tags": {"title": "format title"}}}'
    )
    assert parse_ffprobe_tags(out) == {"title": "format title", "artist": "s"}
    assert parse_ffprobe_tags("") == {}
    assert parse_ffprobe_tags("not json") == {}


def test_read_tags_without_ffprobe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert read_tags_ffprobe(tmp_path / "x.wav") == {}


def test_normalize_failure_leaves_db_unchanged(session: Session, settings: Settings,
                                               tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        _import(session, settings, tmp_path / "missing.wav")
    assert session.scalars(select(Track)).all() == []
    assert session.scalars(select(InputSource)).all() == []


def test_commit_failure_removes_track_dir(session: Session, settings: Settings, tmp_path: Path,
                                          mix: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    src = write_source(tmp_path / "a.wav", mix)

    def failing_commit() -> None:
        raise RuntimeError("commit failed")

    monkeypatch.setattr(session, "commit", failing_commit)
    with pytest.raises(RuntimeError, match="commit failed"):
        _import(session, settings, src)
    monkeypatch.undo()

    # 置いた正規化ファイルのフォルダは消え、DB にも残らない
    assert not any(settings.tracks_dir.iterdir())
    assert session.scalars(select(Track)).all() == []
    assert session.scalars(select(InputSource)).all() == []
    assert not list((settings.cache_dir / "tmp").glob("*"))

    # やり直せば登録できる
    res = _import(session, settings, src)
    assert res.is_new is True
    assert (settings.tracks_dir / str(res.track_id) / "normalized.wav").is_file()


# --- 実際の ffmpeg / ffprobe を使うもの -------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg / ffprobe がありません",
)


def _ffmpeg(*args: str) -> None:
    import subprocess

    subprocess.run(
        [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
    )


@pytest.mark.ffmpeg
@needs_ffmpeg
def test_real_ffmpeg_tags_and_formats(session: Session, settings: Settings, tmp_path: Path,
                                      mix: np.ndarray) -> None:
    wav = write_source(tmp_path / "plain.wav", mix, subtype="PCM_16")
    flac = tmp_path / "tagged.flac"
    _ffmpeg("-i", str(wav), "-metadata", "title=タグ付きの曲", "-metadata", "artist=誰か",
            "-c:a", "flac", str(flac))

    first = import_file(session, settings, flac)
    track = session.get(Track, first.track_id)
    assert track is not None
    assert (track.title, track.artist) == ("タグ付きの曲", "誰か")

    # 同じ音の WAV（タグなし・別名）は既存扱い
    second = import_file(session, settings, wav)
    assert second.is_new is False and second.track_id == first.track_id

    # タグの無いファイルはファイル名がタイトル
    other = write_source(tmp_path / "無題の曲.wav", mix * 0.5, subtype="PCM_16")
    third = import_file(session, settings, other)
    assert third.is_new is True and third.title == "無題の曲"
