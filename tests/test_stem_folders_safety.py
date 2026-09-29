"""T13 レビュー対応: 名前の長さ・正規化、旧 `stems/<job_id>` と新しいフォルダの取り違え防止。"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from job_helpers import make_track
from stemapp.config import Settings
from stemapp.folder_migration import SKIPPED, migrate_folders
from stemapp.jobs import delete_job
from stemapp.jobs.queue import discard_job_outputs
from stemapp.models import SeparationJob, Stem, StemRendition
from stemapp.seed import seed
from stemapp.separation import FakeSeparator
from stemapp.separation.pipeline import separate_track
from stemapp.stem_folders import (
    is_legacy_job_dir,
    remove_job_dir,
    safe_folder_name,
    truncate_utf16,
    utf16_len,
)


@pytest.fixture
def seeded(session: Session) -> Session:
    seed(session)
    return session


# --- 名前 ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2", "2_"), ("123", "123_"), (" 42 . ", "42_"), ("2 (live)", "2 (live)"), ("１２", "１２")],
)
def test_digits_only_gets_underscore(raw: str, expected: str) -> None:
    assert safe_folder_name(raw, 9) == expected


def test_nfc_normalization() -> None:
    decomposed = unicodedata.normalize("NFD", "ガ")  # カ + 濁点
    assert len(decomposed) == 2
    assert safe_folder_name(decomposed, 1) == "ガ"


def test_length_counts_utf16_units() -> None:
    assert utf16_len("a😀") == 3
    name = safe_folder_name("😀" * 60, 1)
    assert name == "😀" * 50 and utf16_len(name) == 100
    # 100 単位目がサロゲートペアの途中になるときは、その手前で切る
    name = safe_folder_name("a" + "😀" * 60, 1)
    assert name == "a" + "😀" * 49 and utf16_len(name) == 99


def test_truncate_keeps_graphemes() -> None:
    family = "\U0001F468\u200d\U0001F469\u200d\U0001F467"  # ZWJ でつないだ1つの絵文字（8 単位）
    assert truncate_utf16("a" * 97 + family, 100) == "a" * 97  # 絵文字の途中で切らない
    accented = "a" * 99 + "e\u0301"  # e + 結合アクセント（NFC にしない場合）
    assert truncate_utf16(accented, 100) == "a" * 99
    jp = "\U0001F1EF\U0001F1F5"  # 国旗（地域指示子 2 つ、4 単位）
    assert truncate_utf16("a" * 97 + jp, 100) == "a" * 97  # 片方だけ残さない
    assert truncate_utf16("a" * 96 + jp, 100) == "a" * 96 + jp
    assert truncate_utf16("a" * 93 + jp * 2, 100) == "a" * 93 + jp


# --- 数字だけの曲名と旧フォルダ ---------------------------------------------------------


def _split(session: Session, settings: Settings, track_id: int, preset: str = "fast") -> int:
    return separate_track(session, settings, track_id, FakeSeparator(), preset_code=preset).job_id


def test_numeric_track_name_does_not_collide_with_legacy(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    """曲「2」を分割 → 別の曲の失敗ジョブ（job 2、output_dir=NULL）を消しても、曲「2」は残る。"""
    t2 = make_track(seeded, settings, tmp_path, name="2")
    job_a = _split(seeded, settings, t2)
    other = make_track(seeded, settings, tmp_path, name="other", seed_offset=0.1)
    failed = SeparationJob(track_id=other, job_kind="full", status="failed")
    seeded.add(failed)
    seeded.commit()
    assert job_a == 1 and failed.job_id == 2
    a = seeded.get(SeparationJob, job_a)
    assert a is not None and a.output_dir == "stems/2_/fast"

    delete_job(seeded, settings, failed.job_id)
    assert (settings.stems_dir / "2_" / "fast" / "vocals.flac").is_file()


def _fake_new_style_folder_named(
    seeded: Session, settings: Settings, job_id: int, name: str
) -> Path:
    """修正前に作られうる「数字だけの曲フォルダ」を再現する（job を stems/<name>/fast へ）。"""
    job = seeded.get(SeparationJob, job_id)
    assert job is not None and job.output_dir
    old_prefix = job.output_dir + "/"
    new_rel = f"stems/{name}/fast"
    dst = settings.data_dir / "stems" / name / "fast"
    dst.parent.mkdir(parents=True)
    os.rename(settings.data_dir / job.output_dir, dst)
    stem_ids = select(Stem.stem_id).where(Stem.job_id == job_id)
    for r in seeded.scalars(select(StemRendition).where(StemRendition.stem_id.in_(stem_ids))):
        r.file_path = new_rel + "/" + r.file_path.removeprefix(old_prefix)
    job.output_dir = new_rel
    seeded.commit()
    return dst


def test_legacy_delete_does_not_remove_other_tracks_folder(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    song = make_track(seeded, settings, tmp_path, name="song")
    job_a = _split(seeded, settings, song)
    other = make_track(seeded, settings, tmp_path, name="other", seed_offset=0.1)
    failed = SeparationJob(track_id=other, job_kind="full", status="failed")
    seeded.add(failed)
    seeded.commit()
    folder = _fake_new_style_folder_named(seeded, settings, job_a, str(failed.job_id))
    assert not is_legacy_job_dir(seeded, settings, failed.job_id)

    # ジョブの削除・分割の失敗（out_rel=None）・後始末のどれでも消さない
    remove_job_dir(seeded, settings, failed.job_id, None)
    assert (folder / "vocals.flac").is_file()
    discard_job_outputs(seeded, settings, failed.job_id)
    assert (folder / "vocals.flac").is_file()
    delete_job(seeded, settings, failed.job_id)
    assert (folder / "vocals.flac").is_file()


def test_legacy_check_looks_at_contents(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    track_id = make_track(seeded, settings, tmp_path, name="song")
    job = SeparationJob(track_id=track_id, job_kind="full", status="failed")
    seeded.add(job)
    seeded.commit()
    legacy = settings.stems_dir / str(job.job_id)
    # DB に無くても、中に分け方のフォルダがあれば旧形式ではない（消さない）
    (legacy / "fast").mkdir(parents=True)
    (legacy / "fast" / "vocals.flac").write_bytes(b"x")
    assert not is_legacy_job_dir(seeded, settings, job.job_id)
    remove_job_dir(seeded, settings, job.job_id, None)
    assert (legacy / "fast" / "vocals.flac").is_file()
    # 旧形式（直下に FLAC、stream / peaks）なら消す
    (legacy / "fast" / "vocals.flac").unlink()
    (legacy / "fast").rmdir()
    (legacy / "stream").mkdir()
    (legacy / "vocals.flac").write_bytes(b"x")
    assert is_legacy_job_dir(seeded, settings, job.job_id)
    remove_job_dir(seeded, settings, job.job_id, None)
    assert not legacy.exists()


def test_migrate_skips_other_tracks_folder(
    seeded: Session, settings: Settings, tmp_path: Path
) -> None:
    """job 2（旧形式・DB は stems/2/ を指す）の stems/2 が別の曲のフォルダなら移さない。"""
    song = make_track(seeded, settings, tmp_path, name="song")
    job_a = _split(seeded, settings, song)
    other = make_track(seeded, settings, tmp_path, name="other", seed_offset=0.1)
    job_b = _split(seeded, settings, other)
    assert job_b == 2
    # job 2 を旧形式にする（DB のパスだけ stems/2/ を指し、フォルダは無い）
    b = seeded.get(SeparationJob, job_b)
    assert b is not None and b.output_dir
    prefix = b.output_dir + "/"
    stem_ids = select(Stem.stem_id).where(Stem.job_id == job_b)
    for r in seeded.scalars(select(StemRendition).where(StemRendition.stem_id.in_(stem_ids))):
        r.file_path = "stems/2/" + r.file_path.removeprefix(prefix)
    b.output_dir = None
    seeded.commit()
    # job 1 の曲のフォルダが stems/2 になっている
    folder = _fake_new_style_folder_named(seeded, settings, job_a, "2")
    before = sorted(p.name for p in folder.iterdir())

    for dry in (True, False):
        report = migrate_folders(seeded, settings, dry_run=dry)
        (item,) = report.items
        assert item.job_id == job_b and item.status == SKIPPED
        assert "古い形の保存フォルダではない" in item.message
    assert sorted(p.name for p in folder.iterdir()) == before
    a = seeded.get(SeparationJob, job_a)
    assert a is not None and a.output_dir == "stems/2/fast"
