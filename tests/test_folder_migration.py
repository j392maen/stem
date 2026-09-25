"""T13: 古い保存フォルダ（data/stems/<job_id>）の移行（`stemapp migrate-folders`）。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from job_helpers import make_track
from stemapp import cli, folder_migration
from stemapp.config import Settings
from stemapp.db import init_db, make_engine, make_session_factory
from stemapp.delivery import fake_encoder, rebuild_delivery_files
from stemapp.folder_migration import (
    FAILED,
    MOVED,
    PLANNED,
    SKIPPED,
    folder_stats,
    migrate_folders,
)
from stemapp.library import resolve_data_path
from stemapp.models import Export, SeparationJob, Stem, StemRendition, Waveform
from stemapp.seed import seed
from stemapp.separation import FakeSeparator
from stemapp.separation.pipeline import separate_track

runner = CliRunner()


@pytest.fixture
def seeded(session: Session) -> Session:
    seed(session)
    return session


def _all_paths(session: Session, job_id: int) -> list[str]:
    stem_ids = select(Stem.stem_id).where(Stem.job_id == job_id)
    paths = list(
        session.scalars(select(StemRendition.file_path).where(StemRendition.stem_id.in_(stem_ids)))
    )
    paths += list(
        session.scalars(select(Waveform.peaks_path).where(Waveform.stem_id.in_(stem_ids)))
    )
    paths += [
        p
        for p in session.scalars(select(Export.output_path).where(Export.job_id == job_id))
        if p
    ]
    return paths


def _legacy_job(
    session: Session, settings: Settings, track_id: int, preset: str, force: bool = False
) -> int:
    """分割してから、T13 より前の形（stems/<job_id>、output_dir=NULL）に戻す。"""
    res = separate_track(
        session, settings, track_id, FakeSeparator(), preset_code=preset, force=force
    )
    job_id = res.job_id
    rebuild_delivery_files(session, settings, job_id, encoder=fake_encoder)
    job = session.get(SeparationJob, job_id)
    assert job is not None and job.output_dir
    new_prefix = job.output_dir + "/"
    old_prefix = f"stems/{job_id}/"
    # 書き出し（T08）の成果物が保存フォルダの中にある場合も書き換わることを確かめる
    export_file = settings.data_dir / job.output_dir / "exports" / "mix.wav"
    export_file.parent.mkdir(parents=True)
    export_file.write_bytes(b"RIFF....")
    session.add(
        Export(job_id=job_id, export_type="mix", format="wav",
               output_path=new_prefix + "exports/mix.wav")
    )
    session.flush()
    stem_ids = select(Stem.stem_id).where(Stem.job_id == job_id)
    for r in session.scalars(select(StemRendition).where(StemRendition.stem_id.in_(stem_ids))):
        r.file_path = old_prefix + r.file_path.removeprefix(new_prefix)
    for w in session.scalars(select(Waveform).where(Waveform.stem_id.in_(stem_ids))):
        w.peaks_path = old_prefix + w.peaks_path.removeprefix(new_prefix)
    for e in session.scalars(select(Export).where(Export.job_id == job_id)):
        assert e.output_path is not None
        e.output_path = old_prefix + e.output_path.removeprefix(new_prefix)
    src = settings.data_dir / job.output_dir
    os.rename(src, settings.stems_dir / str(job_id))
    if not any(src.parent.iterdir()):
        src.parent.rmdir()
    job.output_dir = None
    session.commit()
    return job_id


@pytest.fixture
def legacy(seeded: Session, settings: Settings, tmp_path: Path) -> dict[str, int]:
    """古い形のジョブ: 曲 song に fast・exp_combo・fast（再分割）、曲 other に fast。"""
    song = make_track(seeded, settings, tmp_path, name="song")
    other = make_track(seeded, settings, tmp_path, name="曲：その2", seed_offset=0.1)
    return {
        "song_fast": _legacy_job(seeded, settings, song, "fast"),
        "song_combo": _legacy_job(seeded, settings, song, "exp_combo"),
        "song_fast2": _legacy_job(seeded, settings, song, "fast", force=True),
        "other_fast": _legacy_job(seeded, settings, other, "fast"),
    }


EXPECTED = {
    "song_fast": "stems/song/fast",
    "song_combo": "stems/song/exp_combo",
    "song_fast2": "stems/song/fast (2)",
    "other_fast": "stems/曲：その2/fast",
}


def _snapshot(settings: Settings) -> list[tuple[str, int]]:
    root = settings.stems_dir
    return sorted(
        (p.relative_to(root).as_posix(), p.stat().st_size) for p in root.rglob("*") if p.is_file()
    )


def test_dry_run_changes_nothing(
    seeded: Session, settings: Settings, legacy: dict[str, int]
) -> None:
    before = _snapshot(settings)
    report = migrate_folders(seeded, settings, dry_run=True)
    assert [i.status for i in report.items] == [PLANNED] * 4
    plan = {i.job_id: i.new_dir for i in report.items}
    assert plan == {legacy[k]: v for k, v in EXPECTED.items()}
    for i in report.items:
        assert i.old_dir == f"stems/{i.job_id}"
        assert (i.files, i.bytes) == folder_stats(settings.stems_dir / str(i.job_id))
        assert i.files > 0 and i.db_paths == i.files  # master・stream・peaks・書き出し
    assert _snapshot(settings) == before
    seeded.expire_all()
    assert all(
        j.output_dir is None for j in seeded.scalars(select(SeparationJob))
    )


def test_migrate_moves_and_rewrites(
    seeded: Session, settings: Settings, legacy: dict[str, int]
) -> None:
    before = sorted(size for _, size in _snapshot(settings))
    report = migrate_folders(seeded, settings)
    assert [i.status for i in report.items] == [MOVED] * 4
    assert report.before == report.after
    assert sorted(size for _, size in _snapshot(settings)) == before
    seeded.expire_all()
    for key, rel in EXPECTED.items():
        job_id = legacy[key]
        job = seeded.get(SeparationJob, job_id)
        assert job is not None and job.output_dir == rel
        assert not (settings.stems_dir / str(job_id)).exists()
        paths = _all_paths(seeded, job_id)
        assert paths and all(p.startswith(rel + "/") for p in paths)
        assert all(resolve_data_path(settings, p).is_file() for p in paths)
    # 2回目は何もしない
    again = migrate_folders(seeded, settings)
    assert again.items == []
    assert migrate_folders(seeded, settings, dry_run=True).items == []


def test_migrate_rolls_back_failed_job(
    seeded: Session, settings: Settings, legacy: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = legacy["song_combo"]
    old_paths = sorted(_all_paths(seeded, bad))
    real_stats = folder_migration.folder_stats

    def broken_stats(path: Path) -> tuple[int, int]:
        files, total = real_stats(path)
        if path.name == "exp_combo":  # 移した後の確認で数が合わないふりをする
            return files - 1, total
        return files, total

    monkeypatch.setattr(folder_migration, "folder_stats", broken_stats)
    report = migrate_folders(seeded, settings)
    status = {i.job_id: i.status for i in report.items}
    assert status[bad] == FAILED
    assert [s for j, s in status.items() if j != bad] == [MOVED] * 3
    failed = next(i for i in report.items if i.job_id == bad)
    assert "ファイル数" in failed.message
    # そのジョブの分は元に戻っている（フォルダ・DB とも）
    seeded.expire_all()
    job = seeded.get(SeparationJob, bad)
    assert job is not None and job.output_dir is None
    assert (settings.stems_dir / str(bad)).is_dir()
    assert not (settings.stems_dir / "song" / "exp_combo").exists()
    assert sorted(_all_paths(seeded, bad)) == old_paths
    assert all(resolve_data_path(settings, p).is_file() for p in old_paths)

    # 直してもう一度実行すると、残りの1件だけ移す
    monkeypatch.setattr(folder_migration, "folder_stats", real_stats)
    report = migrate_folders(seeded, settings)
    assert [(i.job_id, i.status, i.new_dir) for i in report.items] == [
        (bad, MOVED, "stems/song/exp_combo")
    ]


def test_migrate_rename_failure_keeps_going(
    seeded: Session, settings: Settings, legacy: dict[str, int]
) -> None:
    bad = legacy["song_fast"]

    def rename(src: Path, dst: Path) -> None:
        if src.name == str(bad):
            raise PermissionError("使用中です")
        os.rename(src, dst)

    report = migrate_folders(seeded, settings, rename=rename)
    status = {i.job_id: i.status for i in report.items}
    assert status[bad] == FAILED and "使用中" in report.items[0].message
    assert list(status.values()).count(MOVED) == 3
    assert (settings.stems_dir / str(bad)).is_dir()
    assert report.before == report.after


def test_active_and_missing_are_skipped(
    seeded: Session, settings: Settings, legacy: dict[str, int]
) -> None:
    running = legacy["song_fast"]
    seeded.execute(
        update(SeparationJob).where(SeparationJob.job_id == running).values(status="running")
    )
    seeded.commit()
    gone = legacy["other_fast"]
    import shutil

    shutil.rmtree(settings.stems_dir / str(gone))
    report = migrate_folders(seeded, settings)
    status = {i.job_id: i.status for i in report.items}
    assert status[running] == SKIPPED and status[gone] == SKIPPED
    assert status[legacy["song_combo"]] == MOVED
    assert (settings.stems_dir / str(running)).is_dir()


def test_cli_migrate_folders(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)
    monkeypatch.setenv("COLUMNS", "250")  # 表を折り返さない
    engine = make_engine(settings.db_path)
    try:
        init_db(engine)
        with make_session_factory(engine)() as s:
            seed(s)
            track_id = make_track(s, settings, tmp_path, name="song")
            job_id = _legacy_job(s, settings, track_id, "fast")
    finally:
        engine.dispose()

    dry = runner.invoke(cli.app, ["migrate-folders", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "dry-run" in dry.output and "stems/song/fast" in dry.output
    assert (settings.stems_dir / str(job_id)).is_dir()

    res = runner.invoke(cli.app, ["migrate-folders"])
    assert res.exit_code == 0, res.output
    assert "ファイル数と合計サイズは同じです" in res.output
    assert (settings.stems_dir / "song" / "fast" / "vocals.flac").is_file()

    again = runner.invoke(cli.app, ["migrate-folders"])
    assert again.exit_code == 0 and "移すフォルダはありません" in again.output
