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
    NEEDS_CHECK,
    PLANNED,
    SKIPPED,
    MigrationBlocked,
    _write_journal,
    folder_stats,
    journal_path,
    migrate_folders,
)
from stemapp.jobs.worker import WorkerLock
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
    # EXPORT.output_path が stems/<job_id>/ を指していれば書き換わることを確かめる
    # （T08 の書き出しは data/exports/ に置くので、ふつうは対象外）
    export_file = settings.data_dir / job.output_dir / "mix_export.wav"
    export_file.write_bytes(b"RIFF....")
    session.add(
        Export(job_id=job_id, export_type="mix", format="wav",
               output_path=new_prefix + "mix_export.wav")
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


# --- 中断への備え（予定の記録） -----------------------------------------------------------


def _state(session: Session, settings: Settings, job_id: int) -> tuple[str | None, list[str]]:
    session.expire_all()
    job = session.get(SeparationJob, job_id)
    assert job is not None
    return job.output_dir, sorted(_all_paths(session, job_id))


def test_keyboard_interrupt_restores_and_reraises(
    seeded: Session, settings: Settings, legacy: dict[str, int]
) -> None:
    bad = legacy["song_combo"]
    before = _state(seeded, settings, bad)

    def rename(src: Path, dst: Path) -> None:
        os.rename(src, dst)
        if src.name == str(bad):
            raise KeyboardInterrupt  # 移した直後に Ctrl+C

    with pytest.raises(KeyboardInterrupt):
        migrate_folders(seeded, settings, rename=rename)
    assert _state(seeded, settings, bad) == before
    assert (settings.stems_dir / str(bad)).is_dir()
    assert not (settings.stems_dir / "song" / "exp_combo").exists()
    assert not journal_path(settings).exists()
    # もう一度実行すると残りを移す
    report = migrate_folders(seeded, settings)
    assert report.notes == []
    assert {i.job_id: i.status for i in report.items}[bad] == MOVED


def test_journal_recovers_after_kill(
    seeded: Session, settings: Settings, legacy: dict[str, int]
) -> None:
    """フォルダを移した直後にプロセスが止まった（DB は古いまま・記録が残った）状態から直る。"""
    bad = legacy["song_fast"]
    before = _state(seeded, settings, bad)
    _write_journal(settings, bad, f"stems/{bad}", "stems/song/fast")
    (settings.stems_dir / "song").mkdir()
    os.rename(settings.stems_dir / str(bad), settings.stems_dir / "song" / "fast")
    assert _state(seeded, settings, bad) == before  # DB は古い場所を指したまま

    # dry-run は何も変えずに、直す予定を知らせる
    dry = migrate_folders(seeded, settings, dry_run=True)
    assert dry.notes and "元に戻します" in dry.notes[0]
    assert journal_path(settings).exists()
    assert (settings.stems_dir / "song" / "fast").is_dir()
    # 中断したジョブは「対象外」ではなく、元に戻してから同じ場所へ移す予定。ほかは重ならない
    plan = {i.job_id: (i.status, i.new_dir) for i in dry.items}
    assert plan[bad] == (PLANNED, "stems/song/fast")
    assert plan == {legacy[k]: (PLANNED, v) for k, v in EXPECTED.items()}
    pending_item = next(i for i in dry.items if i.job_id == bad)
    assert "元に戻してから" in pending_item.message and pending_item.files > 0

    report = migrate_folders(seeded, settings)
    assert report.notes and "元に戻します" in report.notes[0]
    assert not journal_path(settings).exists()
    assert [i.status for i in report.items] == [MOVED] * 4
    output_dir, paths = _state(seeded, settings, bad)
    assert output_dir == "stems/song/fast"
    assert paths and all(p.startswith("stems/song/fast/") for p in paths)
    assert all(resolve_data_path(settings, p).is_file() for p in paths)


@pytest.mark.parametrize("error", [PermissionError("使用中です"), KeyboardInterrupt()])
def test_failure_after_commit_keeps_migration(
    seeded: Session,
    settings: Settings,
    legacy: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    """commit の後（記録を消すとき）に失敗・Ctrl+C があっても、フォルダを戻さない（DB と一致）。"""
    job_id = legacy["song_fast"]
    real_clear = folder_migration._clear_journal
    calls: list[int] = []

    def flaky_clear(s: Settings) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise error
        real_clear(s)

    monkeypatch.setattr(folder_migration, "_clear_journal", flaky_clear)
    if isinstance(error, Exception):
        report = migrate_folders(seeded, settings)
        item = next(i for i in report.items if i.job_id == job_id)
        assert item.status == MOVED and "記録を消せませんでした" in item.message
        assert all(i.status == MOVED for i in report.items)
    else:
        with pytest.raises(KeyboardInterrupt):
            migrate_folders(seeded, settings)
    # DB もフォルダも新しい場所で一致している
    output_dir, paths = _state(seeded, settings, job_id)
    assert output_dir == "stems/song/fast"
    assert (settings.stems_dir / "song" / "fast").is_dir()
    assert not (settings.stems_dir / str(job_id)).exists()
    assert all(resolve_data_path(settings, p).is_file() for p in paths)

    # 残った記録は次の実行で消え、残りも移る
    monkeypatch.setattr(folder_migration, "_clear_journal", real_clear)
    again = migrate_folders(seeded, settings)
    if again.notes:
        assert "完了していました" in again.notes[0]
    assert all(i.status == MOVED for i in again.items)
    assert not journal_path(settings).exists()
    seeded.expire_all()
    assert all(j.output_dir is not None for j in seeded.scalars(select(SeparationJob)))
    for key, rel in EXPECTED.items():
        _, paths = _state(seeded, settings, legacy[key])
        assert all(p.startswith(rel + "/") for p in paths)
        assert all(resolve_data_path(settings, p).is_file() for p in paths)


def test_journal_after_commit_is_cleared(
    seeded: Session, settings: Settings, legacy: dict[str, int]
) -> None:
    report = migrate_folders(seeded, settings)
    assert all(i.status == MOVED for i in report.items)
    job_id = legacy["song_fast"]
    # commit の後、記録を消す前に止まった
    _write_journal(settings, job_id, f"stems/{job_id}", "stems/song/fast")
    again = migrate_folders(seeded, settings)
    assert again.items == [] and "完了していました" in again.notes[0]
    assert not journal_path(settings).exists()


def test_journal_before_rename_is_cleared(
    seeded: Session, settings: Settings, legacy: dict[str, int]
) -> None:
    job_id = legacy["song_fast"]
    _write_journal(settings, job_id, f"stems/{job_id}", "stems/song/fast")  # 移す前に止まった
    report = migrate_folders(seeded, settings)
    assert "移す前" in report.notes[0]
    assert [i.status for i in report.items] == [MOVED] * 4


def test_journal_ambiguous_blocks(
    seeded: Session, settings: Settings, legacy: dict[str, int]
) -> None:
    job_id = legacy["song_fast"]
    _write_journal(settings, job_id, f"stems/{job_id}", "stems/song/fast")
    (settings.stems_dir / "song" / "fast").mkdir(parents=True)  # 元にも先にもある
    with pytest.raises(MigrationBlocked):
        migrate_folders(seeded, settings)
    with pytest.raises(MigrationBlocked):
        migrate_folders(seeded, settings, dry_run=True)
    assert (settings.stems_dir / str(job_id)).is_dir() and journal_path(settings).exists()


def test_rename_back_failure_needs_check_then_recovers(
    seeded: Session, settings: Settings, legacy: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = legacy["song_combo"]
    real_stats = folder_migration.folder_stats

    def broken_stats(path: Path) -> tuple[int, int]:
        files, total = real_stats(path)
        return (files - 1, total) if path.name == "exp_combo" else (files, total)

    def rename(src: Path, dst: Path) -> None:
        if dst.name == str(bad):  # 元に戻す rename だけ失敗させる
            raise PermissionError("使用中です")
        os.rename(src, dst)

    monkeypatch.setattr(folder_migration, "folder_stats", broken_stats)
    report = migrate_folders(seeded, settings, rename=rename)
    item = next(i for i in report.items if i.job_id == bad)
    assert item.status == NEEDS_CHECK
    assert "song" in item.message and "exp_combo" in item.message  # 新しい場所を知らせる
    assert report.items[-1] is item  # そこで止まる（記録を上書きしない）
    assert journal_path(settings).exists()

    monkeypatch.setattr(folder_migration, "folder_stats", real_stats)
    report = migrate_folders(seeded, settings)
    assert report.notes and "元に戻します" in report.notes[0]
    assert all(i.status == MOVED for i in report.items)
    assert not journal_path(settings).exists()
    seeded.expire_all()
    assert all(
        j.output_dir is not None for j in seeded.scalars(select(SeparationJob))
    )


def _cli_setup(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> int:
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)
    monkeypatch.setenv("COLUMNS", "250")  # 表を折り返さない
    engine = make_engine(settings.db_path)
    try:
        init_db(engine)
        with make_session_factory(engine)() as s:
            seed(s)
            track_id = make_track(s, settings, tmp_path, name=name)
            return _legacy_job(s, settings, track_id, "fast")
    finally:
        engine.dispose()


def test_cli_refuses_while_worker_runs(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _cli_setup(settings, tmp_path, monkeypatch, "song")
    lock = WorkerLock(settings.data_root / "worker.lock")
    lock.acquire()
    try:
        res = runner.invoke(cli.app, ["migrate-folders"])
        assert res.exit_code == 1 and "止めてから" in res.output
        assert (settings.stems_dir / str(job_id)).is_dir()
        dry = runner.invoke(cli.app, ["migrate-folders", "--dry-run"])
        assert dry.exit_code == 0 and "ワーカーが動いています" in dry.output
    finally:
        lock.release()
    res = runner.invoke(cli.app, ["migrate-folders"])
    assert res.exit_code == 0, res.output


def test_cli_shows_brackets_in_names(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cli_setup(settings, tmp_path, monkeypatch, "アカ通信ン [sm46822928] [b]")
    dry = runner.invoke(cli.app, ["migrate-folders", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "stems/アカ通信ン [sm46822928] [b]/fast" in dry.output
