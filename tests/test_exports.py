"""T08 書き出し（ファイル名、対象の決定、音声の中身、ZIP、API、片付け、CLI）。"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from job_helpers import make_track, sync_launcher
from stemapp import cli
from stemapp.audio import SAMPLE_RATE, read_audio, run_ffmpeg
from stemapp.config import Settings
from stemapp.delivery import fake_encoder
from stemapp.exports import naming
from stemapp.exports.naming import content_disposition, safe_filename
from stemapp.exports.render import SourceStem, mix_stems, render_mix, render_single
from stemapp.exports.service import (
    DONE,
    FAILED,
    ExportConflict,
    ExportInvalid,
    ExportNotFound,
    ExportRequest,
    cleanup_exports,
    create_export,
    plan_export,
    recover_interrupted_exports,
    run_export,
)
from stemapp.jobs.worker import Worker
from stemapp.library import resolve_data_path
from stemapp.models import (
    Export,
    ExportItem,
    ListenPreset,
    ListenPresetItem,
    SeparationJob,
    Stem,
    StemRendition,
    StemType,
    Track,
)
from test_api import _app

HAS_FFMPEG = shutil.which("ffmpeg") is not None
LEAVES = ["lead_vocal", "backing_vocal", "drums", "bass", "guitar", "piano", "other"]
TOPS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
LSB24 = 1.0 / (1 << 23)


def fake_mp3(args: Sequence[str]) -> None:
    """ffmpeg の代わり: 入力の WAV をそのまま「MP3」として置く（中身の確認用）。"""
    src = Path(args[list(args).index("-i") + 1])
    dst = Path(args[-1])
    assert "libmp3lame" in args and "320k" in args
    shutil.copyfile(src, dst)


# --- ファイル名 ---------------------------------------------------------------------------


def test_safe_filename_japanese_kept() -> None:
    assert safe_filename("夜に駆ける - メインボーカル", "wav") == "夜に駆ける - メインボーカル.wav"


def test_safe_filename_drops_bad_chars() -> None:
    name = safe_filename('a\\b/c:d*e?f"g<h>i|j\x01k\x7fl - x', "flac")
    assert name == "abcdefghijkl - x.flac"


def test_safe_filename_trailing_dots_and_spaces() -> None:
    assert safe_filename("曲名. . ", "mp3") == "曲名.mp3"
    assert safe_filename("  ...  ", "wav") == "export.wav"
    assert safe_filename("", "zip") == "export.zip"


@pytest.mark.parametrize(
    ("base", "expect"),
    [
        ("CON", "CON_.wav"),
        ("con", "con_.wav"),
        ("NUL", "NUL_.wav"),
        ("COM1", "COM1_.wav"),
        ("lpt9", "lpt9_.wav"),
        ("CON.stems", "CON_.stems.wav"),
        ("CONSOLE", "CONSOLE.wav"),
        ("COM10", "COM10.wav"),
        ("CON - vocals", "CON - vocals.wav"),
    ],
)
def test_safe_filename_reserved(base: str, expect: str) -> None:
    assert safe_filename(base, "wav") == expect


def test_safe_filename_length_limit() -> None:
    name = safe_filename("あ" * 300, "flac")
    assert len(name) == naming.MAX_FILENAME_LEN
    assert name.endswith(".flac")
    # 切った位置の末尾の空白・ピリオドも除く
    name2 = safe_filename("x" * 94 + " .yyyyyyy", "wav")
    assert len(name2) <= 100 and not name2[:-4].endswith((" ", "."))


def test_content_disposition_rfc5987() -> None:
    header = content_disposition('夜に駆ける - ボーカル＋ドラム "live".wav')
    assert header.startswith("attachment; ")
    fallback, star = header.split("; ")[1:]
    assert fallback.startswith('filename="') and fallback.endswith('"')
    assert fallback[len('filename="'):-1].isascii()
    assert '"' not in fallback[len('filename="'):-1]
    assert star.startswith("filename*=UTF-8''")
    assert unquote(star.removeprefix("filename*=UTF-8''")) == (
        '夜に駆ける - ボーカル＋ドラム "live".wav'
    )
    assert " " not in star


# --- 準備 ---------------------------------------------------------------------------------


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """MP3 の変換は fake_mp3（ffmpeg を使わない）。"""
    with TestClient(_app(settings)) as c:
        c.app.state.export_manager.runner = fake_mp3  # type: ignore[attr-defined]
        yield c


def _factory(client: TestClient) -> sessionmaker[Session]:
    return client.app.state.session_factory  # type: ignore[attr-defined]


def _settings(client: TestClient) -> Settings:
    return client.app.state.settings  # type: ignore[attr-defined]


def _done_job(
    client: TestClient, tmp_path: Path, name: str = "song", seed_offset: float = 0.0
) -> int:
    """分割済みのジョブ。seed_offset が負なら音が大きくなる（-2.0 で 3 倍）。"""
    settings = _settings(client)
    with _factory(client)() as s:
        track_id = make_track(s, settings, tmp_path, name=name, seed_offset=seed_offset)
        s.get(Track, track_id).title = "夜に駆ける"  # type: ignore[union-attr]
        s.commit()
    job_id = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"}).json()["job"][
        "job_id"
    ]
    worker = Worker(settings, _factory(client), sync_launcher(settings),
                    postprocess_encoder=fake_encoder)
    assert worker.run_one() == job_id
    return job_id


@pytest.fixture
def job_id(client: TestClient, tmp_path: Path) -> int:
    return _done_job(client, tmp_path)


def _master(session: Session, settings: Settings, job_id: int, code: str) -> np.ndarray:
    rend = session.execute(
        select(StemRendition)
        .join(Stem, Stem.stem_id == StemRendition.stem_id)
        .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
        .where(Stem.job_id == job_id, StemType.code == code, StemRendition.purpose == "master")
    ).scalar_one()
    return read_audio(resolve_data_path(settings, rend.file_path)).astype(np.float64)


def _export(client: TestClient, job_id: int, body: dict[str, Any]) -> dict[str, Any]:
    res = client.post(f"/api/jobs/{job_id}/exports", json=body)
    assert res.status_code == 202, res.text
    client.app.state.export_manager.join()  # type: ignore[attr-defined]
    exp = client.get(f"/api/exports/{res.json()['export']['export_id']}").json()["export"]
    assert exp["status"] == DONE, exp
    return exp


def _download(client: TestClient, exp: dict[str, Any]) -> bytes:
    res = client.get(exp["download_url"])
    assert res.status_code == 200
    return res.content


def _preset(session: Session, name: str, items: list[tuple[str, float]]) -> int:
    ids = {t.code: t.stem_type_id for t in session.scalars(select(StemType))}
    p = ListenPreset(name=name, sort_order=999)
    session.add(p)
    session.flush()
    for code, gain in items:
        session.add(
            ListenPresetItem(
                listen_preset_id=p.listen_preset_id, stem_type_id=ids[code], gain_db=gain
            )
        )
    session.commit()
    return p.listen_preset_id


# --- 対象の決定 ---------------------------------------------------------------------------


def _codes(plan: Any) -> list[str]:
    return [i.stem.code for i in plan.items]


def test_plan_single_all_mix(client: TestClient, job_id: int) -> None:
    with _factory(client)() as s:
        p = plan_export(s, job_id, ExportRequest("single", "wav", stem_code="drums"))
        assert _codes(p) == ["drums"]
        assert p.filename == "夜に駆ける - ドラム.wav"
        p = plan_export(s, job_id, ExportRequest("single", "wav", stem_code="vocals"))
        assert _codes(p) == ["vocals"]  # 親も1つの stem として書き出せる

        p = plan_export(s, job_id, ExportRequest("all", "flac"))
        assert _codes(p) == LEAVES
        assert p.filename == "夜に駆ける - stems.zip"
        p = plan_export(s, job_id, ExportRequest("all", "flac", parents_only=True))
        assert _codes(p) == TOPS

        # 選択中の stem（親は子に広げる）。名前は選んだ stem の表示名を「＋」でつなぐ
        p = plan_export(
            s, job_id, ExportRequest("mix", "mp3", stems=[("drums", -3.0), ("vocals", 0.0)])
        )
        assert _codes(p) == ["lead_vocal", "backing_vocal", "drums"]
        assert [i.gain_db for i in p.items] == [0.0, 0.0, -3.0]
        assert p.filename == "夜に駆ける - ボーカル＋ドラム.mp3"
        assert p.listen_preset_id is None
        # 全部を 0dB で選んだときは「全部」
        p = plan_export(s, job_id, ExportRequest("mix", "wav", stems=[(c, 0.0) for c in TOPS]))
        assert p.filename == "夜に駆ける - 全部.wav"

        # 組み合わせプリセット（グループを含む組み込み）
        karaoke = s.scalars(
            select(ListenPreset).where(ListenPreset.name == "カラオケ（伴奏）")
        ).one()
        kid = karaoke.listen_preset_id
        p = plan_export(s, job_id, ExportRequest("mix", "wav", listen_preset_id=kid))
        assert _codes(p) == ["drums", "bass", "guitar", "piano", "other"]
        assert p.filename == "夜に駆ける - カラオケ（伴奏）.wav"
        assert p.listen_preset_id == karaoke.listen_preset_id


def test_plan_errors(client: TestClient, job_id: int, tmp_path: Path) -> None:
    with _factory(client)() as s:
        with pytest.raises(ExportNotFound):
            plan_export(s, 999, ExportRequest("all", "wav"))
        with pytest.raises(ExportInvalid):
            plan_export(s, job_id, ExportRequest("single", "wav"))
        with pytest.raises(ExportInvalid):
            plan_export(s, job_id, ExportRequest("single", "wav", stem_code="kick"))
        with pytest.raises(ExportInvalid):
            plan_export(s, job_id, ExportRequest("mix", "wav"))
        with pytest.raises(ExportInvalid):
            plan_export(s, job_id, ExportRequest("mix", "ogg", stems=[("bass", 0)]))
        with pytest.raises(ExportNotFound):
            plan_export(s, job_id, ExportRequest("mix", "wav", listen_preset_id=999))
        empty = _preset(s, "空", [])
        with pytest.raises(ExportInvalid):
            plan_export(s, job_id, ExportRequest("mix", "wav", listen_preset_id=empty))
        job = s.get(SeparationJob, job_id)
        assert job is not None
        job.status = "running"
        s.commit()
        with pytest.raises(ExportConflict):
            plan_export(s, job_id, ExportRequest("all", "wav"))


# --- 音声の中身（ffmpeg なし） --------------------------------------------------------------


def _src(tmp_path: Path, name: str, data: np.ndarray, gain_db: float = 0.0) -> SourceStem:
    path = tmp_path / f"{name}.flac"
    sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_24", format="FLAC")
    return SourceStem(code=name, display_name=name, path=path, gain_db=gain_db)


def test_mix_sum_with_gain_and_no_clip(tmp_path: Path) -> None:
    n = 4410
    t = np.arange(n) / SAMPLE_RATE
    a = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    b = np.stack([0.2 * np.sin(2 * np.pi * 330 * t), 0.1 * np.cos(2 * np.pi * 50 * t)], axis=1)
    sa, sb = _src(tmp_path, "a", a), _src(tmp_path, "b", b, gain_db=-6.0)
    mixed, gain = mix_stems([sa, sb])
    expect = read_audio(sa.path) + read_audio(sb.path).astype(np.float64) * 10 ** (-6 / 20)
    assert gain == 0.0
    assert np.max(np.abs(mixed - expect)) < 1e-9


def test_mix_over_one_is_scaled(tmp_path: Path) -> None:
    n = 2000
    loud = np.full((n, 2), 0.8)
    loud[100] = [0.9, -0.9]
    s1, s2 = _src(tmp_path, "x", loud), _src(tmp_path, "y", loud)
    raw = read_audio(s1.path).astype(np.float64) * 2
    peak = np.max(np.abs(raw))
    mixed, gain = mix_stems([s1, s2])
    assert np.isclose(np.max(np.abs(mixed)), 1.0)
    assert np.isclose(gain, -20 * np.log10(peak))
    assert np.allclose(mixed, raw / peak)
    # 24bit で書いても音割れ（折り返し）しない
    dst = tmp_path / "m.wav"
    res = render_mix([s1, s2], dst, "wav")
    back, sr = sf.read(str(dst), dtype="float64", always_2d=True)
    assert sr == SAMPLE_RATE and sf.info(str(dst)).subtype == "PCM_24"
    assert np.max(np.abs(back - raw / peak)) <= 2 * LSB24
    assert res.mix_gain_db == pytest.approx(gain)


def test_single_mp3_uses_ffmpeg_args(tmp_path: Path) -> None:
    src = _src(tmp_path, "s", np.zeros((1000, 2)))
    res = render_single(src, tmp_path / "out" / "s.mp3", "mp3", runner=fake_mp3)
    assert res.path.is_file() and res.bytes > 0
    assert not list((tmp_path / "out").glob("*.tmp.wav"))  # 一時ファイルは消える


# --- API・実行 ---------------------------------------------------------------------------


def test_api_single_flac_matches_master(client: TestClient, job_id: int) -> None:
    exp = _export(client, job_id, {"export_type": "single", "format": "flac", "stem_code": "bass"})
    assert exp["filename"] == "夜に駆ける - ベース.flac"
    assert exp["bytes"] > 0 and exp["track_id"] is not None
    body = _download(client, exp)
    data, sr = sf.read(io.BytesIO(body), dtype="float64", always_2d=True)
    info = sf.info(io.BytesIO(body))
    assert (sr, data.shape[1], info.subtype) == (SAMPLE_RATE, 2, "PCM_24")
    with _factory(client)() as s:
        master = _master(s, _settings(client), job_id, "bass")
    assert data.shape == master.shape
    assert np.max(np.abs(data - master)) <= LSB24  # 同じ音量のまま（戻さない）


def test_api_all_zip_contents(client: TestClient, job_id: int) -> None:
    exp = _export(client, job_id, {"export_type": "all", "format": "wav"})
    assert exp["filename"] == "夜に駆ける - stems.zip" and exp["zip"] is True
    zf = zipfile.ZipFile(io.BytesIO(_download(client, exp)))
    names = zf.namelist()
    shown = ["メインボーカル", "サブボーカル", "ドラム", "ベース", "ギター", "ピアノ", "その他"]
    assert names == [f"夜に駆ける - {n}.wav" for n in shown]
    with _factory(client)() as s:
        master = _master(s, _settings(client), job_id, "drums")
    data, sr = sf.read(io.BytesIO(zf.read("夜に駆ける - ドラム.wav")), dtype="float64")
    assert sr == SAMPLE_RATE and np.max(np.abs(data - master)) <= LSB24
    # 親だけ
    exp2 = _export(client, job_id, {"export_type": "all", "format": "flac", "parents_only": True})
    names2 = zipfile.ZipFile(io.BytesIO(_download(client, exp2))).namelist()
    assert [n.split(" - ")[1] for n in names2] == [
        "ボーカル.flac", "ドラム.flac", "ベース.flac", "ギター.flac", "ピアノ.flac", "その他.flac"
    ]
    with _factory(client)() as s:
        items = s.scalars(select(ExportItem).where(ExportItem.export_id == exp2["export_id"])).all()
        assert len(items) == 6


def test_api_mix_preset_gain(client: TestClient, job_id: int) -> None:
    with _factory(client)() as s:
        pid = _preset(s, "ベース強め", [("bass", 3.0), ("drums", -6.0)])
    exp = _export(client, job_id, {"export_type": "mix", "format": "wav", "listen_preset_id": pid})
    assert exp["filename"] == "夜に駆ける - ベース強め.wav"
    assert exp["listen_preset_id"] == pid
    data, _ = sf.read(io.BytesIO(_download(client, exp)), dtype="float64")
    with _factory(client)() as s:
        st = _settings(client)
        expect = _master(s, st, job_id, "bass") * 10 ** (3 / 20) + _master(
            s, st, job_id, "drums"
        ) * 10 ** (-6 / 20)
        gains = {i.stem_id: i.gain_db for i in s.scalars(
            select(ExportItem).where(ExportItem.export_id == exp["export_id"]))}
    assert sorted(gains.values()) == [-6.0, 3.0]
    assert exp["mix_gain_db"] == 0.0
    assert np.max(np.abs(data - expect)) <= 2 * LSB24


def test_api_mix_selection_clipping_recorded(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path, name="loud", seed_offset=-2.0)
    exp = _export(client, job_id, {
        "export_type": "mix", "format": "flac",
        "stems": [{"code": c, "gain_db": 12.0} for c in LEAVES],
    })
    assert exp["mix_gain_db"] < 0
    data, _ = sf.read(io.BytesIO(_download(client, exp)), dtype="float64")
    assert np.max(np.abs(data)) <= 1.0
    with _factory(client)() as s:
        st = _settings(client)
        raw = sum(_master(s, st, job_id, c) for c in LEAVES) * 10 ** (12 / 20)
        row = s.get(Export, exp["export_id"])
        assert row is not None and row.mix_gain_db == pytest.approx(exp["mix_gain_db"])
    expect = raw * 10 ** (exp["mix_gain_db"] / 20)
    assert np.max(np.abs(data - expect)) <= 2 * LSB24


def test_api_errors(client: TestClient, job_id: int) -> None:
    assert client.get("/api/exports/999").status_code == 404
    assert client.get("/api/exports/999/download").status_code == 404
    res = client.post("/api/jobs/999/exports", json={"export_type": "all", "format": "wav"})
    assert res.status_code == 404
    res = client.post(f"/api/jobs/{job_id}/exports", json={"export_type": "x", "format": "wav"})
    assert res.status_code == 422
    res = client.post(f"/api/jobs/{job_id}/exports",
                      json={"export_type": "single", "format": "wav", "stem_code": "nope"})
    assert res.status_code == 400 and "nope" in res.json()["detail"]
    with _factory(client)() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None
        job.status = "failed"
        s.commit()
    res = client.post(f"/api/jobs/{job_id}/exports", json={"export_type": "all", "format": "wav"})
    assert res.status_code == 409
    assert "分割" in res.json()["detail"]


def test_download_before_done_is_409(client: TestClient, job_id: int) -> None:
    with _factory(client)() as s:
        plan = plan_export(s, job_id, ExportRequest("single", "wav", stem_code="bass"))
        exp = create_export(s, plan)  # 実行はしない（queued のまま）
        export_id = exp.export_id
    res = client.get(f"/api/exports/{export_id}/download")
    assert res.status_code == 409
    got = client.get(f"/api/exports/{export_id}").json()["export"]
    assert got["status"] == "queued" and got["download_url"] is None


def test_download_range(client: TestClient, job_id: int) -> None:
    exp = _export(client, job_id, {"export_type": "single", "format": "wav", "stem_code": "piano"})
    full = _download(client, exp)
    res = client.get(exp["download_url"], headers={"Range": "bytes=10-19"})
    assert res.status_code == 206
    assert res.content == full[10:20]
    assert res.headers["content-range"] == f"bytes 10-19/{len(full)}"
    cd = res.headers["content-disposition"]
    assert "filename*=UTF-8''" in cd
    assert unquote(cd.split("filename*=UTF-8''")[1]) == "夜に駆ける - ピアノ.wav"
    assert res.headers["content-type"] == "audio/wav"


def test_failed_export_reports_message(client: TestClient, job_id: int) -> None:
    with _factory(client)() as s:
        rend = s.execute(
            select(StemRendition)
            .join(Stem, Stem.stem_id == StemRendition.stem_id)
            .join(StemType, StemType.stem_type_id == Stem.stem_type_id)
            .where(Stem.job_id == job_id, StemType.code == "bass",
                   StemRendition.purpose == "master")
        ).scalar_one()
        resolve_data_path(_settings(client), rend.file_path).unlink()
    res = client.post(f"/api/jobs/{job_id}/exports",
                      json={"export_type": "single", "format": "wav", "stem_code": "bass"})
    client.app.state.export_manager.join()  # type: ignore[attr-defined]
    exp = client.get(f"/api/exports/{res.json()['export']['export_id']}").json()["export"]
    assert exp["status"] == FAILED
    assert "ベース" in exp["error_message"]
    assert client.get(f"/api/exports/{exp['export_id']}/download").status_code == 409
    assert not (_settings(client).exports_dir / str(exp["export_id"])).exists()


def test_run_export_skips_non_queued_and_deleted(client: TestClient, job_id: int) -> None:
    settings, factory = _settings(client), _factory(client)
    with factory() as s:
        exp = create_export(s, plan_export(s, job_id, ExportRequest("all", "wav")))
        export_id = exp.export_id
        s.delete(exp)
        s.commit()
    assert run_export(settings, factory, export_id) is None
    assert not (settings.exports_dir / str(export_id)).exists()


# --- 片付け --------------------------------------------------------------------------------


def _fake_done(session: Session, settings: Settings, job_id: int, created: datetime,
               size: int) -> int:
    exp = Export(job_id=job_id, export_type="single", format="wav", status=DONE,
                 filename="x.wav", created_at=created, bytes=size)
    session.add(exp)
    session.flush()
    path = settings.exports_dir / str(exp.export_id) / "x.wav"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\0" * 16)
    exp.output_path = path.relative_to(settings.data_dir).as_posix()
    session.commit()
    return exp.export_id


def test_cleanup_expired_capacity_orphans(client: TestClient, job_id: int) -> None:
    settings = _settings(client).model_copy(update={"export_ttl_hours": 24, "export_max_mb": 1})
    now = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    mb = 1024 * 1024
    with _factory(client)() as s:
        old = _fake_done(s, settings, job_id, now - timedelta(hours=25), 10)
        a = _fake_done(s, settings, job_id, now - timedelta(hours=3), mb // 2)
        b = _fake_done(s, settings, job_id, now - timedelta(hours=2), mb // 2)
        c = _fake_done(s, settings, job_id, now - timedelta(hours=1), mb // 2)
        running = Export(job_id=job_id, export_type="all", format="wav", status="running",
                         created_at=now - timedelta(hours=30))
        s.add(running)
        s.commit()
        orphan = settings.exports_dir / "9999"
        orphan.mkdir()
        (settings.exports_dir / "not-a-number").mkdir()
        removed = cleanup_exports(s, settings, now=now)
        assert sorted(removed) == sorted([old, a])  # 期限切れ・容量超え（古い順）
        left = {e.export_id for e in s.scalars(select(Export))}
        assert left == {b, c, running.export_id}
        assert not (settings.exports_dir / str(old)).exists()
        assert not (settings.exports_dir / str(a)).exists()
        assert (settings.exports_dir / str(c) / "x.wav").is_file()
        assert not orphan.exists()
        assert not (settings.exports_dir / "not-a-number").exists()


def test_cleanup_keeps_newest_even_if_too_big(client: TestClient, job_id: int) -> None:
    settings = _settings(client).model_copy(update={"export_max_mb": 1})
    now = datetime.now(UTC)
    with _factory(client)() as s:
        big = _fake_done(s, settings, job_id, now, 5 * 1024 * 1024)
        assert cleanup_exports(s, settings, now=now) == []
        assert s.get(Export, big) is not None


def test_recover_interrupted(client: TestClient, job_id: int) -> None:
    settings = _settings(client)
    with _factory(client)() as s:
        exp = create_export(s, plan_export(s, job_id, ExportRequest("all", "wav")))
        exp.status = "running"
        s.commit()
        d = settings.exports_dir / str(exp.export_id)
        d.mkdir(parents=True)
        (d / "half.zip").write_bytes(b"x")
        assert recover_interrupted_exports(s, settings) == [exp.export_id]
        s.refresh(exp)
        assert exp.status == FAILED and "中断" in (exp.error_message or "")
        assert not d.exists()


def test_delete_track_removes_exports(client: TestClient, job_id: int) -> None:
    settings = _settings(client)
    exp = _export(client, job_id, {"export_type": "single", "format": "wav", "stem_code": "bass"})
    folder = settings.exports_dir / str(exp["export_id"])
    assert folder.is_dir()
    with _factory(client)() as s:
        track_id = s.get(SeparationJob, job_id).track_id  # type: ignore[union-attr]
    assert client.delete(f"/api/tracks/{track_id}").status_code == 200
    assert client.get(f"/api/exports/{exp['export_id']}").status_code == 404
    assert not folder.exists()
    with _factory(client)() as s:
        assert s.scalars(select(ExportItem)).all() == []


def test_deleted_job_exports_removed_by_cleanup(client: TestClient, job_id: int) -> None:
    """ジョブの行が（曲の削除以外で）消えても、次の片付けでフォルダが消える。"""
    settings = _settings(client)
    exp = _export(client, job_id, {"export_type": "single", "format": "wav", "stem_code": "bass"})
    with _factory(client)() as s:
        s.delete(s.get(Export, exp["export_id"]))
        s.commit()
        cleanup_exports(s, settings)
    assert not (settings.exports_dir / str(exp["export_id"])).exists()


# --- CLI ---------------------------------------------------------------------------------


def test_cli_export(client: TestClient, job_id: int, tmp_path: Path,
                    monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(client)
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    out = tmp_path / "出力先"
    runner = CliRunner()
    res = runner.invoke(cli.app, ["export", str(job_id), "--type", "single", "--format", "flac",
                                  "--stem", "bass", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "夜に駆ける - ベース.flac").is_file()
    # 同じ名前があれば上書きしない
    res = runner.invoke(cli.app, ["export", str(job_id), "--type", "single", "--format", "flac",
                                  "--stem", "bass", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "夜に駆ける - ベース (2).flac").is_file()
    res = runner.invoke(cli.app, ["export", str(job_id), "--type", "all", "--format", "wav",
                                  "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "夜に駆ける - stems.zip").is_file()
    res = runner.invoke(cli.app, ["export", str(job_id), "--type", "mix", "--format", "wav",
                                  "--preset", "カラオケ（伴奏）", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "夜に駆ける - カラオケ（伴奏）.wav").is_file()
    res = runner.invoke(cli.app, ["export", str(job_id), "--type", "mix", "--format", "wav",
                                  "--stem", "drums", "--stem", "bass", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "夜に駆ける - ドラム＋ベース.wav").is_file()
    # 作業用の一時フォルダは残らない、data/exports は使わない、DB に行を作らない
    assert not [p for p in out.iterdir() if p.name.startswith(".stemapp-export-")]
    assert not any(settings.exports_dir.iterdir())
    with _factory(client)() as s:
        assert s.scalars(select(Export)).all() == []
    # 誤り
    res = runner.invoke(cli.app, ["export", str(job_id), "--type", "single", "--format", "wav",
                                  "-o", str(out)])
    assert res.exit_code == 1
    res = runner.invoke(cli.app, ["export", str(job_id), "--type", "mix", "--format", "wav",
                                  "--preset", "無い名前", "-o", str(out)])
    assert res.exit_code == 1 and "見つかりません" in res.output


# --- ffmpeg を使う（各種類×各形式） ----------------------------------------------------------


def _probe(path: Path) -> dict[str, Any]:
    from stemapp.proc import run_bound

    exe = shutil.which("ffprobe")
    assert exe is not None
    proc = run_bound(
        [exe, "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate"
         ":format=duration", "-of", "json", str(path)],
        text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.fixture
def ff_client(settings: Settings) -> Iterator[TestClient]:
    """書き出しの MP3 に本物の ffmpeg を使うサーバー。"""
    app = _app(settings)
    with TestClient(app) as c:
        c.app.state.export_manager.runner = run_ffmpeg  # type: ignore[attr-defined]
        yield c


@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3"])
@pytest.mark.parametrize("export_type", ["single", "all", "mix"])
@pytest.mark.ffmpeg
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg がありません")
def test_each_type_and_format(
    ff_client: TestClient, tmp_path: Path, export_type: str, fmt: str
) -> None:
    client = ff_client
    job_id = _done_job(client, tmp_path)
    body: dict[str, Any] = {"export_type": export_type, "format": fmt}
    if export_type == "single":
        body["stem_code"] = "guitar"
    if export_type == "mix":
        body["stems"] = [{"code": "drums", "gain_db": -3.0}, {"code": "vocals", "gain_db": 2.0}]
    exp = _export(client, job_id, body)
    content = _download(client, exp)
    settings = _settings(client)
    with _factory(client)() as s:
        guitar = _master(s, settings, job_id, "guitar")
        expect_mix = (
            _master(s, settings, job_id, "drums") * 10 ** (-3 / 20)
            + (_master(s, settings, job_id, "lead_vocal")
               + _master(s, settings, job_id, "backing_vocal")) * 10 ** (2 / 20)
        )
    n = guitar.shape[0]
    files: list[tuple[str, bytes]]
    if export_type == "all":
        assert exp["filename"].endswith(".zip")
        zf = zipfile.ZipFile(io.BytesIO(content))
        assert len(zf.namelist()) == len(LEAVES)
        assert all(name.endswith(f".{fmt}") for name in zf.namelist())
        files = [(name, zf.read(name)) for name in zf.namelist()]
    else:
        assert exp["filename"].endswith(f".{fmt}")
        files = [(exp["filename"], content)]
    for name, blob in files:
        path = tmp_path / "check" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(blob)
        info = _probe(path)
        stream = info["streams"][0]
        assert int(stream["sample_rate"]) == SAMPLE_RATE
        assert int(stream["channels"]) == 2
        duration = float(info["format"]["duration"])
        if fmt == "mp3":
            assert stream["codec_name"] == "mp3"
            assert abs(int(stream.get("bit_rate") or 320000) - 320000) <= 1000
            assert abs(duration - n / SAMPLE_RATE) < 0.08  # MP3 は符号化の遅延・詰め物がある
        else:
            data, sr = sf.read(str(path), dtype="float64", always_2d=True)
            assert sf.info(str(path)).subtype == "PCM_24"
            assert data.shape == (n, 2)
            if export_type == "single":
                assert np.max(np.abs(data - guitar)) <= LSB24
            if export_type == "mix":
                assert np.max(np.abs(data - expect_mix)) <= 2 * LSB24


def test_old_db_gets_export_columns(tmp_path: Path) -> None:
    """T08 より前の DB（EXPORT に状態などの列が無い）にも列が足される。"""
    from sqlalchemy import inspect

    from stemapp.db import init_db, make_engine

    engine = make_engine(tmp_path / "old.db")
    try:
        init_db(engine)
        new_cols = ["status", "progress", "stage", "error_message", "filename", "bytes",
                    "mix_gain_db", "finished_at"]
        with engine.begin() as conn:
            for col in new_cols:
                conn.exec_driver_sql(f"ALTER TABLE export DROP COLUMN {col}")
            conn.exec_driver_sql("ALTER TABLE export_item DROP COLUMN gain_db")
        init_db(engine)
        cols = {c["name"] for c in inspect(engine).get_columns("export")}
        assert set(new_cols) <= cols
        assert "gain_db" in {c["name"] for c in inspect(engine).get_columns("export_item")}
    finally:
        engine.dispose()
