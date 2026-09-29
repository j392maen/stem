"""T11 速度変更（ピッチを保つ方式）: 伸縮ジョブ、キャッシュの上限、キャンセル、削除、API。

通常のテストは FakeStretcher（線形補間で長さだけを変える）。本物の rubberband は `-m ffmpeg`。
"""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from job_helpers import make_track, sync_launcher
from stemapp.audio import SAMPLE_RATE, run_ffmpeg
from stemapp.config import Settings
from stemapp.delivery import fake_encoder
from stemapp.jobs.worker import Worker
from stemapp.library import resolve_data_path
from stemapp.models import SeparationJob, Stem, StemRendition, TempoRender, TempoRendition
from stemapp.tempo import service as tempo
from stemapp.tempo.stretch import (
    FakeStretcher,
    FfmpegStretcher,
    StretchCanceled,
    latency_shift,
    rubberband_filter,
    stretch_args,
)
from test_api import _app

HAS_FFMPEG = shutil.which("ffmpeg") is not None
LEAVES = ["lead_vocal", "backing_vocal", "drums", "bass", "guitar", "piano", "other"]


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(_app(settings)) as c:
        yield c


def _factory(client: TestClient) -> sessionmaker[Session]:
    return client.app.state.session_factory  # type: ignore[attr-defined]


def _settings(client: TestClient) -> Settings:
    return client.app.state.settings  # type: ignore[attr-defined]


def _done_job(
    client: TestClient, tmp_path: Path, name: str = "song", seconds: float = 1.0,
    offset: float = 0.0,
) -> int:
    settings = _settings(client)
    with _factory(client)() as s:
        track_id = make_track(
            s, settings, tmp_path, name=name, seconds=seconds, seed_offset=offset
        )
    job_id = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"}).json()["job"][
        "job_id"
    ]
    worker = Worker(settings, _factory(client), sync_launcher(settings),
                    postprocess_encoder=fake_encoder)
    assert worker.run_one() == job_id
    return job_id


def _worker(client: TestClient, stretcher: Any, settings: Settings | None = None) -> Worker:
    return Worker(
        settings or _settings(client), _factory(client), sync_launcher(_settings(client)),
        tempo_stretcher=stretcher,
    )


def _request(client: TestClient, job_id: int, ratio: float) -> dict[str, Any]:
    res = client.post(f"/api/jobs/{job_id}/tempo", json={"ratio": ratio})
    assert res.status_code in (200, 202), res.text
    return res.json()["render"]


def _render(client: TestClient, render_id: int) -> dict[str, Any]:
    return client.get(f"/api/tempo/{render_id}").json()


def _track_of(client: TestClient, job_id: int) -> int:
    with _factory(client)() as s:
        return s.get(SeparationJob, job_id).track_id  # type: ignore[union-attr]


# --- 純粋な部品 ---------------------------------------------------------------------------


def test_normalize_ratio_and_key() -> None:
    assert tempo.normalize_ratio(1.10004) == 1.1
    assert tempo.ratio_key(1.1) == "1.100"
    assert tempo.ratio_key(0.9) == "0.900"
    for bad in (0.49, 2.01, 1.0, 1.0004, float("nan")):
        with pytest.raises(tempo.TempoInvalid):
            tempo.normalize_ratio(bad)


def test_rubberband_args_trim_to_frames() -> None:
    f = rubberband_filter(1.1, 40091)
    # 速くするときは遅れの分だけ先頭を落とし、遅くするときは先頭に無音を足す
    assert f == (
        "rubberband=tempo=1.100000:channels=together,atrim=start_sample=145,"
        "asetpts=PTS-STARTPTS,apad,atrim=end_sample=40091,aresample=48000"
    )
    assert rubberband_filter(0.9, 49000) == (
        "rubberband=tempo=0.900000:channels=together,adelay=delays=178S:all=1,"
        "apad,atrim=end_sample=49000,aresample=48000"
    )
    assert latency_shift(2.0) == 800 and latency_shift(0.5) == -1600
    args = stretch_args(Path("in.flac"), Path("out.webm"), 1.1, 40091)
    assert args[args.index("-af") + 1] == f
    assert ["-c:a", "libopus", "-b:a", "128k"] == args[args.index("-c:a"):args.index("-c:a") + 4]
    assert args[-3:] == ["-f", "webm", "out.webm"]
    assert "-progress" in args


def test_worker_count(settings: Settings) -> None:
    assert tempo.worker_count(settings.model_copy(update={"tempo_workers": 3}), 7) == 3
    assert tempo.worker_count(settings.model_copy(update={"tempo_workers": 3}), 2) == 2
    auto = tempo.worker_count(settings, 100)
    assert 1 <= auto <= tempo.MAX_WORKERS


# --- 伸縮ジョブ -----------------------------------------------------------------------------


def test_render_with_fake_stretcher(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    settings = _settings(client)
    r = _request(client, job_id, 1.1)
    assert r["status"] == "queued" and r["ratio_key"] == "1.100" and r["files"] == {}
    fake = FakeStretcher()
    assert _worker(client, fake).run_tempo_one() == r["render_id"]
    done = _render(client, r["render_id"])
    assert done["status"] == "done" and done["progress"] == 1.0
    # 子に分かれていない stem（画面で鳴らす stem）だけを伸縮する
    assert sorted(done["files"]) == sorted(LEAVES)
    # 長さ = master の長さ ÷ 倍率（全 stem で同じ）
    expected = round(SAMPLE_RATE * 1.0 / 1.1)
    assert done["frames"] == expected
    assert {frames for _src, _r, frames in fake.calls} == {expected}
    out_dir = settings.data_dir / "cache" / "tempo" / str(job_id) / "1.100"
    lengths = set()
    for code in LEAVES:
        path = out_dir / f"{code}.webm"
        assert path.is_file()
        lengths.add(sf.info(str(path)).frames)
    assert lengths == {expected}
    with _factory(client)() as s:
        render = s.get(TempoRender, r["render_id"])
        assert render is not None
        assert render.dir_path == f"cache/tempo/{job_id}/1.100"
        rows = s.scalars(select(TempoRendition).where(TempoRendition.render_id == r["render_id"]))
        rows = list(rows)
        assert len(rows) == len(LEAVES)
        assert render.bytes == sum(x.bytes or 0 for x in rows)
        assert all(x.codec == "opus" and x.bitrate_kbps == 128 for x in rows)
    # ファイルの配信
    url = done["files"]["drums"]
    res = client.get(url)
    assert res.status_code == 200 and res.headers["content-type"] == "audio/webm"
    assert res.content == (out_dir / "drums.webm").read_bytes()
    # 同じ倍率をもう一度選ぶと作り直さない（200）
    again = client.post(f"/api/jobs/{job_id}/tempo", json={"ratio": 1.1})
    assert again.status_code == 200 and again.json()["created"] is False
    assert again.json()["render"]["status"] == "done"
    assert _worker(client, fake).run_tempo_one() is None
    # 一覧
    listed = client.get(f"/api/jobs/{job_id}/tempo").json()["renders"]
    assert [x["ratio_key"] for x in listed] == ["1.100"]
    # 分け方の stem 一覧には混ざらない
    stems = client.get(f"/api/jobs/{job_id}/stems").json()["stems"]
    assert all(r["purpose"] in ("master", "stream") for s in stems for r in s["renditions"])


def test_output_frames_uses_shortest_stem(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    settings = _settings(client)
    with _factory(client)() as s:
        sources = tempo.leaf_sources(s, settings, job_id)
    # 1つの stem だけ短くする → 全 stem をいちばん短い長さに合わせる
    short = sources[0].master
    data, sr = sf.read(str(short), dtype="float32", always_2d=True)
    sf.write(str(short), data[: sr // 2], sr, subtype="PCM_24", format="FLAC")
    assert tempo.output_frames(sources, 0.5) == sr // 2 * 2
    assert tempo.output_frames(sources, 2.0) == sr // 4


def test_input_master_path_comes_from_db(client: TestClient, tmp_path: Path) -> None:
    """伸縮の入力（master の FLAC）は STEM_RENDITION.file_path から引く（フォルダを推測しない）。"""
    job_id = _done_job(client, tmp_path)
    settings = _settings(client)
    moved = settings.data_dir / "elsewhere"
    moved.mkdir()
    with _factory(client)() as s:
        rows = s.scalars(
            select(StemRendition)
            .join(Stem, Stem.stem_id == StemRendition.stem_id)
            .where(Stem.job_id == job_id, StemRendition.purpose == "master")
        ).all()
        for rend in rows:
            src = resolve_data_path(settings, rend.file_path)
            dst = moved / f"{rend.stem_id}.flac"
            shutil.move(src, dst)
            rend.file_path = f"elsewhere/{rend.stem_id}.flac"
        s.commit()
    _request(client, job_id, 1.1)
    fake = FakeStretcher()
    _worker(client, fake).run_tempo_one()
    assert fake.calls and all(src.parent == moved for src, _r, _f in fake.calls)


def test_render_failure_cleans_up(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    r = _request(client, job_id, 0.9)
    _worker(client, FakeStretcher(fail=True)).run_tempo_one()
    failed = _render(client, r["render_id"])
    assert failed["status"] == "failed" and "失敗" in failed["error_message"]
    assert failed["files"] == {}
    assert not (_settings(client).data_dir / "cache" / "tempo" / str(job_id) / "0.900").exists()
    # もう一度選ぶと作成待ちに戻る
    again = client.post(f"/api/jobs/{job_id}/tempo", json={"ratio": 0.9})
    assert again.status_code == 202 and again.json()["render"]["status"] == "queued"
    _worker(client, FakeStretcher()).run_tempo_one()
    assert _render(client, r["render_id"])["status"] == "done"


def test_cancel_queued_and_running(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    queued = _request(client, job_id, 1.2)
    res = client.post(f"/api/tempo/{queued['render_id']}/cancel")
    assert res.status_code == 200 and res.json()["status"] == "canceled"
    assert client.post(f"/api/tempo/{queued['render_id']}/cancel").status_code == 409

    running = _request(client, job_id, 1.3)
    worker = _worker(client, FakeStretcher(delay_sec=3.0, steps=30))
    t = threading.Thread(target=worker.run_tempo_one)
    t.start()
    deadline = time.monotonic() + 10
    while True:
        cur = _render(client, running["render_id"])
        if cur["status"] == "running" and cur["progress"] > 0:
            break
        assert time.monotonic() < deadline, cur
        time.sleep(0.05)
    started = time.monotonic()
    assert client.post(f"/api/tempo/{running['render_id']}/cancel").status_code == 200
    t.join(timeout=10)
    assert not t.is_alive()
    assert time.monotonic() - started < 2.0  # 伸縮の途中でやめる
    cur = _render(client, running["render_id"])
    assert cur["status"] == "canceled" and cur["files"] == {}
    assert not (_settings(client).data_dir / "cache" / "tempo" / str(job_id) / "1.300").exists()
    with _factory(client)() as s:
        assert not list(s.scalars(select(TempoRendition)))


def test_worker_stop_requeues(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    r = _request(client, job_id, 1.05)
    worker = _worker(client, FakeStretcher(delay_sec=3.0, steps=30))
    t = threading.Thread(target=worker.run_tempo_one)
    t.start()
    deadline = time.monotonic() + 10
    while _render(client, r["render_id"])["progress"] <= 0:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    worker.stop()
    t.join(timeout=10)
    cur = _render(client, r["render_id"])
    assert cur["status"] == "queued" and cur["progress"] == 0.0


def test_postprocess_rebuild_invalidates_tempo_cache(client: TestClient, tmp_path: Path) -> None:
    """配信用データを作り直したら、そのジョブの速度変更のキャッシュ（行とフォルダ）を消す。"""
    job_id = _done_job(client, tmp_path)
    other = _done_job(client, tmp_path / "o", name="o", offset=0.3)
    settings = _settings(client)
    _run(client, job_id, 1.1)
    _run(client, other, 1.1)
    job_dir = settings.data_dir / "cache" / "tempo" / str(job_id)
    assert job_dir.is_dir()
    # stream を1つ消して作り直しを頼む
    with _factory(client)() as s:
        rend = s.scalars(
            select(StemRendition)
            .join(Stem, Stem.stem_id == StemRendition.stem_id)
            .where(Stem.job_id == job_id, StemRendition.purpose == "stream")
        ).first()
        assert rend is not None
        s.delete(rend)
        s.commit()
    assert client.post(f"/api/jobs/{job_id}/postprocess").status_code == 202
    worker = Worker(settings, _factory(client), sync_launcher(settings),
                    postprocess_encoder=fake_encoder)
    assert worker.run_postprocess_one() == job_id
    assert not job_dir.exists()
    assert client.get(f"/api/jobs/{job_id}/tempo").json()["renders"] == []
    # ほかのジョブのキャッシュは残る
    assert _done_ratios(client, other) == ["1.100"]
    # 公開関数としても使える（何も無くても失敗しない）
    with _factory(client)() as s:
        assert tempo.invalidate_job_tempo(s, settings, job_id) == 0
        assert tempo.invalidate_job_tempo(s, settings, other) == 1
    assert _done_ratios(client, other) == []


def test_request_render_stale_state(client: TestClient, tmp_path: Path) -> None:
    """読んだ後に別の処理が状態を変えても（片付け・失敗）、500 にせず正しく登録する。"""
    job_id = _done_job(client, tmp_path)
    rid = _run(client, job_id, 1.1)
    factory = _factory(client)
    with factory() as a:
        stale = a.get(TempoRender, rid)
        assert stale is not None and stale.status == "done"
        with factory() as b:  # 別の処理が失敗にした
            b.get(TempoRender, rid).status = "failed"  # type: ignore[union-attr]
            b.commit()
        res = tempo.request_render(a, job_id, 1.1)
        assert res.created and res.render.status == "queued"
        with factory() as b:  # 別の処理が消した（キャッシュの片付け）
            tempo.remove_render(b, _settings(client), b.get(TempoRender, rid))  # type: ignore[arg-type]
        res = tempo.request_render(a, job_id, 1.1)
        assert res.created and res.render.status == "queued"
        again = tempo.request_render(a, job_id, 1.1)
        assert not again.created and again.render.render_id == res.render.render_id


def test_recover_interrupted(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    settings = _settings(client)
    r = _request(client, job_id, 1.5)
    with _factory(client)() as s:
        assert tempo.claim_next_render(s) == r["render_id"]
    half = settings.data_dir / "cache" / "tempo" / str(job_id) / "1.500"
    half.mkdir(parents=True)
    (half / "drums.webm").write_bytes(b"x")
    stray = settings.data_dir / "cache" / "tempo" / "9999" / "1.100"
    stray.mkdir(parents=True)
    _worker(client, FakeStretcher()).recover()
    cur = _render(client, r["render_id"])
    assert cur["status"] == "failed" and "中断" in cur["error_message"]
    assert not half.exists() and not stray.parent.exists()


def _run(client: TestClient, job_id: int, ratio: float, settings: Settings | None = None) -> int:
    r = _request(client, job_id, ratio)
    _worker(client, FakeStretcher(), settings).run_tempo_one()
    assert _render(client, r["render_id"])["status"] == "done"
    return int(r["render_id"])


def _done_ratios(client: TestClient, job_id: int) -> list[str]:
    return [
        x["ratio_key"]
        for x in client.get(f"/api/jobs/{job_id}/tempo").json()["renders"]
        if x["status"] == "done"
    ]


def test_cache_limit_per_track_lru(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    settings = _settings(client)
    root = settings.data_dir / "cache" / "tempo" / str(job_id)
    for ratio in (0.8, 0.9, 1.1):
        _run(client, job_id, ratio)
        time.sleep(0.01)
    assert _done_ratios(client, job_id) == ["0.800", "0.900", "1.100"]
    # 0.8 を使い直す → 4つ目を作ると、最も長く使っていない 0.9 が消える
    time.sleep(0.01)
    _request(client, job_id, 0.8)
    time.sleep(0.01)
    _run(client, job_id, 1.2)
    assert _done_ratios(client, job_id) == ["0.800", "1.100", "1.200"]
    assert sorted(p.name for p in root.iterdir()) == ["0.800", "1.100", "1.200"]
    with _factory(client)() as s:
        assert s.scalar(select(TempoRender).where(TempoRender.ratio == 0.9)) is None


def test_cache_limit_is_per_track(client: TestClient, tmp_path: Path) -> None:
    a = _done_job(client, tmp_path, name="a")
    b = _done_job(client, tmp_path / "b", name="b", offset=0.3)
    assert _track_of(client, a) != _track_of(client, b)
    for ratio in (0.8, 0.9, 1.1):
        _run(client, a, ratio)
        _run(client, b, ratio)
    assert len(_done_ratios(client, a)) == 3 and len(_done_ratios(client, b)) == 3


def test_cache_limit_total_bytes(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    first = _run(client, job_id, 0.8)
    with _factory(client)() as s:
        size = s.get(TempoRender, first).bytes  # type: ignore[union-attr]
    assert size and size > 0
    # 全体の上限を 1 つ分と少しにする → 2つ目を作ると古い方が消える
    small = _settings(client).model_copy(
        update={"tempo_cache_max_mb": 2}
    )
    assert size < 2 * 1024 * 1024 < size * 2, size
    time.sleep(0.01)
    second = _run(client, job_id, 0.9, small)
    assert _done_ratios(client, job_id) == ["0.900"]
    with _factory(client)() as s:
        assert s.get(TempoRender, first) is None and s.get(TempoRender, second) is not None
    # 作ったばかりのものは、それだけで上限を超えても消さない
    tiny = _settings(client).model_copy(update={"tempo_cache_max_mb": 0})
    _run(client, job_id, 1.1, tiny)
    assert _done_ratios(client, job_id) == ["1.100"]


def test_delete_track_and_job_remove_cache(client: TestClient, tmp_path: Path) -> None:
    settings = _settings(client)
    a = _done_job(client, tmp_path, name="a")
    b = _done_job(client, tmp_path / "b", name="b", offset=0.3)
    _run(client, a, 1.1)
    _run(client, b, 1.1)
    dir_a = settings.data_dir / "cache" / "tempo" / str(a)
    dir_b = settings.data_dir / "cache" / "tempo" / str(b)
    assert dir_a.is_dir() and dir_b.is_dir()

    def stems_dir(job_id: int) -> Path:
        with _factory(client)() as s:
            job = s.get(SeparationJob, job_id)
            assert job is not None and job.output_dir
            return resolve_data_path(settings, job.output_dir)

    def export_dir(job_id: int) -> Path:
        res = client.post(f"/api/jobs/{job_id}/exports",
                          json={"export_type": "single", "format": "wav", "stem_code": "drums"})
        assert res.status_code == 202, res.text
        client.app.state.export_manager.join()  # type: ignore[attr-defined]
        d = settings.data_dir / "exports" / str(res.json()["export"]["export_id"])
        assert d.is_dir()
        return d

    # 曲の削除: 保存フォルダ（T13）・書き出し・速度変更のキャッシュがすべて消える
    stems_a, export_a = stems_dir(a), export_dir(a)
    assert stems_a.is_dir()
    assert client.delete(f"/api/tracks/{_track_of(client, a)}").status_code == 200
    assert not dir_a.exists() and dir_b.is_dir()
    assert not stems_a.exists() and not export_a.exists()
    stems_b, export_b = stems_dir(b), export_dir(b)
    assert stems_b.is_dir() and export_b.is_dir()
    # 分け方（ジョブ）の削除でも消える（同じ曲に別の分け方を作ってから消す）
    track_b = _track_of(client, b)
    job2 = client.post(f"/api/tracks/{track_b}/jobs", json={"preset": "standard"}).json()["job"]
    Worker(settings, _factory(client), sync_launcher(settings),
           postprocess_encoder=fake_encoder).run_one()
    assert client.get(f"/api/jobs/{job2['job_id']}").json()["status"] == "done"
    assert client.delete(f"/api/jobs/{b}").status_code == 200
    assert not dir_b.exists()
    assert not stems_b.exists() and not export_b.exists()
    assert stems_dir(job2["job_id"]).is_dir()  # 同じ曲のほかの分け方は残る
    with _factory(client)() as s:
        assert not list(s.scalars(select(TempoRender)))
        assert not list(s.scalars(select(TempoRendition)))


def test_delete_track_while_rendering(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    r = _request(client, job_id, 1.1)
    worker = _worker(client, FakeStretcher(delay_sec=3.0, steps=30))
    t = threading.Thread(target=worker.run_tempo_one)
    t.start()
    deadline = time.monotonic() + 10
    while _render(client, r["render_id"])["progress"] <= 0:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert client.delete(f"/api/tracks/{_track_of(client, job_id)}").status_code == 200
    t.join(timeout=10)
    assert not t.is_alive()
    assert not (_settings(client).data_dir / "cache" / "tempo" / str(job_id)).exists()


# --- API ---------------------------------------------------------------------------------


def test_api_errors(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    assert client.post(f"/api/jobs/{job_id}/tempo", json={"ratio": 1.0}).status_code == 400
    assert client.post(f"/api/jobs/{job_id}/tempo", json={"ratio": 3.0}).status_code == 422
    assert client.post(f"/api/jobs/{job_id}/tempo", json={"ratio": 0.3}).status_code == 422
    assert client.post("/api/jobs/9999/tempo", json={"ratio": 1.1}).status_code == 404
    assert client.get("/api/jobs/9999/tempo").status_code == 404
    assert client.get("/api/tempo/9999").status_code == 404
    assert client.post("/api/tempo/9999/cancel").status_code == 404
    assert client.get("/api/tempo/9999/events").status_code == 404
    assert client.get("/api/files/tempo/9999/1").status_code == 404
    # 分割が終わっていないジョブ
    settings = _settings(client)
    with _factory(client)() as s:
        track_id = make_track(s, settings, tmp_path / "q", name="q", seed_offset=0.4)
    queued = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"}).json()["job"]
    res = client.post(f"/api/jobs/{queued['job_id']}/tempo", json={"ratio": 1.1})
    assert res.status_code == 409 and "分割" in res.json()["detail"]


def test_api_events_until_done(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path)
    r = _request(client, job_id, 1.1)
    worker = _worker(client, FakeStretcher(delay_sec=0.5))
    t = threading.Thread(target=worker.run_tempo_one)
    t.start()
    events: list[str] = []
    with client.stream("GET", f"/api/tempo/{r['render_id']}/events") as res:
        assert res.headers["content-type"].startswith("text/event-stream")
        for line in res.iter_lines():
            if line.startswith("event: "):
                events.append(line.removeprefix("event: "))
            if line.startswith("data: ") and '"status": "done"' in line:
                break
    t.join(timeout=10)
    assert events and set(events) == {"tempo"}
    assert _render(client, r["render_id"])["status"] == "done"


def test_worker_order_tempo_after_separation(client: TestClient, tmp_path: Path) -> None:
    """分割待ちがあるときは分割を先に行う（伸縮は後回し）。"""
    job_id = _done_job(client, tmp_path)
    r = _request(client, job_id, 1.1)
    settings = _settings(client)
    with _factory(client)() as s:
        track_id = make_track(s, settings, tmp_path / "n", name="n", seed_offset=0.4)
    queued = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"}).json()["job"]
    order: list[str] = []
    worker = _worker(client, FakeStretcher())
    orig_one, orig_tempo = worker.run_one, worker.run_tempo_one

    def run_one() -> int | None:
        res = orig_one()
        if res is not None:
            order.append("job")
        return res

    def run_tempo_one() -> int | None:
        res = orig_tempo()
        if res is not None:
            order.append("tempo")
            worker.stop()
        return res

    worker.run_one = run_one  # type: ignore[method-assign]
    worker.run_tempo_one = run_tempo_one  # type: ignore[method-assign]
    worker.poll_interval = 0.01
    worker.run_forever()
    assert order == ["job", "tempo"]
    assert client.get(f"/api/jobs/{queued['job_id']}").json()["status"] == "done"
    assert _render(client, r["render_id"])["status"] == "done"


# --- 本物の rubberband（ffmpeg） --------------------------------------------------------------


def _decode(path: Path, tmp: Path) -> np.ndarray:
    wav = tmp / (path.stem + ".wav")
    run_ffmpeg(["-y", "-i", str(path), "-c:a", "pcm_f32le", "-f", "wav", str(wav)])
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    assert sr == 48000
    return data


@pytest.mark.ffmpeg
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg がありません")
def test_real_rubberband_lengths_match(client: TestClient, tmp_path: Path) -> None:
    job_id = _done_job(client, tmp_path, seconds=3.0)
    settings = _settings(client)
    r = _request(client, job_id, 1.1)
    _worker(client, FfmpegStretcher()).run_tempo_one()
    done = _render(client, r["render_id"])
    assert done["status"] == "done", done
    expected = round(3.0 * SAMPLE_RATE / 1.1)
    assert done["frames"] == expected
    out = tmp_path / "decoded"
    out.mkdir()
    lengths = {}
    with _factory(client)() as s:
        rows = s.scalars(
            select(TempoRendition).where(TempoRendition.render_id == r["render_id"])
        )
        for rend in rows:
            lengths[rend.stem_id] = len(_decode(resolve_data_path(settings, rend.file_path), out))
    # 全 stem が同じ長さ（48kHz に直した長さ）
    assert len(set(lengths.values())) == 1, lengths
    n = next(iter(lengths.values()))
    assert abs(n - expected * 48000 / SAMPLE_RATE) < 48 * 2, (n, expected)  # 2ms 以内


@pytest.mark.ffmpeg
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg がありません")
def test_real_rubberband_keeps_pitch_and_aligns(tmp_path: Path) -> None:
    """440Hz の音を 1.25 倍にしてもピッチが変わらず、同じ入力の 2 本はずれない。"""
    sr = SAMPLE_RATE
    t = np.arange(sr * 4) / sr
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    clicks = np.zeros_like(t)
    clicks[(np.arange(8) * sr // 2)] = 0.9
    src_a = tmp_path / "a.flac"
    src_b = tmp_path / "b.flac"
    sf.write(str(src_a), np.stack([tone, tone], 1), sr, subtype="PCM_24", format="FLAC")
    sf.write(str(src_b), np.stack([clicks, clicks], 1), sr, subtype="PCM_24", format="FLAC")
    frames = round(len(t) / 1.25)
    seen: list[float] = []
    st = FfmpegStretcher()
    st(src_a, tmp_path / "a.webm", 1.25, frames, seen.append, lambda: False)
    st(src_b, tmp_path / "b.webm", 1.25, frames, lambda _f: None, lambda: False)
    assert seen and seen[-1] == 1.0 and all(0 <= f <= 1 for f in seen)
    a = _decode(tmp_path / "a.webm", tmp_path)[:, 0]
    b = _decode(tmp_path / "b.webm", tmp_path)[:, 0]
    assert len(a) == len(b)
    spec = np.abs(np.fft.rfft(a[48000:96000]))  # 1 秒分 → 1Hz 刻み
    peak_hz = float(np.argmax(spec))
    assert abs(peak_hz - 440) < 10, peak_hz  # ピッチが変わると 550Hz になる
    # 0.5 秒おきのクリックは 0.4 秒おきになる（位置のずれは 10ms 未満）
    for k in range(1, 6):
        at = int(k * 0.4 * 48000)
        lo = at - 2400
        pos = lo + int(np.argmax(np.abs(b[lo:at + 2400])))
        assert abs(pos - at) / 48000 < 0.01, (k, pos, at)


@pytest.mark.ffmpeg
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg がありません")
@pytest.mark.ffmpeg
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg がありません")
@pytest.mark.parametrize("ratio", [0.5, 0.9, 1.1, 2.0])
def test_real_rubberband_time_matches_song_time(tmp_path: Path, ratio: float) -> None:
    """伸縮した音の位置が「元の時刻 ÷ 倍率」と ±5ms 以内（曲の時刻に直して）。

    1 秒おきの 440Hz ガウス形バースト（σ=10ms）の重心で測る。補正しないと
    0.5 倍で約 −17ms、2.0 倍で約 +36ms ずれる（rubberband フィルタの遅延補正の食い違い）。
    """
    sr = SAMPLE_RATE
    n_sec = 12
    t = np.arange(sr * n_sec) / sr
    centers = np.arange(1.0, n_sec - 1)
    x = 0.5 * sum(
        np.exp(-0.5 * ((t - c) / 0.01) ** 2) * np.sin(2 * np.pi * 440 * t) for c in centers
    )
    src = tmp_path / "bursts.flac"
    sf.write(str(src), np.stack([x, 0.7 * x], 1), sr, subtype="PCM_24", format="FLAC")
    frames = round(len(t) / ratio)
    FfmpegStretcher()(src, tmp_path / "out.webm", ratio, frames, lambda _f: None, lambda: False)
    y = _decode(tmp_path / "out.webm", tmp_path)[:, 0].astype(np.float64)
    assert abs(len(y) - frames * 48000 / sr) <= 1  # 48kHz へのリサンプルの丸め
    env = y**2
    errors = []
    for c in centers:
        at = c / ratio
        lo, hi = int((at - 0.3 / ratio) * 48000), int((at + 0.3 / ratio) * 48000)
        seg = env[lo:hi]
        centroid = float((seg * np.arange(lo, hi)).sum() / seg.sum()) / 48000
        errors.append((centroid - at) * ratio * 1000)  # 曲の時刻で ms
    mean = float(np.mean(errors))
    assert abs(mean) < 5.0, (ratio, mean, errors)


def test_real_rubberband_stop_kills_ffmpeg(tmp_path: Path) -> None:
    sr = SAMPLE_RATE
    data = (0.1 * np.random.default_rng(0).standard_normal((sr * 120, 2))).astype(np.float32)
    src = tmp_path / "long.flac"
    sf.write(str(src), data, sr, subtype="PCM_24", format="FLAC")
    seen: list[float] = []
    started = time.monotonic()
    with pytest.raises(StretchCanceled):
        FfmpegStretcher()(
            src, tmp_path / "out.webm", 0.8, round(len(data) / 0.8), seen.append,
            lambda: time.monotonic() - started > 1.0,
        )
    assert time.monotonic() - started < 5.0


def test_stretch_rendition_not_in_stem_rendition(client: TestClient, tmp_path: Path) -> None:
    """伸縮済みの音声は STEM_RENDITION に入れない（master / stream だけ）。"""
    job_id = _done_job(client, tmp_path)
    _run(client, job_id, 1.1)
    with _factory(client)() as s:
        purposes = set(
            s.scalars(
                select(StemRendition.purpose)
                .join(Stem, Stem.stem_id == StemRendition.stem_id)
                .where(Stem.job_id == job_id)
            )
        )
    assert purposes == {"master", "stream"}
