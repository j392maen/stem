"""T05 で足した API（組み合わせプリセットの編集、キュー、配信用データの作り直し、画面の配信）。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from job_helpers import make_track, sync_launcher
from stemapp.config import Settings
from stemapp.delivery import fake_encoder, missing_delivery
from stemapp.jobs.queue import recover_interrupted_jobs
from stemapp.jobs.worker import Worker
from stemapp.models import (
    CuePoint,
    ListenPreset,
    ListenPresetItem,
    SeparationJob,
    Stem,
    StemRendition,
    Waveform,
)
from stemapp.seed import seed
from test_api import PASS, _app


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(_app(settings)) as c:
        yield c


def _factory(client: TestClient) -> sessionmaker[Session]:
    return client.app.state.session_factory  # type: ignore[attr-defined]


def _worker(client: TestClient, encoder: Any = fake_encoder) -> Worker:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    return Worker(
        settings, _factory(client), sync_launcher(settings), postprocess_encoder=encoder
    )


@pytest.fixture
def done_job(client: TestClient, tmp_path: Path) -> tuple[int, int]:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _factory(client)() as s:
        track_id = make_track(s, settings, tmp_path)
    job_id = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"}).json()["job"][
        "job_id"
    ]
    assert _worker(client).run_one() == job_id
    return track_id, job_id


def _type_id(client: TestClient, code: str) -> int:
    types = client.get("/api/stem-types").json()["stem_types"]
    return next(t["stem_type_id"] for t in types if t["code"] == code)


def _group_id(client: TestClient, code: str) -> int:
    groups = client.get("/api/stem-groups").json()["stem_groups"]
    return next(g["group_id"] for g in groups if g["code"] == code)


# --- 画面の配信 -----------------------------------------------------------------------


def test_static_index_and_api_untouched(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "<title>stemapp</title>" in res.text
    assert res.headers["cache-control"] == "no-cache"
    js = client.get("/js/main.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert client.get("/css/app.css").status_code == 200
    # /api は従来どおり（画面のファイルとして扱わない）
    assert client.get("/api/health").json()["status"] == "ok"
    res = client.get("/api/no-such-thing")
    assert res.status_code == 404 and res.json() == {"detail": "見つかりません。"}
    assert client.get("/api").status_code == 404
    assert client.get("/no-such-file.js").status_code == 404


def test_static_is_public_but_api_needs_login(settings: Settings) -> None:
    with TestClient(_app(settings, passcode=PASS)) as c:
        assert c.get("/").status_code == 200
        assert c.get("/js/api.js").status_code == 200
        assert c.get("/api/me").status_code == 401
        assert c.get("/api/tracks/1/cues").status_code == 401
        assert c.post("/api/listen-presets", json={"name": "x"}).status_code == 401


# --- 組み合わせプリセット -------------------------------------------------------------


def test_listen_preset_crud(client: TestClient) -> None:
    bass = _type_id(client, "bass")
    chords = _group_id(client, "chords")
    before = client.get("/api/listen-presets").json()["listen_presets"]

    res = client.post(
        "/api/listen-presets",
        json={
            "name": "  練習用  ",
            "items": [{"stem_type_id": bass, "gain_db": -3}, {"group_id": chords}],
        },
    )
    assert res.status_code == 201
    p = res.json()
    assert p["name"] == "練習用"
    assert p["sort_order"] > max(x["sort_order"] for x in before)
    assert [(i["stem_type_code"], i["group_code"], i["gain_db"]) for i in p["items"]] == [
        ("bass", None, -3.0),
        (None, "chords", 0.0),
    ]
    assert p["items"][0]["stem_type_id"] == bass and p["items"][1]["group_id"] == chords
    listed = client.get("/api/listen-presets").json()["listen_presets"]
    assert listed[-1]["listen_preset_id"] == p["listen_preset_id"]
    assert listed[-1]["items"] == p["items"]

    pid = p["listen_preset_id"]
    # 名前だけ変える（items はそのまま）
    res = client.put(f"/api/listen-presets/{pid}", json={"name": "新しい名前"})
    assert res.status_code == 200
    assert res.json()["name"] == "新しい名前" and len(res.json()["items"]) == 2
    # items を置き換える
    res = client.put(f"/api/listen-presets/{pid}", json={"items": [{"stem_type_id": bass}]})
    assert [i["stem_type_code"] for i in res.json()["items"]] == ["bass"]
    # 並び替え（いちばん上へ）
    res = client.put(f"/api/listen-presets/{pid}", json={"sort_order": -5})
    assert res.json()["sort_order"] == -5
    listed = client.get("/api/listen-presets").json()["listen_presets"]
    assert listed[0]["listen_preset_id"] == pid

    assert client.delete(f"/api/listen-presets/{pid}").status_code == 204
    assert client.delete(f"/api/listen-presets/{pid}").status_code == 404
    with _factory(client)() as s:
        count = s.scalar(
            select(func.count())
            .select_from(ListenPresetItem)
            .where(ListenPresetItem.listen_preset_id == pid)
        )
        assert count == 0  # 中身も消える


def test_delete_builtin_preset_hides_it(client: TestClient) -> None:
    """組み込みを削除すると隠す（再起動の seed で復活しない）。ユーザー作成は行ごと消す。"""
    listed = client.get("/api/listen-presets").json()["listen_presets"]
    builtin = next(p for p in listed if p["name"] == "サブボーカルのみ")
    assert builtin["builtin"] is True
    pid = builtin["listen_preset_id"]
    assert client.delete(f"/api/listen-presets/{pid}").status_code == 204
    def listed_presets() -> list[dict[str, Any]]:
        return client.get("/api/listen-presets").json()["listen_presets"]

    assert pid not in [p["listen_preset_id"] for p in listed_presets()]
    assert client.put(f"/api/listen-presets/{pid}", json={"name": "x"}).status_code == 404
    assert client.delete(f"/api/listen-presets/{pid}").status_code == 404
    with _factory(client)() as s:
        row = s.get(ListenPreset, pid)
        assert row is not None and row.hidden and row.seed_code == "backing_only"
        seed(s)  # 次の起動
    after = listed_presets()
    assert pid not in [p["listen_preset_id"] for p in after]
    assert "サブボーカルのみ" not in [p["name"] for p in after]

    user = client.post("/api/listen-presets", json={"name": "自作", "items": []}).json()
    assert user["builtin"] is False
    assert client.delete(f"/api/listen-presets/{user['listen_preset_id']}").status_code == 204
    with _factory(client)() as s:
        assert s.get(ListenPreset, user["listen_preset_id"]) is None


def test_listen_preset_validation(client: TestClient) -> None:
    bass = _type_id(client, "bass")
    chords = _group_id(client, "chords")
    # stem_type_id と group_id の両方・どちらも無し → 422（日本語の理由つき）
    for item in ({"stem_type_id": bass, "group_id": chords}, {"gain_db": 0}):
        res = client.post("/api/listen-presets", json={"name": "x", "items": [item]})
        assert res.status_code == 422, item
        assert "どちらか一方" in res.json()["detail"]
    # 名前が空
    res = client.post("/api/listen-presets", json={"name": "   ", "items": []})
    assert res.status_code == 422 and "name" in res.json()["detail"]
    # 音量が範囲外
    res = client.post(
        "/api/listen-presets", json={"name": "x", "items": [{"stem_type_id": bass, "gain_db": 99}]}
    )
    assert res.status_code == 422
    # 存在しない stem の種類・グループ → 400
    res = client.post("/api/listen-presets", json={"name": "x", "items": [{"stem_type_id": 9999}]})
    assert res.status_code == 400 and "stem の種類が見つかりません" in res.json()["detail"]
    res = client.post("/api/listen-presets", json={"name": "x", "items": [{"group_id": 9999}]})
    assert res.status_code == 400 and "グループが見つかりません" in res.json()["detail"]
    # 存在しないプリセット
    assert client.put("/api/listen-presets/9999", json={"name": "y"}).status_code == 404
    res = client.put("/api/listen-presets/9999", json={"items": [{"stem_type_id": bass}]})
    assert res.status_code == 404
    # 失敗した作成は何も残さない
    names = [p["name"] for p in client.get("/api/listen-presets").json()["listen_presets"]]
    assert "x" not in names


def test_preset_update_with_bad_item_keeps_old_items(client: TestClient) -> None:
    bass = _type_id(client, "bass")
    pid = client.post(
        "/api/listen-presets", json={"name": "a", "items": [{"stem_type_id": bass}]}
    ).json()["listen_preset_id"]
    res = client.put(
        f"/api/listen-presets/{pid}", json={"name": "b", "items": [{"stem_type_id": 9999}]}
    )
    assert res.status_code == 400
    p = next(
        x for x in client.get("/api/listen-presets").json()["listen_presets"]
        if x["listen_preset_id"] == pid
    )
    assert p["name"] == "a" and [i["stem_type_code"] for i in p["items"]] == ["bass"]


# --- キュー ---------------------------------------------------------------------------


def test_cue_crud(client: TestClient, tmp_path: Path) -> None:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _factory(client)() as s:
        track_id = make_track(s, settings, tmp_path, seconds=2.0)
    assert client.get(f"/api/tracks/{track_id}/cues").json() == {"cues": []}

    res = client.post(
        f"/api/tracks/{track_id}/cues",
        json={"position_sec": 1.5, "label": " サビ ", "color": "#ff3b4e"},
    )
    assert res.status_code == 201
    cue = res.json()
    assert cue["label"] == "サビ" and cue["color"] == "#FF3B4E" and cue["loop_end_sec"] is None
    second = client.post(
        f"/api/tracks/{track_id}/cues", json={"position_sec": 0.5, "loop_end_sec": 1.0}
    ).json()
    assert second["label"] is None
    listed = client.get(f"/api/tracks/{track_id}/cues").json()["cues"]
    assert [c["cue_id"] for c in listed] == [second["cue_id"], cue["cue_id"]]  # 位置の順

    cid = cue["cue_id"]
    # 終点を付ける（A-B ループ）→ 外す（null）
    res = client.put(f"/api/cues/{cid}", json={"loop_end_sec": 1.9})
    assert res.status_code == 200 and res.json()["loop_end_sec"] == 1.9
    assert res.json()["label"] == "サビ"  # 送っていない項目はそのまま
    res = client.put(f"/api/cues/{cid}", json={"loop_end_sec": None, "label": "", "color": None})
    body = res.json()
    assert body["loop_end_sec"] is None and body["label"] is None and body["color"] is None
    moved = client.put(f"/api/cues/{cid}", json={"position_sec": 0.25}).json()
    assert moved["position_sec"] == 0.25

    assert client.delete(f"/api/cues/{cid}").status_code == 204
    assert client.delete(f"/api/cues/{cid}").status_code == 404
    assert len(client.get(f"/api/tracks/{track_id}/cues").json()["cues"]) == 1

    # 曲を消すとキューも消える
    assert client.delete(f"/api/tracks/{track_id}").status_code == 200
    with _factory(client)() as s:
        assert s.scalar(select(func.count()).select_from(CuePoint)) == 0


def test_cue_validation(client: TestClient, tmp_path: Path) -> None:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _factory(client)() as s:
        track_id = make_track(s, settings, tmp_path, seconds=2.0)
    url = f"/api/tracks/{track_id}/cues"
    assert client.get("/api/tracks/9999/cues").status_code == 404
    assert client.post("/api/tracks/9999/cues", json={"position_sec": 1}).status_code == 404
    # 終点が始点より前 → 422
    res = client.post(url, json={"position_sec": 1.0, "loop_end_sec": 0.5})
    assert res.status_code == 422 and "終点は始点より後" in res.json()["detail"]
    assert client.post(url, json={"position_sec": -1}).status_code == 422
    assert client.post(url, json={"position_sec": 1, "color": "red"}).status_code == 422
    assert client.post(url, json={}).status_code == 422
    # 曲の長さを超える
    res = client.post(url, json={"position_sec": 5.0})
    assert res.status_code == 400 and "曲の長さ" in res.json()["detail"]

    cid = client.post(url, json={"position_sec": 1.0}).json()["cue_id"]
    # PUT でも終点の前後を確かめる（今の位置 1.0 より前の終点）
    res = client.put(f"/api/cues/{cid}", json={"loop_end_sec": 0.5})
    assert res.status_code == 400
    res = client.put(f"/api/cues/{cid}", json={"loop_end_sec": 1.5})
    assert res.status_code == 200
    res = client.put(f"/api/cues/{cid}", json={"position_sec": 1.8})  # 終点より後へ
    assert res.status_code == 400
    assert client.put("/api/cues/9999", json={"label": "x"}).status_code == 404


# --- 配信用データの作り直し -------------------------------------------------------------


def _drop_streams(client: TestClient, job_id: int) -> None:
    with _factory(client)() as s:
        ids = select(Stem.stem_id).where(Stem.job_id == job_id)
        s.execute(
            delete(StemRendition).where(
                StemRendition.stem_id.in_(ids), StemRendition.purpose == "stream"
            )
        )
        s.execute(delete(Waveform).where(Waveform.stem_id.in_(ids), Waveform.samples_per_px == 256))
        s.commit()


def test_postprocess_when_ready(client: TestClient, done_job: tuple[int, int]) -> None:
    track_id, job_id = done_job
    stems = client.get(f"/api/jobs/{job_id}/stems").json()
    assert stems["delivery_ready"] is True and stems["delivery_missing"] == []
    assert all("stem_type_id" in s for s in stems["stems"])
    res = client.post(f"/api/jobs/{job_id}/postprocess")
    assert res.status_code == 200
    assert res.json()["reason"] == "ready" and res.json()["job"]["postprocess_status"] is None
    tracks = client.get("/api/tracks").json()["tracks"]
    assert tracks[0]["playable_job_id"] == job_id
    assert client.get(f"/api/tracks/{track_id}").json()["playable_job_id"] == job_id


def test_postprocess_rebuilds_missing(client: TestClient, done_job: tuple[int, int]) -> None:
    _, job_id = done_job
    _drop_streams(client, job_id)
    stems = client.get(f"/api/jobs/{job_id}/stems").json()
    assert stems["delivery_ready"] is False and len(stems["delivery_missing"]) == 8

    res = client.post(f"/api/jobs/{job_id}/postprocess")
    assert res.status_code == 202
    assert res.json()["created"] is True and res.json()["job"]["postprocess_status"] == "queued"
    again = client.post(f"/api/jobs/{job_id}/postprocess")
    assert again.status_code == 200 and again.json()["reason"] == "active"
    assert client.get(f"/api/jobs/{job_id}").json()["postprocess_status"] == "queued"

    worker = _worker(client)
    assert worker.run_one() is None  # 分割のジョブではない
    assert worker.run_postprocess_one() == job_id
    assert worker.run_postprocess_one() is None
    stems = client.get(f"/api/jobs/{job_id}/stems").json()
    assert stems["delivery_ready"] is True and stems["postprocess_status"] == "done"
    for s in stems["stems"]:
        assert [r["purpose"] for r in s["renditions"]].count("stream") == 1
        assert [p["samples_per_px"] for p in s["peaks"]] == [256, 1024, 4096, 16384]
    # master は残っている
    assert all(
        [r["purpose"] for r in s["renditions"]].count("master") == 1 for s in stems["stems"]
    )


def test_postprocess_failure_leaves_no_partial_rows(
    client: TestClient, done_job: tuple[int, int]
) -> None:
    _, job_id = done_job
    _drop_streams(client, job_id)
    calls: list[int] = []

    def broken(args: Any) -> None:
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("ffmpeg が失敗しました")
        fake_encoder(args)

    assert client.post(f"/api/jobs/{job_id}/postprocess").status_code == 202
    assert _worker(client, encoder=broken).run_postprocess_one() == job_id
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["postprocess_status"] == "failed" and job["status"] == "done"
    with _factory(client)() as s:
        ids = select(Stem.stem_id).where(Stem.job_id == job_id)
        streams = s.scalar(
            select(func.count())
            .select_from(StemRendition)
            .where(StemRendition.stem_id.in_(ids), StemRendition.purpose == "stream")
        )
        assert streams == 0
        assert len(missing_delivery(s, job_id)) == 8
    settings = client.app.state.settings  # type: ignore[attr-defined]
    assert not (settings.stems_dir / str(job_id) / "stream").exists()
    # もう一度頼める
    assert client.post(f"/api/jobs/{job_id}/postprocess").status_code == 202


def test_postprocess_errors(client: TestClient, tmp_path: Path) -> None:
    assert client.post("/api/jobs/9999/postprocess").status_code == 404
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _factory(client)() as s:
        track_id = make_track(s, settings, tmp_path)
    job_id = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"}).json()["job"][
        "job_id"
    ]
    res = client.post(f"/api/jobs/{job_id}/postprocess")  # queued のジョブ
    assert res.status_code == 409 and "分割が終わっていない" in res.json()["detail"]
    stems = client.get(f"/api/jobs/{job_id}/stems").json()
    assert stems["delivery_ready"] is False


def test_interrupted_postprocess_is_failed_on_worker_start(
    client: TestClient, done_job: tuple[int, int]
) -> None:
    _, job_id = done_job
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _factory(client)() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None
        job.postprocess_status = "running"
        s.commit()
        recover_interrupted_jobs(s, settings)
        s.expire_all()
        job = s.get(SeparationJob, job_id)
        assert job is not None and job.postprocess_status == "failed" and job.status == "done"
