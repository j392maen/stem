"""端末（DEVICE）と続きから再生（PLAYBACK_STATE）の API（T06b）。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from job_helpers import make_track
from stemapp.config import Settings
from stemapp.db import init_db, make_engine, migrate_db
from stemapp.models import PlaybackState, SeparationJob
from test_api import _app

KEY_PC = "pc-0123456789abcdef"
KEY_PHONE = "ph-0123456789abcdef"


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(_app(settings)) as c:
        yield c


def _factory(client: TestClient) -> sessionmaker[Session]:
    return client.app.state.session_factory  # type: ignore[attr-defined]


@pytest.fixture
def track(client: TestClient, tmp_path: Path) -> tuple[int, int]:
    """曲と、その曲の（完了扱いの）ジョブ。"""
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _factory(client)() as s:
        track_id = make_track(s, settings, tmp_path, seconds=2.0)
        job = SeparationJob(track_id=track_id, job_kind="full", status="done", run_on="gpu")
        s.add(job)
        s.commit()
        return track_id, job.job_id


def _register(client: TestClient, key: str, kind: str, name: str | None = None) -> dict:
    body: dict = {"device_key": key, "kind": kind}
    if name is not None:
        body["name"] = name
    res = client.post("/api/devices", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_register_device_names_and_rename(client: TestClient) -> None:
    pc = _register(client, KEY_PC, "pc")
    assert pc["name"] == "PC" and pc["kind"] == "pc" and pc["last_seen_at"]
    phone = _register(client, KEY_PHONE, "iphone")
    assert phone["name"] == "iPhone" and phone["device_id"] != pc["device_id"]
    # 同じ ID で登録し直すと同じ行（名前は省略すれば今のまま）
    again = _register(client, KEY_PC, "pc")
    assert again["device_id"] == pc["device_id"] and again["name"] == "PC"
    # 名前を変える
    res = client.put(f"/api/devices/{pc['device_id']}", json={"name": "  居間の PC  "})
    assert res.status_code == 200 and res.json()["name"] == "居間の PC"
    assert _register(client, KEY_PC, "pc")["name"] == "居間の PC"
    names = [d["name"] for d in client.get("/api/devices").json()["devices"]]
    assert names == ["居間の PC", "iPhone"]
    # 不正な ID・空の名前・無い端末
    assert client.post("/api/devices", json={"device_key": "短い", "kind": "pc"}).status_code == 422
    assert client.post("/api/devices", json={"device_key": KEY_PC, "kind": "tv"}).status_code == 422
    assert client.put(f"/api/devices/{pc['device_id']}", json={"name": " "}).status_code == 422
    assert client.put("/api/devices/999", json={"name": "x"}).status_code == 404


def test_save_and_read_playback_per_device(client: TestClient, track: tuple[int, int]) -> None:
    track_id, job_id = track
    pc = _register(client, KEY_PC, "pc")
    phone = _register(client, KEY_PHONE, "iphone")
    presets = client.get("/api/listen-presets").json()["listen_presets"]
    preset = presets[0]

    assert client.get(f"/api/tracks/{track_id}/playback").json() == {"states": []}
    res = client.put(
        f"/api/tracks/{track_id}/playback/{pc['device_id']}",
        json={
            "position_sec": 1.25, "job_id": job_id, "selected": ["drums", "bass"],
            "gains_db": {"bass": -3, "drums": 0}, "listen_preset_id": preset["listen_preset_id"],
            "tempo_ratio": 1.05, "tempo_mode": "instant",
        },
    )
    assert res.status_code == 200, res.text
    st = res.json()
    assert st["position_sec"] == 1.25 and st["job_id"] == job_id
    assert st["selected"] == ["drums", "bass"]
    assert st["gains_db"] == {"bass": -3.0}  # 0 dB は入れない
    assert st["listen_preset_id"] == preset["listen_preset_id"]
    assert st["listen_preset_name"] == preset["name"]
    assert (st["tempo_ratio"], st["tempo_mode"]) == (1.05, "instant")
    assert st["device_name"] == "PC" and st["device_kind"] == "pc"

    # iPhone は別の行（PC の状態は変わらない）。曲の長さを超える位置は長さに収める
    res = client.put(
        f"/api/tracks/{track_id}/playback/{phone['device_id']}",
        json={"position_sec": 999, "selected": ["vocals"]},
    )
    assert res.status_code == 200
    assert res.json()["position_sec"] == pytest.approx(2.0, abs=0.05)
    states = client.get(f"/api/tracks/{track_id}/playback").json()["states"]
    assert [s["device_name"] for s in states] == ["iPhone", "PC"]  # 新しい順
    assert states[1]["position_sec"] == 1.25 and states[1]["selected"] == ["drums", "bass"]
    assert states[0]["job_id"] is None and states[0]["tempo_mode"] is None

    # 上書き（同じ端末・同じ曲は1行）
    client.put(
        f"/api/tracks/{track_id}/playback/{pc['device_id']}",
        json={"position_sec": 0.5, "job_id": job_id},
    )
    with _factory(client)() as s:
        rows = s.scalars(select(PlaybackState).where(PlaybackState.track_id == track_id)).all()
        assert len(rows) == 2
    states = client.get(f"/api/tracks/{track_id}/playback").json()["states"]
    assert states[0]["device_name"] == "PC" and states[0]["position_sec"] == 0.5
    assert states[0]["selected"] is None and states[0]["listen_preset_id"] is None


def test_playback_validation(client: TestClient, track: tuple[int, int], tmp_path: Path) -> None:
    track_id, job_id = track
    pc = _register(client, KEY_PC, "pc")
    url = f"/api/tracks/{track_id}/playback/{pc['device_id']}"
    assert client.put(url, json={"position_sec": -1}).status_code == 422
    assert client.put(url, json={"position_sec": 1, "tempo_ratio": 3}).status_code == 422
    assert client.put(url, json={"position_sec": 1, "tempo_mode": "fast"}).status_code == 422
    assert client.put(url, json={"position_sec": 1, "selected": ["a b"]}).status_code == 422
    assert client.put(url, json={"position_sec": 1, "job_id": 9999}).status_code == 400
    # 別の曲のジョブは使えない
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _factory(client)() as s:
        other = make_track(s, settings, tmp_path, name="other", seed_offset=0.3)
    res = client.put(f"/api/tracks/{other}/playback/{pc['device_id']}",
                     json={"position_sec": 0, "job_id": job_id})
    assert res.status_code == 400
    # 無い組み合わせは覚えない（エラーにしない）
    res = client.put(url, json={"position_sec": 1, "listen_preset_id": 99999})
    assert res.status_code == 200 and res.json()["listen_preset_id"] is None
    zero = {"position_sec": 0}
    assert client.put(f"/api/tracks/{track_id}/playback/999", json=zero).status_code == 404
    assert client.put(f"/api/tracks/999/playback/{pc['device_id']}", json=zero).status_code == 404
    assert client.get("/api/tracks/999/playback").status_code == 404


def test_playback_follows_deletes(client: TestClient, track: tuple[int, int]) -> None:
    track_id, job_id = track
    pc = _register(client, KEY_PC, "pc")
    url = f"/api/tracks/{track_id}/playback/{pc['device_id']}"
    assert client.put(url, json={"position_sec": 1, "job_id": job_id}).status_code == 200
    # ジョブを消すと job_id は null（行は残る）
    with _factory(client)() as s:
        s.delete(s.get(SeparationJob, job_id))
        s.commit()
    states = client.get(f"/api/tracks/{track_id}/playback").json()["states"]
    assert states[0]["job_id"] is None and states[0]["position_sec"] == 1
    # 曲を消すと行も消える
    assert client.delete(f"/api/tracks/{track_id}").status_code in (200, 204)
    with _factory(client)() as s:
        assert s.scalars(select(PlaybackState)).all() == []


def test_migration_adds_device_and_playback_columns(tmp_path: Path) -> None:
    """T06b より前の DB（device_key・playback_state の新しい列が無い）に列と索引を足す。"""
    engine = make_engine(tmp_path / "old.db")
    try:
        init_db(engine)
        with engine.begin() as conn:
            # T01 の device と playback_state に作り直す（playback_state.job_id は外部キーがあり
            # DROP COLUMN できないため、表ごと作る）
            conn.exec_driver_sql("DROP TABLE playback_state")
            conn.exec_driver_sql("DROP TABLE offline_cache")
            conn.exec_driver_sql("DROP TABLE device")
            conn.exec_driver_sql(
                "CREATE TABLE device (device_id INTEGER NOT NULL PRIMARY KEY, "
                "name VARCHAR(200) NOT NULL, kind VARCHAR(10) NOT NULL, "
                "push_subscription_json JSON, last_seen_at DATETIME)"
            )
            conn.exec_driver_sql(
                "CREATE TABLE playback_state (device_id INTEGER NOT NULL, "
                "track_id INTEGER NOT NULL, listen_preset_id INTEGER, channel_gains_json JSON, "
                "position_sec FLOAT NOT NULL, updated_at DATETIME NOT NULL, "
                "PRIMARY KEY (device_id, track_id))"
            )
            conn.exec_driver_sql(
                "INSERT INTO device (device_id, name, kind) VALUES (1, '古い端末', 'pc')"
            )
        # 起動し直したときと同じく新しい接続で移行する（表を作り直す前から接続を持っていると、
        # SQLite がその接続の古い表の定義で ALTER TABLE を解釈して「列が重複」と言うことがある）
        engine.dispose()
        added = migrate_db(engine)
        assert set(added) == {
            "device.device_key", "playback_state.job_id", "playback_state.selected_json",
            "playback_state.tempo_ratio", "playback_state.tempo_mode",
        }
        indexes = {i["name"]: i for i in inspect(engine).get_indexes("device")}
        assert indexes["ux_device_key"]["unique"]
        with engine.connect() as conn:
            row = conn.execute(text("SELECT name, device_key FROM device")).one()
            assert tuple(row) == ("古い端末", None)
        assert migrate_db(engine) == []
    finally:
        engine.dispose()
