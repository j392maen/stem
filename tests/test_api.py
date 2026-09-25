from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from job_helpers import make_track, no_tags, sync_launcher
from stemapp.api import auth
from stemapp.api.imports import ImportDeps, safe_filename
from stemapp.app import create_app
from stemapp.config import Settings
from stemapp.jobs import enqueue_full_job
from stemapp.jobs.worker import Worker
from stemapp.models import InputSource, SeparationJob, StemRendition, Track
from stemapp.peaks import MAGIC, decode_peaks
from test_url import FakeYtDlp

PASS = "ひらけごま-123"


def _app(settings: Settings, passcode: str | None = None, ytdlp: FakeYtDlp | None = None):
    s = settings.model_copy(update={"passcode": passcode})
    runner = ytdlp or FakeYtDlp(synth_mix(1.5))
    deps = ImportDeps(
        ffmpeg_runner=fake_ffmpeg, tag_reader=no_tags, ytdlp_runner=lambda _s: runner, threads=1
    )
    app = create_app(s, import_deps=deps)
    app.state.sse_poll_sec = 0.02
    return app


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(_app(settings)) as c:
        yield c


def _factory(client: TestClient) -> sessionmaker[Session]:
    return client.app.state.session_factory  # type: ignore[attr-defined]


def _new_track(client: TestClient, tmp_path: Path, name: str = "song", offset: float = 0.0) -> int:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with _factory(client)() as s:
        return make_track(s, settings, tmp_path, name=name, seed_offset=offset)


def _run_worker_once(client: TestClient) -> int | None:
    settings = client.app.state.settings  # type: ignore[attr-defined]
    return Worker(settings, _factory(client), sync_launcher(settings)).run_one()


@pytest.fixture
def done_job(client: TestClient, tmp_path: Path) -> tuple[int, int]:
    """分割済みの曲（track_id, job_id）。"""
    track_id = _new_track(client, tmp_path)
    res = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"})
    job_id = res.json()["job"]["job_id"]
    assert _run_worker_once(client) == job_id
    return track_id, job_id


# --- 基本 ---------------------------------------------------------------------------


def test_health_and_japanese_errors(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    res = client.get("/api/no-such-thing")
    assert res.status_code == 404 and res.json() == {"detail": "見つかりません。"}
    res = client.get("/api/tracks/abc")
    assert res.status_code == 422 and "正しくありません" in res.json()["detail"]


# --- 認証 ---------------------------------------------------------------------------


def test_no_passcode_means_no_auth(client: TestClient) -> None:
    assert client.get("/api/tracks").status_code == 200
    assert client.get("/api/me").json() == {
        "authenticated": True, "passcode_required": False,
        "local_client": False, "can_open_folder": False,
    }
    assert client.post("/api/login", json={"passcode": ""}).json()["authenticated"] is True


def test_passcode_login_flow(settings: Settings) -> None:
    with TestClient(_app(settings, passcode=PASS)) as c:
        for path in ("/api/tracks", "/api/me", "/api/stem-types", "/api/jobs/1"):
            res = c.get(path)
            assert res.status_code == 401, path
            assert res.json() == {"detail": "ログインしてください。"}
        assert c.get("/api/health").status_code == 200
        assert c.post("/api/logout").status_code == 401

        bad = c.post("/api/login", json={"passcode": "ちがう"})
        assert bad.status_code == 401 and bad.json()["detail"] == "パスコードが違います。"
        assert auth.COOKIE_NAME not in c.cookies

        ok = c.post("/api/login", json={"passcode": PASS})
        assert ok.status_code == 200
        cookie = ok.headers["set-cookie"]
        assert f"{auth.COOKIE_NAME}=" in cookie
        assert "HttpOnly" in cookie and "SameSite=lax" in cookie
        assert "Max-Age=2592000" in cookie
        assert c.get("/api/tracks").status_code == 200
        assert c.get("/api/me").json() == {
            "authenticated": True, "passcode_required": True,
            "local_client": False, "can_open_folder": False,
        }

        # 改ざんした Cookie は通らない
        token = c.cookies[auth.COOKIE_NAME]
        c.cookies.set(auth.COOKIE_NAME, token[:-1] + ("0" if token[-1] != "0" else "1"))
        assert c.get("/api/tracks").status_code == 401
        c.cookies.set(auth.COOKIE_NAME, token)
        assert c.get("/api/tracks").status_code == 200

        assert c.post("/api/logout").status_code == 200
        c.cookies.clear()
        assert c.get("/api/tracks").status_code == 401
    # 秘密鍵は data/secret.key に保存され、次回も同じものを使う
    key_file = settings.data_dir / "secret.key"
    assert key_file.is_file()
    assert auth.load_or_create_secret(key_file) == auth.load_or_create_secret(key_file)


def test_login_rate_limit(settings: Settings) -> None:
    with TestClient(_app(settings, passcode=PASS)) as c:
        for _ in range(5):
            assert c.post("/api/login", json={"passcode": "x"}).status_code == 401
        res = c.post("/api/login", json={"passcode": PASS})  # 正しくても受け付けない
        assert res.status_code == 429 and "1分" in res.json()["detail"]


def test_limiter_window() -> None:
    now = [0.0]
    lim = auth.LoginLimiter(clock=lambda: now[0])
    for _ in range(5):
        lim.record_failure("a")
    assert lim.is_blocked("a") and not lim.is_blocked("b")
    now[0] = 61.0
    assert not lim.is_blocked("a")


def test_token_expiry_and_passcode_change() -> None:
    key = b"k" * 32
    tok = auth.make_token(key, "p", now=1000.0)
    assert auth.verify_token(key, "p", tok, now=1001.0)
    assert not auth.verify_token(key, "p", tok, now=1000.0 + auth.SESSION_MAX_AGE_SEC + 1)
    assert not auth.verify_token(key, "q", tok, now=1001.0)
    assert not auth.verify_token(b"x" * 32, "p", tok, now=1001.0)
    assert not auth.verify_token(key, "p", "garbage", now=1001.0)


# --- 曲 ---------------------------------------------------------------------------


def test_tracks_list_and_detail(client: TestClient, done_job: tuple[int, int]) -> None:
    track_id, job_id = done_job
    tracks = client.get("/api/tracks").json()["tracks"]
    assert [t["track_id"] for t in tracks] == [track_id]
    t = tracks[0]
    assert t["title"] == "song" and t["duration_sec"] == pytest.approx(1.0)
    assert t["latest_job"]["job_id"] == job_id and t["latest_job"]["status"] == "done"
    assert t["latest_job"]["preset"] == "fast"

    detail = client.get(f"/api/tracks/{track_id}").json()
    assert [j["job_id"] for j in detail["jobs"]] == [job_id]
    assert detail["sources"][0]["source_type"] == "file"
    assert client.get("/api/tracks/999").status_code == 404
    assert client.get("/api/tracks/999").json()["detail"] == "曲が見つかりません。"


def test_delete_track(client: TestClient, done_job: tuple[int, int]) -> None:
    track_id, job_id = done_job
    settings = client.app.state.settings  # type: ignore[attr-defined]
    assert (settings.tracks_dir / str(track_id)).is_dir()
    assert (settings.stems_dir / str(job_id)).is_dir()

    # 分割待ちのジョブがあると 409
    queued = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast", "force": True})
    assert queued.status_code == 201
    res = client.delete(f"/api/tracks/{track_id}")
    assert res.status_code == 409 and "キャンセル" in res.json()["detail"]
    client.post(f"/api/jobs/{queued.json()['job']['job_id']}/cancel")

    res = client.delete(f"/api/tracks/{track_id}")
    assert res.status_code == 200 and res.json()["deleted"] is True
    assert not (settings.tracks_dir / str(track_id)).exists()
    assert not (settings.stems_dir / str(job_id)).exists()
    with _factory(client)() as s:
        assert s.scalars(select(Track)).all() == []
        assert s.scalars(select(SeparationJob)).all() == []
        assert s.scalars(select(InputSource)).all() == []
    assert client.delete(f"/api/tracks/{track_id}").status_code == 404


# --- ジョブ -------------------------------------------------------------------------


def test_create_get_cancel_job(client: TestClient, tmp_path: Path) -> None:
    track_id = _new_track(client, tmp_path)
    res = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"})
    assert res.status_code == 201
    body = res.json()
    assert body["created"] is True and body["job"]["status"] == "queued"
    job_id = body["job"]["job_id"]

    again = client.post(f"/api/tracks/{track_id}/jobs", json={})
    assert again.status_code == 200
    assert again.json()["created"] is False and again.json()["reason"] == "active"

    assert client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "zzz"}).status_code == 400
    assert client.post("/api/tracks/999/jobs", json={}).status_code == 404

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["preset"] == "fast" and job["progress"] == 0.0
    assert client.get("/api/jobs/999").status_code == 404

    res = client.post(f"/api/jobs/{job_id}/cancel")
    assert res.status_code == 200 and res.json()["status"] == "canceled"
    res = client.post(f"/api/jobs/{job_id}/cancel")
    assert res.status_code == 409 and "終わっています" in res.json()["detail"]
    assert client.post("/api/jobs/999/cancel").status_code == 404


def test_cancel_running_job_sets_flag(client: TestClient, tmp_path: Path) -> None:
    track_id = _new_track(client, tmp_path)
    job_id = client.post(f"/api/tracks/{track_id}/jobs", json={}).json()["job"]["job_id"]
    with _factory(client)() as s:
        s.get(SeparationJob, job_id).status = "running"  # type: ignore[union-attr]
        s.commit()
    res = client.post(f"/api/jobs/{job_id}/cancel").json()
    assert res["status"] == "running" and res["cancel_requested"] is True


def _read_events(res: Any) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    name = ""
    for line in res.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return events


def test_events_stream_until_done(client: TestClient, tmp_path: Path) -> None:
    track_id = _new_track(client, tmp_path)
    job_id = client.post(f"/api/tracks/{track_id}/jobs", json={}).json()["job"]["job_id"]
    factory = _factory(client)

    def progress_then_done() -> None:
        steps = [
            ("running", 0.1, "分離中"),
            ("running", 0.5, "分離中（2/2）"),
            ("done", 1.0, "完了"),
        ]
        for status, p, stage in steps:
            time.sleep(0.2)
            with factory() as s:
                job = s.get(SeparationJob, job_id)
                assert job is not None
                job.status, job.progress, job.stage = status, p, stage
                s.commit()

    t = threading.Thread(target=progress_then_done)
    t.start()
    with client.stream("GET", f"/api/jobs/{job_id}/events") as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        events = _read_events(res)
    t.join()
    assert all(name == "job" for name, _ in events)
    seen = [(e["status"], e["progress"]) for _, e in events]
    assert seen[0] == ("queued", 0.0)
    assert ("running", 0.1) in seen and ("running", 0.5) in seen
    assert seen[-1] == ("done", 1.0)
    assert len(seen) == len(set(seen))  # 変わったときだけ送る
    assert client.get("/api/jobs/999/events").status_code == 404


def test_events_for_finished_job_closes_immediately(
    client: TestClient, done_job: tuple[int, int]
) -> None:
    _, job_id = done_job
    with client.stream("GET", f"/api/jobs/{job_id}/events") as res:
        events = _read_events(res)
    assert len(events) == 1 and events[0][1]["status"] == "done"


# --- stem とファイル ---------------------------------------------------------------------


def test_stems_and_files(client: TestClient, done_job: tuple[int, int]) -> None:
    _, job_id = done_job
    body = client.get(f"/api/jobs/{job_id}/stems").json()
    assert body["job_id"] == job_id and body["status"] == "done"
    stems = {s["code"]: s for s in body["stems"]}
    assert set(stems) == {
        "vocals", "lead_vocal", "backing_vocal", "drums", "bass", "guitar", "piano", "other",
    }
    lead = stems["lead_vocal"]
    assert lead["parent_code"] == "vocals" and lead["color"].startswith("#")
    assert stems["backing_vocal"]["is_residual"] is True
    assert lead["display_name"] == "メインボーカル"
    assert {r["purpose"] for r in lead["renditions"]} == {"master", "stream"}
    assert [p["samples_per_px"] for p in lead["peaks"]] == [256, 1024, 4096, 16384]

    stream = next(r for r in lead["renditions"] if r["purpose"] == "stream")
    master = next(r for r in lead["renditions"] if r["purpose"] == "master")
    full = client.get(stream["url"])
    assert full.status_code == 200 and full.headers["content-type"] == "audio/webm"
    assert len(full.content) == stream["bytes"]
    assert full.headers.get("accept-ranges") == "bytes"

    part = client.get(master["url"], headers={"Range": "bytes=0-9"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 0-9/{master['bytes']}"
    assert part.content == b"fLaC" + part.content[4:] and len(part.content) == 10
    assert part.headers["content-type"] == "audio/flac"
    tail = client.get(master["url"], headers={"Range": "bytes=-4"})
    assert tail.status_code == 206 and len(tail.content) == 4

    pk = client.get(lead["peaks"][0]["url"])
    assert pk.status_code == 200 and pk.content[:4] == MAGIC
    assert decode_peaks(pk.content).samples_per_px == 256

    assert client.get(f"/api/files/peaks/{lead['stem_id']}/999").status_code == 404
    assert client.get("/api/files/renditions/99999").status_code == 404
    assert client.get("/api/jobs/999/stems").status_code == 404


def test_files_outside_data_dir_are_refused(
    client: TestClient, done_job: tuple[int, int], tmp_path: Path
) -> None:
    _, job_id = done_job
    secret = tmp_path / "outside.txt"
    secret.write_text("secret", encoding="utf-8")
    with _factory(client)() as s:
        rend = s.scalars(select(StemRendition)).first()
        assert rend is not None
        rid = rend.rendition_id
        for bad in ("../outside.txt", str(secret), "stems/../../outside.txt"):
            rend.file_path = bad
            s.commit()
            res = client.get(f"/api/files/renditions/{rid}")
            assert res.status_code == 404, bad
            assert "secret" not in res.text


# --- マスタ ------------------------------------------------------------------------------


def test_master_endpoints(client: TestClient) -> None:
    types = client.get("/api/stem-types").json()["stem_types"]
    by_code = {t["code"]: t for t in types}
    assert by_code["kick"]["parent_code"] == "drums" and by_code["kick"]["tier"] == "detail"
    groups = {g["code"]: g for g in client.get("/api/stem-groups").json()["stem_groups"]}
    assert groups["rhythm"]["members"] == ["bass", "drums"]
    presets = client.get("/api/listen-presets").json()["listen_presets"]
    assert presets and all("items" in p for p in presets)
    items = presets[0]["items"]
    assert all((i["stem_type_code"] is None) != (i["group_code"] is None) for i in items)
    sep = client.get("/api/presets").json()["presets"]
    assert {p["code"] for p in sep if not p["experimental"]} == {"fast", "standard", "best"}
    assert [p["code"] for p in sep if p["is_default"]] == ["standard"]
    exp = {p["code"] for p in sep if p["experimental"]}
    assert {"exp_resid_vocals", "exp_kara_mix", "exp_kara_anvuew", "exp_combo"} <= exp


# --- 取り込み ----------------------------------------------------------------------------


def _wait_import(client: TestClient, source_id: int) -> dict[str, Any]:
    client.app.state.import_manager.join()  # type: ignore[attr-defined]
    return client.get(f"/api/imports/{source_id}").json()


def test_import_file_with_separate(client: TestClient, tmp_path: Path) -> None:
    src = write_source(tmp_path / "曲.wav", synth_mix(1.0))
    with open(src, "rb") as fh:
        res = client.post(
            "/api/imports",
            files={"file": ("../dir/曲.wav", fh, "audio/wav")},
            data={"separate": "true", "preset": "fast"},
        )
    assert res.status_code == 202
    source_id = res.json()["source_id"]
    assert res.json()["status"] == "queued"

    info = _wait_import(client, source_id)
    assert info["status"] == "done" and info["error_code"] is None
    assert info["original_name"] == "曲.wav"
    assert info["track_id"] is not None
    assert info["job_id"] is not None and info["job_created"] is True
    job = client.get(f"/api/jobs/{info['job_id']}").json()
    assert job["status"] == "queued" and job["preset"] == "fast"
    assert job["track_id"] == info["track_id"]
    # 受け取ったファイルは消える
    settings = client.app.state.settings  # type: ignore[attr-defined]
    assert not list((settings.cache_dir / "uploads").glob("*"))

    # 分割済み（ここでは分割待ち）の曲をもう一度: ジョブは増えない
    assert _run_worker_once(client) == info["job_id"]
    with open(src, "rb") as fh:
        again = client.post(
            "/api/imports", files={"file": ("曲.wav", fh)}, data={"separate": "1"}
        ).json()
    info2 = _wait_import(client, again["source_id"])
    assert info2["track_id"] == info["track_id"]
    assert info2["job_created"] is False and info2["job_id"] == info["job_id"]
    # force なら登録される
    with open(src, "rb") as fh:
        forced = client.post(
            "/api/imports", files={"file": ("曲.wav", fh)},
            data={"separate": "1", "force": "1", "preset": "fast"},
        ).json()
    info3 = _wait_import(client, forced["source_id"])
    assert info3["job_created"] is True and info3["job_id"] != info["job_id"]


def test_import_file_without_separate(client: TestClient, tmp_path: Path) -> None:
    src = write_source(tmp_path / "a.wav", synth_mix(1.0))
    with open(src, "rb") as fh:
        res = client.post("/api/imports", files={"file": ("a.wav", fh)})
    info = _wait_import(client, res.json()["source_id"])
    assert info["status"] == "done" and info["job_id"] is None
    with _factory(client)() as s:
        assert s.scalars(select(SeparationJob)).all() == []


def test_import_broken_file(client: TestClient) -> None:
    res = client.post("/api/imports", files={"file": ("x.mp3", b"not audio")})
    info = _wait_import(client, res.json()["source_id"])
    assert info["status"] == "failed" and info["error_code"] == "invalid_audio"
    assert "読み込めません" in info["message"]
    assert info["track_id"] is None


def test_import_url(settings: Settings) -> None:
    ytdlp = FakeYtDlp(synth_mix(1.2))
    with TestClient(_app(settings, ytdlp=ytdlp)) as c:
        res = c.post("/api/imports", json={"url": "https://example.com/v?id=1", "separate": True})
        assert res.status_code == 202
        info = _wait_import(c, res.json()["source_id"])
        assert info["status"] == "done" and info["source_type"] == "url"
        assert info["url"] == "https://example.com/v?id=1"
        assert info["job_created"] is True
        track = c.get(f"/api/tracks/{info['track_id']}").json()
        assert track["title"] == "テスト動画" and track["artist"] == "投稿者"
    assert len(ytdlp.calls) == 1


def test_import_url_failure(settings: Settings) -> None:
    ytdlp = FakeYtDlp(returncode=1, stderr="ERROR: [youtube] abc: Private video. Sign in")
    with TestClient(_app(settings, ytdlp=ytdlp)) as c:
        res = c.post("/api/imports", json={"url": "https://example.com/x"})
        info = _wait_import(c, res.json()["source_id"])
        assert info["status"] == "failed" and info["error_code"] == "private_or_removed"
        assert "非公開" in info["message"] and "Private video" in info["error_detail"]
        with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
            # 行は1つだけ（先に作った行を failed にする）
            assert len(s.scalars(select(InputSource)).all()) == 1


def test_import_bad_requests(client: TestClient) -> None:
    assert client.post("/api/imports", json={"url": "ftp://x"}).status_code == 400
    assert client.post("/api/imports", json=["x"]).status_code == 400
    res = client.post("/api/imports", json={"url": "https://e.com", "separate": True,
                                            "preset": "zzz"})
    assert res.status_code == 400 and "zzz" in res.json()["detail"]
    assert client.post("/api/imports", content=b"x",
                       headers={"content-type": "text/plain"}).status_code == 415
    assert client.post("/api/imports", data={"separate": "1"},
                       files={"other": ("a", b"x")}).status_code == 400
    assert client.get("/api/imports/999").status_code == 404
    with _factory(client)() as s:
        assert s.scalars(select(InputSource)).all() == []


def test_interrupted_imports_are_failed_on_startup(settings: Settings) -> None:
    with TestClient(_app(settings)) as c, c.app.state.session_factory() as s:  # type: ignore[attr-defined]
        s.add(InputSource(source_type="url", url="https://e.com", fetch_status="fetching"))
        s.commit()
    with TestClient(_app(settings)) as c:
        info = c.get("/api/imports/1").json()
        assert info["status"] == "failed" and "中断" in info["error_detail"]


def test_enqueue_via_function_matches_api(client: TestClient, tmp_path: Path) -> None:
    """API から登録したジョブと enqueue_full_job の結果が同じ規則に従う。"""
    track_id = _new_track(client, tmp_path)
    with _factory(client)() as s:
        assert enqueue_full_job(s, track_id, "fast").created is True
    res = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"})
    assert res.json()["created"] is False


def test_safe_filename() -> None:
    assert safe_filename("../../evil.wav") == "evil.wav"
    assert safe_filename("C:\\x\\a:b?.mp3") == "a_b_.mp3"
    assert safe_filename("") == "upload"
    assert safe_filename("..") == "upload"


def test_non_ascii_cookie_is_401(settings: Settings) -> None:
    assert not auth.verify_token(b"k" * 32, "p", "1.a.\xe9")
    with TestClient(_app(settings, passcode=PASS)) as c:
        res = c.get("/api/tracks", headers={"cookie": f"{auth.COOKIE_NAME}=1.a.%C3%A9"})
        assert res.status_code == 401


def test_docs_disabled_with_passcode(settings: Settings) -> None:
    with TestClient(_app(settings, passcode=PASS)) as c:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert c.get(path).status_code == 404, path
    with TestClient(_app(settings)) as c:
        assert c.get("/openapi.json").status_code == 200
        assert c.get("/docs").status_code == 200


def test_upload_too_large(settings: Settings) -> None:
    small = settings.model_copy(update={"max_upload_mb": 1})
    with TestClient(_app(small)) as c:
        res = c.post("/api/imports", files={"file": ("big.wav", b"\0" * (1024 * 1024 + 10))})
        assert res.status_code == 413 and "上限 1 MB" in res.json()["detail"]
        assert not list((small.cache_dir / "uploads").glob("*"))
        with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
            assert s.scalars(select(InputSource)).all() == []


# --- 分け方の聴き比べ（T12） -------------------------------------------------------------


def test_multiple_jobs_per_track_and_delete_job(
    client: TestClient, done_job: tuple[int, int]
) -> None:
    track_id, fast_job = done_job
    settings = client.app.state.settings  # type: ignore[attr-defined]
    # 別の分け方なら force なしで登録できる
    res = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "exp_resid_vocals"})
    assert res.status_code == 201 and res.json()["created"] is True
    exp_job = res.json()["job"]["job_id"]
    assert res.json()["job"]["preset_experimental"] is True
    assert res.json()["job"]["preset_name"] == "残差をボーカルへ"
    # 分割待ちは消せない
    busy = client.delete(f"/api/jobs/{exp_job}")
    assert busy.status_code == 409 and "キャンセル" in busy.json()["detail"]
    assert _run_worker_once(client) == exp_job
    # 同じ分け方は分割済み
    again = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "exp_resid_vocals"})
    assert again.status_code == 200 and again.json()["reason"] == "done"
    assert "この分け方" in again.json()["message"]

    track = client.get(f"/api/tracks/{track_id}").json()
    assert track["playable_job_id"] == fast_job  # 実験のジョブは既定にしない
    jobs = {j["job_id"]: j for j in track["jobs"]}
    assert set(jobs) == {fast_job, exp_job}
    for j in jobs.values():
        assert j["status"] == "done"
        assert j["residual_rms_db"] is not None and j["mixture_rms_db"] is not None
        levels = j["stem_rms_db"]
        assert list(levels)[:3] == ["vocals", "lead_vocal", "backing_vocal"]
        assert all(isinstance(v, float) for v in levels.values())
    assert jobs[fast_job]["preset"] == "fast" and jobs[fast_job]["preset_experimental"] is False

    # ジョブ単位で消す（曲とほかのジョブは残る）
    res = client.delete(f"/api/jobs/{exp_job}")
    assert res.status_code == 200
    assert res.json() == {"deleted": True, "job_id": exp_job, "track_id": track_id}
    assert not (settings.stems_dir / str(exp_job)).exists()
    assert (settings.stems_dir / str(fast_job)).is_dir()
    track = client.get(f"/api/tracks/{track_id}").json()
    assert track["playable_job_id"] == fast_job
    assert [j["job_id"] for j in track["jobs"]] == [fast_job]
    # 実験のジョブしか無ければそれを再生する
    assert client.delete(f"/api/jobs/{fast_job}").status_code == 200
    res = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "exp_kara_mix"})
    only_exp = res.json()["job"]["job_id"]
    assert _run_worker_once(client) == only_exp
    assert client.get(f"/api/tracks/{track_id}").json()["playable_job_id"] == only_exp
    assert client.delete(f"/api/jobs/{exp_job}").status_code == 404
