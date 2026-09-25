"""保存フォルダを開く API（POST /api/jobs/{id}/open-folder）と /api/me の local 判定。

エクスプローラーは実際には開かない（app.state.folder_opener を差し替える）。
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from job_helpers import make_track, sync_launcher
from stemapp import proc
from stemapp.api import folders
from stemapp.config import Settings
from stemapp.jobs.worker import Worker
from stemapp.library import resolve_data_path
from stemapp.models import Stem, StemRendition
from test_api import _app

LOCAL = ("127.0.0.1", 50123)


class Opener:
    def __init__(self) -> None:
        self.opened: list[Path] = []

    def __call__(self, folder: Path) -> None:
        self.opened.append(folder)


@pytest.fixture
def opener() -> Opener:
    return Opener()


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux でも Windows として扱う（起動関数は差し替えてあるので何も起動しない）。"""
    monkeypatch.setattr(folders, "supports_open_folder", lambda: True)


def _client(settings: Settings, opener: Opener, client: tuple[str, int]) -> TestClient:
    app = _app(settings)
    app.state.folder_opener = opener
    return TestClient(app, client=client)


@pytest.fixture
def local(settings: Settings, opener: Opener) -> Iterator[TestClient]:
    with _client(settings, opener, LOCAL) as c:
        yield c


@pytest.fixture
def remote(settings: Settings, opener: Opener) -> Iterator[TestClient]:
    with _client(settings, opener, ("100.64.0.5", 50123)) as c:
        yield c


def _done_job(client: TestClient, tmp_path: Path) -> int:
    app: Any = client.app
    settings = app.state.settings
    factory = app.state.session_factory
    with factory() as s:
        track_id = make_track(s, settings, tmp_path)
    job_id = client.post(f"/api/tracks/{track_id}/jobs", json={"preset": "fast"}).json()["job"][
        "job_id"
    ]
    assert Worker(settings, factory, sync_launcher(settings)).run_one() == job_id
    return int(job_id)


def _master_dir(client: TestClient, job_id: int) -> Path:
    app: Any = client.app
    with app.state.session_factory() as s:
        stored = (
            s.query(StemRendition.file_path)
            .join(Stem, Stem.stem_id == StemRendition.stem_id)
            .filter(Stem.job_id == job_id, StemRendition.purpose == "master")
            .first()[0]
        )
    return resolve_data_path(app.state.settings, stored).resolve().parent


def test_local_opens_master_folder(
    local: TestClient, opener: Opener, tmp_path: Path, windows: None
) -> None:
    job_id = _done_job(local, tmp_path)
    res = local.post(f"/api/jobs/{job_id}/open-folder")
    assert res.status_code == 200, res.text
    folder = _master_dir(local, job_id)
    assert opener.opened == [folder]
    assert res.json()["folder"] == str(folder)
    assert (folder / "drums.flac").is_file()  # ファイル名は <stem code>.flac のまま


def test_ipv6_loopback_is_local(
    settings: Settings, opener: Opener, tmp_path: Path, windows: None
) -> None:
    with _client(settings, opener, ("::1", 50123)) as c:
        job_id = _done_job(c, tmp_path)
        assert c.post(f"/api/jobs/{job_id}/open-folder").status_code == 200
    assert len(opener.opened) == 1


def test_remote_is_forbidden(
    remote: TestClient, opener: Opener, tmp_path: Path, windows: None
) -> None:
    job_id = _done_job(remote, tmp_path)
    res = remote.post(f"/api/jobs/{job_id}/open-folder")
    assert res.status_code == 403
    assert "同じ PC" in res.json()["detail"]
    assert opener.opened == []


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Forwarded-For": "100.64.0.5"},  # Tailscale Serve などの中継
        {"Forwarded": "for=100.64.0.5"},
        {"Tailscale-User-Login": "someone@example.com"},
        {"Origin": "http://evil.example"},  # 他のサイトのページからの POST
    ],
)
def test_local_but_proxied_or_cross_site_is_forbidden(
    local: TestClient, opener: Opener, tmp_path: Path, windows: None, headers: dict[str, str]
) -> None:
    job_id = _done_job(local, tmp_path)
    res = local.post(f"/api/jobs/{job_id}/open-folder", headers=headers)
    assert res.status_code == 403
    assert opener.opened == []


def test_same_origin_header_is_allowed(
    local: TestClient, opener: Opener, tmp_path: Path, windows: None
) -> None:
    job_id = _done_job(local, tmp_path)
    res = local.post(f"/api/jobs/{job_id}/open-folder", headers={"Origin": "http://testserver"})
    assert res.status_code == 200
    assert len(opener.opened) == 1


def test_missing_job_is_404(local: TestClient, opener: Opener, windows: None) -> None:
    res = local.post("/api/jobs/9999/open-folder")
    assert res.status_code == 404
    assert "ジョブ" in res.json()["detail"]
    assert opener.opened == []


def test_not_windows_is_501(
    local: TestClient, opener: Opener, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(folders, "supports_open_folder", lambda: False)
    job_id = _done_job(local, tmp_path)
    res = local.post(f"/api/jobs/{job_id}/open-folder")
    assert res.status_code == 501
    assert "Windows" in res.json()["detail"]
    assert opener.opened == []


def test_opener_failure_is_500(
    local: TestClient, tmp_path: Path, windows: None
) -> None:
    def broken(_folder: Path) -> None:
        raise OSError("起動できない")

    local.app.state.folder_opener = broken  # type: ignore[attr-defined]
    job_id = _done_job(local, tmp_path)
    res = local.post(f"/api/jobs/{job_id}/open-folder")
    assert res.status_code == 500
    assert "エクスプローラー" in res.json()["detail"]


def test_me_reports_local(
    local: TestClient, remote: TestClient, windows: None
) -> None:
    me = local.get("/api/me").json()
    assert me["local_client"] is True
    assert me["can_open_folder"] is True
    assert local.get("/api/me", headers={"X-Forwarded-For": "1.2.3.4"}).json()[
        "local_client"
    ] is False
    me = remote.get("/api/me").json()
    assert me["local_client"] is False
    assert me["can_open_folder"] is False


# --- 起動方法（Job に入れない） ----------------------------------------------------


class FakePopen:
    calls: list[dict[str, Any]] = []
    fail_breakaway = False

    def __init__(self, cmd: list[str], **kwargs: Any) -> None:
        flags = kwargs.get("creationflags", 0)
        FakePopen.calls.append({"cmd": cmd, **kwargs})
        if FakePopen.fail_breakaway and flags & proc._CREATE_BREAKAWAY_FROM_JOB:
            err = OSError("アクセスが拒否されました")
            err.winerror = proc._ERROR_ACCESS_DENIED  # type: ignore[attr-defined]
            raise err

    def wait(self) -> int:
        return 0


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> type[FakePopen]:
    FakePopen.calls = []
    FakePopen.fail_breakaway = False
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    assign: list[Any] = []
    monkeypatch.setattr(proc, "_assign_to_job", lambda p: assign.append(p))
    FakePopen.assign = assign  # type: ignore[attr-defined]
    return FakePopen


@pytest.mark.skipif(proc.os.name != "nt", reason="Windows の起動フラグの確認")
def test_open_in_explorer_breaks_away_from_job(fake_popen: type[FakePopen], tmp_path: Path) -> None:
    proc.open_in_explorer(tmp_path)
    assert len(fake_popen.calls) == 1
    call = fake_popen.calls[0]
    assert call["cmd"] == ["explorer.exe", str(tmp_path)]
    assert call["creationflags"] & proc._CREATE_BREAKAWAY_FROM_JOB
    assert call["stdin"] is subprocess.DEVNULL
    assert fake_popen.assign == []  # type: ignore[attr-defined]  # このプロセスの Job に入れない


@pytest.mark.skipif(proc.os.name != "nt", reason="Windows の起動フラグの確認")
def test_open_in_explorer_retries_without_breakaway(
    fake_popen: type[FakePopen], tmp_path: Path
) -> None:
    fake_popen.fail_breakaway = True
    proc.open_in_explorer(tmp_path)
    assert len(fake_popen.calls) == 2
    assert not fake_popen.calls[1]["creationflags"] & proc._CREATE_BREAKAWAY_FROM_JOB
    assert fake_popen.assign == []  # type: ignore[attr-defined]


def test_start_detached_other_os(
    fake_popen: type[FakePopen], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(proc.os, "name", "posix")
    proc.start_detached(["xdg-open", "/tmp"])
    assert fake_popen.calls[0]["start_new_session"] is True
    assert "creationflags" not in fake_popen.calls[0]
