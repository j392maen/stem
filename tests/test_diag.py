"""端末の診断の API（テスト音声・結果の保存・一覧）。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stemapp import diag
from stemapp.audio import AudioError
from stemapp.config import Settings
from test_api import _app

RESULT = {
    "device": "iPhone-Safari",
    "ua": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)",
    "decode": {"webm": {"ok": False, "error": "EncodingError"}, "m4a": {"ok": True}},
}


class FakeFfmpeg:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args: Sequence[str]) -> None:
        self.calls.append(list(args))
        Path(args[-1]).write_bytes(b"FAKE-AUDIO")


@pytest.fixture
def ffmpeg() -> FakeFfmpeg:
    return FakeFfmpeg()


@pytest.fixture
def client(settings: Settings, ffmpeg: FakeFfmpeg) -> Iterator[TestClient]:
    app = _app(settings)
    app.state.diag_ffmpeg_runner = ffmpeg
    with TestClient(app) as c:
        yield c


def _post(c: TestClient, body: object, **headers: str) -> object:
    return c.post(
        "/api/diag",
        content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )


# --- テスト音声 ------------------------------------------------------------------------


def test_sample_list(client: TestClient) -> None:
    samples = client.get("/api/diag/samples").json()["samples"]
    assert [s["code"] for s in samples] == ["webm", "m4a", "mp3", "flac", "wav"]
    assert all(s["url"] == f"/api/diag/samples/{s['code']}" for s in samples)


@pytest.mark.parametrize(
    ("code", "ctype"),
    [("webm", "audio/webm"), ("m4a", "audio/mp4"), ("mp3", "audio/mpeg"),
     ("flac", "audio/flac"), ("wav", "audio/wav")],
)
def test_sample_is_generated_once_and_cached(
    client: TestClient, ffmpeg: FakeFfmpeg, settings: Settings, code: str, ctype: str
) -> None:
    for _ in range(2):
        res = client.get(f"/api/diag/samples/{code}")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith(ctype)
        assert res.content == b"FAKE-AUDIO"
    assert len(ffmpeg.calls) == 1
    args = ffmpeg.calls[0]
    assert args[args.index("-f") + 1] == "lavfi"
    assert args[-1].endswith(".part")
    assert (settings.cache_dir / "diag" / f"sample.{code}").is_file()
    assert not list((settings.cache_dir / "diag").glob("*.part"))


def test_unknown_sample_is_404(client: TestClient, ffmpeg: FakeFfmpeg) -> None:
    for code in ("ogg", "WAV", "wav.exe", "%2e%2e"):
        assert client.get(f"/api/diag/samples/{code}").status_code == 404
    assert ffmpeg.calls == []


def test_sample_ffmpeg_failure_is_503(settings: Settings) -> None:
    def broken(_args: Sequence[str]) -> None:
        raise AudioError("ffmpeg が見つかりません。")

    app = _app(settings)
    app.state.diag_ffmpeg_runner = broken
    with TestClient(app) as c:
        res = c.get("/api/diag/samples/mp3")
    assert res.status_code == 503
    assert "テスト音声" in res.json()["detail"]
    assert not list((settings.cache_dir / "diag").iterdir())


def test_sample_args_per_format(tmp_path: Path) -> None:
    opus = diag.sample_args(diag.SAMPLE_FORMATS["webm"], tmp_path / "x")
    assert "libopus" in opus and "48000" in opus and opus[opus.index("-f", 3) + 1] == "webm"
    aac = diag.sample_args(diag.SAMPLE_FORMATS["m4a"], tmp_path / "x")
    assert "aac" in aac and "44100" in aac


# --- 結果の保存と一覧 --------------------------------------------------------------------


def test_post_and_list(client: TestClient, settings: Settings) -> None:
    res = _post(client, RESULT, **{"User-Agent": "iPhone-test", "X-Forwarded-Proto": "https"})
    assert res.status_code == 200, res.text
    name = res.json()["name"]
    assert name.endswith("-iPhone-Safari.json")
    path = settings.data_dir / "diag" / name
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["result"] == RESULT
    assert saved["server"]["user_agent"] == "iPhone-test"
    assert saved["server"]["proxied"] is True and saved["server"]["https"] is True

    _post(client, {"device": "Windows-Edge"})
    listed = client.get("/api/diag").json()["results"]
    assert len(listed) == 2
    assert {item["result"]["device"] for item in listed} == {"iPhone-Safari", "Windows-Edge"}
    assert all(item["size"] > 0 and item["saved_at"] for item in listed)
    assert len(client.get("/api/diag", params={"limit": 1}).json()["results"]) == 1
    assert client.get("/api/diag", params={"limit": 0}).status_code == 422
    assert client.get("/api/diag", params={"limit": 1000}).status_code == 422


def test_same_second_does_not_overwrite(settings: Settings) -> None:
    now = datetime(2026, 9, 26, 12, 0, 0).astimezone()
    a = diag.save_result(settings, {"device": "x", "n": 1}, {}, now=now)
    b = diag.save_result(settings, {"device": "x", "n": 2}, {}, now=now)
    assert a == "20260926-120000-x.json"
    assert b == "20260926-120000-x-2.json"
    results = diag.list_results(settings)
    assert sorted(r["result"]["n"] for r in results) == [1, 2]


@pytest.mark.parametrize(
    ("device", "slug"),
    [
        ("../../etc/passwd", "etc-passwd"),
        ("C:\\Windows\\x", "C-Windows-x"),
        ("アイフォン", "unknown"),
        (None, "unknown"),
        (123, "unknown"),
        ("a" * 100, "a" * 32),
        ("iPhone 13 mini", "iPhone-13-mini"),
    ],
)
def test_device_slug(device: object, slug: str) -> None:
    assert diag.device_slug(device) == slug


def test_saved_file_stays_in_diag_dir(client: TestClient, settings: Settings) -> None:
    name = _post(client, {"device": "../../../evil"}).json()["name"]
    assert (settings.data_dir / "diag" / name).is_file()
    assert "/" not in name and "\\" not in name and ".." not in name


def test_too_large_is_413(client: TestClient, settings: Settings) -> None:
    big = {"device": "x", "pad": "a" * diag.MAX_DIAG_BYTES}
    res = _post(client, big)
    assert res.status_code == 413
    assert "大きすぎ" in res.json()["detail"]
    # Content-Length を付けずに（分割して）送っても止める
    def chunks() -> Iterator[bytes]:
        data = json.dumps(big).encode()
        for i in range(0, len(data), 8192):
            yield data[i : i + 8192]

    res = client.post("/api/diag", content=chunks(), headers={"Content-Type": "application/json"})
    assert res.status_code == 413
    assert not (settings.data_dir / "diag").exists() or not list(
        (settings.data_dir / "diag").iterdir()
    )


@pytest.mark.parametrize(
    ("content", "ctype", "status"),
    [
        (b"not json", "application/json", 400),
        (b"\xff\xfe", "application/json", 400),
        (b"[1, 2]", "application/json", 400),
        (b'"text"', "application/json", 400),
        (b"", "application/json", 400),
        (b'{"device": "x"}', "text/plain", 415),
        (b"device=x", "application/x-www-form-urlencoded", 415),
    ],
)
def test_bad_input(client: TestClient, content: bytes, ctype: str, status: int) -> None:
    res = client.post("/api/diag", content=content, headers={"Content-Type": ctype})
    assert res.status_code == status
    assert res.json()["detail"]


def test_list_skips_broken_and_foreign_files(client: TestClient, settings: Settings) -> None:
    folder = settings.data_dir / "diag"
    folder.mkdir(parents=True)
    (folder / "20260101-000000-broken.json").write_text("{oops", encoding="utf-8")
    (folder / "notes.json").write_text("{}", encoding="utf-8")
    items = client.get("/api/diag").json()["results"]
    assert [i["name"] for i in items] == ["20260101-000000-broken.json"]
    assert "読めません" in items[0]["error"]


def test_diag_requires_login(settings: Settings) -> None:
    with TestClient(_app(settings, passcode="pw")) as c:
        assert _post(c, RESULT).status_code == 401
        assert c.get("/api/diag").status_code == 401
        assert c.get("/api/diag/samples/wav").status_code == 401


# --- 本物の ffmpeg -----------------------------------------------------------------------

MAGIC = {
    "webm": (0, b"\x1a\x45\xdf\xa3"),
    "m4a": (4, b"ftyp"),
    "flac": (0, b"fLaC"),
    "wav": (0, b"RIFF"),
}


@pytest.mark.ffmpeg
@pytest.mark.parametrize("code", list(diag.SAMPLE_FORMATS))
def test_real_samples(settings: Settings, code: str) -> None:
    import soundfile as sf

    path = diag.ensure_sample(settings, code)
    data = path.read_bytes()
    assert 200 < len(data) < 400_000
    if code in MAGIC:
        offset, magic = MAGIC[code]
        assert data[offset : offset + len(magic)] == magic
    if code == "mp3":
        assert data[:3] == b"ID3" or data[0] == 0xFF
    if code in ("wav", "flac"):
        info = sf.info(str(path))
        assert info.channels == 2
        assert info.samplerate == 44100
        assert abs(info.duration - diag.SAMPLE_SECONDS) < 0.01
    assert diag.ensure_sample(settings, code) == path  # 2回目は作り直さない
