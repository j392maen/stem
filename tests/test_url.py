from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from audio_helpers import fake_ffmpeg, synth_mix, write_source
from stemapp.config import Settings
from stemapp.ingest import url as url_mod
from stemapp.ingest.url import (
    ERROR_MESSAGES,
    ERROR_RULES,
    ExeYtDlpRunner,
    ModuleYtDlpRunner,
    UrlImportError,
    YtDlpProcess,
    choose_runner,
    classify_error,
    download_args,
    error_detail,
    fetch_url,
    is_url,
    parse_update_output,
    update_ytdlp,
)
from stemapp.models import InputSource, Track

URL = "https://example.com/watch?v=abc"


def no_tags(_path: Path) -> Mapping[str, str]:
    return {}


class FakeYtDlp:
    """yt-dlp の代わり。成功時は -o の場所に音声ファイルを書き、JSON を stdout に返す。"""

    name = "fake"

    def __init__(
        self,
        data: np.ndarray | None = None,
        info: dict[str, object] | None = None,
        *,
        returncode: int = 0,
        stderr: str = "",
        ext: str = "wav",
        write_file: bool = True,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.data = data
        self.info = info or {"id": "abc", "title": "テスト動画", "uploader": "投稿者",
                             "duration": 1.0}
        self.returncode = returncode
        self.stderr = stderr
        self.ext = ext
        self.write_file = write_file
        self.raise_exc = raise_exc
        self.calls: list[list[str]] = []
        self.out_dirs: list[Path] = []

    def run(self, args: Sequence[str], *, timeout: float | None = None) -> YtDlpProcess:
        args = list(args)
        self.calls.append(args)
        if self.raise_exc is not None:
            raise self.raise_exc
        template = Path(args[args.index("-o") + 1])
        self.out_dirs.append(template.parent)
        if self.returncode != 0:
            return YtDlpProcess(self.returncode, "", self.stderr)
        if self.write_file:
            assert self.data is not None
            name = template.name.replace("%(ext)s", self.ext)
            write_source(template.parent / name, self.data)
            (template.parent / f"{template.stem}.part").write_bytes(b"")  # 途中ファイルは無視
        stdout = "[info] something\n" + json.dumps(self.info, ensure_ascii=False) + "\n"
        return YtDlpProcess(0, stdout, "WARNING: some warning\n")


@pytest.fixture
def mix() -> np.ndarray:
    return synth_mix(1.0)


def _fetch(session: Session, settings: Settings, runner: FakeYtDlp, url: str = URL):
    return fetch_url(session, settings, url, runner, ffmpeg_runner=fake_ffmpeg, tag_reader=no_tags)


# --- 取得 --------------------------------------------------------------------------


def test_fetch_success(session: Session, settings: Settings, mix: np.ndarray) -> None:
    runner = FakeYtDlp(mix)
    res = _fetch(session, settings, runner)
    assert res.is_new is True

    args = runner.calls[0]
    assert args[args.index("-f") + 1] == "bestaudio/best"
    assert "--no-playlist" in args and args[-1] == URL
    assert runner.out_dirs[0].parent == settings.cache_dir / "downloads"

    track = session.get(Track, res.track_id)
    assert track is not None
    assert (track.title, track.artist) == ("テスト動画", "投稿者")
    assert track.duration_sec == pytest.approx(1.0)
    source = session.scalars(select(InputSource)).one()
    assert (source.source_type, source.url, source.fetch_status) == ("url", URL, "done")
    assert source.track_id == track.track_id
    assert source.fetched_at is not None
    assert source.error_code is None

    # ダウンロードした一時ファイルは消す
    assert not runner.out_dirs[0].exists()
    assert not list((settings.cache_dir / "downloads").iterdir())


def test_fetch_same_audio_is_existing(session: Session, settings: Settings,
                                      mix: np.ndarray) -> None:
    first = _fetch(session, settings, FakeYtDlp(mix))
    second = _fetch(session, settings, FakeYtDlp(mix), url=URL + "&t=1")
    assert second.is_new is False and second.track_id == first.track_id
    assert len(session.scalars(select(Track)).all()) == 1
    assert len(session.scalars(select(InputSource)).all()) == 2


def test_fetch_title_falls_back_to_id(session: Session, settings: Settings,
                                      mix: np.ndarray) -> None:
    res = _fetch(session, settings, FakeYtDlp(mix, {"id": "xyz", "channel": "ch"}))
    track = session.get(Track, res.track_id)
    assert track is not None and (track.title, track.artist) == ("xyz", "ch")


def test_fetch_failure_records_source(session: Session, settings: Settings) -> None:
    stderr = (
        "WARNING: [youtube] something harmless\n"
        "ERROR: [youtube] xxxxxxxxxxx: Video unavailable\n"
    )
    runner = FakeYtDlp(returncode=1, stderr=stderr)
    with pytest.raises(UrlImportError) as ei:
        _fetch(session, settings, runner)
    assert ei.value.code == "private_or_removed"
    assert ei.value.message == ERROR_MESSAGES["private_or_removed"]

    assert session.scalars(select(Track)).all() == []
    source = session.scalars(select(InputSource)).one()
    assert source.source_id == ei.value.source_id
    assert source.track_id is None
    assert (source.source_type, source.url, source.fetch_status) == ("url", URL, "failed")
    assert source.error_code == "private_or_removed"
    assert source.error_detail is not None and "Video unavailable" in source.error_detail
    assert not runner.out_dirs[0].exists()


def test_fetch_no_file_is_unknown(session: Session, settings: Settings) -> None:
    with pytest.raises(UrlImportError) as ei:
        _fetch(session, settings, FakeYtDlp(write_file=False))
    assert ei.value.code == "unknown"
    assert session.scalars(select(InputSource)).one().fetch_status == "failed"


def test_fetch_timeout_is_network(session: Session, settings: Settings) -> None:
    runner = FakeYtDlp(raise_exc=subprocess.TimeoutExpired(["yt-dlp"], 5))
    with pytest.raises(UrlImportError) as ei:
        _fetch(session, settings, runner)
    assert ei.value.code == "network"
    assert not list((settings.cache_dir / "downloads").iterdir())


def test_fetch_broken_audio_is_recorded(session: Session, settings: Settings) -> None:
    class BrokenFile(FakeYtDlp):
        def run(self, args: Sequence[str], *, timeout: float | None = None) -> YtDlpProcess:
            args = list(args)
            template = Path(args[args.index("-o") + 1])
            (template.parent / "audio.webm").write_bytes(b"not audio")
            return YtDlpProcess(0, json.dumps({"title": "t"}), "")

    with pytest.raises(UrlImportError) as ei:
        _fetch(session, settings, BrokenFile())
    assert ei.value.code == "unknown"
    assert session.scalars(select(Track)).all() == []
    source = session.scalars(select(InputSource)).one()
    assert source.fetch_status == "failed" and source.track_id is None
    assert source.error_detail is not None and "取り込めません" in source.error_detail


def test_download_args() -> None:
    args = download_args("https://x/y", Path("C:/d"))
    assert args[args.index("-f") + 1] == "bestaudio/best"
    for flag in ("--no-playlist", "--ignore-config", "--no-simulate", "--dump-json"):
        assert flag in args
    assert args[args.index("--playlist-items") + 1] == "1"
    assert "-x" not in args and "--recode-video" not in args  # 再エンコードしない
    assert args[-2:] == ["--", "https://x/y"]


def test_is_url() -> None:
    assert is_url("https://a") and is_url("HTTP://a")
    assert not is_url("C:\\music\\a.mp3") and not is_url("ftp://a")


# --- 実行器の選択 -----------------------------------------------------------------------


def test_exe_runner_is_chosen_when_exe_exists(tmp_path: Path) -> None:
    exe = tmp_path / "yt-dlp.exe"
    exe.write_bytes(b"")
    s = Settings(_env_file=None, data_dir=tmp_path / "data", ytdlp_path=exe)  # type: ignore[call-arg]
    runner = choose_runner(s)
    assert isinstance(runner, ExeYtDlpRunner) and runner.exe == exe


def test_module_runner_is_chosen_without_exe(tmp_path: Path) -> None:
    s = Settings(  # type: ignore[call-arg]
        _env_file=None, data_dir=tmp_path / "data", ytdlp_path=tmp_path / "none.exe"
    )
    assert isinstance(choose_runner(s), ModuleYtDlpRunner)


def test_module_runner_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ModuleYtDlpRunner, "available", staticmethod(lambda: False))
    proc = ModuleYtDlpRunner().run(["--version"])
    assert proc.returncode != 0 and "yt-dlp が見つかりません" in proc.stderr


def test_module_runner_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: Sequence[str], timeout: float | None) -> YtDlpProcess:
        seen.append(list(cmd))
        return YtDlpProcess(0, "ok", "")

    monkeypatch.setattr(ModuleYtDlpRunner, "available", staticmethod(lambda: True))
    monkeypatch.setattr(url_mod, "_run_command", fake_run)
    ModuleYtDlpRunner(python="py").run(["--version"])
    assert seen == [["py", "-m", "yt_dlp", "--version"]]


# --- 理由コード ------------------------------------------------------------------------

# 実際の yt-dlp のメッセージ例（各理由コード最低1つ）
REAL_MESSAGES: list[tuple[str, str]] = [
    ("ERROR: Unsupported URL: https://example.com/", "unsupported_site"),
    ("ERROR: [generic] 'not a url' is not a valid URL. Set --default-search", "unsupported_site"),
    (
        "ERROR: [youtube] abc: Sign in to confirm you’re not a bot. Use --cookies-from-browser "
        "or --cookies for the authentication.",
        "login_required",
    ),
    (
        "ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate for "
        "some users. Use --cookies-from-browser or --cookies for the authentication.",
        "login_required",
    ),
    (
        "ERROR: [youtube] abc: The uploader has not made this video available in your country",
        "geo_blocked",
    ),
    (
        "ERROR: [youtube] abc: Video unavailable. This video is not available in your country",
        "geo_blocked",
    ),
    (
        "ERROR: [youtube] abc: Private video. Sign in if you've been granted access to this "
        "video. Use --cookies-from-browser or --cookies for the authentication.",
        "private_or_removed",
    ),
    ("ERROR: [youtube] xxxxxxxxxxx: Video unavailable", "private_or_removed"),
    # 2026-09 に存在しない ID で実際に出たメッセージ（yt-dlp 2026.08.19）
    ("ERROR: [youtube] xxxxxxxxxxx: This video is unavailable", "private_or_removed"),
    (
        "ERROR: [youtube] abc: Video unavailable. This video has been removed by the uploader",
        "private_or_removed",
    ),
    ("ERROR: [youtube] abc: This video is DRM protected", "drm"),
    ("ERROR: [Spotify] abc: The requested site is known to use DRM protection.", "drm"),
    (
        "ERROR: [youtube] abc: Unable to download webpage: <urlopen error [Errno 11001] "
        "getaddrinfo failed> (caused by TransportError(\"<urlopen error [Errno 11001] "
        "getaddrinfo failed>\"))",
        "network",
    ),
    ("ERROR: [youtube] abc: Unable to download API page: The read operation timed out",
     "network"),
    (
        "ERROR: unable to download video data: HTTP Error 403: Forbidden",
        "needs_update",
    ),
    (
        "ERROR: [youtube] abc: nsig extraction failed: Some formats may be missing; please "
        "report this issue on https://github.com/yt-dlp/yt-dlp/issues",
        "needs_update",
    ),
    (
        "ERROR: [youtube] abc: Unable to extract uploader id; please report this issue on "
        "https://github.com/yt-dlp/yt-dlp/issues?q= , filling out the appropriate issue "
        "template. Confirm you are on the latest version using  yt-dlp -U",
        "needs_update",
    ),
]


@pytest.mark.parametrize(("stderr", "code"), REAL_MESSAGES)
def test_classify_real_messages(stderr: str, code: str) -> None:
    assert classify_error(stderr, 1) == code


def test_every_code_has_message_and_example() -> None:
    codes = {
        "unsupported_site", "login_required", "geo_blocked", "private_or_removed", "drm",
        "network", "needs_update",
    }
    assert {c for _, c in REAL_MESSAGES} == codes
    assert set(ERROR_MESSAGES) == codes | {"unknown"}
    for rule in ERROR_RULES:
        assert classify_error(rule.example, 1) == rule.code, rule.example


def test_classify_unknown() -> None:
    assert classify_error("ERROR: something completely different", 1) == "unknown"
    assert classify_error("", 1) == "unknown"


def test_classify_ignores_warning_lines() -> None:
    stderr = (
        "WARNING: [youtube] abc: nsig extraction failed: Some formats may be missing\n"
        "ERROR: [youtube] abc: Private video. Sign in if you've been granted access\n"
    )
    assert classify_error(stderr, 1) == "private_or_removed"


def test_error_detail_keeps_last_lines() -> None:
    stderr = "\n".join(f"line {i}" for i in range(20))
    detail = error_detail(stderr)
    assert detail.splitlines() == [f"line {i}" for i in range(15, 20)]


# --- 更新 -------------------------------------------------------------------------------


class UpdateRunner:
    name = "fake"

    def __init__(self, proc: YtDlpProcess) -> None:
        self.proc = proc
        self.calls: list[list[str]] = []

    def run(self, args: Sequence[str], *, timeout: float | None = None) -> YtDlpProcess:
        self.calls.append(list(args))
        return self.proc


def test_update_results(settings: Settings) -> None:
    latest = UpdateRunner(YtDlpProcess(
        0, "Latest version: stable@2026.08.19 from yt-dlp/yt-dlp\n"
        "yt-dlp is up to date (stable@2026.08.19 from yt-dlp/yt-dlp)\n", ""))
    res = update_ytdlp(settings, latest)
    assert res.status == "latest" and latest.calls == [["-U"]]

    updated = YtDlpProcess(
        0, "Current version: stable@2026.08.01\nLatest version: stable@2026.08.19\n"
        "Current Build Hash: abc\nUpdating to stable@2026.08.19 ...\n"
        "Updated yt-dlp to stable@2026.08.19\n", "")
    assert parse_update_output(updated).status == "updated"

    failed = YtDlpProcess(1, "", "ERROR: Unable to write to C:\\mine\\yt-dlp.exe\n")
    res = parse_update_output(failed)
    assert res.status == "failed" and "Unable to write" in res.detail


def test_update_without_exe(tmp_path: Path) -> None:
    s = Settings(  # type: ignore[call-arg]
        _env_file=None, data_dir=tmp_path / "data", ytdlp_path=tmp_path / "none.exe"
    )
    res = update_ytdlp(s)
    assert res.status == "failed" and "exe 版のみ" in res.detail


def test_fake_runner_writes_source(tmp_path: Path, mix: np.ndarray) -> None:
    # Fake 自体の確認（-o のテンプレートどおりに書く）
    r = FakeYtDlp(mix)
    r.run(["-o", str(tmp_path / "audio.%(ext)s"), "--", URL])
    assert (tmp_path / "audio.wav").is_file()
