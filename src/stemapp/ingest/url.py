"""URL からの取り込み（yt-dlp）。

- 音声だけを最高音質で取得する（`-f bestaudio/best`、再エンコードしない）。
- 保存先は `data/cache/downloads/<一時ID>/`。取り込み後に消す。
- 失敗したら TRACK は作らず、INPUT_SOURCE を track_id=NULL・fetch_status=failed で残す。
- 失敗の理由は `classify_error` で理由コード（SPEC 8章）に分類する。規則は `ERROR_RULES`（データ）。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy.orm import Session

from stemapp.audio import FfmpegRunner
from stemapp.config import Settings
from stemapp.ingest.service import (
    FETCH_FAILED,
    SOURCE_URL,
    ImportResult,
    TagReader,
    import_file,
)
from stemapp.models import InputSource

log = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SEC = 3600.0
UPDATE_TIMEOUT_SEC = 300.0
ERROR_DETAIL_LINES = 5
ERROR_DETAIL_MAX_CHARS = 2000

# --- 実行器 -------------------------------------------------------------------------


@dataclass(frozen=True)
class YtDlpProcess:
    returncode: int
    stdout: str
    stderr: str


class YtDlpRunner(Protocol):
    """yt-dlp を引数（先頭の実行ファイル名を除く）で実行する。"""

    name: str

    def run(self, args: Sequence[str], *, timeout: float | None = None) -> YtDlpProcess: ...


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    # 日本語タイトルなどを UTF-8 で受け取る（exe 版も PyInstaller 製の Python なので効く）
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _run_command(cmd: Sequence[str], timeout: float | None) -> YtDlpProcess:
    proc = subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
        env=_subprocess_env(),
        stdin=subprocess.DEVNULL,
    )
    return YtDlpProcess(proc.returncode, proc.stdout or "", proc.stderr or "")


class ExeYtDlpRunner:
    """yt-dlp.exe（設定 `ytdlp_path`）を呼ぶ。"""

    name = "exe"

    def __init__(self, exe: Path) -> None:
        self.exe = Path(exe)

    def run(self, args: Sequence[str], *, timeout: float | None = None) -> YtDlpProcess:
        return _run_command([str(self.exe), *args], timeout)


class ModuleYtDlpRunner:
    """Python パッケージ yt-dlp（optional `url`）を `python -m yt_dlp` で呼ぶ。"""

    name = "module"

    def __init__(self, python: str | None = None) -> None:
        self.python = python or sys.executable

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("yt_dlp") is not None

    def run(self, args: Sequence[str], *, timeout: float | None = None) -> YtDlpProcess:
        if not self.available():
            return YtDlpProcess(
                127,
                "",
                "ERROR: yt-dlp が見つかりません（yt-dlp.exe も Python パッケージ yt-dlp も"
                "ありません）。STEMAPP_YTDLP_PATH を設定するか uv sync --extra url を"
                "実行してください。",
            )
        return _run_command([self.python, "-m", "yt_dlp", *args], timeout)


def choose_runner(settings: Settings) -> YtDlpRunner:
    """exe があれば exe 版、無ければ Python パッケージ版。"""
    if settings.ytdlp_path.is_file():
        return ExeYtDlpRunner(settings.ytdlp_path)
    log.info("%s が無いため Python パッケージ版 yt-dlp を使います。", settings.ytdlp_path)
    return ModuleYtDlpRunner()


# --- 理由コード -----------------------------------------------------------------------

UNSUPPORTED_SITE = "unsupported_site"
LOGIN_REQUIRED = "login_required"
GEO_BLOCKED = "geo_blocked"
PRIVATE_OR_REMOVED = "private_or_removed"
DRM = "drm"
NETWORK = "network"
NEEDS_UPDATE = "needs_update"
UNKNOWN = "unknown"

ERROR_MESSAGES: dict[str, str] = {
    UNSUPPORTED_SITE: "このサイトの URL には対応していません。",
    LOGIN_REQUIRED: (
        "ログインが必要なため取得できません（年齢確認・メンバー限定・ボット確認など）。"
    ),
    GEO_BLOCKED: "この地域では公開されていないため取得できません。",
    PRIVATE_OR_REMOVED: "非公開か削除済み、または存在しない動画です。URL を確認してください。",
    DRM: "DRM（コピー防止）で保護されているため取得できません。",
    NETWORK: "通信に失敗しました。ネットワークを確認し、時間をおいて再度お試しください。",
    NEEDS_UPDATE: "yt-dlp が古い可能性があります。更新してから再度お試しください。",
    UNKNOWN: "取得に失敗しました（原因を特定できませんでした）。詳細を確認してください。",
}


@dataclass(frozen=True)
class ErrorRule:
    code: str
    pattern: str  # 正規表現（大文字小文字を区別しない）
    example: str  # 実際の yt-dlp のメッセージ例（テストで使う）


# 上から順に調べ、最初に当たった規則の理由コードにする。
# 「Video unavailable. ... not available in your country」のように複数に当たるものがあるので、
# 具体的なもの（DRM・地域・ログイン）を先に置く。
ERROR_RULES: list[ErrorRule] = [
    # DRM
    ErrorRule(DRM, r"\bDRM\b", "ERROR: [youtube] abc: This video is DRM protected"),
    ErrorRule(
        DRM, r"Widevine|PlayReady", "ERROR: [Netflix] abc: Widevine content is not supported"
    ),
    # 地域制限
    ErrorRule(
        GEO_BLOCKED,
        r"not (?:made this video )?available in your (?:country|region|location)",
        "ERROR: [youtube] abc: Video unavailable. The uploader has not made this video "
        "available in your country",
    ),
    ErrorRule(
        GEO_BLOCKED,
        r"geo[ -]?(?:restrict|block)",
        "ERROR: [niconico] sm9: This video is not available from your location due to "
        "geo restriction",
    ),
    ErrorRule(
        GEO_BLOCKED,
        r"not available from your location",
        "ERROR: [abematv] abc: This video is not available from your location",
    ),
    # 非公開（案内文に --cookies が含まれるので、ログイン必須より先に調べる）
    ErrorRule(
        PRIVATE_OR_REMOVED,
        r"Private video|This video is private",
        "ERROR: [youtube] abc: Private video. Sign in if you've been granted access to this "
        "video. Use --cookies-from-browser or --cookies for the authentication.",
    ),
    # ログイン必須
    ErrorRule(
        LOGIN_REQUIRED,
        r"Sign in to confirm",
        "ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate "
        "for some users.",
    ),
    ErrorRule(
        LOGIN_REQUIRED,
        r"--cookies|login required|requires? (?:login|authentication|an account)"
        r"|need to log ?in|only available (?:for|to) (?:registered users|members|subscribers)"
        r"|members[- ]only|Join this channel",
        "ERROR: [youtube] abc: Join this channel to get access to members-only content like "
        "this video, and other exclusive perks.",
    ),
    # 削除・存在しない
    ErrorRule(
        PRIVATE_OR_REMOVED,
        r"Video unavailable|This video is unavailable|content isn't available",
        "ERROR: [youtube] xxxxxxxxxxx: Video unavailable",
    ),
    ErrorRule(
        PRIVATE_OR_REMOVED,
        r"has been removed|no longer available|does not exist|account .* terminated"
        r"|HTTP Error 404|HTTP Error 410|Incomplete YouTube ID",
        "ERROR: [generic] Unable to download webpage: HTTP Error 404: Not Found",
    ),
    # 対応していないサイト
    ErrorRule(
        UNSUPPORTED_SITE,
        r"Unsupported URL|is not a valid URL|no suitable InfoExtractor",
        "ERROR: Unsupported URL: https://example.com/",
    ),
    # yt-dlp の更新が必要
    ErrorRule(
        NEEDS_UPDATE,
        r"nsig extraction failed|Signature extraction failed|n challenge solving failed"
        r"|Unable to extract|Confirm you are on the latest version"
        r"|yt-dlp -U|update yt-dlp",
        "ERROR: [youtube] abc: nsig extraction failed: You may experience throttling for some "
        "formats",
    ),
    # 403 は YouTube の仕様変更で起きることが多く、更新で直ることが多い
    ErrorRule(
        NEEDS_UPDATE,
        r"HTTP Error 403",
        "ERROR: unable to download video data: HTTP Error 403: Forbidden",
    ),
    # 通信
    ErrorRule(
        NETWORK,
        r"Unable to download (?:webpage|API page|JSON metadata)|urlopen error|timed out"
        r"|getaddrinfo failed|Name or service not known|Temporary failure in name resolution"
        r"|Connection (?:reset|refused|aborted)|Failed to resolve|\bSSL\b|HTTP Error 5\d\d"
        r"|HTTP Error 429|Too Many Requests|Network is unreachable|IncompleteRead",
        "ERROR: [youtube] abc: Unable to download webpage: <urlopen error [Errno 11001] "
        "getaddrinfo failed> (caused by URLError(gaierror(11001, 'getaddrinfo failed')))",
    ),
]


def _error_lines(stderr: str) -> list[str]:
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    errors = [ln for ln in lines if ln.upper().startswith("ERROR")]
    return errors or lines


def classify_error(stderr: str, returncode: int) -> str:
    """yt-dlp の stderr と終了コードから理由コードを返す。どの規則にも当たらなければ unknown。

    ERROR 行があれば ERROR 行だけを見る（WARNING 行の文言で誤分類しないため）。
    終了コードは今のところ分類に使わない（yt-dlp は失敗の種類によらずほぼ 1 を返すため）。
    """
    text = "\n".join(_error_lines(stderr))
    if returncode != 0 or text:
        for rule in ERROR_RULES:
            if re.search(rule.pattern, text, flags=re.IGNORECASE):
                return rule.code
    return UNKNOWN


def error_message(code: str) -> str:
    return ERROR_MESSAGES.get(code, ERROR_MESSAGES[UNKNOWN])


def error_detail(stderr: str) -> str:
    """INPUT_SOURCE.error_detail に入れる stderr の最後の数行。"""
    lines = [ln.rstrip() for ln in (stderr or "").splitlines() if ln.strip()]
    detail = "\n".join(lines[-ERROR_DETAIL_LINES:])
    return detail[-ERROR_DETAIL_MAX_CHARS:]


# --- 取得 ---------------------------------------------------------------------------


class UrlImportError(RuntimeError):
    """URL 取り込みの失敗。INPUT_SOURCE（failed）は記録済み。"""

    def __init__(self, code: str, detail: str, source_id: int | None) -> None:
        super().__init__(f"{error_message(code)}（理由コード: {code}）")
        self.code = code
        self.message = error_message(code)
        self.detail = detail
        self.source_id = source_id


def is_url(text: str) -> bool:
    return text.lower().startswith(("http://", "https://"))


OUTPUT_BASENAME = "audio"


def download_args(url: str, out_dir: Path) -> list[str]:
    """音声だけを最高音質で、再エンコードせずに取得する引数。"""
    return [
        "--ignore-config",  # ユーザーの設定ファイル（--recode-video 等）の影響を受けない
        "--no-playlist",
        "--playlist-items", "1",  # プレイリスト URL は先頭1曲だけ
        "-f", "bestaudio/best",
        "--no-progress",
        "--no-simulate",
        "--dump-json",  # 情報を JSON で stdout に出す（--no-simulate と組で使う）
        "-o", str(out_dir / f"{OUTPUT_BASENAME}.%(ext)s"),
        "--",
        url,
    ]


def parse_info_json(stdout: str) -> dict[str, Any]:
    """stdout の JSON（1行1件）のうち最初のものを返す。無ければ空。"""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def find_downloaded_file(out_dir: Path) -> Path | None:
    """ダウンロードしたファイル（途中ファイルや JSON を除く、いちばん大きいもの）。"""
    if not out_dir.is_dir():
        return None
    skip = {".part", ".ytdl", ".json", ".tmp"}
    files = [p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() not in skip]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_size)


def _str_or_none(v: object) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def _record_failure(
    session: Session, url: str, code: str, detail: str
) -> int | None:
    session.rollback()
    src = InputSource(
        track_id=None,
        source_type=SOURCE_URL,
        original_name=None,
        url=url,
        fetch_status=FETCH_FAILED,
        error_code=code,
        error_detail=detail or None,
    )
    session.add(src)
    session.commit()
    log.warning("URL の取得に失敗しました（%s）: %s", code, detail)
    return src.source_id


def fetch_url(
    session: Session,
    settings: Settings,
    url: str,
    runner: YtDlpRunner,
    *,
    ffmpeg_runner: FfmpegRunner | None = None,
    tag_reader: TagReader | None = None,
    timeout: float | None = DOWNLOAD_TIMEOUT_SEC,
) -> ImportResult:
    """URL の音声を取得して取り込む。失敗したら INPUT_SOURCE（failed）を残して UrlImportError。"""
    out_dir = settings.cache_dir / "downloads" / uuid.uuid4().hex
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        try:
            proc = runner.run(download_args(url, out_dir), timeout=timeout)
        except subprocess.TimeoutExpired as e:
            detail = f"yt-dlp が {e.timeout:.0f} 秒で終わりませんでした。"
            sid = _record_failure(session, url, NETWORK, detail)
            raise UrlImportError(NETWORK, detail, sid) from e
        except OSError as e:
            detail = f"yt-dlp を起動できませんでした: {e}"
            sid = _record_failure(session, url, UNKNOWN, detail)
            raise UrlImportError(UNKNOWN, detail, sid) from e

        if proc.returncode != 0:
            code = classify_error(proc.stderr, proc.returncode)
            detail = error_detail(proc.stderr) or f"終了コード {proc.returncode}"
            sid = _record_failure(session, url, code, detail)
            raise UrlImportError(code, detail, sid)

        audio_file = find_downloaded_file(out_dir)
        if audio_file is None:
            detail = error_detail(proc.stderr) or "ダウンロードしたファイルが見つかりません。"
            sid = _record_failure(session, url, UNKNOWN, detail)
            raise UrlImportError(UNKNOWN, detail, sid)

        info = parse_info_json(proc.stdout)
        title = _str_or_none(info.get("title")) or _str_or_none(info.get("id"))
        artist = (
            _str_or_none(info.get("artist"))
            or _str_or_none(info.get("uploader"))
            or _str_or_none(info.get("channel"))
        )
        try:
            return import_file(
                session,
                settings,
                audio_file,
                original_name=_str_or_none(info.get("title")),
                source_type=SOURCE_URL,
                url=url,
                fetched_at=datetime.now(UTC),
                title=title,
                artist=artist,
                ffmpeg_runner=ffmpeg_runner,
                tag_reader=tag_reader,
            )
        except Exception as e:
            detail = f"取得した音声を取り込めませんでした: {type(e).__name__}: {e}"
            sid = _record_failure(session, url, UNKNOWN, detail[-ERROR_DETAIL_MAX_CHARS:])
            raise UrlImportError(UNKNOWN, detail, sid) from e
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# --- 更新 ---------------------------------------------------------------------------

UPDATE_UPDATED = "updated"
UPDATE_LATEST = "latest"
UPDATE_FAILED = "failed"


@dataclass(frozen=True)
class UpdateResult:
    status: str  # updated / latest / failed
    detail: str


def update_ytdlp(settings: Settings, runner: YtDlpRunner | None = None) -> UpdateResult:
    """`yt-dlp.exe -U` で更新する（exe 版のみ対応）。"""
    if runner is None:
        if not settings.ytdlp_path.is_file():
            return UpdateResult(
                UPDATE_FAILED,
                f"{settings.ytdlp_path} が見つかりません（更新は exe 版のみ対応）。",
            )
        runner = ExeYtDlpRunner(settings.ytdlp_path)
    try:
        proc = runner.run(["-U"], timeout=UPDATE_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError) as e:
        return UpdateResult(UPDATE_FAILED, f"yt-dlp を実行できませんでした: {e}")
    return parse_update_output(proc)


def parse_update_output(proc: YtDlpProcess) -> UpdateResult:
    text = "\n".join(s for s in (proc.stdout.strip(), proc.stderr.strip()) if s)
    last = error_detail(text)
    if proc.returncode != 0:
        return UpdateResult(UPDATE_FAILED, last or f"終了コード {proc.returncode}")
    if re.search(r"Updated yt-dlp to|Updating to", text, flags=re.IGNORECASE):
        return UpdateResult(UPDATE_UPDATED, last)
    if re.search(r"up to date|up-to-date", text, flags=re.IGNORECASE):
        return UpdateResult(UPDATE_LATEST, last)
    if re.search(r"^ERROR", text, flags=re.IGNORECASE | re.MULTILINE):
        return UpdateResult(UPDATE_FAILED, last)
    return UpdateResult(UPDATE_LATEST, last)
