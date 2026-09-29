"""環境診断（`stemapp doctor`）。

各診断は `Settings` を受け取り `CheckResult` を返す関数。テストでは差し替えられる。
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import func, select, text

from stemapp.config import Settings


class Status(Enum):
    OK = "OK"
    WARN = "注意"
    NG = "NG"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    hint: str = ""


Check = Callable[[Settings], CheckResult]


def _run(cmd: Sequence[str | Path], timeout: float = 30.0) -> str:
    """外部コマンドを実行し、標準出力の1行目を返す。失敗時は例外。"""
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=True,
    )
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    return lines[0].strip() if lines else ""


# --- 必須（失敗は NG） ------------------------------------------------------------


def check_python(_settings: Settings) -> CheckResult:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 12):
        return CheckResult("Python", Status.OK, ver)
    return CheckResult("Python", Status.NG, ver, "Python 3.12 以上が必要です（uv sync で入ります）")


def check_data_dir(settings: Settings) -> CheckResult:
    name = "データフォルダ"
    try:
        root = settings.data_root
        probe = root / f".write_test_{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return CheckResult(name, Status.OK, str(root))
    except OSError as e:
        return CheckResult(
            name, Status.NG, f"{settings.data_dir}: {e}",
            "STEMAPP_DATA_DIR に書き込めるフォルダを指定してください",
        )


def check_db(settings: Settings) -> CheckResult:
    from stemapp.db import init_db, make_engine
    from stemapp.models import StemType

    name = "データベース"
    try:
        engine = make_engine(settings.db_path)
        try:
            init_db(engine)
            with engine.connect() as conn:
                fk = conn.execute(text("PRAGMA foreign_keys")).scalar()
                n_types = conn.execute(select(func.count()).select_from(StemType)).scalar()
        finally:
            engine.dispose()
    except Exception as e:  # noqa: BLE001 - どんな失敗も診断結果として表示する
        return CheckResult(
            name, Status.NG, str(e), "データフォルダの権限と空き容量を確認してください"
        )
    if fk != 1:
        return CheckResult(name, Status.NG, "外部キー制約が無効", "SQLite の設定を確認してください")
    if not n_types:
        return CheckResult(
            name, Status.WARN, f"{settings.db_path}（初期データなし）",
            "`uv run stemapp init-db` を実行してください",
        )
    return CheckResult(name, Status.OK, f"{settings.db_path}（stem 種類 {n_types}）")


# --- 任意（無ければ注意） -----------------------------------------------------------


def check_ffmpeg(_settings: Settings) -> CheckResult:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return CheckResult(
            "ffmpeg", Status.WARN, "見つかりません",
            "winget install Gyan.FFmpeg で導入し、新しいシェルを開いてください",
        )
    try:
        first = _run([exe, "-version"])
    except (OSError, subprocess.SubprocessError) as e:
        return CheckResult("ffmpeg", Status.WARN, f"{exe}: {e}", "ffmpeg を入れ直してください")
    detail = first.removeprefix("ffmpeg ").split(" Copyright")[0]
    return CheckResult("ffmpeg", Status.OK, detail)


def check_torch_cuda(_settings: Settings) -> CheckResult:
    name = "torch / CUDA"
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return CheckResult(
            name, Status.WARN, "torch が入っていません",
            "uv sync --extra gpu を実行してください",
        )
    if not torch.cuda.is_available():
        return CheckResult(
            name, Status.WARN, f"torch {torch.__version__}（CUDA 使用不可）",
            "CUDA 版 torch と NVIDIA ドライバを確認してください（CPU で動作します）",
        )
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    detail = (
        f"{props.name} / VRAM {vram_gb:.1f}GB / CUDA {torch.version.cuda}"
        f" / torch {torch.__version__}"
    )
    return CheckResult(name, Status.OK, detail)


def check_audio_separator(_settings: Settings) -> CheckResult:
    name = "audio-separator"
    try:
        ver = importlib.metadata.version("audio-separator")
    except importlib.metadata.PackageNotFoundError:
        return CheckResult(
            name, Status.WARN, "入っていません", "uv sync --extra gpu を実行してください"
        )
    return CheckResult(name, Status.OK, ver)


def check_ytdlp(settings: Settings) -> CheckResult:
    name = "yt-dlp"
    exe = settings.ytdlp_path
    if exe.is_file():
        try:
            ver = _run([exe, "--version"])
        except (OSError, subprocess.SubprocessError) as e:
            return CheckResult(name, Status.WARN, f"{exe}: {e}", "yt-dlp.exe を入れ直してください")
        return CheckResult(name, Status.OK, f"{exe}（{ver}）")
    try:
        ver = importlib.metadata.version("yt-dlp")
    except importlib.metadata.PackageNotFoundError:
        return CheckResult(
            name, Status.WARN, f"{exe} が見つかりません",
            "yt-dlp.exe を置くか STEMAPP_YTDLP_PATH を設定してください",
        )
    return CheckResult(
        name, Status.WARN, f"{exe} が無いため Python 版 {ver} を使います",
        "yt-dlp.exe を置くか STEMAPP_YTDLP_PATH を設定してください",
    )


def check_deno(_settings: Settings) -> CheckResult:
    exe = shutil.which("deno")
    if exe is None:
        return CheckResult(
            "deno", Status.WARN, "見つかりません（YouTube 取得に必要）",
            "winget install DenoLand.Deno で導入し、新しいシェルを開いてください",
        )
    try:
        first = _run([exe, "--version"])
    except (OSError, subprocess.SubprocessError) as e:
        return CheckResult("deno", Status.WARN, f"{exe}: {e}", "deno を入れ直してください")
    return CheckResult("deno", Status.OK, first)


LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def check_host(settings: Settings) -> CheckResult:
    name = "待ち受けアドレス"
    host = settings.host.strip()
    if host.lower() in LOCAL_HOSTS:
        return CheckResult(name, Status.OK, f"{host}:{settings.port}（この PC からのみ）")
    return CheckResult(
        name, Status.WARN,
        f"{host}:{settings.port}（外部に公開される可能性があります）",
        "STEMAPP_HOST=127.0.0.1 にし、外出先からは Tailscale Serve を使ってください",
    )


TAILSCALE_DEFAULT_PATH = Path(r"C:\Program Files\Tailscale\tailscale.exe")
SERVE_SCRIPT = r"scripts\tailscale-serve.ps1"


def find_tailscale() -> str | None:
    exe = shutil.which("tailscale")
    if exe:
        return exe
    if TAILSCALE_DEFAULT_PATH.is_file():
        return str(TAILSCALE_DEFAULT_PATH)
    return None


def _tailscale_json(exe: str, args: Sequence[str]) -> Any:
    """tailscale の --json の出力を読む。失敗は例外。"""
    proc = subprocess.run(
        [exe, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=15, check=True,
    )
    out = proc.stdout.strip()
    return json.loads(out) if out else {}


def serve_targets(serve: Any) -> list[tuple[str, str]]:
    """serve の設定から（公開している URL, 転送先）の一覧を作る。"""
    targets: list[tuple[str, str]] = []
    web = serve.get("Web") if isinstance(serve, dict) else None
    for hostport, conf in (web or {}).items():
        handlers = conf.get("Handlers") if isinstance(conf, dict) else None
        for path, handler in (handlers or {}).items():
            if isinstance(handler, dict) and handler.get("Proxy"):
                targets.append((f"https://{hostport}{path}", str(handler["Proxy"])))
    return targets


def funnel_enabled(serve: Any) -> bool:
    allow = serve.get("AllowFunnel") if isinstance(serve, dict) else None
    return isinstance(allow, dict) and any(bool(v) for v in allow.values())


def _points_to_app(proxy: str, port: int) -> bool:
    parts = urlsplit(proxy if "://" in proxy else f"http://{proxy}")
    return parts.hostname in ("127.0.0.1", "localhost", "::1") and parts.port == port


def check_tailscale(settings: Settings) -> CheckResult:
    """外出先（iPhone）から使うための Tailscale の状態。無くても使えるので「注意」止まり。"""
    name = "Tailscale"
    exe = find_tailscale()
    if exe is None:
        return CheckResult(
            name, Status.WARN, "入っていません（外出先から使うときに必要）",
            "https://tailscale.com/download から入れてログインしてください",
        )
    try:
        status = _tailscale_json(exe, ["status", "--json"])
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return CheckResult(
            name, Status.WARN, f"状態を取れません: {e}", "Tailscale を起動してください"
        )
    state = status.get("BackendState") if isinstance(status, dict) else None
    if state != "Running":
        return CheckResult(
            name, Status.WARN, f"ログインしていないか停止中です（{state}）",
            "タスクトレイの Tailscale からログインしてください",
        )
    self_info = status.get("Self") or {}
    dns = str(self_info.get("DNSName") or "").rstrip(".").lower()
    try:
        serve = _tailscale_json(exe, ["serve", "status", "--json"])
    except (OSError, subprocess.SubprocessError, ValueError):
        serve = None
    if serve is not None and funnel_enabled(serve):
        return CheckResult(
            name, Status.WARN, f"{dns}: Funnel（インターネット全体への公開）が有効です",
            f"Funnel は使いません。`tailscale funnel reset` で止め、{SERVE_SCRIPT} start を"
            "使ってください",
        )
    to_app = [
        url for url, proxy in serve_targets(serve) if _points_to_app(proxy, settings.port)
    ]
    serve_text = (
        "serve の状態を取れません" if serve is None
        else f"serve 有効（{', '.join(to_app)}）" if to_app
        else "serve 未設定"
    )
    detail = f"{dns or '名前不明'}（{serve_text}）"
    hints = []
    host_allowed = bool(dns) and dns in settings.allowed_host_names
    if dns and not host_allowed:
        hints.append(f".env に STEMAPP_ALLOWED_HOSTS={dns} を書いてください")
    if not to_app:
        hints.append(f"外から使うときは {SERVE_SCRIPT} start")
    status_level = Status.WARN if to_app and not host_allowed else Status.OK
    return CheckResult(name, status_level, detail, "／".join(hints))


DEFAULT_CHECKS: tuple[Check, ...] = (
    check_python,
    check_data_dir,
    check_db,
    check_host,
    check_ffmpeg,
    check_torch_cuda,
    check_audio_separator,
    check_ytdlp,
    check_deno,
    check_tailscale,
)


def run_checks(settings: Settings, checks: Sequence[Check] = DEFAULT_CHECKS) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        try:
            results.append(check(settings))
        except Exception as e:  # noqa: BLE001 - 診断関数自体の失敗も NG として表示
            results.append(CheckResult(check.__name__, Status.NG, f"診断中にエラー: {e}"))
    return results


def exit_code(results: Sequence[CheckResult]) -> int:
    """NG が1つでもあれば 1。"""
    return 1 if any(r.status is Status.NG for r in results) else 0
