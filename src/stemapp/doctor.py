"""環境診断（`stemapp doctor`）。

各診断は `Settings` を受け取り `CheckResult` を返す関数。テストでは差し替えられる。
"""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
