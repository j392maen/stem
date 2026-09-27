"""端末の診断（iPhone などのブラウザで何が使えるかを調べる）のサーバー側。

- 診断ページが使う小さなテスト音声（1秒のサイン波）を形式ごとに ffmpeg で作り、
  `data/cache/diag/` にキャッシュする。
- 診断結果は `data/diag/<日時>-<端末>.json` に保存する（DB には入れない）。
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stemapp.audio import FfmpegRunner, run_ffmpeg
from stemapp.config import Settings

SAMPLE_SECONDS = 1.0
# 保存する診断結果1件の大きさの上限
MAX_DIAG_BYTES = 128 * 1024
# 一覧で返す件数の上限
MAX_LIST = 100


@dataclass(frozen=True)
class SampleFormat:
    code: str
    label: str
    media_type: str
    suffix: str
    muxer: str
    codec_args: tuple[str, ...]
    sample_rate: int = 44100


SAMPLE_FORMATS: dict[str, SampleFormat] = {
    f.code: f
    for f in (
        # 今の配信（stream）と同じ Opus/WebM。Opus は 48kHz
        SampleFormat("webm", "Opus / WebM", "audio/webm", ".webm", "webm",
                     ("-c:a", "libopus", "-b:a", "96k"), 48000),
        SampleFormat("m4a", "AAC / M4A", "audio/mp4", ".m4a", "ipod",
                     ("-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart")),
        SampleFormat("mp3", "MP3", "audio/mpeg", ".mp3", "mp3",
                     ("-c:a", "libmp3lame", "-b:a", "128k")),
        SampleFormat("flac", "FLAC", "audio/flac", ".flac", "flac", ("-c:a", "flac")),
        SampleFormat("wav", "WAV", "audio/wav", ".wav", "wav", ("-c:a", "pcm_s16le")),
    )
}

_sample_lock = threading.Lock()


def sample_dir(settings: Settings) -> Path:
    path = settings.cache_dir / "diag"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_args(fmt: SampleFormat, dst: Path) -> list[str]:
    """1秒・440Hz・ステレオのテスト音声を作る ffmpeg の引数（先頭の "ffmpeg" を除く）。"""
    return [
        "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:sample_rate={fmt.sample_rate}:duration={SAMPLE_SECONDS}",
        "-ac", "2",
        "-ar", str(fmt.sample_rate),
        *fmt.codec_args,
        "-f", fmt.muxer,
        str(dst),
    ]


def ensure_sample(settings: Settings, code: str, runner: FfmpegRunner | None = None) -> Path:
    """テスト音声のファイル（無ければ作る）。未知の形式は KeyError、変換の失敗は AudioError。"""
    fmt = SAMPLE_FORMATS[code]
    path = sample_dir(settings) / f"sample{fmt.suffix}"
    with _sample_lock:
        if path.is_file() and path.stat().st_size > 0:
            return path
        tmp = path.with_name(path.name + ".part")
        try:
            (runner or run_ffmpeg)(sample_args(fmt, tmp))
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return path


# --- 診断結果の保存 -------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_NAME_RE = re.compile(r"^\d{8}-\d{6}-[A-Za-z0-9_-]{1,32}(?:-\d+)?\.json$")


def diag_dir(settings: Settings) -> Path:
    path = settings.data_root / "diag"
    path.mkdir(parents=True, exist_ok=True)
    return path


def device_slug(value: Any) -> str:
    """ファイル名に使う端末の名前（英数字・_・- だけ、32文字まで）。"""
    text = value if isinstance(value, str) else ""
    slug = _SLUG_RE.sub("-", text).strip("-")[:32].strip("-")
    return slug or "unknown"


def save_result(
    settings: Settings, data: dict[str, Any], server_info: dict[str, Any],
    now: datetime | None = None,
) -> str:
    """診断結果を保存し、ファイル名を返す。"""
    now = now or datetime.now().astimezone()
    base = f"{now:%Y%m%d-%H%M%S}-{device_slug(data.get('device'))}"
    record = {"saved_at": now.isoformat(), "server": server_info, "result": data}
    text = json.dumps(record, ensure_ascii=False, indent=1)
    folder = diag_dir(settings)
    for n in range(1, 1000):
        name = f"{base}.json" if n == 1 else f"{base}-{n}.json"
        try:
            with open(folder / name, "x", encoding="utf-8", newline="\n") as f:
                f.write(text)
            return name
        except FileExistsError:
            continue
    raise RuntimeError("診断結果のファイル名を決められませんでした。")


def list_results(settings: Settings, limit: int = 20) -> list[dict[str, Any]]:
    """保存した診断結果（新しい順）。読めないファイルは error を付けて返す。"""
    files: Sequence[Path] = sorted(
        (p for p in diag_dir(settings).glob("*.json") if _NAME_RE.match(p.name)),
        key=lambda p: p.name,
        reverse=True,
    )
    items: list[dict[str, Any]] = []
    for path in files[: max(0, min(limit, MAX_LIST))]:
        item: dict[str, Any] = {"name": path.name, "size": path.stat().st_size}
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            item["saved_at"] = record.get("saved_at")
            item["server"] = record.get("server")
            item["result"] = record.get("result")
        except (OSError, ValueError, AttributeError) as e:
            item["error"] = f"読めませんでした: {e}"
        items.append(item)
    return items
