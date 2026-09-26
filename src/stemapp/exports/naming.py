"""書き出しのファイル名と、ダウンロード時の Content-Disposition。

規則:
- 使えない文字（`\\ / : * ? " < > |`、制御文字）は飛ばす。
- 末尾のピリオド・空白を除く。
- Windows の予約名（CON, PRN, AUX, NUL, COM1〜9, LPT1〜9）は末尾に `_` を付ける。
- 拡張子を含めて 100 文字まで（超えた分は名前の末尾を切る）。
"""

from __future__ import annotations

import re
from urllib.parse import quote

MAX_FILENAME_LEN = 100
FALLBACK_NAME = "export"
MIX_JOINER = "＋"

_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')
_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_NOT_IN_QUOTED = frozenset('"\\;')


def _clean(text: str) -> str:
    return _UNSAFE.sub("", text)


def _fix_reserved(base: str) -> str:
    """最初のピリオドまでが予約名なら、その後ろに `_` を入れる（"CON" → "CON_"）。"""
    head, dot, rest = base.partition(".")
    if head.strip().upper() in _RESERVED:
        return f"{head}_{dot}{rest}"
    return base


def safe_filename(base: str, extension: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """base（拡張子なし）と extension（"wav" など）から、保存に使えるファイル名を作る。"""
    ext = _clean(extension).strip(" .")
    suffix = f".{ext}" if ext else ""
    room = max(2, max_len - len(suffix))
    name = _clean(base)[:room].rstrip(" .") or FALLBACK_NAME
    # 予約名の `_` は名前の先頭側（最初のピリオドの前）に入るので、末尾を切っても残る
    name = _fix_reserved(name)[:room].rstrip(" .")
    return name + suffix


def export_base_name(track_title: str, label: str) -> str:
    """`<曲名> - <ラベル>`（ラベルは stem の表示名・組み合わせ名・"stems"）。"""
    title = (track_title or "").strip() or "曲"
    return f"{title} - {label}"


def mix_label(names: list[str]) -> str:
    """mix のラベル（選んだ stem の表示名を「＋」でつなぐ）。"""
    return MIX_JOINER.join(names)


def _ascii_fallback(filename: str) -> str:
    """RFC 5987 に対応していない相手向けの filename=（ASCII だけ。それ以外は _）。"""
    out = "".join(
        c if 0x20 <= ord(c) < 0x7F and c not in _NOT_IN_QUOTED else "_" for c in filename
    )
    return out or FALLBACK_NAME


def content_disposition(filename: str, disposition: str = "attachment") -> str:
    """`attachment; filename="..."; filename*=UTF-8''...`（日本語名が化けないよう RFC 5987）。"""
    encoded = quote(filename, safe="")
    return f"{disposition}; filename=\"{_ascii_fallback(filename)}\"; filename*=UTF-8''{encoded}"
