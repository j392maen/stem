"""T16: cp932 の出力先でも CLI が落ちないこと。"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from rich.console import Console
from rich.table import Table

from stemapp import cli
from stemapp.config import Settings
from test_folder_migration import _cli_setup

NAME = "春嵐 ⧸ 初音ミク 🎵"


def _cp932() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp932", newline="")


def _text(stream: io.TextIOWrapper) -> str:
    stream.flush()
    buf = stream.buffer
    assert isinstance(buf, io.BytesIO)
    return buf.getvalue().decode("cp932")


def test_plain_cp932_stream_fails() -> None:
    """前提の確認: 何もしなければ cp932 では書けない。"""
    with pytest.raises(UnicodeEncodeError):
        _cp932().write(NAME)


def test_safe_stream_prints_table() -> None:
    out = _cp932()
    cli.make_stream_safe(out)
    table = Table("曲名")
    table.add_row(NAME)
    Console(file=out, width=80).print(table)
    text = _text(out)
    assert "春嵐 ? 初音ミク ?" in text  # 日本語はそのまま、表せない文字は ?


def test_utf8_stream_untouched() -> None:
    out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    cli.make_stream_safe(out)
    assert out.errors == "strict"
    out.write(NAME)
    out.flush()
    assert out.buffer.getvalue().decode("utf-8") == NAME  # type: ignore[attr-defined]


def test_stream_without_reconfigure_is_ignored() -> None:
    cli.make_stream_safe(object())  # 例外にならない


def test_main_migrate_dry_run_on_cp932(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cli_setup(settings, tmp_path, monkeypatch, "春嵐 ⧸ 初音ミク")
    out, err = _cp932(), _cp932()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "argv", ["stemapp", "migrate-folders", "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code in (0, None), _text(err)
    text = _text(out)
    assert "stems/春嵐 ? 初音ミク/fast" in text
    assert "dry-run" in text
