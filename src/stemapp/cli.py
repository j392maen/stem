"""コマンドライン（`stemapp ...`）。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.config import Settings, get_settings
from stemapp.doctor import CheckResult, Status, exit_code, run_checks

if TYPE_CHECKING:
    from stemapp.ingest import ImportResult
    from stemapp.ingest.url import YtDlpRunner
    from stemapp.separation.base import Separator
    from stemapp.separation.bench import BenchReport
    from stemapp.separation.pipeline import SeparateResult

app = typer.Typer(help="stemapp: 楽曲 stem 分割・再生アプリ", no_args_is_help=True)

_STATUS_STYLE = {Status.OK: "green", Status.WARN: "yellow", Status.NG: "bold red"}


def _settings() -> Settings:
    return get_settings()


@app.command("init-db")
def init_db_cmd() -> None:
    """DB を作成し、初期データを投入する（何度実行しても同じ結果）。"""
    from stemapp.db import init_db, make_engine, make_session_factory
    from stemapp.seed import seed

    settings = _settings()
    engine = make_engine(settings.db_path)
    try:
        init_db(engine)
        with make_session_factory(engine)() as session:
            seed(session)
    finally:
        engine.dispose()
    typer.echo(f"DB を準備しました: {settings.db_path}")


def render_results(results: list[CheckResult], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="stemapp 環境診断")
    table.add_column("項目")
    table.add_column("結果")
    table.add_column("内容")
    table.add_column("対処のヒント")
    for r in results:
        style = _STATUS_STYLE[r.status]
        table.add_row(r.name, f"[{style}]{r.status.value}[/{style}]", r.detail, r.hint)
    console.print(table)


@app.command()
def doctor() -> None:
    """環境を診断する。NG があれば終了コード 1。"""
    results = run_checks(_settings())
    render_results(results)
    code = exit_code(results)
    if code:
        typer.echo("NG の項目があります。対処のヒントを確認してください。")
    raise typer.Exit(code)


@app.command()
def serve() -> None:
    """Web サーバーを起動する（設定の host/port）。"""
    import uvicorn

    from stemapp.app import create_app

    settings = _settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


# --- 分離 --------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # audio-separator の細かいログは抑える
    logging.getLogger("audio_separator").setLevel(logging.WARNING)


def make_separator(settings: Settings) -> Separator:
    """実際の分離器（audio-separator）。テストでは差し替える。"""
    from stemapp.separation.audio_separator_backend import AudioSeparatorBackend

    return AudioSeparatorBackend(
        models_dir=settings.models_dir, work_dir=settings.cache_dir / "audio-separator"
    )


@contextmanager
def _db_session(settings: Settings) -> Iterator[Session]:
    """DB を開く。テーブルが無ければ作り、プリセットが無ければ初期データを入れる。"""
    from stemapp.db import init_db, make_engine, make_session_factory
    from stemapp.models import SeparationPreset
    from stemapp.seed import seed

    engine = make_engine(settings.db_path)
    try:
        init_db(engine)
        with make_session_factory(engine)() as session:
            if session.scalars(select(SeparationPreset)).first() is None:
                seed(session)
            yield session
    finally:
        engine.dispose()


def render_separate_result(result: SeparateResult, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title=f"stem 一覧（track {result.track_id} / job {result.job_id}）")
    table.add_column("stem")
    table.add_column("rms_db", justify="right")
    table.add_column("無音")
    table.add_column("ファイル")
    for st in result.stems:
        indent = "  └ " if st.parent_code else ""
        name = f"{indent}{st.display_name}（{st.code}）"
        if st.is_residual:
            name += " ※残り"
        level = "-" if st.rms_db is None else f"{st.rms_db:.1f}"
        table.add_row(name, level, "はい" if st.is_silent else "", str(st.file_path))
    console.print(table)


def make_ytdlp_runner(settings: Settings) -> YtDlpRunner:
    """yt-dlp の実行器（exe があれば exe 版）。テストでは差し替える。"""
    from stemapp.ingest.url import choose_runner

    return choose_runner(settings)


def _import_target(
    session: Session, settings: Settings, target: str, console: Console
) -> ImportResult:
    """ファイルか URL を取り込む。失敗したら理由を表示して終了コード 1。"""
    from stemapp.ingest import import_file
    from stemapp.ingest.url import UrlImportError, fetch_url, is_url

    try:
        if is_url(target):
            console.print(f"URL から取得しています: {target}")
            return fetch_url(session, settings, target, make_ytdlp_runner(settings))
        return import_file(session, settings, Path(target))
    except UrlImportError as e:
        console.print(f"[bold red]{e.message}[/bold red]（理由コード: {e.code}）")
        if e.detail:
            console.print(f"詳細:\n{e.detail}", markup=False)
        raise typer.Exit(1) from e
    except Exception as e:  # 正規化の失敗など
        console.print(f"[bold red]取り込みに失敗しました: {e}[/bold red]")
        raise typer.Exit(1) from e


def render_import_result(result: ImportResult, console: Console) -> None:
    kind = "新規" if result.is_new else "既存"
    console.print(f"track_id: {result.track_id}")
    console.print(f"登録: {kind}")
    console.print(f"タイトル: {result.title}", markup=False)
    if result.has_done_full_job:
        console.print("この曲は分割済みです。")


@app.command("import")
def import_cmd(
    target: Annotated[str, typer.Argument(help="音声ファイルのパス、または URL（http/https）")],
) -> None:
    """曲を取り込む（分割はしない）。同じ音の曲が登録済みなら新しく作らない。"""
    _setup_logging()
    settings = _settings()
    console = Console()
    with _db_session(settings) as session:
        result = _import_target(session, settings, target, console)
    render_import_result(result, console)


@app.command()
def separate(
    target: Annotated[
        str,
        typer.Argument(help="分割する音声ファイル（mp3/m4a/flac/wav など）または URL"),
    ],
    preset: Annotated[
        str | None,
        typer.Option("--preset", help="品質プリセット fast / standard / best（省略時は既定）"),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="分割済みでも分割し直す")] = False,
    cpu: Annotated[bool, typer.Option("--cpu", help="GPU を使わず CPU で実行する")] = False,
) -> None:
    """1曲を取り込み、stem に分割して保存する。"""
    import time

    from stemapp.separation.base import DEVICE_CPU, DEVICE_CUDA
    from stemapp.separation.pipeline import SeparationError, load_plan, separate_track

    _setup_logging()
    settings = _settings()
    console = Console()
    t0 = time.perf_counter()
    with _db_session(settings) as session:
        try:
            load_plan(session, preset)  # プリセットの誤りは取り込む前に知らせる
        except SeparationError as e:
            console.print(f"[bold red]{e}[/bold red]")
            raise typer.Exit(1) from e
        imported = _import_target(session, settings, target, console)
        try:
            result = separate_track(
                session,
                settings,
                imported.track_id,
                make_separator(settings),
                preset_code=preset,
                force=force,
                device=DEVICE_CPU if cpu else DEVICE_CUDA,
                progress=lambda p, stage: console.print(f"[{p * 100:5.1f}%] {stage}"),
                started=t0,
            )
        except SeparationError as e:
            console.print(f"[bold red]{e}[/bold red]")
            raise typer.Exit(1) from e
        except Exception as e:
            console.print(f"[bold red]失敗しました: {e}[/bold red]")
            raise typer.Exit(1) from e
        if result.skipped:
            console.print(
                "この曲は分割済みです（分割し直すには --force）。保存済みの stem を表示します。"
            )
        render_separate_result(result, console)
        console.print(f"所要時間: {result.seconds:.1f} 秒")


@app.command("ytdlp-update")
def ytdlp_update() -> None:
    """yt-dlp.exe を更新する（yt-dlp.exe -U）。"""
    from stemapp.ingest import url as url_mod

    settings = _settings()
    res = url_mod.update_ytdlp(settings)
    label = {
        url_mod.UPDATE_UPDATED: "更新しました",
        url_mod.UPDATE_LATEST: "最新です",
        url_mod.UPDATE_FAILED: "更新に失敗しました",
    }[res.status]
    typer.echo(f"yt-dlp: {label}")
    if res.detail:
        typer.echo(res.detail)
    if res.status == url_mod.UPDATE_FAILED:
        raise typer.Exit(1)


def render_bench(report: BenchReport, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title=f"bench（{report.duration_sec:.1f} 秒の音源, {report.device}）")
    table.add_column("プリセット")
    table.add_column("ステップ")
    table.add_column("時間(秒)", justify="right")
    table.add_column("GPU最大(MB)", justify="right")
    table.add_column("備考")

    def mb(v: float | None) -> str:
        return "-" if v is None else f"{v:,.0f}"

    for pb in report.presets:
        table.add_row(pb.preset, "全体", f"{pb.seconds:.1f}", mb(pb.peak_memory_mb), "")
        for st in pb.steps:
            note = f"{st.device}, チャンク {st.chunk_scale:g} 倍"
            if st.attempts:
                note += f", やり直し {len(st.attempts)} 回"
            table.add_row(
                "",
                f"{st.order}. {st.model_filename}（{st.role}）",
                f"{st.seconds:.1f}",
                mb(st.peak_memory_mb),
                note,
            )
    console.print(table)


@app.command()
def bench(
    file: Annotated[Path, typer.Argument(help="計測に使う音声ファイル")],
    presets: Annotated[
        str, typer.Option("--presets", help="カンマ区切りのプリセット")
    ] = "fast,standard",
    cpu: Annotated[bool, typer.Option("--cpu", help="GPU を使わず CPU で実行する")] = False,
) -> None:
    """プリセットごとの処理時間と GPU メモリ最大使用量を測る（結果は DB に登録しない）。"""
    from stemapp.separation.base import DEVICE_CPU, DEVICE_CUDA
    from stemapp.separation.bench import run_bench, save_report

    _setup_logging()
    settings = _settings()
    console = Console()
    codes = [c.strip() for c in presets.split(",") if c.strip()]
    with _db_session(settings) as session:
        try:
            report = run_bench(
                session,
                settings,
                file,
                make_separator(settings),
                codes,
                device=DEVICE_CPU if cpu else DEVICE_CUDA,
            )
        except Exception as e:
            console.print(f"[bold red]失敗しました: {e}[/bold red]")
            raise typer.Exit(1) from e
    render_bench(report, console)
    path = save_report(settings, report)
    console.print(f"結果を保存しました: {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
