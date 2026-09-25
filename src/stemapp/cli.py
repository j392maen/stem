"""コマンドライン（`stemapp ...`）。"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from stemapp.config import Settings, get_settings
from stemapp.doctor import CheckResult, Status, exit_code, run_checks

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
