"""コマンドライン（`stemapp ...`）。"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stemapp.config import Settings, get_settings
from stemapp.doctor import CheckResult, Status, exit_code, run_checks

if TYPE_CHECKING:
    from stemapp.beats.base import BeatAnalyzer
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


WORKER_STOP_WAIT_SEC = 20.0


def worker_stop_file(settings: Settings) -> Path:
    """serve が起動したワーカーへの停止の合図（このファイルができたら止まる）。"""
    return settings.data_root / "run" / f"worker-{os.getpid()}.stop"


def _start_worker_process(stop_file: Path) -> subprocess.Popen[bytes]:
    """ワーカーを別プロセスで起動する（このプロセスが終わると OS が一緒に終了させる）。"""
    from stemapp.proc import start_bound_process

    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.unlink(missing_ok=True)
    return start_bound_process(
        [sys.executable, "-m", "stemapp.cli", "worker", "--stop-file", str(stop_file)]
    )


def _stop_worker_process(proc: subprocess.Popen[bytes], stop_file: Path) -> None:
    """停止ファイルを置いてワーカーに後始末させ、止まるのを待つ。止まらなければ強制終了。"""
    try:
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.write_text("stop", encoding="utf-8")
        try:
            proc.wait(timeout=WORKER_STOP_WAIT_SEC)
        except subprocess.TimeoutExpired:
            typer.echo("ワーカーが止まらないため強制終了します。")
            proc.kill()
            proc.wait()
    finally:
        stop_file.unlink(missing_ok=True)


@app.command()
def serve() -> None:
    """Web サーバーを起動する（設定の host/port）。ワーカーも一緒に起動し、終了時に止める。"""
    import uvicorn

    from stemapp.app import create_app
    from stemapp.db import init_db, make_engine, make_session_factory
    from stemapp.seed import seed

    settings = _settings()
    # DB の作成・列の追加・初期データ投入は、ワーカーを起動する前にここで済ませる
    # （サーバーとワーカーが同時に初回作成・列追加をして競合しないように）
    engine = make_engine(settings.db_path)
    try:
        init_db(engine)
        with make_session_factory(engine)() as session:
            seed(session)
    finally:
        engine.dispose()
    stop_file = worker_stop_file(settings)
    worker_proc = _start_worker_process(stop_file)
    try:
        uvicorn.run(
            create_app(settings),
            host=settings.host,
            port=settings.port,
            timeout_graceful_shutdown=3,
        )
    finally:
        _stop_worker_process(worker_proc, stop_file)


@app.command()
def worker(
    stop_file: Annotated[
        Path | None,
        typer.Option("--stop-file", hidden=True, help="このファイルができたら止まる"),
    ] = None,
) -> None:
    """分割ワーカーだけを起動する（queued のジョブを1件ずつ実行する）。Ctrl+C で止まる。"""
    from stemapp.jobs.worker import (
        Worker,
        WorkerLock,
        WorkerLockError,
        subprocess_beat_runner,
        subprocess_launcher,
    )
    from stemapp.proc import watch_parent

    _setup_logging()
    watch_parent()  # serve から起動されたとき（Linux）: serve がいなくなったら終了する
    settings = _settings()
    lock = WorkerLock(settings.data_root / "worker.lock")
    try:
        lock.acquire()
    except WorkerLockError as e:
        typer.echo(str(e))
        raise typer.Exit(1) from e
    try:
        with _db_engine(settings) as factory:
            w = Worker(
                settings,
                factory,
                subprocess_launcher(settings),
                stop_file=stop_file,
                beat_runner=subprocess_beat_runner(settings),
            )
            try:
                w.run_forever()
            except KeyboardInterrupt:
                typer.echo("ワーカーを止めました。")
    finally:
        lock.release()


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
def _db_engine(settings: Settings) -> Iterator[sessionmaker[Session]]:
    """DB を開き（無ければ作って初期データを入れ）、セッションの作り方を返す。"""
    from stemapp.db import init_db, make_engine, make_session_factory
    from stemapp.models import SeparationPreset
    from stemapp.seed import seed

    engine = make_engine(settings.db_path)
    try:
        init_db(engine)
        factory = make_session_factory(engine)
        with factory() as session:
            if session.scalars(select(SeparationPreset)).first() is None:
                seed(session)
        yield factory
    finally:
        engine.dispose()


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
        console.print(
            f"[bold red]{e.message}[/bold red]（理由コード: {e.code}）", soft_wrap=True
        )
        if e.detail:
            console.print(f"詳細:\n{e.detail}", markup=False, soft_wrap=True)
        raise typer.Exit(1) from e
    except Exception as e:  # 正規化の失敗など
        console.print(f"[bold red]取り込みに失敗しました: {e}[/bold red]", soft_wrap=True)
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
    """1曲を取り込み、stem に分割して保存する。配信用データ（Opus・波形）も作る。"""
    import time

    from stemapp import audio
    from stemapp.delivery import create_delivery_files, missing_delivery, rebuild_delivery_files
    from stemapp.separation.base import DEVICE_CPU, DEVICE_CUDA
    from stemapp.separation.pipeline import SeparationError, load_plan, separate_track

    def postprocess(
        s: Session, st: Settings, job_id: int, progress: Callable[[float, str], None]
    ) -> None:
        # audio.run_ffmpeg は呼ぶときに探す（テストで差し替えられるように）
        create_delivery_files(s, st, job_id, encoder=audio.run_ffmpeg, progress=progress)

    _setup_logging()
    settings = _settings()
    console = Console()
    t0 = time.perf_counter()
    with _db_session(settings) as session:
        try:
            load_plan(session, preset)  # プリセットの誤りは取り込む前に知らせる
        except SeparationError as e:
            console.print(f"[bold red]{e}[/bold red]", soft_wrap=True)
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
                progress=lambda p, stage: console.print(
                        f"[{p * 100:5.1f}%] {stage}", markup=False
                    ),
                started=t0,
                postprocess=postprocess,
            )
            if result.skipped and missing_delivery(session, result.job_id):
                # 分割済みでも配信用データが欠けていれば作る（T04 以前に CLI で分割した曲など）
                console.print("配信用データ（Opus・波形）を作成しています。")
                rebuild_delivery_files(session, settings, result.job_id, encoder=audio.run_ffmpeg)
        except SeparationError as e:
            console.print(f"[bold red]{e}[/bold red]", soft_wrap=True)
            raise typer.Exit(1) from e
        except Exception as e:
            console.print(f"[bold red]失敗しました: {e}[/bold red]", soft_wrap=True)
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
            console.print(f"[bold red]失敗しました: {e}[/bold red]", soft_wrap=True)
            raise typer.Exit(1) from e
    render_bench(report, console)
    path = save_report(settings, report)
    console.print(f"結果を保存しました: {path}")


# --- 拍 ----------------------------------------------------------------------------


def make_beat_analyzer(settings: Settings, cpu: bool = False) -> BeatAnalyzer:
    """実際の拍の解析器（beat_this）。テストでは差し替える。"""
    from stemapp.beats.beat_this_backend import BeatThisAnalyzer

    return BeatThisAnalyzer(settings.models_dir, device="cpu" if cpu else None)


@app.command()
def beats(
    track_id: Annotated[int, typer.Argument(help="曲の番号（track_id）")],
    force: Annotated[bool, typer.Option("--force", help="解析済みでも解析し直す")] = False,
    cpu: Annotated[bool, typer.Option("--cpu", help="GPU を使わず CPU で解析する")] = False,
) -> None:
    """曲の拍・小節の頭を解析して保存し、区間ごとの BPM と拍子を表示する。"""
    from stemapp.beats.service import analyze_track
    from stemapp.beats.tempo import tempo_segments

    _setup_logging()
    settings = _settings()
    console = Console()
    with _db_session(settings) as session:
        try:
            outcome = analyze_track(
                session, settings, track_id, make_beat_analyzer(settings, cpu), force=force
            )
            session.commit()
        except Exception as e:
            console.print(f"[bold red]拍を解析できませんでした: {e}[/bold red]", soft_wrap=True)
            raise typer.Exit(1) from e
        grid = outcome.grid
        beat_list = list(grid.beats_json or [])
        if outcome.skipped:
            console.print("解析済みです（解析し直すには --force）。保存済みの結果を表示します。")
        segments = tempo_segments(beat_list)
        table = Table(title=f"track {track_id} の区間ごとの BPM")
        table.add_column("開始", justify="right")
        table.add_column("終了", justify="right")
        table.add_column("BPM", justify="right")
        for seg in segments:
            table.add_row(f"{seg.start_sec:.2f}", f"{seg.end_sec:.2f}", f"{seg.bpm:.2f}")
        console.print(table)
        console.print(
            f"拍 {len(beat_list)}・小節の頭 {len(grid.downbeats_json or [])}・"
            f"拍子 {grid.time_signature}/4・解析器 {grid.analyzer}"
        )
        if outcome.result is not None:
            console.print(
                f"装置: {outcome.result.device}　解析: {outcome.result.seconds:.1f} 秒"
                f"（読み込み等を含む全体 {outcome.seconds:.1f} 秒）"
            )


# --- 書き出し ------------------------------------------------------------------------


def _free_path(path: Path) -> Path:
    """同じ名前のファイルがあれば「名前 (2).拡張子」のように空いている名前にする。"""
    if not path.exists():
        return path
    n = 2
    while True:
        cand = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not cand.exists():
            return cand
        n += 1


@app.command("export")
def export_cmd(
    job_id: Annotated[int, typer.Argument(help="ジョブの番号（job_id）")],
    export_type: Annotated[
        str, typer.Option("--type", help="single（stem 1つ）/ all（全部を ZIP）/ mix（ミックス）")
    ],
    fmt: Annotated[str, typer.Option("--format", help="wav / flac / mp3")] = "wav",
    stems: Annotated[
        list[str] | None,
        typer.Option(
            "--stem",
            help="stem の code。single は1つ、mix は複数指定できる（--preset を使わないとき）",
        ),
    ] = None,
    preset: Annotated[
        str | None, typer.Option("--preset", help="mix に使う組み合わせプリセットの名前")
    ] = None,
    parents_only: Annotated[
        bool, typer.Option("--parents-only", help="all: 子に分かれていても親の stem だけにする")
    ] = False,
    output: Annotated[
        Path, typer.Option("-o", "--output", help="出力先フォルダ（無ければ作る）")
    ] = Path("."),
) -> None:
    """分割した stem を書き出す（出力先はデータフォルダではなく -o のフォルダ）。"""
    import tempfile

    from stemapp.exports import ExportError, ExportRequest, plan_export
    from stemapp.exports.service import render_plan
    from stemapp.models import ListenPreset

    _setup_logging()
    settings = _settings()
    console = Console()
    codes = list(stems or [])
    with _db_session(settings) as session:
        preset_id: int | None = None
        if preset is not None:
            if export_type != "mix":
                console.print("[bold red]--preset は --type mix のときだけ使えます。[/bold red]")
                raise typer.Exit(1)
            found = session.scalars(
                select(ListenPreset).where(
                    ListenPreset.name == preset, ListenPreset.hidden.is_(False)
                )
            ).first()
            if found is None:
                console.print(f"[bold red]組み合わせ「{preset}」が見つかりません。[/bold red]")
                raise typer.Exit(1)
            preset_id = found.listen_preset_id
        if export_type == "single" and len(codes) != 1:
            console.print("[bold red]single では --stem を1つ指定してください。[/bold red]")
            raise typer.Exit(1)
        req = ExportRequest(
            export_type=export_type,
            format=fmt,
            stem_code=codes[0] if export_type == "single" else None,
            parents_only=parents_only,
            listen_preset_id=preset_id,
            stems=[(c, 0.0) for c in codes] if export_type == "mix" else (),
        )
        try:
            plan = plan_export(session, job_id, req)
            # EXPORT の行は作らない（data/exports の書き出しとは別。片付けの対象にしない）
            output.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=output, prefix=".stemapp-export-") as tmp:
                result = render_plan(
                    session, settings, plan, Path(tmp),
                    progress=lambda p, stage: console.print(
                        f"[{p * 100:5.1f}%] {stage}", markup=False
                    ),
                )
                dst = _free_path(output / result.path.name)
                result.path.replace(dst)
        except ExportError as e:
            console.print(f"[bold red]{e}[/bold red]", soft_wrap=True)
            raise typer.Exit(1) from e
        except Exception as e:
            console.print(f"[bold red]書き出しに失敗しました: {e}[/bold red]", soft_wrap=True)
            raise typer.Exit(1) from e
    console.print(f"書き出しました: {dst}", markup=False, soft_wrap=True)
    console.print(f"大きさ: {result.bytes / 1024 / 1024:.1f} MB")
    if result.mix_gain_db:
        console.print(f"音割れしないよう、全体を {result.mix_gain_db:.1f} dB 下げました。")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
