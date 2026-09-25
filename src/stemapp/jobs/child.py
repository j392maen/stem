"""分割の子プロセス（`python -m stemapp.jobs.child <job_id>`）。

ワーカーが1ジョブごとに起動する。job_id だけを受け取り、DB からジョブを読んで
T02 の `separate_track` を実行する（進捗は JOB に書く）。最後の段階で配信用データ
（Opus と波形 peaks）を作り、続けて曲の拍を解析する（T10。失敗してもジョブは done のまま、
警告を JOB.beat_warning に残す）。キャンセル時はワーカーがこのプロセスを終了させる
（GPU メモリを確実に解放するため）。

終了コード: 0=完了、1=失敗（JOB は failed）、2=想定外の例外、3=親（ワーカー）がいなくなった。

ワーカーが強制終了されたときは、Windows ではワーカーの Job Object により OS がこのプロセスを
終了させる。Linux では `watch_parent()` が親の終了に気づいて終了する（`stemapp.proc`）。

`--fake` を付けると FakeSeparator・FakeBeatAnalyzer とダミーのエンコーダで動く
（テスト用。GPU・ffmpeg 不要）。
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.orm import Session

from stemapp.audio import FfmpegRunner
from stemapp.beats.base import BeatAnalyzer
from stemapp.beats.service import STAGE_BEATS, analyze_job_beats
from stemapp.config import Settings, get_settings
from stemapp.db import make_engine, make_session_factory
from stemapp.delivery import create_delivery_files, fake_encoder
from stemapp.jobs.queue import ACTIVE_STATUSES, FAILED, finish_job
from stemapp.models import SeparationJob
from stemapp.proc import watch_parent
from stemapp.separation.base import DEVICE_CPU, DEVICE_CUDA, Separator
from stemapp.separation.pipeline import ProgressCallback, SeparationError, separate_track

log = logging.getLogger(__name__)

EXIT_DONE = 0
EXIT_FAILED = 1
EXIT_ERROR = 2


def run_job(
    settings: Settings,
    job_id: int,
    separator: Separator,
    *,
    encoder: FfmpegRunner | None = None,
    beat_analyzer: BeatAnalyzer | None = None,
) -> int:
    """登録済みのジョブを実行する（子プロセスの本体。テストではそのまま呼べる）。

    beat_analyzer を渡すと、配信用データの後に拍を解析する（None なら拍の段階を飛ばす）。
    """
    engine = make_engine(settings.db_path)
    factory = make_session_factory(engine)
    try:
        with factory() as session:
            job = session.get(SeparationJob, job_id)
            if job is None:
                log.error("ジョブが見つかりません（job %d）。", job_id)
                return EXIT_FAILED
            device = DEVICE_CPU if job.run_on == "cpu" else DEVICE_CUDA

            def postprocess(
                s: Session, st: Settings, jid: int, progress: ProgressCallback
            ) -> None:
                if beat_analyzer is None:
                    create_delivery_files(s, st, jid, encoder=encoder, progress=progress)
                    return
                create_delivery_files(
                    s, st, jid, encoder=encoder, progress=lambda p, m: progress(0.7 * p, m)
                )
                progress(0.75, STAGE_BEATS)
                analyze_job_beats(s, st, jid, beat_analyzer)

            try:
                separate_track(
                    session,
                    settings,
                    job.track_id,
                    separator,
                    job_id=job_id,
                    device=device,
                    postprocess=postprocess,
                )
                return EXIT_DONE
            except SeparationError as e:
                # 分割に入る前の失敗（プリセットが無い等）は JOB がまだ running のまま
                session.rollback()
                job = session.get(SeparationJob, job_id)
                if job is not None and job.status in ACTIVE_STATUSES:
                    finish_job(session, settings, job_id, FAILED, message=str(e))
                log.error("job %d を実行できませんでした: %s", job_id, e)
                return EXIT_FAILED
    finally:
        engine.dispose()


def _real_separator(settings: Settings) -> Separator:
    from stemapp.separation.audio_separator_backend import AudioSeparatorBackend

    return AudioSeparatorBackend(
        models_dir=settings.models_dir, work_dir=settings.cache_dir / "audio-separator"
    )


def _real_beat_analyzer(settings: Settings) -> BeatAnalyzer:
    from stemapp.beats.beat_this_backend import BeatThisAnalyzer

    return BeatThisAnalyzer(settings.models_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stemapp 分割の子プロセス")
    parser.add_argument("job_id", type=int)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--fake", action="store_true", help="FakeSeparator で動かす（テスト用）")
    parser.add_argument("--fake-delay", type=float, default=0.0)
    parser.add_argument("--fake-fail", action="store_true")
    parser.add_argument(
        "--preimport", action="append", default=[],
        help="起動直後に import するモジュール（テスト用）",
    )
    args = parser.parse_args(argv)
    watch_parent()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("audio_separator").setLevel(logging.WARNING)

    settings = (
        Settings(data_dir=args.data_dir) if args.data_dir is not None else get_settings()
    )
    encoder: FfmpegRunner | None = None
    separator: Separator
    beat_analyzer: BeatAnalyzer
    if args.fake:
        from stemapp.beats.fake import FakeBeatAnalyzer
        from stemapp.seed import SW
        from stemapp.separation.fake import FakeSeparator

        beat_analyzer = FakeBeatAnalyzer()
        separator = FakeSeparator(
            delay_sec=args.fake_delay, fail_models={SW} if args.fake_fail else ()
        )
        encoder = fake_encoder
    else:
        separator = _real_separator(settings)
        beat_analyzer = _real_beat_analyzer(settings)
    try:
        for name in args.preimport:
            importlib.import_module(name)
        return run_job(
            settings, args.job_id, separator, encoder=encoder, beat_analyzer=beat_analyzer
        )
    except Exception:
        log.exception("job %d で想定外のエラー", args.job_id)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
