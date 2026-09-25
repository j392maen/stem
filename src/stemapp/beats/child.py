"""拍の解析の子プロセス（`python -m stemapp.beats.child <job_id>`）。

ワーカーが「配信用データの作り直し」（postprocess）の中で、拍が無い曲のために起動する。
GPU（torch・beat_this）をワーカー本体に読み込まず、終われば GPU メモリも確実に解放される。
解析の失敗は `analyze_job_beats` が JOB.beat_warning に書く（終了コード 0）。

終了コード: 0=完了（解析の失敗を含む）、1=ジョブが無い、2=想定外の例外、3=親がいなくなった。

`--fake` を付けると FakeBeatAnalyzer で動く（テスト用）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from stemapp.beats.base import BeatAnalyzer
from stemapp.beats.service import analyze_job_beats
from stemapp.config import Settings, get_settings
from stemapp.db import make_engine, make_session_factory
from stemapp.models import SeparationJob
from stemapp.proc import watch_parent

log = logging.getLogger(__name__)

EXIT_DONE = 0
EXIT_NOT_FOUND = 1
EXIT_ERROR = 2


def run_beats(
    settings: Settings, job_id: int, analyzer: BeatAnalyzer, *, force: bool = False
) -> int:
    engine = make_engine(settings.db_path)
    try:
        with make_session_factory(engine)() as session:
            if session.get(SeparationJob, job_id) is None:
                log.error("ジョブが見つかりません（job %d）。", job_id)
                return EXIT_NOT_FOUND
            analyze_job_beats(session, settings, job_id, analyzer, force=force)
            return EXIT_DONE
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stemapp 拍の解析の子プロセス")
    parser.add_argument("job_id", type=int)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="拍があっても解析し直す")
    parser.add_argument("--fake", action="store_true", help="FakeBeatAnalyzer で動かす（テスト用）")
    args = parser.parse_args(argv)
    watch_parent()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = (
        Settings(data_dir=args.data_dir) if args.data_dir is not None else get_settings()
    )
    analyzer: BeatAnalyzer
    if args.fake:
        from stemapp.beats.fake import FakeBeatAnalyzer

        analyzer = FakeBeatAnalyzer()
    else:
        from stemapp.beats.beat_this_backend import BeatThisAnalyzer

        analyzer = BeatThisAnalyzer(settings.models_dir)
    try:
        return run_beats(settings, args.job_id, analyzer, force=args.force)
    except Exception:
        log.exception("job %d の拍の解析で想定外のエラー", args.job_id)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
