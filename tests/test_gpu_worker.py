"""ワーカー経由（実際の子プロセス・AudioSeparatorBackend）で分割するテスト。`-m gpu` で実行する。"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from job_helpers import make_track
from stemapp.config import Settings
from stemapp.db import make_session_factory
from stemapp.jobs import enqueue_full_job
from stemapp.jobs.worker import Worker, subprocess_launcher
from stemapp.models import SeparationJob, Stem, StemRendition
from stemapp.seed import seed

pytestmark = pytest.mark.gpu


def test_fast_via_worker_child_process(
    session: Session, settings: Settings, engine: Engine, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = Settings()  # .env の設定（モデルの置き場）
    # 子プロセスは環境変数で実際のモデルの置き場を使う（データは一時フォルダ）
    monkeypatch.setenv("STEMAPP_MODEL_DIR", str(real.models_dir))
    seed(session)
    track_id = make_track(session, settings, tmp_path, seconds=10.0)
    job_id = enqueue_full_job(session, track_id, "fast").job.job_id

    factory = make_session_factory(engine)
    worker = Worker(settings, factory, subprocess_launcher(settings))
    assert worker.run_one() == job_id

    with factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None
        assert job.status == "done", job.error_message
        stems = s.scalars(select(Stem).where(Stem.job_id == job_id)).all()
        assert len(stems) == 8
        streams = s.scalars(
            select(StemRendition).where(
                StemRendition.stem_id.in_([st.stem_id for st in stems]),
                StemRendition.purpose == "stream",
            )
        ).all()
        assert len(streams) == 8 and all(r.bytes and r.bytes > 1000 for r in streams)
