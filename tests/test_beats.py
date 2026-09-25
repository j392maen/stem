"""拍の解析（T10）: 区間 BPM・拍子の計算、Fake 解析器、後処理の流れ（GPU 不要）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from job_helpers import make_track, sync_launcher
from stemapp.beats import (
    FakeBeatAnalyzer,
    analyze_job_beats,
    analyze_track,
    beats_payload,
    clean_beats,
    estimate_time_signature,
    get_grid,
    segment_at,
    tempo_segments,
)
from stemapp.beats.service import STAGE_BEATS
from stemapp.config import Settings
from stemapp.db import make_session_factory
from stemapp.delivery import fake_encoder
from stemapp.jobs import enqueue_full_job, request_postprocess
from stemapp.jobs.worker import Worker, subprocess_beat_runner
from stemapp.models import BeatGrid, SeparationJob
from stemapp.seed import seed

# --- 区間 BPM・拍子（純粋関数） ------------------------------------------------------


def regular(bpm_sections: list[tuple[float, float]], until: float, per_bar: int = 4):
    """[(開始秒, BPM)] の規則的な拍と小節の頭。"""
    res = FakeBeatAnalyzer(bpm_sections, beats_per_bar=per_bar).analyze(
        np.zeros((int(until * 44100), 2), np.float32)
    )
    return np.array(res.beats), np.array(res.downbeats)


def quantize(beats: np.ndarray, seed_: int = 0, jitter: float = 0.008) -> np.ndarray:
    """beat_this と同じ 20ms 刻みに丸め、小さな揺れを足す。"""
    rng = np.random.default_rng(seed_)
    return np.round((beats + rng.normal(0, jitter, len(beats))) / 0.02) * 0.02


def bpms(beats) -> list[float]:
    return [round(s.bpm, 1) for s in tempo_segments(beats)]


def test_constant_tempo() -> None:
    beats, downbeats = regular([(0, 128)], 60)
    segs = tempo_segments(beats)
    assert len(segs) == 1
    assert segs[0].bpm == pytest.approx(128, abs=0.01)
    assert segs[0].start_sec == pytest.approx(0) and segs[0].end_sec == pytest.approx(beats[-1])
    assert estimate_time_signature(beats, downbeats) == 4


def test_tempo_change_120_to_140() -> None:
    beats, _ = regular([(0, 120), (30, 140)], 60)
    segs = tempo_segments(beats)
    assert [round(s.bpm, 2) for s in segs] == [120.0, 140.0]
    assert segs[0].end_sec == pytest.approx(30.0, abs=0.01)
    assert segs[1].start_sec == segs[0].end_sec
    assert segment_at(segs, 10).bpm == pytest.approx(120)
    assert segment_at(segs, 45).bpm == pytest.approx(140)
    assert segment_at(segs, -1) is segs[0]  # 最初の拍より前
    assert segment_at(segs, 999) is segs[-1]
    assert segment_at([], 1) is None


def test_small_jitter_is_smoothed() -> None:
    """20ms 刻み＋揺れ（標準偏差 8ms）でも1区間・BPM は ±0.5 以内。"""
    for seed_ in range(5):
        beats, downbeats = regular([(0, 124)], 90)
        noisy = quantize(beats, seed_)
        segs = tempo_segments(noisy)
        assert len(segs) == 1, bpms(noisy)
        assert segs[0].bpm == pytest.approx(124, abs=0.5)
        assert estimate_time_signature(noisy, quantize(downbeats, seed_)) == 4


def test_jitter_with_tempo_change() -> None:
    beats, _ = regular([(0, 120), (30, 140)], 60)
    got = bpms(quantize(beats, 3))
    assert len(got) == 2
    assert got[0] == pytest.approx(120, abs=0.5) and got[1] == pytest.approx(140, abs=0.5)


def test_gradual_drift_within_tolerance_is_one_segment() -> None:
    """演奏のゆれ（120→121.5 に少しずつ）は区間を分けない。"""
    t = np.concatenate([[0.0], np.cumsum(60 / np.linspace(120, 121.5, 240))])
    segs = tempo_segments(t)
    assert len(segs) == 1
    assert 120 < segs[0].bpm < 121.5


def test_missing_and_extra_beats() -> None:
    """拍の抜け（1拍・2拍続けて）と余分な拍（倍取りに近い半拍の位置）があっても BPM は保たれる。"""
    beats, downbeats = regular([(0, 120), (30, 140)], 60)
    noisy = quantize(beats, 1)
    b = np.delete(noisy, [10, 50, 51, 90])  # 抜け
    b = np.sort(np.concatenate([b, [noisy[20] + 0.25, noisy[70] + 0.2]]))  # 余分な拍
    got = bpms(b)
    assert len(got) == 2
    assert got[0] == pytest.approx(120, abs=0.5) and got[1] == pytest.approx(140, abs=0.5)
    assert estimate_time_signature(b, downbeats) == 4
    # 抜けは仮の拍で埋まり、余分な拍は消える
    cleaned = clean_beats(b)
    assert len(cleaned) == len(noisy)


def test_long_double_time_section_is_reported_as_is() -> None:
    """長く（16 拍以上かつ 8 秒以上）倍で取れた区間は、倍の BPM として出す（補正は T10c）。"""
    first = np.arange(0, 30, 0.5)
    doubled = np.arange(30, 45, 0.25)
    last = np.arange(45, 60, 0.5)
    got = bpms(np.concatenate([first, doubled, last]))
    assert got == [120.0, 240.0, 120.0]


@pytest.mark.parametrize(
    ("interval", "seconds"),
    [
        (0.25, 3.0),  # 倍（12 拍・3 秒）
        (1.0, 12.0),  # 半分（12 拍・12 秒。拍の数が 16 未満）
        (0.375, 5.25),  # 4/3（3連のノリ。14 拍・5.25 秒）
        (2 / 3, 7.33),  # 3/4（ハーフタイム寄り。11 拍・7.3 秒）
    ],
)
def test_short_related_section_is_absorbed(interval: float, seconds: float) -> None:
    """倍・半分・4/3・3/4 の関係にある短い区間は前後に吸収し、表示を 120 のままにする。"""
    mid = np.arange(30, 30 + seconds, interval)
    after = mid[-1] + interval
    b = np.concatenate([np.arange(0, 30, 0.5), mid, np.arange(after, after + 30, 0.5)])
    segs = tempo_segments(b)
    assert [round(x.bpm, 1) for x in segs] == [120.0]
    assert segs[0].start_sec == 0.0 and segs[0].end_sec == pytest.approx(b[-1])


def test_short_unrelated_section_is_kept() -> None:
    """関係の無い比（120→100）の変化は短くても残す（本当のテンポの変化かもしれない）。"""
    mid = np.arange(30, 30 + 12 * 0.6, 0.6)
    after = mid[-1] + 0.6
    b = np.concatenate([np.arange(0, 30, 0.5), mid, np.arange(after, after + 30, 0.5)])
    assert bpms(b) == [120.0, 100.0, 120.0]


def test_mixed_half_intervals_are_not_filled() -> None:
    """0.3 秒と 0.6 秒が混ざる所（曲の主な間隔 0.446 秒）で、0.6 秒を「抜け」として埋めない。"""
    main = 60 / 134.5
    b = list(np.arange(0, 30, main))
    t = b[-1]
    for gap in [0.3, 0.3, 0.6, 0.3, 0.3, 0.6, 0.3, 0.3, 0.6]:
        t += gap
        b.append(t)
    b += list(np.arange(t + main, t + 30, main))
    cleaned = clean_beats(b)
    assert len(cleaned) == len(b)  # 仮の拍を足していない
    assert np.all(np.diff(cleaned) > 0.25)


def test_isolated_beats_do_not_make_a_segment() -> None:
    """ブレイクの中の孤立した数拍は区間にしない（直前の区間の BPM のまま）。"""
    b = np.concatenate([np.arange(0, 30.1, 0.5), [60.0, 60.3, 60.6], np.arange(90, 110, 0.5)])
    segs = tempo_segments(b)
    assert [round(x.bpm, 1) for x in segs] == [120.0]
    assert segment_at(segs, 75).bpm == pytest.approx(120)
    # 曲の頭の孤立した2拍（0.0, 0.7）も区間にしない
    head = np.concatenate([[0.0, 0.7], np.arange(10, 40, 0.5)])
    segs = tempo_segments(head)
    assert [round(x.bpm, 2) for x in segs] == [120.0]
    assert segs[0].start_sec == pytest.approx(10.0)


def test_break_splits_runs() -> None:
    """長いブレイク（拍が 10 秒途切れる）は埋めない。前後でテンポが違ってよい。"""
    b = np.concatenate([np.arange(0, 20, 0.5), np.arange(30, 50, 60 / 126)])
    segs = tempo_segments(b)
    assert [round(s.bpm, 1) for s in segs] == [120.0, 126.0]
    assert segs[0].end_sec == pytest.approx(19.5)
    assert segs[1].start_sec == pytest.approx(30.0)
    # 前後が同じテンポならまとめる（ブレイクの長さは BPM に含めない）
    same = np.concatenate([np.arange(0, 20, 0.5), np.arange(30, 50, 0.5)])
    assert bpms(same) == [120.0]


def test_short_inputs() -> None:
    assert tempo_segments([]) == []
    assert tempo_segments([1.0]) == []
    # 拍の間隔が 8 未満の列は区間にしない
    assert bpms([0.0, 0.5]) == []
    assert bpms([0.5 * i for i in range(8)]) == []
    assert bpms([0.5 * i for i in range(9)]) == [120.0]
    assert estimate_time_signature([], []) == 4
    assert estimate_time_signature([0, 0.5, 1.0], [0.0]) == 4  # 小節の頭が1つだけ


def test_time_signature_three_and_six() -> None:
    beats, downbeats = regular([(0, 90)], 40, per_bar=3)
    assert estimate_time_signature(beats, downbeats) == 3
    beats, downbeats = regular([(0, 150)], 40, per_bar=6)
    assert estimate_time_signature(quantize(beats), quantize(downbeats)) == 6
    # 小節の頭が1つ抜けても（8拍の小節が1つ）最頻値は 4
    beats, downbeats = regular([(0, 120)], 40)
    assert estimate_time_signature(beats, np.delete(downbeats, 3)) == 4


def test_fake_analyzer_sections_and_offset(tmp_path: Path) -> None:
    fake = FakeBeatAnalyzer([(0, 120), (4, 60)], beats_per_bar=3, offset=0.25)
    res = fake.analyze(np.zeros((44100 * 8, 2), np.float32))
    assert res.beats[:3] == [0.25, 0.75, 1.25]
    assert res.downbeats[:2] == [0.25, 1.75]
    assert np.diff(res.beats)[-1] == pytest.approx(1.0)
    assert res.beats[-1] < 8
    with pytest.raises(RuntimeError):
        FakeBeatAnalyzer(fail=True).analyze(np.zeros((10, 2), np.float32))


# --- 保存・後処理 -------------------------------------------------------------------


def test_analyze_track_saves_and_skips(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    track_id = make_track(session, settings, tmp_path, seconds=6.0)
    fake = FakeBeatAnalyzer(100)
    out = analyze_track(session, settings, track_id, fake)
    session.commit()
    assert not out.skipped and len(fake.calls) == 1
    assert isinstance(fake.calls[0], Path) and fake.calls[0].name == "normalized.wav"
    grid = session.get(BeatGrid, track_id)
    assert grid is not None and grid.analyzer == "fake 1" and grid.time_signature == 4
    assert grid.beats_json[:3] == [0.0, 0.6, 1.2]
    assert grid.downbeats_json[:2] == [0.0, 2.4]

    again = analyze_track(session, settings, track_id, fake)
    assert again.skipped and len(fake.calls) == 1
    analyze_track(session, settings, track_id, FakeBeatAnalyzer(120), force=True)
    session.commit()
    payload = beats_payload(session.get(BeatGrid, track_id))
    assert payload["segments"] == [{"start_sec": 0.0, "end_sec": 5.5, "bpm": 120.0}]
    assert payload["time_signature"] == 4 and payload["analyzer"] == "fake 1"
    assert payload["created_at"].endswith("+00:00")


def _done_job(session: Session, settings: Settings, tmp_path: Path, analyzer=None) -> int:
    seed(session)
    track_id = make_track(session, settings, tmp_path, seconds=4.0)
    job = enqueue_full_job(session, track_id, "fast").job
    factory = make_session_factory(session.get_bind())
    Worker(settings, factory, sync_launcher(settings, beat_analyzer=analyzer)).run_one()
    session.expire_all()
    return job.job_id


def test_separation_job_analyzes_beats(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    stages: list[str] = []
    fake = FakeBeatAnalyzer(90)
    original = fake.analyze

    def spy(audio):
        job = session.query(SeparationJob).one()
        session.refresh(job)
        stages.append(job.stage)
        return original(audio)

    fake.analyze = spy  # type: ignore[method-assign]
    job_id = _done_job(session, settings, tmp_path, fake)
    job = session.get(SeparationJob, job_id)
    assert job.status == "done" and job.beat_warning is None
    assert stages == [STAGE_BEATS]
    grid = get_grid(session, job.track_id)
    assert grid is not None and grid.beats_json[1] == pytest.approx(60 / 90, abs=1e-3)


def test_beat_failure_keeps_job_done(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    job_id = _done_job(session, settings, tmp_path, FakeBeatAnalyzer(fail=True))
    job = session.get(SeparationJob, job_id)
    assert job.status == "done" and job.error_message is None
    assert job.beat_warning is not None and "拍を解析できませんでした" in job.beat_warning
    assert get_grid(session, job.track_id) is None
    # 後で解析に成功すると警告が消える
    assert analyze_job_beats(session, settings, job_id, FakeBeatAnalyzer()) is None
    session.refresh(job)
    assert job.beat_warning is None and get_grid(session, job.track_id) is not None


def test_existing_grid_is_not_reanalyzed_by_new_job(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    job_id = _done_job(session, settings, tmp_path, FakeBeatAnalyzer(100))
    track_id = session.get(SeparationJob, job_id).track_id
    fake = FakeBeatAnalyzer(150)
    assert analyze_job_beats(session, settings, job_id, fake) is None
    assert fake.calls == []  # 既にあるので解析しない
    assert analyze_job_beats(session, settings, job_id, fake, force=True) is None
    session.expire_all()
    assert get_grid(session, track_id).beats_json[1] == pytest.approx(0.4)


def _postprocess_worker(settings: Settings, session: Session, runner) -> Worker:
    factory = make_session_factory(session.get_bind())
    return Worker(
        settings, factory, sync_launcher(settings), postprocess_encoder=fake_encoder,
        beat_runner=runner,
    )


def test_postprocess_makes_missing_beats(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    """配信用データはそろっていて拍だけ無い（T10 より前に分割した曲）→ 拍だけ作る。"""
    job_id = _done_job(session, settings, tmp_path, FakeBeatAnalyzer())
    track_id = session.get(SeparationJob, job_id).track_id
    session.delete(session.get(BeatGrid, track_id))
    session.commit()
    assert request_postprocess(session, job_id, ["beats"]).created

    calls: list[int] = []

    def runner(jid: int, should_stop) -> None:
        calls.append(jid)
        assert should_stop() is False
        with make_session_factory(session.get_bind())() as s:
            analyze_job_beats(s, settings, jid, FakeBeatAnalyzer(110))

    worker = _postprocess_worker(settings, session, runner)
    assert worker.run_postprocess_one() == job_id
    session.expire_all()
    assert calls == [job_id]
    job = session.get(SeparationJob, job_id)
    assert job.postprocess_status == "done" and job.beat_warning is None
    assert get_grid(session, track_id) is not None
    # 拍があれば解析しない
    assert request_postprocess(session, job_id, ["beats"]).created
    assert worker.run_postprocess_one() == job_id
    assert calls == [job_id]


def test_postprocess_beat_runner_failure_is_warning(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    job_id = _done_job(session, settings, tmp_path, FakeBeatAnalyzer(fail=True))
    assert request_postprocess(session, job_id, ["beats"]).created

    def runner(_jid: int, _stop) -> None:
        raise RuntimeError("子プロセスが異常終了しました")

    assert _postprocess_worker(settings, session, runner).run_postprocess_one() == job_id
    session.expire_all()
    job = session.get(SeparationJob, job_id)
    assert job.status == "done" and job.postprocess_status == "done"
    assert "子プロセスが異常終了しました" in job.beat_warning


def test_subprocess_beat_runner_with_fake(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    """本物の子プロセス（`python -m stemapp.beats.child --fake`）で拍を作る。"""
    job_id = _done_job(session, settings, tmp_path, FakeBeatAnalyzer(fail=True))
    run = subprocess_beat_runner(settings, ["--fake"])
    run(job_id, lambda: False)
    session.expire_all()
    job = session.get(SeparationJob, job_id)
    assert job.beat_warning is None
    assert get_grid(session, job.track_id).analyzer == "fake 1"
    from stemapp.jobs.worker import BeatRunInterrupted

    with pytest.raises(BeatRunInterrupted):
        run(job_id, lambda: True)
    with pytest.raises(RuntimeError, match="異常終了"):
        run(99999, lambda: False)  # ジョブが無い → 終了コード 1


def test_beat_this_is_not_imported_by_other_modules() -> None:
    """beat_this・torch は解析器のモジュールの中でだけ import する。"""
    code = (
        "import sys, stemapp.cli, stemapp.app, stemapp.beats, stemapp.jobs.child, "
        "stemapp.beats.child, stemapp.beats.beat_this_backend, stemapp.jobs.worker;"
        "print('beat_this' in sys.modules, 'torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == ["False", "False"]


# --- レビュー後の追加 ------------------------------------------------------------------


def test_no_beats_is_warning(session: Session, settings: Settings, tmp_path: Path) -> None:
    """拍が1つも見つからない（無音など）ときは、保存せず警告を残す。"""
    job_id = _done_job(session, settings, tmp_path, FakeBeatAnalyzer(offset=999))
    job = session.get(SeparationJob, job_id)
    assert job.status == "done"
    assert "拍が見つかりませんでした" in job.beat_warning
    assert get_grid(session, job.track_id) is None
    track_id = job.track_id
    with pytest.raises(RuntimeError, match="拍が見つかりませんでした"):
        analyze_track(session, settings, track_id, FakeBeatAnalyzer(offset=999))


def test_save_failure_is_warning(
    session: Session, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stemapp.beats import service

    job_id = _done_job(session, settings, tmp_path, FakeBeatAnalyzer(fail=True))

    def broken(*_a, **_k):
        raise RuntimeError("保存できません（テスト）")

    monkeypatch.setattr(service, "save_grid", broken)
    msg = analyze_job_beats(session, settings, job_id, FakeBeatAnalyzer())
    assert msg is not None and "保存できません" in msg
    session.expire_all()
    job = session.get(SeparationJob, job_id)
    assert job.status == "done" and "保存できません" in job.beat_warning


def test_concurrent_grid_insert_is_treated_as_existing(
    session: Session, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同じ曲の拍を別の処理が先に保存していた（主キーの重複）→ 警告にせず、そちらを使う。"""
    from stemapp.beats import service

    job_id = _done_job(session, settings, tmp_path, FakeBeatAnalyzer(fail=True))
    track_id = session.get(SeparationJob, job_id).track_id
    real_save = service.save_grid

    def racing_save(s, tid, result):
        # 解析の間に別の処理（別の接続）が同じ曲の拍を保存した
        with make_session_factory(session.get_bind())() as other:
            real_save(other, tid, FakeBeatAnalyzer(90).analyze(np.zeros((44100 * 4, 2))))
            other.commit()
        s.add(BeatGrid(track_id=tid, analyzer="x", beats_json=[], downbeats_json=[]))
        s.flush()

    monkeypatch.setattr(service, "save_grid", racing_save)
    session.expire_all()
    assert analyze_job_beats(session, settings, job_id, FakeBeatAnalyzer()) is None
    session.expire_all()
    grid = get_grid(session, track_id)
    assert grid is not None and grid.beats_json[1] == pytest.approx(60 / 90, abs=1e-3)
    # 先に保存した処理が警告を消している（こちらは警告を書かない）
    assert session.get(SeparationJob, job_id).beat_warning is None


def test_postprocess_interrupted_by_stop_goes_back_to_queue(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    from stemapp.jobs.worker import BeatRunInterrupted

    job_id = _done_job(session, settings, tmp_path, FakeBeatAnalyzer(fail=True))
    session.execute(
        SeparationJob.__table__.update().values(beat_warning=None)  # type: ignore[attr-defined]
    )
    session.commit()
    assert request_postprocess(session, job_id, ["beats"]).created

    def runner(_jid: int, _stop) -> None:
        raise BeatRunInterrupted("停止の指示で拍の解析を中断しました。")

    assert _postprocess_worker(settings, session, runner).run_postprocess_one() == job_id
    session.expire_all()
    job = session.get(SeparationJob, job_id)
    assert job.postprocess_status == "queued" and job.beat_warning is None
