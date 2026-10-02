"""ピッチを保つ・すぐ（PC。ブラウザ内の伸縮器 signalsmith-stretch）のテスト（T11c）。

`-m browser` で実行する（PC の Edge を Playwright で）。

- 純粋関数（位置の履歴、半音数、方式の既定）。
- 時刻のずれ: アプリの Engine を OfflineAudioContext で動かし、本物の伸縮器（AudioWorklet）を通した
  出力を調べる。1 秒おきの 440Hz ガウス形バースト（σ=10ms。T11 と同じ測り方）の重心が、画面に出す
  再生位置（engine.position）がその曲の時刻を指す時刻と ±5ms 以内（曲の時刻に直して）に合うこと。
  速度の変更・シーク・ループを途中に入れても合うこと。音の高さが変わらないこと。
- プレイヤーの画面: 速度を変えても止まらない、stem の ON/OFF、シーク・ループ、方式の切り替え、
  スマホ幅の既定。スクリーンショットは data/cache/screens/ に保存する（コミットしない）。
"""

from __future__ import annotations

import base64
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from browser_helpers import LiveServer, run_server
from stemapp.beats import FakeBeatAnalyzer
from stemapp.config import Settings
from stemapp.tempo.stretch import FakeStretcher
from test_browser import (  # noqa: F401  fixture を使う
    PHONE,
    _shot,
    _wait_gains,
    browser,
    page,
)
from test_browser_beats import TEMPO, _done_track

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

VIEW = "window.__stemapp.view"
SR = 48000


@pytest.fixture
def server(tmp_path: Path) -> Iterator[LiveServer]:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    with run_server(
        settings, fake_delay=0.2, beat_analyzer=FakeBeatAnalyzer(TEMPO),
        tempo_stretcher=FakeStretcher(delay_sec=1.0, steps=10),
    ) as srv:
        yield srv


def _open(pg: Any, server: LiveServer, track_id: int) -> None:
    pg.goto(f"{server.base_url}/#/track/{track_id}")
    pg.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)


def _engine(pg: Any, expr: str) -> Any:
    return pg.evaluate(f"() => {{ const e = {VIEW}.engine; return {expr}; }}")


def _hide_toast(pg: Any) -> None:
    pg.evaluate("() => { document.getElementById('toast').hidden = true; }")


def _speed_over(pg: Any, ms: int = 600) -> float:
    """実時間 ms の間に曲の時刻がどれだけ進んだか（曲の秒 / 実時間の秒）。"""
    return float(pg.evaluate(
        """async (ms) => {
        const e = window.__stemapp.view.engine;
        const c0 = e.ctx.currentTime, p0 = e.position;
        await new Promise((r) => setTimeout(r, ms));
        const c1 = e.ctx.currentTime, p1 = e.position;
        let d = p1 - p0;
        if (d < 0 && e.loop) d += e.loop.end - e.loop.start;
        return d / (c1 - c0);
    }""",
        ms,
    ))


def _positions(pg: Any, n: int) -> list[float]:
    return list(pg.evaluate(
        """async (n) => {
        const e = window.__stemapp.view.engine;
        const out = [];
        for (let i = 0; i < n; i++) {
          out.push(e.position);
          await new Promise((r) => setTimeout(r, 50));
        }
        return out;
    }""",
        n,
    ))


# --- 純粋関数 -------------------------------------------------------------------------


def test_stretch_pure_functions(page: Any, server: LiveServer) -> None:  # noqa: F811
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    res = page.evaluate(
        """() => {
        const { engine: E, tempo: T } = window.__stemapp.modules;
        const past = [{ offset: 0, ctxTime: 0, speed: 1 }, { offset: 5, ctxTime: 5, speed: 2 }];
        const cur = { offset: 9, ctxTime: 7, speed: 0.5 };
        localStorage.removeItem("stemapp.tempo.987");
        const put = (id, mode) => localStorage.setItem(
          `stemapp.tempo.${id}`, JSON.stringify({ ratio: 1.1, mode, range: 8 }));
        put(988, "instant");
        put(989, "???");
        return {
          // 今の基準より前は、その時刻に効いていた基準で数える
          hist: [E.positionFromHistory(past, cur, 3, null, 100),   // 1 倍の区間
                 E.positionFromHistory(past, cur, 6, null, 100),   // 2 倍の区間: 5 + 2
                 E.positionFromHistory(past, cur, 8, null, 100),   // 今の基準: 9 + 0.5
                 E.positionFromHistory([], cur, 6, null, 100),     // 履歴なし: 基準で止まる
                 E.positionFromHistory(past, cur, -1, null, 100), // 最も古い基準より前
                 // 当時のループを持つ基準はそれで折り返す（今はループ解除: null）
                 E.positionFromHistory(
                   [{ offset: 3, ctxTime: 0, speed: 1, loop: { start: 2, end: 4 } }],
                   cur, 2, null, 100),
                 // 当時ループが無かった基準は、今のループで折り返さない
                 E.positionFromHistory([{ offset: 3, ctxTime: 0, speed: 1, loop: null }],
                                       cur, 2, { start: 2, end: 4 }, 100)],
          semi: [E.semitonesFor(1), E.semitonesFor(2), E.semitonesFor(0.5), E.semitonesFor(0)],
          modes: T.MODES,
          def: [T.defaultMode(true, false), T.defaultMode(true, true), T.defaultMode(false, false)],
          state: [T.loadTempoState(987, "instant").mode, T.loadTempoState(987, "keep").mode,
                  T.loadTempoState(988, "keep").mode, T.loadTempoState(989, "keep").mode],
          // T11 の位置の関数（即時方式でも同じものを使う）
          song: E.songPositionAt(5, 2, 1.5, null, 100),
        };
    }"""
    )
    assert res["hist"] == [pytest.approx(3), pytest.approx(7), pytest.approx(9.5), 9, 0,
                           pytest.approx(3), pytest.approx(5)]
    assert res["semi"][0] == 0
    assert res["semi"][1] == pytest.approx(-12) and res["semi"][2] == pytest.approx(12)
    assert res["semi"][3] == 0
    assert res["modes"] == ["pitch", "instant", "keep"]
    assert res["def"] == ["instant", "keep", "keep"]
    assert res["state"] == ["instant", "keep", "instant", "keep"]
    assert res["song"] == pytest.approx(8.0)


# --- 時刻のずれ（OfflineAudioContext で本物の伸縮器を通す） -------------------------------------

OFFLINE_JS = """
async ({ ratio, actions, nSec, renderSec, tone }) => {
  const { engine: E } = window.__stemapp.modules;
  const sr = 48000;
  const ctx = new OfflineAudioContext(2, Math.round(renderSec * sr), sr);
  const e = new E.Engine(() => ctx);
  const n = nSec * sr;
  const buf = ctx.createBuffer(2, n, sr);
  const d0 = buf.getChannelData(0), d1 = buf.getChannelData(1);
  if (tone) {
    for (let i = 0; i < n; i++) {
      const v = 0.3 * Math.sin(2 * Math.PI * 440 * i / sr);
      d0[i] = v; d1[i] = v;
    }
  }
  for (let c = 1; c < nSec && !tone; c++) {
    for (let i = Math.floor((c - 0.06) * sr); i < Math.min(n, (c + 0.06) * sr); i++) {
      const t = i / sr;
      const v = 0.5 * Math.exp(-0.5 * ((t - c) / 0.01) ** 2) * Math.sin(2 * Math.PI * 440 * t);
      d0[i] += v; d1[i] += 0.7 * v;
    }
  }
  e.addTrack("a", buf, 1);
  await e.ensureStretch();
  e.setRate(ratio);
  e.setAligned(true);
  e.setPitchLock(ratio !== 1);
  await e.stretch.latency(); // 予約（schedule）が伸縮器に届くのを待つ
  e._startSources(0, 0);
  e.playing = true;
  const trace = [];
  const step = 0.01;
  const applied = [];
  for (let k = 1; k * step < renderSec - 0.05; k++) {
    const t = Math.round(k * step * sr / 128) * 128 / sr;
    ctx.suspend(t).then(async () => {
      trace.push([ctx.currentTime, e.position]);
      for (const a of actions) {
        if (a.done || a.at > ctx.currentTime + 1e-9) continue;
        a.done = true;
        if (a.op === "rate") e.setRate(a.v);
        else if (a.op === "rate+lock") { e.setRate(a.v); e.setPitchLock(a.v !== 1); }
        else if (a.op === "seek") e.seek(a.v);
        else if (a.op === "loop") e.setLoop(a.v);
        applied.push([a.op, ctx.currentTime]);
        await e.stretch.latency();
      }
      ctx.resume();
    });
  }
  const out = await ctx.startRendering();
  const y = out.getChannelData(0);
  const bytes = new Uint8Array(y.buffer, y.byteOffset, y.byteLength);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return { y: btoa(bin), trace, latency: e.latency, applied, starts: e.startCtxTime };
}
"""


def _render(pg: Any, ratio: float, actions: list[dict[str, Any]], n_sec: int,
            render_sec: float, tone: bool = False) -> dict[str, Any]:
    res = pg.evaluate(OFFLINE_JS, {
        "ratio": ratio, "actions": actions, "nSec": n_sec, "renderSec": render_sec, "tone": tone,
    })
    res["y"] = np.frombuffer(base64.b64decode(res["y"]), dtype=np.float32).astype(np.float64)
    return res


def _burst_errors(
    res: dict[str, Any], n_sec: int, quiet: list[float]
) -> list[tuple[float, float, float]]:
    """画面の再生位置がバーストの曲の時刻 c を指す時刻と、出力のバーストの重心の差。

    戻り値は (c, 曲の時刻に直したずれ ms, その時刻の速さ)。quiet（操作した時刻）の直後 0.4 秒は
    フェード・鳴らし直しの途中なので数えない。
    """
    y = res["y"]
    env = y**2
    trace = np.array(res["trace"])
    out: list[tuple[float, float, float]] = []
    for (t0, p0), (t1, p1) in zip(trace, trace[1:], strict=False):
        dp = p1 - p0
        if not (0 < dp < 0.1):  # シーク・ループの折り返し・止まっている所は使わない
            continue
        speed = dp / (t1 - t0)
        for c in range(1, n_sec):
            if not (p0 <= c < p1):
                continue
            t = t0 + (c - p0) / dp * (t1 - t0)
            if any(q - 0.05 <= t <= q + 0.4 + res["latency"] for q in quiet):
                continue
            half = min(0.2, 0.45 / speed)
            lo, hi = int((t - half) * SR), int((t + half) * SR)
            if lo < 0 or hi > len(y):
                continue
            seg = env[lo:hi]
            if seg.sum() < 1e-3:
                out.append((c, float("nan"), speed))
                continue
            centroid = float((seg * np.arange(lo, hi)).sum() / seg.sum()) / SR
            out.append((c, (centroid - t) * speed * 1000, speed))
    return out


def _peak_hz(res: dict[str, Any], t: float) -> float:
    y = res["y"]
    lo = int((t - 0.25) * SR)
    seg = y[lo:lo + SR // 2] * np.hanning(SR // 2)
    spec = np.abs(np.fft.rfft(seg, n=SR))  # 1Hz 刻み
    return float(np.argmax(spec))


@pytest.mark.parametrize("ratio", [0.5, 0.9, 1.1, 2.0])
def test_offline_time_offset(page: Any, server: LiveServer, ratio: float) -> None:  # noqa: F811
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    n_sec = 12
    res = _render(page, ratio, [], n_sec, n_sec / ratio + 0.6)
    assert 0.05 < res["latency"] < 0.3
    errs = _burst_errors(res, n_sec, [0.0])
    assert len(errs) >= 8, errs
    values = [e for _, e, _ in errs]
    assert all(np.isfinite(values)), errs
    mean = float(np.mean(values))
    print(f"ratio={ratio} latency={res['latency'] * 1000:.1f}ms "
          f"mean={mean:+.2f}ms max|e|={max(abs(v) for v in values):.2f}ms")
    assert abs(mean) < 5.0, (ratio, mean, errs)
    assert all(abs(v) < 8.0 for v in values), errs
    # 音の高さは変わらない（ピッチも変わる方式なら 440 × 倍率 Hz になる）
    c = 5
    t = c / ratio + res["latency"]
    hz = _peak_hz(res, t)
    assert abs(hz - 440) < 15, (ratio, hz)


def test_offline_time_offset_with_changes(page: Any, server: LiveServer) -> None:  # noqa: F811
    """速度の変更・シーク・ループを途中に入れても、画面の位置と聞こえる音が合う。"""
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    n_sec = 12
    actions: list[dict[str, Any]] = [
        {"at": 2.0, "op": "rate", "v": 1.25},
        {"at": 4.5, "op": "seek", "v": 1.5},
        {"at": 6.0, "op": "loop", "v": {"start": 3.5, "end": 5.6}},
        {"at": 8.0, "op": "rate", "v": 0.8},
        {"at": 11.0, "op": "loop", "v": None},
    ]
    res = _render(page, 0.9, actions, n_sec, 15.0)
    assert [a[0] for a in res["applied"]] == ["rate", "seek", "loop", "rate", "loop"]
    errs = _burst_errors(res, n_sec, [a[1] for a in res["applied"]])
    values = [e for _, e, _ in errs]
    speeds = sorted({round(s, 2) for _, _, s in errs})
    print("errors(ms):", [(c, round(e, 2), round(s, 2)) for c, e, s in errs])
    assert len(errs) >= 10, errs
    assert {0.8, 0.9, 1.25} <= set(speeds), speeds
    # ループで同じバースト（4 秒・5 秒）を何回も通る
    assert sum(1 for c, _, _ in errs if c == 4) >= 3, errs
    assert all(np.isfinite(values)), errs
    assert abs(float(np.mean(values))) < 5.0, errs
    assert all(abs(v) < 8.0 for v in values), errs


def _semitone_track(y: np.ndarray, center: float, span: float = 0.1) -> list[tuple[float, float]]:
    """center の前後 span 秒を、21ms 窓（1024 点）・5ms 刻みで 440Hz からのずれ（半音）にする。"""
    w = 1024
    hop = SR * 5 // 1000
    out = []
    for c in np.arange(center - span, center + span, hop / SR):
        lo = int(c * SR - w / 2)
        seg = y[lo:lo + w] * np.hanning(w)
        sp = np.abs(np.fft.rfft(seg, n=SR * 2))  # 0.5Hz 刻み
        k = int(np.argmax(sp))
        a, b, g = sp[k - 1], sp[k], sp[k + 1]
        k_f = k + 0.5 * (a - g) / (a - 2 * b + g)  # 放物線で山の位置を細かく
        hz = k_f / 2
        out.append((float(c - center), float(12 * np.log2(hz / 440))))
    return out


@pytest.mark.parametrize(
    ("r0", "r1"),
    [(1.1, 1.25), (0.9, 0.92), (1.0, 1.1), (1.1, 1.0), (1.25, 0.8)],
)
def test_offline_pitch_stays_at_rate_change(
    page: Any, server: LiveServer, r0: float, r1: float  # noqa: F811
) -> None:
    """速度を変えた瞬間の前後 100ms も音の高さが外れない（440Hz の音で ±0.3 半音の外れは数窓まで）。

    1.0 ⇔ 1.1 は伸縮器の音（wet）と通さない音（dry）のクロスフェードも含む。
    """
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    res = _render(page, r0, [{"at": 2.0, "op": "rate+lock", "v": r1}], 10, 4.0, tone=True)
    heard = res["starts"] + res["latency"]  # 速度を変えた音が聞こえる時刻
    track = _semitone_track(res["y"], heard)
    off = [(round(t * 1000), round(d, 2)) for t, d in track if abs(d) > 0.3]
    print(f"{r0}->{r1}: 外れた窓 {len(off)} 個 {off}")
    assert len(off) <= 3, off
    # 変える前後の落ち着いた所は 440Hz（±0.1 半音）
    for t in (heard - 0.5, heard + 0.5):
        assert all(abs(d) < 0.1 for _, d in _semitone_track(res["y"], t, 0.05))


# --- プレイヤーの画面 ------------------------------------------------------------------


def _wait_lock(pg: Any, on: bool, timeout_ms: int = 10_000) -> None:
    pg.wait_for_function(f"() => {VIEW}.engine.lock === {str(on).lower()}", timeout=timeout_ms)


def test_instant_mode_player(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    track_id, _ = _done_track(server, tmp_path, seconds=24.0)
    _open(page, server, track_id)
    # PC の既定は「ピッチを保つ・すぐ」。元の速度では伸縮器を通さない
    assert page.get_attribute("#tp-mode .seg-btn[data-value='instant']", "aria-pressed") == "true"
    assert page.is_visible("#tp-instant") and page.is_hidden("#tp-keep")
    page.wait_for_function(f"() => {VIEW}.tempo.stretchState === 'ready'", timeout=10_000)
    assert _engine(page, "e.lock") is False
    page.evaluate(f"() => {VIEW}.seek(1.0)")
    page.click("#play-btn")
    page.wait_for_function(f"() => {VIEW}.engine.playing && {VIEW}.engine.position > 1.3")
    page.wait_for_timeout(150)
    all_on = _engine(page, "e.gainValues()")
    page.locator("body").focus()
    page.keyboard.press("1")
    page.wait_for_timeout(150)
    gains = _engine(page, "e.gainValues()")
    assert sorted(set(gains.values())) == [0.0, 1.0]
    n_sources = _engine(page, "e.activeSources()")
    under0 = _engine(page, "e.ctx.playbackStats ? e.ctx.playbackStats.underrunEvents : -1")

    # 「すぐ」の方式の間は、1.000 でも伸縮器に音を入れて遅れをそろえている（位置にも遅れを入れる）
    assert _engine(page, "e.aligned") is True
    assert _engine(page, "e.dryDelay.delayTime.value") == pytest.approx(_engine(page, "e.latency"))
    # +10%: 伸縮器の音にクロスフェードし、全 stem の playbackRate = 1.1、音の高さを戻す。
    # 鳴らし直さない（音源はそのまま、位置は止まらず実時間 × 速さで進む）
    before = _engine(page, "e.position")
    srcs = page.evaluate(
        f"() => {{ window.__srcs = [...{VIEW}.engine.tracks.values()].map((t) => t.source); }}")
    trace = page.evaluate(
        """async () => {
        const v = window.__stemapp.view, e = v.engine;
        const out = [];
        for (let i = 0; i < 40; i++) {
          if (i === 10) v.tempo.setRatio(1.1);
          out.push([e.ctx.currentTime, e.position]);
          await new Promise((r) => setTimeout(r, 10));
        }
        return out;
    }"""
    )
    del srcs
    assert page.evaluate(
        f"() => [...{VIEW}.engine.tracks.values()].every((t, i) => t.source === window.__srcs[i])"
    ), "鳴らし直している"
    for (c0, p0), (c1, p1) in zip(trace, trace[1:], strict=False):
        dt = c1 - c0
        if dt <= 0:
            continue
        assert 0.9 * dt - 0.004 <= p1 - p0 <= 1.2 * dt + 0.004, trace  # 止まらず飛ばない
    _wait_lock(page, True)
    page.wait_for_timeout(100)
    assert _engine(page, "e.playing") is True
    rates = _engine(page, "e.sourceRates()")
    assert len(rates) == n_sources and all(r == pytest.approx(1.1) for r in rates), rates
    assert _engine(page, "e.position") >= before - 0.01
    assert _engine(page, "e.latency") == pytest.approx(_engine(page, "e.stretchLatency"))
    assert 0.05 < _engine(page, "e.latency") < 0.3
    assert _speed_over(page) == pytest.approx(1.1, abs=0.05)
    page.wait_for_function(
        "(t) => document.querySelector('#tp-inst-status').textContent.startsWith(t)",
        arg="ブラウザ内でピッチを保っています",
    )
    # stem の ON/OFF は今までどおり効く（GainNode）
    assert _engine(page, "e.gainValues()") == gains
    page.locator("body").focus()
    page.keyboard.press("1")
    _wait_gains(page, all_on)

    # スライダーを動かしている間（1.000 を通っても）止まらず、伸縮器も外さない
    trace = page.evaluate(
        """async () => {
        const v = window.__stemapp.view, e = v.engine;
        const out = [];
        const steps = [];
        for (let s = 100; s >= -60; s -= 4) steps.push(s);
        for (const s of steps) {
          v.tempo.slider.value = String(s);
          v.tempo.slider.dispatchEvent(new Event("input"));
          await new Promise((r) => setTimeout(r, 16));
          out.push([e.ctx.currentTime, e.position, e.lock, e.playing, e.rate]);
        }
        return out;
    }"""
    )
    assert all(lock and playing for _, _, lock, playing, _ in trace), trace
    assert any(r == 1 for *_, r in trace)  # 1.000 を通った
    pos = [p for _, p, *_ in trace]
    assert all(b >= a - 1e-3 for a, b in zip(pos, pos[1:], strict=False)), pos  # 戻らない
    # 進み方は実時間 × 速さ程度（飛ばない）
    for (c0, p0, *_), (c1, p1, *_) in zip(trace, trace[1:], strict=False):
        assert p1 - p0 <= (c1 - c0) * 1.2 + 0.02, trace
    assert page.inner_text("#tp-readout") == "−6.0%"
    page.dispatch_event("#tp-slider", "change")  # 離す
    assert _engine(page, "e.lock") is True  # 0.94 なので通したまま
    assert _speed_over(page) == pytest.approx(0.94, abs=0.05)

    # シーク: 曲の時刻で動き、その後も 0.94 倍で進む
    page.evaluate(f"() => {VIEW}.seek(8.0)")
    page.wait_for_function(f"() => Math.abs({VIEW}.engine.position - 8.0) < 0.3")
    page.wait_for_timeout(300)  # 鳴らし直し（伸縮器の遅れの分）を待つ
    assert _speed_over(page) == pytest.approx(0.94, abs=0.05)
    assert _engine(page, "e.position") > 8.0

    # 小節ループ（曲の時刻）: 速度を変えても同じ区間で折り返す
    page.evaluate(f"() => {VIEW}.setBarLoop(1)")
    loop = page.evaluate(f"() => {VIEW}.engine.loop")
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.2)")
    seen = _positions(page, 40)
    assert all(loop["start"] - 0.01 <= p <= loop["end"] + 0.01 for p in seen), seen
    assert any(b < a for a, b in zip(seen, seen[1:], strict=False)), "折り返していない"
    page.evaluate(f"() => {{ {VIEW}.loopOn = false; {VIEW}.applyLoop(); }}")

    # 一時停止 → 再生でも伸縮器を通したまま
    page.click("#play-btn")
    page.wait_for_function(f"() => !{VIEW}.engine.playing")
    paused = _engine(page, "e.position")
    page.click("#play-btn")
    page.wait_for_function(f"() => {VIEW}.engine.playing")
    assert _engine(page, "e.lock") is True
    assert _engine(page, "e.position") >= paused - 0.01

    under1 = _engine(page, "e.ctx.playbackStats ? e.ctx.playbackStats.underrunEvents : -1")
    print(f"underrun events: {under0} -> {under1}")
    _hide_toast(page)
    _shot(page, "tempo_instant_desktop.png")

    # 元の速度に戻すと伸縮器を外す
    page.click("#tp-reset")
    _wait_lock(page, False)
    assert _engine(page, "e.rate") == 1 and _engine(page, "e.playing") is True
    assert page.inner_text("#tp-inst-status").startswith("元の速度です")
    # 方式は曲ごとに覚える
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.05)")
    _wait_lock(page, True)
    page.reload()
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    assert page.evaluate(f"() => {VIEW}.tempo.mode") == "instant"
    assert page.inner_text("#tp-readout") == "+5.0%"
    _wait_lock(page, True)  # 止まっている間も、開き直したら伸縮器を通す準備をする
    assert not page.errors  # type: ignore[attr-defined]


def test_instant_mode_stretch_timeout(
    browser: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    """伸縮器が 5 秒で用意できなければ、使えないものとしてピッチも変わる方式で速度を変える。"""
    track_id, _ = _done_track(server, tmp_path, seconds=24.0)
    # Service Worker を通すと route が効かないので止める
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, locale="ja-JP",
                              service_workers="block")
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    try:
        # 伸縮器のファイルに応答しない（読み込みが終わらない）
        pg.route("**/SignalsmithStretch.mjs", lambda route: None)
        _open(pg, server, track_id)
        assert pg.evaluate(f"() => {VIEW}.tempo.mode") == "instant"
        assert pg.evaluate(f"() => {VIEW}.tempo.stretchState") == "loading"
        pg.wait_for_function(f"() => {VIEW}.tempo.stretchState === 'failed'", timeout=10_000)
        assert pg.inner_text("#tp-inst-status") == (
            "ブラウザ内の伸縮を使えません（ピッチも変わる方式で再生中）")
        pg.click("#play-btn")
        pg.wait_for_function(f"() => {VIEW}.engine.playing")
        pg.evaluate(f"() => {VIEW}.tempo.setRatio(1.1)")
        pg.wait_for_timeout(100)
        assert _engine(pg, "e.rate") == pytest.approx(1.1)
        assert all(r == pytest.approx(1.1) for r in _engine(pg, "e.sourceRates()"))
        assert _engine(pg, "e.aligned") is False and _engine(pg, "e.lock") is False
        assert _engine(pg, "e.latency") == 0
        assert _speed_over(pg) == pytest.approx(1.1, abs=0.05)
        assert not errors
    finally:
        ctx.close()


def _wait_status(pg: Any, text: str, timeout_ms: int = 20_000) -> None:
    pg.wait_for_function(
        "(t) => document.querySelector('#tp-status').textContent.startsWith(t)", arg=text,
        timeout=timeout_ms,
    )


def test_switch_instant_and_server_modes(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    """同じ倍率で「すぐ（PC）」と「高音質（サーバー）」を切り替えて聴き比べられる。"""
    track_id, _ = _done_track(server, tmp_path, seconds=24.0)
    _open(page, server, track_id)
    page.click("#play-btn")
    page.wait_for_function(f"() => {VIEW}.engine.playing")
    page.locator("body").focus()
    page.keyboard.press("2")
    page.wait_for_timeout(150)
    gains = _engine(page, "e.gainValues()")
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.2)")
    _wait_lock(page, True)

    page.click("#tp-mode .seg-btn[data-value='keep']")
    assert page.is_visible("#tp-keep") and page.is_hidden("#tp-instant")
    _wait_lock(page, False)  # サーバーの方式では伸縮器を通さない
    _wait_status(page, "ピッチを保って再生中（×1.200）")
    assert _engine(page, "e.bufScale") == pytest.approx(1.2) and _engine(page, "e.rate") == 1
    assert _engine(page, "e.lock") is False
    assert _engine(page, "e.playing") is True
    assert _speed_over(page) == pytest.approx(1.2, abs=0.05)
    assert _engine(page, "e.gainValues()") == gains

    # 「すぐ」に戻す: 元の音声を読み直して、playbackRate 1.2 ＋ 伸縮器
    page.click("#tp-mode .seg-btn[data-value='instant']")
    page.wait_for_function(
        f"() => {VIEW}.engine.bufScale === 1 && {VIEW}.engine.lock && {VIEW}.engine.rate === 1.2",
        timeout=10_000,
    )
    assert _engine(page, "e.playing") is True
    page.wait_for_timeout(400)  # 鳴らし直し（伸縮器の遅れの分、位置が止まる）が終わるのを待つ
    assert _speed_over(page) == pytest.approx(1.2, abs=0.05)
    assert _engine(page, "e.gainValues()") == gains
    # 作成済みの倍率なので、もう一度サーバーの方式にするとすぐ切り替わる
    page.click("#tp-mode .seg-btn[data-value='keep']")
    _wait_status(page, "ピッチを保って再生中（×1.200）", timeout_ms=5000)
    # ピッチも変わる方式: 伸縮器を通さない
    page.click("#tp-mode .seg-btn[data-value='pitch']")
    page.wait_for_function(f"() => {VIEW}.engine.bufScale === 1 && {VIEW}.engine.rate === 1.2",
                           timeout=10_000)
    assert _engine(page, "e.lock") is False
    assert page.is_hidden("#tp-instant") and page.is_hidden("#tp-keep")
    assert not page.errors  # type: ignore[attr-defined]


def test_instant_mode_phone(browser: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    """スマホ幅の既定はサーバーで作る方式。「すぐ」を選んでもはみ出さない。"""
    track_id, _ = _done_track(server, tmp_path)
    ctx = browser.new_context(viewport=PHONE, locale="ja-JP", is_mobile=True, has_touch=True)
    pg = ctx.new_page()
    try:
        pg.goto(f"{server.base_url}/#/track/{track_id}")
        pg.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
        assert pg.evaluate(f"() => {VIEW}.tempo.mode") == "keep"
        assert pg.get_attribute("#tp-mode .seg-btn[data-value='keep']", "aria-pressed") == "true"
        assert pg.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
        pg.click("#tp-mode .seg-btn[data-value='instant']")
        pg.evaluate(f"() => {VIEW}.tempo.setRatio(1.08)")
        pg.wait_for_function(f"() => {VIEW}.engine.lock", timeout=10_000)
        assert pg.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
        pg.evaluate("() => { document.getElementById('toast').hidden = true; }")
        _shot(pg, "tempo_instant_phone.png")
    finally:
        ctx.close()
