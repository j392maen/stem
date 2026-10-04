"""iPhone 向けの再生の画面テスト（T06b。PC の Edge を Playwright で、スマホ幅・タッチを再現）。

`-m browser` で実行する。Edge には navigator.audioSession が無いので、スマホの再現では C の経路
（Web Audio → createMediaStreamDestination → <audio>.srcObject）で鳴る。
拍は FakeBeatAnalyzer、ピッチを保つ方式の伸縮は FakeStretcher。
スクリーンショットは data/cache/screens/ に保存する（コミットしない）。
"""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from browser_helpers import LiveServer, run_server
from stemapp.beats import FakeBeatAnalyzer
from stemapp.config import Settings
from stemapp.tempo.stretch import FakeStretcher
from test_browser import (  # noqa: F401  fixture を使う
    PHONE,
    _shot,
    browser,
    page,
)
from test_browser_beats import TEMPO, _done_track
from test_browser_tempo import _speed_over

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

VIEW = "window.__stemapp.view"
IPHONE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"
)
RENDITION = re.compile(r"/api/files/(renditions|tempo)/")

# Media Session の呼び出しを記録する（ハンドラを後から呼べるように）
MEDIA_SPY = """
(() => {
  const ms = navigator.mediaSession;
  window.__ms = { handlers: {}, positions: [] };
  if (!ms) return;
  const setH = ms.setActionHandler.bind(ms);
  ms.setActionHandler = (a, h) => { window.__ms.handlers[a] = h; try { setH(a, h); } catch (e) {} };
  const setP = ms.setPositionState ? ms.setPositionState.bind(ms) : null;
  ms.setPositionState = (st) => { window.__ms.positions.push(st); if (setP) setP(st); };
})();
"""


@pytest.fixture
def server(tmp_path: Path) -> Iterator[LiveServer]:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    with run_server(
        settings, fake_delay=0.2, beat_analyzer=FakeBeatAnalyzer(TEMPO),
        tempo_stretcher=FakeStretcher(delay_sec=1.0, steps=10),
    ) as srv:
        yield srv


def _phone(browser: Any) -> tuple[Any, Any, list[str]]:  # noqa: F811
    """スマホ（幅 390px・タッチ・iPhone の UA）の画面。fetch した音声の URL を記録する。"""
    ctx = browser.new_context(
        viewport=PHONE, locale="ja-JP", is_mobile=True, has_touch=True, user_agent=IPHONE_UA,
    )
    ctx.add_init_script(MEDIA_SPY)
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.errors = errors  # type: ignore[attr-defined]
    audio: list[str] = []
    pg.on("request", lambda r: audio.append(r.url) if RENDITION.search(r.url) else None)
    return ctx, pg, audio


def _open(pg: Any, server: LiveServer, track_id: int) -> None:
    pg.goto(f"{server.base_url}/#/track/{track_id}")
    pg.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)


def _leave(pg: Any) -> None:
    """ライブラリへ移る（プレイヤーを閉じる。最後の状態を保存する）。"""
    pg.evaluate("() => { location.hash = '#/library'; }")
    pg.wait_for_selector(".track-row", timeout=20_000)


def _states(server: LiveServer, track_id: int) -> list[dict[str, Any]]:
    with httpx.Client(base_url=server.base_url) as c:
        return list(c.get(f"/api/tracks/{track_id}/playback").json()["states"])


def _wait_state(server: LiveServer, track_id: int, pred: Any, timeout: float = 10.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while True:
        st = _states(server, track_id)
        if pred(st):
            return st
        assert time.monotonic() < deadline, f"保存されません: {st}"
        time.sleep(0.1)


def _engine(pg: Any, expr: str) -> Any:
    return pg.evaluate(f"() => {{ const e = {VIEW}.engine; return {expr}; }}")


def _loaded(pg: Any) -> list[str]:
    return sorted(_engine(pg, "[...e.tracks].filter(([, t]) => t.buffer).map(([c]) => c)"))


def _hide_toast(pg: Any) -> None:
    pg.evaluate("() => { document.getElementById('toast').hidden = true; }")


# --- 純粋な処理 ---------------------------------------------------------------------------


def test_iphone_pure_functions(page: Any, server: LiveServer) -> None:  # noqa: F811
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    res = page.evaluate(
        """async () => {
        const { audioroute: R, stemload: L, mediasession: M, resume: P } = window.__stemapp.modules;
        const D = await import('/js/device.js');
        const c = (o) => R.chooseRoute(o);
        const MB = 1024 * 1024;
        return {
          routes: [
            c({}),                                                   // PC・何も無い
            c({ hasSession: true }),                    // audioSession がある（PC でも A）
            c({ hasSession: true, hasStream: true, coarse: true }),  // iPhone（A）
            c({ hasStream: true, coarse: true }),                    // A が無いスマホ → C
            c({ hasStream: true, coarse: false }),                   // PC → そのまま
            c({ choice: 'direct', hasSession: true, coarse: true }),
            c({ choice: 'stream', hasStream: true }),
            c({ choice: 'stream', hasStream: false, hasSession: true }),
          ],
          evict: [
            // 2 分たった OFF の stem だけ捨てる（選択中は捨てない）
            L.pickEvictions([
              { code: 'a', bytes: 10, offSince: 0 }, { code: 'b', bytes: 10, offSince: 100000 },
              { code: 'c', bytes: 10, offSince: null },
            ], 125000),
            // 上限を超えたら OFF の古いものから
            L.pickEvictions([
              { code: 'a', bytes: 200 * MB, offSince: 5 },
              { code: 'b', bytes: 200 * MB, offSince: 1 },
              { code: 'c', bytes: 200 * MB, offSince: null },
            ], 10),
            L.pickEvictions([{ code: 'c', bytes: 900 * MB, offSince: null }], 10),
          ],
          mb: [L.formatMB(85 * MB), L.formatMB(1.5 * MB)],
          pos: [M.positionStateOf(100, 120, 1.25), M.positionStateOf(100, -1, 0),
                M.positionStateOf(NaN, 3, 1)],
          picked: P.pickStates([
            { device_id: 2, position_sec: 0.5 }, { device_id: 1, position_sec: 9 },
            { device_id: 3, position_sec: 83 },
          ], 1),
          other: P.describeOther({ device_name: 'PC', position_sec: 83.4 }),
          restore: (() => {
            const r = P.restoreFromState({
              position_sec: 12.5, selected: ['drums'], gains_db: { bass: -3 }, listen_preset_id: 4,
            });
            return [r.position, [...r.sel], [...r.gainsDb], r.presetId, r.playing, r.fromServer];
          })(),
          kinds: [
            D.guessKind('Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)', 5),
            D.guessKind('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 5),
            D.guessKind('Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/140', 0),
            D.guessKind('Mozilla/5.0 (Linux; Android 14) Mobile', 5),
          ],
          key: [D.deviceKey() === D.deviceKey(), /^[A-Za-z0-9_-]{8,64}$/.test(D.deviceKey())],
        };
    }"""
    )
    assert res["routes"] == [
        "direct", "session", "session", "stream", "direct", "direct", "stream", "session",
    ]
    assert res["evict"] == [["a"], ["b"], []]
    assert res["mb"] == ["85 MB", "1.5 MB"]
    assert res["pos"] == [
        {"duration": 100, "position": 100, "playbackRate": 1.25},
        {"duration": 100, "position": 0, "playbackRate": 1},
        {"duration": 0, "position": 0, "playbackRate": 1},
    ]
    assert res["picked"]["mine"]["device_id"] == 1
    assert res["picked"]["other"]["device_id"] == 3  # 位置が 1 秒未満の端末は出さない
    assert res["other"] == "PC で 1:23 まで聴いた"
    assert res["restore"] == [12.5, ["drums"], [["bass", -3]], 4, False, True]
    assert res["kinds"] == ["iphone", "ipad", "pc", "other"]
    assert res["key"] == [True, True]


def test_late_stem_sync_offline(page: Any, server: LiveServer) -> None:  # noqa: F811
    """あとから読み込んだ stem が、鳴っている stem と同じ曲の時刻で鳴り始めるか（音で測る）。

    OfflineAudioContext で Engine を動かす。同じ雑音を、鳴らしておく stem は左だけ・
    あとから足す stem は右だけに入れ、足した後の左右の相互相関のずれ（サンプル）と差の大きさを測る。
    """
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    res = page.evaluate(
        """async () => {
        const { engine: E } = window.__stemapp.modules;
        const sr = 44100;
        const noise = new Float32Array(sr * 8);
        let x = 12345;
        for (let i = 0; i < noise.length; i++) {
          x = (x * 1103515245 + 12345) & 0x7fffffff;
          noise[i] = (x / 0x7fffffff) * 2 - 1;
        }
        const scenarios = [
          { name: 'rate1', rate: 1, add: 2.0 },
          { name: 'rate1.25', rate: 1.25, add: 2.0 },
          { name: 'rate0.8', rate: 0.8, add: 1.7 },
          { name: 'loop', rate: 1, add: 2.2, loop: { start: 1.0, end: 2.5 } },
          { name: 'rate_change', rate: 1, add: 2.0, change: 1.5 },
          // サーバーで伸縮した音声（1 秒 = 曲の 1.25 秒）
          { name: 'scaled', rate: 1, add: 2.0, scale: 1.25 },
        ];
        const out = {};
        for (const sc of scenarios) {
          const off = new OfflineAudioContext(2, sr * 5, sr);
          const eng = new E.Engine(() => off);
          const mk = (ch) => {
            const b = off.createBuffer(2, noise.length, sr);
            b.getChannelData(ch).set(noise);
            return b;
          };
          eng.addTrack('a', mk(0), 1);
          eng.addTrack('b', null, 1);
          eng.rate = sc.rate;
          if (sc.loop) eng.loop = sc.loop;
          if (sc.scale) eng.bufScale = sc.scale;
          eng._startSources(0.5, 0);
          eng.playing = true;
          let info = null;
          off.suspend(sc.add).then(() => {
            if (sc.change) eng.setRate(sc.change);
            eng.addBuffer('b', mk(1));
            info = eng.tracks.get('b').startInfo;
            off.resume();
          });
          const buf = await off.startRendering();
          const L = buf.getChannelData(0);
          const R = buf.getChannelData(1);
          const from = Math.ceil((info.when + 0.05) * sr);
          const n = Math.min(sr, L.length - from);
          let best = 0, bestLag = 0;
          for (let lag = -40; lag <= 40; lag++) {
            let s = 0;
            for (let i = 0; i < n; i++) s += L[from + i] * R[from + i + lag];
            if (s > best) { best = s; bestLag = lag; }
          }
          let maxDiff = 0, energy = 0;
          for (let i = 0; i < n; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(L[from + i] - R[from + i]));
            energy += L[from + i] * L[from + i];
          }
          out[sc.name] = { lag: bestLag, maxDiff, rms: Math.sqrt(energy / n), when: info.when };
        }
        return out;
    }"""
    )
    for name, r in res.items():
        assert r["rms"] > 0.3, (name, r)  # 雑音が鳴っている
        assert r["lag"] == 0, (name, r)  # ずれは 0 サンプル
        # 同じ位置から鳴らしているので、左右はほぼ同じ（伸縮の補間の誤差だけ）
        assert r["maxDiff"] < 0.02, (name, r)
    print("あとから読み込んだ stem のずれ:", res)


def test_stream_route_delay(page: Any, server: LiveServer) -> None:  # noqa: F811
    """C の経路（MediaStream）で増える遅れを測る。

    位置の計算に入れる値（STREAM_ROUTE_LATENCY_SEC）の根拠。
    """
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    res = page.evaluate(
        """async () => {
        const R = window.__stemapp.modules.audioroute;
        const out = [];
        for (let i = 0; i < 3; i++) {
          const ctx = new AudioContext();
          out.push(await R.measureStreamLoopback(ctx));
          await ctx.close();
        }
        return { delays: out, fix: R.STREAM_ROUTE_LATENCY_SEC };
    }"""
    )
    print("MediaStream を通した音の遅れ（秒）:", res)
    assert all(d is not None for d in res["delays"]), res
    # 位置の計算に入れている値と、測った遅れの差は 1 描画単位（約 3ms）以内
    assert all(abs(d - res["fix"]) <= 0.003 for d in res["delays"]), res


# --- スマホ: 選択中の stem だけ読み込む・C の経路・ロック画面 ------------------------------------


def test_phone_lazy_route_media_session(
    browser: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    track_id, _ = _done_track(server, tmp_path, seconds=24.0)
    ctx, pg, audio = _phone(browser)
    try:
        _open(pg, server, track_id)
        view = pg.evaluate(f"() => ({{ lazy: {VIEW}.lazy, route: {VIEW}.route.mode, "
                           f"leaves: {VIEW}.tree.leaves }})")
        assert view["lazy"] is True
        # Edge には audioSession が無いので C（<audio> 経由）
        assert view["route"] == "stream"
        assert "C:" in pg.inner_text("#route-info")
        leaves = view["leaves"]
        # 開いた直後は全部 ON（初めて開いた曲）なので全部読み込む
        assert len([u for u in audio if "/renditions/" in u]) == len(leaves)
        assert pg.inner_text("#device-btn") == "端末: iPhone"

        # ソロで drums だけにして、位置 3.0 で離れる
        # → 次に開くと drums だけ読み込み、位置と選択が戻る
        pg.click("#solo-btn")
        pg.click(".stem-btn[data-code='drums'] .name")
        pg.click("#solo-btn")
        pg.evaluate(f"() => {VIEW}.seek(3.0)")
        _leave(pg)
        _wait_state(server, track_id, lambda st: st and st[0]["selected"] == ["drums"])
        audio.clear()
        _open(pg, server, track_id)
        assert len(audio) == 1 and "/renditions/" in audio[0], audio
        assert _loaded(pg) == ["drums"]
        assert pg.evaluate(f"() => [...{VIEW}.sel]") == ["drums"]
        assert _engine(pg, "e.position") == pytest.approx(3.0, abs=0.01)
        assert f"1/{len(leaves)}" in pg.inner_text("#mem-info")
        bytes_one = int(pg.get_attribute("#mem-info", "data-bytes"))
        rate = _engine(pg, "e.ctx.sampleRate")  # デコードは AudioContext の周波数になる
        assert bytes_one == pytest.approx(24.0 * rate * 2 * 4, rel=0.02)  # 1 stem・24 秒・2ch

        # 再生中に bass を ON → 読み込んでから、鳴っている drums と同じ曲の時刻で鳴り始める
        pg.click("#play-btn")
        pg.wait_for_function(f"() => {VIEW}.engine.playing && {VIEW}.engine.position > 3.2")
        assert pg.evaluate(f"() => {VIEW}.route.elementPlaying") is True
        pg.route("**/api/files/renditions/**", lambda route: (time.sleep(0.4), route.continue_()))
        pg.click(".stem-btn[data-code='bass'] .name")
        pg.wait_for_selector(".stem-btn[data-code='bass'].loading", timeout=5000)
        pg.wait_for_function(f"() => {VIEW}.engine.tracks.get('bass').source", timeout=10_000)
        pg.unroute("**/api/files/renditions/**")
        assert pg.locator(".stem-btn[data-code='bass'].loading").count() == 0
        assert _loaded(pg) == ["bass", "drums"]
        sync = pg.evaluate(
            """() => {
            const e = window.__stemapp.view.engine;
            const ref = e.tracks.get('drums').startInfo;
            const late = e.tracks.get('bass').startInfo;
            // drums の音源が late.when に鳴らしている音声データ上の位置（同じ速さで進む）
            const refPos = ref.offset + (late.when - ref.when) * ref.rate;
            return { diffSamples: (late.offset - refPos) * e.ctx.sampleRate, late, ref,
                     rate: e.rate, active: e.activeSources() };
        }"""
        )
        assert abs(sync["diffSamples"]) < 0.5, sync
        assert sync["active"] == 2
        print("あとから読み込んだ stem のずれ（Edge・C の経路）:", sync)

        # C の経路でもシーク・stem の切り替え・速度変更が効く
        pg.evaluate(f"() => {VIEW}.seek(10.0)")
        pg.wait_for_function(f"() => {VIEW}.engine.position > 10.1 && {VIEW}.engine.position < 11")
        pg.click("#tp-mode .seg-btn[data-value='pitch']")
        pg.evaluate(f"() => {VIEW}.tempo.setRatio(1.2)")
        assert _speed_over(pg) == pytest.approx(1.2, abs=0.05)
        pg.click(".stem-btn[data-code='drums'] .name")  # OFF（音声はしばらく残す）
        pg.wait_for_timeout(100)
        gains = _engine(pg, "e.gainValues()")
        assert gains["drums"] == 0 and gains["bass"] == 1
        assert "drums" in _loaded(pg)

        # Media Session: 曲名・操作・位置（速度の倍率も）
        pg.wait_for_timeout(1200)  # 位置は 1 秒ごとに更新
        ms = pg.evaluate(
            """() => {
            const m = navigator.mediaSession.metadata;
            const pos = window.__ms.positions;
            return { title: m.title, artist: m.artist, album: m.album, art: m.artwork.length,
                     handlers: Object.keys(window.__ms.handlers).sort(), last: pos[pos.length - 1],
                     state: navigator.mediaSession.playbackState };
        }"""
        )
        title = pg.evaluate(f"() => {VIEW}.track.title")
        assert ms["title"] == title and ms["art"] == 2 and ms["album"] == ""
        assert ms["handlers"] == [
            "pause", "play", "seekbackward", "seekforward", "seekto", "stop",
        ]
        assert ms["state"] == "playing"
        assert ms["last"]["playbackRate"] == pytest.approx(1.2)
        assert ms["last"]["duration"] == pytest.approx(24.0, abs=0.1)
        # ロック画面の操作: シーク・10 秒戻る・一時停止・再生
        pg.evaluate("() => window.__ms.handlers.seekto({ action: 'seekto', seekTime: 5 })")
        pg.wait_for_function(f"() => Math.abs({VIEW}.engine.position - 5.1) < 0.3")
        pg.evaluate("() => window.__ms.handlers.seekbackward({ action: 'seekbackward' })")
        pg.wait_for_function(f"() => {VIEW}.engine.position < 1")
        pg.evaluate("() => window.__ms.handlers.pause({ action: 'pause' })")
        pg.wait_for_function(f"() => !{VIEW}.engine.playing")
        assert pg.evaluate("() => navigator.mediaSession.playbackState") == "paused"
        pg.wait_for_function(f"() => {VIEW}.route.elementPlaying === false")
        pg.evaluate("() => window.__ms.handlers.play({ action: 'play' })")
        pg.wait_for_function(f"() => {VIEW}.engine.playing && {VIEW}.route.elementPlaying")
        # 組み合わせプリセットを選ぶと、その名前をアルバム欄に出す
        pg.click("#presets li .name >> nth=0")
        name = pg.inner_text("#presets li.active .name")
        assert pg.evaluate("() => navigator.mediaSession.metadata.album") == name

        # OFF にしてしばらくたった stem の音声は捨てる（ここでは時間を進めて確かめる）
        pg.evaluate(
            f"""() => {{
            const v = {VIEW};
            const old = Date.now() - 200000;
            for (const c of v.tree.leaves) if (!v.sel.has(c)) v.offSince.set(c, old);
            v.housekeeping();
        }}"""
        )
        sel = set(pg.evaluate(f"() => [...{VIEW}.sel]"))
        assert set(_loaded(pg)) <= sel
        pg.wait_for_function(f"() => {VIEW}.engine.playing")
        # はみ出さない（横スクロールが出ない）
        assert pg.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
        _hide_toast(pg)
        pg.click(".pi-more summary")
        _shot(pg, "iphone_player_phone.png")
        assert not pg.errors  # type: ignore[attr-defined]
    finally:
        ctx.close()


def test_phone_keep_mode_loads_selected_only(
    browser: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    """ピッチを保つ方式（サーバーで伸縮した音声）でも、選択中の stem だけ読み込む。"""
    track_id, _ = _done_track(server, tmp_path, seconds=12.0)
    ctx, pg, audio = _phone(browser)
    try:
        _open(pg, server, track_id)
        assert pg.evaluate(f"() => {VIEW}.tempo.mode") == "keep"  # スマホの既定
        pg.click("#solo-btn")
        pg.click(".stem-btn[data-code='drums'] .name")
        pg.click("#solo-btn")
        pg.click("#play-btn")
        audio.clear()
        pg.evaluate(f"() => {VIEW}.tempo.setRatio(1.1)")
        pg.wait_for_function(f"() => {VIEW}.tempo.activeKey === '1.100'", timeout=30_000)
        tempo_urls = [u for u in audio if "/api/files/tempo/" in u]
        assert len(tempo_urls) == 1, audio
        assert _loaded(pg) == ["drums"]
        assert _engine(pg, "e.bufScale") == pytest.approx(1.1)
        # ON にした stem は伸縮済みの音声から読み込む
        pg.click(".stem-btn[data-code='bass'] .name")
        pg.wait_for_function(f"() => {VIEW}.engine.tracks.get('bass').buffer", timeout=10_000)
        assert len([u for u in audio if "/api/files/tempo/" in u]) == 2
        dur = _engine(pg, "e.tracks.get('bass').buffer.duration")
        assert dur == pytest.approx(12.0 / 1.1, abs=0.01)
        assert not pg.errors  # type: ignore[attr-defined]
    finally:
        ctx.close()


def test_pc_with_audio_session_is_harmless(
    browser: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    """audioSession がある PC のブラウザ（Safari など）では A を使う。音の経路は今までどおり。"""
    track_id, _ = _done_track(server, tmp_path, seconds=12.0)
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, locale="ja-JP")
    # Edge には無いので、Safari と同じ形の navigator.audioSession を用意する
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'audioSession', "
        "{ value: { type: 'auto', state: 'inactive' }, configurable: true });"
    )
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    try:
        _open(pg, server, track_id)
        assert pg.evaluate(f"() => [{VIEW}.route.mode, {VIEW}.lazy]") == ["session", False]
        assert pg.evaluate("() => navigator.audioSession.type") == "auto"  # 再生するまで変えない
        pg.click("#play-btn")
        pg.wait_for_function(f"() => {VIEW}.engine.position > 0.3")
        assert pg.evaluate("() => navigator.audioSession.type") == "playback"
        # 出力は今までどおり AudioContext の destination（<audio> を使わない）・位置の遅れも無い
        assert pg.evaluate(f"() => [{VIEW}.route.audio, {VIEW}.engine.routeLatency]") == [None, 0]
        assert _speed_over(pg) == pytest.approx(1.0, abs=0.05)
        pg.click("#play-btn")
        pg.wait_for_function(f"() => !{VIEW}.engine.playing")
        assert not errors
    finally:
        ctx.close()


# --- 続きから再生 ---------------------------------------------------------------------------


def test_resume_per_device(
    page: Any, browser: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    track_id, job_id = _done_track(server, tmp_path, seconds=12.0)
    # PC: 位置・選択・速度を変えて離れる
    _open(page, server, track_id)
    assert page.evaluate(f"() => {VIEW}.lazy") is False  # PC は全部読み込む（今までどおり）
    assert page.evaluate(f"() => {VIEW}.route.mode") == "direct"
    assert page.inner_text("#device-btn") == "端末: PC"
    page.locator("body").focus()
    page.keyboard.press("1")  # 1 番目の stem を OFF
    page.evaluate(f"() => {VIEW}.tempo.setMode('pitch')")
    page.evaluate(f"() => {VIEW}.tempo.setRatio(1.05)")
    page.evaluate(f"() => {VIEW}.seek(11.0)")
    first = page.evaluate(f"() => {VIEW}.tree.order[0]")
    # 一定間隔（5 秒）でも保存される
    st = _wait_state(server, track_id, lambda s: bool(s) and s[0]["position_sec"] > 10.9, timeout=8)
    assert st[0]["device_name"] == "PC" and st[0]["job_id"] == job_id
    assert first not in st[0]["selected"]
    assert (st[0]["tempo_ratio"], st[0]["tempo_mode"]) == (1.05, "pitch")
    page.evaluate(f"() => {VIEW}.seek(2.5)")
    _leave(page)
    _wait_state(server, track_id, lambda s: bool(s) and abs(s[0]["position_sec"] - 2.5) < 0.05)
    # 開き直すと位置・選択・速度が戻る
    _open(page, server, track_id)
    assert _engine(page, "e.position") == pytest.approx(2.5, abs=0.05)
    assert first not in page.evaluate(f"() => [...{VIEW}.sel]")
    assert page.evaluate(f"() => [{VIEW}.tempo.ratio, {VIEW}.tempo.mode]") == [1.05, "pitch"]
    assert _engine(page, "e.rate") == pytest.approx(1.05)
    # 端末の名前を変える
    page.click("#device-btn")
    page.fill(".modal-back input", "居間の PC")
    page.click(".modal-back .primary")
    page.wait_for_function(
        "() => document.querySelector('#device-btn').textContent === '端末: 居間の PC'")

    # iPhone（別の端末）: 自分の状態は無いので最初から。PC の位置を「〜で 0:02 まで聴いた」と出す
    ctx, pg, _ = _phone(browser)
    try:
        _open(pg, server, track_id)
        assert _engine(pg, "e.position") == 0
        btn = pg.locator("#resume-other")
        assert btn.is_visible()
        assert btn.inner_text().startswith("居間の PC で 0:02 まで聴いた")
        _hide_toast(pg)
        _shot(pg, "iphone_resume_phone.png")
        btn.click()
        assert _engine(pg, "e.position") == pytest.approx(2.5, abs=0.05)
        assert not pg.errors  # type: ignore[attr-defined]
    finally:
        ctx.close()
    assert not page.errors  # type: ignore[attr-defined]
