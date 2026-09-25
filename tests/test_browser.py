"""画面のテスト（PC の Microsoft Edge を Playwright で動かす）。

`uv run pytest -m browser` で実行する。

サーバーは一時データフォルダで動かし、分割は FakeSeparator、配信用データ（Opus/WebM）は
本物の ffmpeg で作る。スクリーンショットは data/cache/screens/ に保存する（コミットしない）。
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, select

from audio_helpers import synth_mix, write_source
from browser_helpers import EDGE_ARGS, SCREENS_DIR, LiveServer, run_server
from stemapp.config import Settings
from stemapp.models import ListenPreset, SeparationJob, Stem, StemRendition, Track

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

playwright_api = pytest.importorskip("playwright.sync_api")

DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveServer]:
    data = tmp_path_factory.mktemp("browser") / "data"
    settings = Settings(_env_file=None, data_dir=data)  # type: ignore[call-arg]
    with run_server(settings, fake_delay=0.8) as srv:
        yield srv


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge", headless=True, args=EDGE_ARGS)
        except Exception as e:  # Edge が無い環境
            pytest.skip(f"Microsoft Edge を起動できません: {e}")
        yield b
        b.close()


@pytest.fixture
def page(browser: Any) -> Iterator[Any]:
    ctx = browser.new_context(viewport=DESKTOP, locale="ja-JP")
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.errors = errors  # type: ignore[attr-defined]
    yield pg
    ctx.close()


def _shot(page: Any, name: str) -> Path:
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENS_DIR / name
    page.screenshot(path=str(path), full_page=True)
    return path


def _gains(page: Any) -> dict[str, float]:
    return page.evaluate("() => window.__stemapp.view.engine.gainValues()")


def _wait_gains(page: Any, expect: dict[str, float], timeout: float = 3.0) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    while True:
        g = _gains(page)
        if all(abs(g[k] - v) < 1e-3 for k, v in expect.items()):
            return g
        if time.monotonic() > deadline:
            raise AssertionError(f"GainNode の値が変わりません: {g}（期待 {expect}）")
        time.sleep(0.05)


LEAVES = ["lead_vocal", "backing_vocal", "drums", "bass", "guitar", "piano", "other"]


# --- 純粋な処理（ブラウザの中で評価） ----------------------------------------------


def test_pure_functions(page: Any, server: LiveServer) -> None:
    page.goto(server.base_url + "/#/library")
    page.wait_for_function("() => window.__stemapp && window.__stemapp.modules")
    res = page.evaluate(
        """() => {
        const { peaks, selection: S, engine } = window.__stemapp.modules;
        // STPK を組み立てて読む
        const n = 3;
        const buf = new ArrayBuffer(20 + n * 2);
        const v = new DataView(buf);
        [..."STPK"].forEach((c, i) => v.setUint8(i, c.charCodeAt(0)));
        v.setUint16(4, 1, true); v.setUint32(8, 256, true); v.setUint32(12, 44100, true);
        v.setUint32(16, n, true);
        new Int8Array(buf, 20).set([-10, 20, -127, 127, 0, 5]);
        const pk = peaks.parsePeaks(buf);
        let badMagic = null;
        try { peaks.parsePeaks(new ArrayBuffer(24)); } catch (e) { badMagic = e.message; }
        const levels = [256, 1024, 4096, 16384];
        // 概観: 240 秒（10584000 サンプル）を 1400px → 点の数が 1400 以下の最小 = 16384（646 点）
        const ov = peaks.chooseOverviewLevel(levels, 10584000, 1400);
        const ovWide = peaks.chooseOverviewLevel(levels, 44100 * 10, 2000);  // 10 秒 → 256
        const zm = peaks.chooseZoomLevel(levels, 8 * 44100 / 2800);  // 126 spp → 最小の 256
        const zm2 = peaks.chooseZoomLevel(levels, 32 * 44100 / 1000);  // 1411 → 1024
        const mm = peaks.rangeMinMax(pk, 0, 2);
        // 時刻の計算（ループの折り返し）
        const loop = { start: 10, end: 14 };
        const pos = [
          engine.positionAt(5, 3, null, 100),
          engine.positionAt(11, 5, loop, 100),   // 16 → 12
          engine.positionAt(8, 7, loop, 100),    // 15 → 11（始点より前から始めても終点で折り返す）
          engine.positionAt(20, 3, loop, 100),   // 終点より後から始めたら折り返さない
          engine.positionAt(99, 5, null, 100),   // 曲の長さで止まる
        ];
        // 親子のルール
        const stems = [
          { code: "vocals", parent_code: null }, { code: "lead_vocal", parent_code: "vocals" },
          { code: "backing_vocal", parent_code: "vocals" }, { code: "drums", parent_code: null },
          { code: "bass", parent_code: null },
        ];
        const t = S.buildTree(stems);
        let sel = S.allOn(t);
        const leaves = [...sel].sort();
        const gainsAll = S.targetGains(t, sel);
        sel = S.toggle(t, sel, "lead_vocal");
        const partial = S.stateOf(t, sel, "vocals");
        const gainsPartial = S.targetGains(t, sel);
        sel = S.toggle(t, sel, "vocals");  // 一部 → 全部 ON
        const afterParent = S.stateOf(t, sel, "vocals");
        sel = S.toggle(t, sel, "vocals");  // 全部 ON → 全部 OFF
        const offParent = [
          S.stateOf(t, sel, "vocals"), sel.has("lead_vocal"), sel.has("backing_vocal"),
        ];
        const solo = [...S.solo(t, "vocals")].sort();
        const group = S.groupLeaves(t, { members: ["vocals", "bass", "piano"] });
        const items = S.selectionToItems(t, new Set(["lead_vocal", "backing_vocal", "bass"]),
          (c) => ({ vocals: 1, lead_vocal: 2, backing_vocal: 3, drums: 4, bass: 5 })[c]);
        const items2 = S.selectionToItems(t, new Set(["backing_vocal"]),
          (c) => ({ vocals: 1, lead_vocal: 2, backing_vocal: 3, drums: 4, bass: 5 })[c]);
        const fromPreset = S.presetToSelection(t,
          { items: [{ group_code: "g" }, { stem_type_code: "drums", gain_db: -6 }] },
          [{ code: "g", group_id: 9, members: ["vocals"] }]);
        return {
          pk: [pk.samplesPerPx, pk.sampleRate, pk.points, Array.from(pk.data)], badMagic,
          ov, ovWide, zm, zm2, mm, pos, leaves, gainsAll, partial, gainsPartial, afterParent,
          offParent, solo, group, items, items2,
          fromPreset: [[...fromPreset.sel].sort(), fromPreset.gainsDb.get("drums")],
          time: [window.__stemapp.modules.engine.clampTime(-3, 10)],
        };
    }"""
    )
    assert res["pk"] == [256, 44100, 3, [-10, 20, -127, 127, 0, 5]]
    assert res["badMagic"] == "波形データの形式が違います。"
    assert res["ov"] == 16384 and res["ovWide"] == 256
    assert res["zm"] == 256 and res["zm2"] == 1024
    assert res["mm"] == pytest.approx([-1.0, 1.0])
    assert res["pos"] == pytest.approx([8, 12, 11, 23, 100])
    assert res["leaves"] == ["backing_vocal", "bass", "drums", "lead_vocal"]
    # 子に分かれた親（vocals）は鳴らさない
    assert res["gainsAll"] == {
        "vocals": 0, "lead_vocal": 1, "backing_vocal": 1, "drums": 1, "bass": 1,
    }
    assert res["partial"] == "partial"
    assert res["gainsPartial"]["vocals"] == 0 and res["gainsPartial"]["lead_vocal"] == 0
    assert res["afterParent"] == "on"
    assert res["offParent"] == ["off", False, False]
    assert res["solo"] == ["backing_vocal", "lead_vocal"]
    assert res["group"] == ["lead_vocal", "backing_vocal", "bass"]
    assert res["items"] == [{"stem_type_id": 1, "gain_db": 0}, {"stem_type_id": 5, "gain_db": 0}]
    assert res["items2"] == [{"stem_type_id": 3, "gain_db": 0}]
    assert res["fromPreset"] == [["backing_vocal", "drums", "lead_vocal"], -6]
    assert res["time"] == [0]
    assert not page.errors  # type: ignore[attr-defined]


# --- 取り込みから再生・切り替え・プリセット・キュー・ループまで ------------------------------


def test_full_flow(page: Any, server: LiveServer, tmp_path: Path) -> None:
    page.goto(server.base_url + "/")
    page.wait_for_selector("text=まだ曲がありません")
    _shot(page, "library_empty_pc.png")

    src = write_source(tmp_path / "テスト曲.wav", synth_mix(6.0), subtype="PCM_16")
    page.set_input_files("#file-input", str(src))
    # 分割中の表示（進捗バーと段階）。速く終わって見えなかったときは撮らない
    try:
        page.wait_for_selector(".track-row .progress", timeout=20_000)
        _shot(page, "library_progress_pc.png")
    except playwright_api.TimeoutError:
        pass
    # 取り込み → 分割（Fake）→ 配信用データ（本物の ffmpeg）が終わると「再生」が出る
    page.wait_for_selector(".track-row .btn:has-text('再生')", timeout=90_000)
    assert page.locator(".track-row .badge", has_text="分割済み").count() == 1
    _shot(page, "library_pc.png")
    page.set_viewport_size(PHONE)
    _shot(page, "library_phone.png")
    page.set_viewport_size(DESKTOP)

    page.click(".track-row .btn:has-text('再生')")
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    assert page.locator(".stem-btn").count() == 8

    # 再生が始まる（AudioContext の時刻と再生位置が進む）
    page.click("#play-btn")
    page.wait_for_function(
        "() => window.__stemapp.view.engine.playing && window.__stemapp.view.engine.position > 0.3",
        timeout=10_000,
    )
    t0 = page.evaluate("() => window.__stemapp.view.engine.ctx.currentTime")
    time.sleep(0.4)
    t1 = page.evaluate("() => window.__stemapp.view.engine.ctx.currentTime")
    assert t1 > t0
    # 全 stem の長さがそろっている（Opus の先頭の無音でずれていない）
    durations = page.evaluate(
        """() => Object.fromEntries([...window.__stemapp.view.engine.tracks]
            .filter(([, t]) => t.buffer).map(([c, t]) => [c, t.buffer.duration]))"""
    )
    assert set(durations) == set(LEAVES)
    assert max(durations.values()) - min(durations.values()) < 0.001
    assert abs(max(durations.values()) - 6.0) < 0.05, durations

    # 最初は全部 ON（親の vocals は鳴らさない）
    _wait_gains(page, {**dict.fromkeys(LEAVES, 1.0), "vocals": 0.0})
    # stem ボタン（ドラム）で OFF
    page.click(".stem-btn[data-code='drums']")
    _wait_gains(page, {"drums": 0.0, "bass": 1.0})
    assert page.evaluate("() => window.__stemapp.view.engine.playing")  # 止まらない
    # キーボード 1（ボーカル＝親）で子をまとめて OFF
    page.keyboard.press("1")
    _wait_gains(page, {"lead_vocal": 0.0, "backing_vocal": 0.0, "vocals": 0.0, "bass": 1.0})
    # グループ「リズム」（drums+bass）: 一部 ON → 全部 ON
    page.click(".chip[data-group='rhythm']")
    _wait_gains(page, {"drums": 1.0, "bass": 1.0})
    # ソロ（Shift＋クリック）
    page.click(".stem-btn[data-code='bass']", modifiers=["Shift"])
    _wait_gains(page, {**dict.fromkeys(LEAVES, 0.0), "bass": 1.0})
    # 全部
    page.click("#all-btn")
    _wait_gains(page, dict.fromkeys(LEAVES, 1.0))

    # 組み合わせプリセット: ベースだけを保存 → 全部に戻す → 選ぶと切り替わる
    page.click(".stem-btn[data-code='bass']", modifiers=["Shift"])
    page.click("#save-preset-btn")
    page.fill(".modal input", "ベースだけ")
    page.click(".modal button[type='submit']")
    page.wait_for_selector("#presets li:has-text('ベースだけ')")
    with server.session_factory() as s:
        assert s.scalar(select(ListenPreset).where(ListenPreset.name == "ベースだけ")) is not None
    page.click("#all-btn")
    _wait_gains(page, dict.fromkeys(LEAVES, 1.0))
    page.click("#presets li:has-text('ベースだけ') .name")
    _wait_gains(page, {**dict.fromkeys(LEAVES, 0.0), "bass": 1.0})
    assert "active" in (page.get_attribute("#presets li:has-text('ベースだけ')", "class") or "")
    # 並び替え（いちばん上へ）
    for _ in range(10):
        first = page.locator("#presets li").first.inner_text()
        if "ベースだけ" in first:
            break
        page.click("#presets li:has-text('ベースだけ') button[aria-label$='を上へ']")
        page.wait_for_timeout(300)
    assert "ベースだけ" in page.locator("#presets li").first.inner_text()
    page.click("#all-btn")

    # キュー: 1.0 秒に追加 → 3.0 秒で「終点」→ A-B ループ
    page.evaluate("() => { const e = window.__stemapp.view.engine; e.pause(); e.seek(1.0); }")
    page.click("#add-cue-btn")
    page.wait_for_selector("#cues li[data-cue-id]")
    cue_pos = page.evaluate("() => window.__stemapp.view.cues[0].position_sec")
    assert cue_pos == pytest.approx(1.0, abs=0.01)
    page.evaluate("() => window.__stemapp.view.engine.seek(3.0)")
    page.click("#cues li[data-cue-id] button:has-text('終点')")
    page.wait_for_selector("#cues li.active")
    loop = page.evaluate("() => window.__stemapp.view.engine.loop")
    assert loop["start"] == pytest.approx(1.0, abs=0.01)
    assert loop["end"] == pytest.approx(3.0, abs=0.01)
    page.click("#play-btn")  # 再生（3.0 は区間の外なので始点 1.0 から）
    # 2 秒の区間を 3 秒以上鳴らし、位置が区間内に収まり、折り返したことを確かめる
    samples = page.evaluate(
        """async () => {
        const e = window.__stemapp.view.engine;
        const out = [];
        for (let i = 0; i < 32; i++) {
          out.push(e.position);
          await new Promise(r => setTimeout(r, 100));
        }
        return out;
    }"""
    )
    assert all(0.99 <= p <= 3.01 for p in samples), samples
    assert any(samples[i + 1] < samples[i] for i in range(len(samples) - 1)), samples
    _shot(page, "player_pc.png")
    # L でループを切る
    page.keyboard.press("l")
    page.wait_for_function("() => window.__stemapp.view.engine.loop === null")
    assert page.evaluate("() => window.__stemapp.view.loopOn") is False
    # ←→ で 5 秒戻る・進む
    page.evaluate("() => window.__stemapp.view.engine.pause()")
    page.evaluate("() => window.__stemapp.view.engine.seek(0.5)")
    position = "() => window.__stemapp.view.engine.position"
    page.keyboard.press("ArrowRight")
    assert page.evaluate(position) == pytest.approx(5.5, abs=0.01)
    page.keyboard.press("ArrowLeft")
    assert page.evaluate(position) == pytest.approx(0.5, abs=0.01)
    # スペースで再生・停止（stem ボタンにフォーカスがあっても、ボタンは押されない）
    page.focus(".stem-btn[data-code='drums']")
    before = _gains(page)["drums"]
    page.keyboard.press("Space")
    page.wait_for_function("() => window.__stemapp.view.engine.playing")
    page.keyboard.press("Space")
    page.wait_for_function("() => !window.__stemapp.view.engine.playing")
    page.wait_for_timeout(100)
    assert _gains(page)["drums"] == before

    page.set_viewport_size(PHONE)
    page.wait_for_timeout(300)
    _shot(page, "player_phone.png")
    page.set_viewport_size(DESKTOP)
    assert not page.errors  # type: ignore[attr-defined]


def _only_track(server: LiveServer) -> tuple[int, int]:
    with server.session_factory() as s:
        track = s.scalars(select(Track).order_by(Track.track_id)).first()
        assert track is not None
        job = s.scalars(
            select(SeparationJob).where(
                SeparationJob.track_id == track.track_id, SeparationJob.status == "done"
            )
        ).first()
        assert job is not None
        return track.track_id, job.job_id


def test_rebuild_delivery(page: Any, server: LiveServer) -> None:
    """配信用データが無いと知らせ、「作り直す」でワーカーが作り、再生できるようになる。"""
    track_id, job_id = _only_track(server)
    with server.session_factory() as s:
        ids = select(Stem.stem_id).where(Stem.job_id == job_id)
        s.execute(
            delete(StemRendition).where(
                StemRendition.stem_id.in_(ids), StemRendition.purpose == "stream"
            )
        )
        s.commit()
    page.goto(f"{server.base_url}/#/track/{track_id}")
    page.wait_for_selector("text=配信用データがありません")
    page.click("#rebuild-btn")
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    assert not page.errors  # type: ignore[attr-defined]


def test_delete_track(page: Any, server: LiveServer) -> None:
    track_id, _ = _only_track(server)
    page.goto(server.base_url + "/#/library")
    row = f".track-row[data-track-id='{track_id}']"
    page.wait_for_selector(row)
    page.click(f"{row} .btn:has-text('削除')")
    page.wait_for_selector(".modal")
    page.click(".modal .btn:has-text('やめる')")
    assert page.locator(row).count() == 1
    page.click(f"{row} .btn:has-text('削除')")
    page.click(".modal .btn:has-text('削除する')")
    page.wait_for_selector("text=まだ曲がありません")
    with server.session_factory() as s:
        assert s.get(Track, track_id) is None


def test_login_screen(browser: Any, tmp_path: Path) -> None:
    data = tmp_path / "data"
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, data_dir=data, passcode="ひらけごま"
    )
    with run_server(settings, with_worker=False) as srv:
        ctx = browser.new_context(viewport=PHONE, locale="ja-JP")
        page = ctx.new_page()
        page.goto(srv.base_url + "/")
        page.wait_for_selector("#passcode")
        _shot(page, "login_phone.png")
        page.fill("#passcode", "ちがう")
        page.click("button[type='submit']")
        page.wait_for_selector("text=パスコードが違います。")
        page.fill("#passcode", "ひらけごま")
        page.click("button[type='submit']")
        page.wait_for_selector("text=まだ曲がありません")
        page.click("text=ログアウト")
        page.wait_for_selector("#passcode")
        ctx.close()
