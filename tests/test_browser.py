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

import httpx
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


@pytest.fixture
def server(tmp_path: Path) -> Iterator[LiveServer]:
    """テストごとに空のデータフォルダで起動する（テストの順序に依存しない）。"""
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    with run_server(settings, fake_delay=0.8) as srv:
        yield srv


def _done_track(server: LiveServer, tmp_path: Path, name: str = "用意した曲") -> tuple[int, int]:
    """API で曲を取り込み・分割し、終わるまで待つ（track_id, job_id）。"""
    src = write_source(tmp_path / f"{name}.wav", synth_mix(3.0), subtype="PCM_16")
    with httpx.Client(base_url=server.base_url, timeout=30) as c:
        res = c.post(
            "/api/imports",
            files={"file": (src.name, src.read_bytes(), "audio/wav")},
            data={"separate": "true", "preset": "fast"},
        )
        assert res.status_code == 202, res.text
        source_id = res.json()["source_id"]
        deadline = time.monotonic() + 90
        while True:
            imp = c.get(f"/api/imports/{source_id}").json()
            if imp["status"] == "done" and imp["job_id"]:
                job = c.get(f"/api/jobs/{imp['job_id']}").json()
                if job["status"] == "done":
                    return imp["track_id"], job["job_id"]
            assert imp["status"] != "failed", imp
            assert time.monotonic() < deadline, "分割が終わりません"
            time.sleep(0.2)


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
    # 再生ボタン: ▶ のアイコンが見え、赤の塗りつぶしではない
    play = page.locator("#play-btn")
    assert play.is_visible()
    assert play.get_attribute("aria-label") == "再生"
    assert play.locator("svg path").count() == 1
    style = page.evaluate(
        """() => { const s = getComputedStyle(document.querySelector("#play-btn"));
            return [s.backgroundColor, s.backgroundImage, s.borderRadius]; }"""
    )
    assert "255, 59, 78" not in style[0] + style[1], style
    assert style[2] != "50%", style

    # 保存フォルダを開く（同じ PC からなので出る。エクスプローラーは開かず、呼ばれたことを確かめる）
    folder_btn = page.locator("#open-folder-btn")
    assert folder_btn.is_visible()
    assert "FLAC（24bit）" in (folder_btn.get_attribute("title") or "")
    folder_btn.click()
    page.wait_for_selector("#toast:has-text('保存フォルダを開きました')")
    assert len(server.opened_folders) == 1
    assert (server.opened_folders[0] / "drums.flac").is_file()

    # 再生が始まる（AudioContext の時刻と再生位置が進む）
    # 二重の play()（resume を待つ間の2回目）でも、音源は stem ごとに1つだけ
    counts = page.evaluate(
        """async () => {
        const e = window.__stemapp.view.engine;
        await Promise.all([e.play(), e.play(), e.play()]);
        const playing = e.activeSources();
        e.pause();
        return [playing, e.activeSources(), e.playing];
    }"""
    )
    assert counts == [len(LEAVES), 0, False]
    # play() の直後に pause() したら鳴らさない
    page.evaluate(
        """async () => {
        const e = window.__stemapp.view.engine;
        const p = e.play(); e.pause(); await p;
    }"""
    )
    assert page.evaluate("() => window.__stemapp.view.engine.activeSources()") == 0
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
    # 0 キー: 全部 ⇔ 直前の組み合わせ（ベースだけ）を行き来する
    page.keyboard.press("0")
    _wait_gains(page, {**dict.fromkeys(LEAVES, 0.0), "bass": 1.0})
    page.keyboard.press("0")
    _wait_gains(page, dict.fromkeys(LEAVES, 1.0))
    assert "on" in (page.get_attribute("#all-btn", "class") or "")
    page.keyboard.press("0")
    _wait_gains(page, {**dict.fromkeys(LEAVES, 0.0), "bass": 1.0})
    # 組み合わせを変えてから 0 → 全部 → 0 で、変えた後のものに戻る
    page.click(".stem-btn[data-code='drums']")
    _wait_gains(page, {"drums": 1.0, "bass": 1.0, "guitar": 0.0})
    page.keyboard.press("0")
    _wait_gains(page, dict.fromkeys(LEAVES, 1.0))
    page.keyboard.press("0")
    _wait_gains(page, {**dict.fromkeys(LEAVES, 0.0), "drums": 1.0, "bass": 1.0})
    page.keyboard.press("0")
    _wait_gains(page, dict.fromkeys(LEAVES, 1.0))
    assert page.locator(".keys-help kbd", has_text="0").count() == 1

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
    # Space の押しっぱなし（repeat）は無視する
    page.evaluate(
        """() => document.dispatchEvent(new KeyboardEvent("keydown",
            { key: " ", code: "Space", repeat: true, bubbles: true }))"""
    )
    page.wait_for_timeout(100)
    assert page.evaluate("() => window.__stemapp.view.engine.playing") is False

    page.set_viewport_size(PHONE)
    page.wait_for_timeout(300)
    _shot(page, "player_phone.png")
    page.set_viewport_size(DESKTOP)
    assert not page.errors  # type: ignore[attr-defined]


def test_rebuild_delivery(page: Any, server: LiveServer, tmp_path: Path) -> None:
    """配信用データが無いと知らせ、「作り直す」でワーカーが作り、再生できるようになる。"""
    track_id, job_id = _done_track(server, tmp_path)
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


def test_delete_track(page: Any, server: LiveServer, tmp_path: Path) -> None:
    track_id, _ = _done_track(server, tmp_path)
    page.goto(server.base_url + "/#/library")
    row = f".track-row[data-track-id='{track_id}']"
    page.wait_for_selector(row)
    page.click(f"{row} .btn:has-text('削除')")
    page.wait_for_selector(".modal")
    page.click(".modal .btn:has-text('やめる')")
    assert page.locator(row).count() == 1
    page.click(f"{row} .btn:has-text('削除')")
    page.click(".modal .btn:has-text('削除する')")
    page.wait_for_selector(row, state="detached")
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


def test_load_failure_aborts_rest(page: Any, server: LiveServer, tmp_path: Path) -> None:
    """1つの stem の音声が読めなければ、残りの取得を中断して理由を出す。"""
    track_id, _ = _done_track(server, tmp_path)
    requested: list[str] = []
    failed_once: list[str] = []

    def handle(route: Any) -> None:
        requested.append(route.request.url)
        if not failed_once:
            failed_once.append(route.request.url)
            route.fulfill(status=500, content_type="application/json",
                          body='{"detail": "テスト用の失敗"}')
        else:
            route.continue_()

    page.route("**/api/files/renditions/*", handle)
    page.goto(f"{server.base_url}/#/track/{track_id}")
    page.wait_for_selector("text=読み込めませんでした", timeout=30_000)
    assert "テスト用の失敗" in page.inner_text("#loading")
    page.wait_for_timeout(500)
    # 並行して取得していた分（最大 3）を除き、残りの stem は取りに行かない
    assert len(requested) <= 3, requested
    assert page.locator("#play-btn[disabled]").count() == 1


def test_open_folder_hidden_for_remote(page: Any, server: LiveServer, tmp_path: Path) -> None:
    """同じ PC 以外（/api/me が can_open_folder=false）では「保存フォルダを開く」を出さない。"""
    track_id, _job_id = _done_track(server, tmp_path)

    def remote_me(route: Any) -> None:
        route.fulfill(
            json={"authenticated": True, "passcode_required": False,
                  "local_client": False, "can_open_folder": False}
        )

    page.route("**/api/me", remote_me)
    page.goto(f"{server.base_url}/#/track/{track_id}")
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    assert page.locator("#open-folder-btn").count() == 0
    assert page.locator("#play-btn svg").count() == 1
    assert not page.errors  # type: ignore[attr-defined]


def _wait_jobs_done(server: LiveServer, track_id: int, count: int) -> list[dict[str, Any]]:
    with httpx.Client(base_url=server.base_url, timeout=30) as c:
        deadline = time.monotonic() + 90
        while True:
            jobs = c.get(f"/api/tracks/{track_id}").json()["jobs"]
            done = [j for j in jobs if j["status"] == "done"]
            if len(done) >= count:
                return done
            assert not any(j["status"] == "failed" for j in jobs), jobs
            assert time.monotonic() < deadline, "分割が終わりません"
            time.sleep(0.2)


def test_switch_separation_job(page: Any, server: LiveServer, tmp_path: Path) -> None:
    """同じ曲を別の分け方（実験プリセット）で分割し、プレイヤーで切り替えて聴き比べる。"""
    track_id, fast_job = _done_track(server, tmp_path)
    page.goto(server.base_url + "/#/library")
    row = f".track-row[data-track-id='{track_id}']"
    page.wait_for_selector(row)
    # 実験プリセットは通常隠れていて、「実験を表示」で出る
    exp_opt = "#preset-select optgroup option[value='exp_resid_vocals']"
    assert page.locator(exp_opt).count() == 0
    page.check("#show-experimental")
    assert page.locator(exp_opt).count() == 1
    page.select_option("#preset-select", "exp_resid_vocals")
    page.click(f"{row} .btn:has-text('別の分け方で分割')")
    page.wait_for_selector("#toast:has-text('分割ジョブを登録しました')")
    done = _wait_jobs_done(server, track_id, 2)
    exp_job = next(j["job_id"] for j in done if j["preset"] == "exp_resid_vocals")

    page.goto(f"{server.base_url}/#/track/{track_id}")
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    view = "window.__stemapp.view"
    # 既定は実験でない方。切り替えの選択肢は2つ
    assert page.evaluate(f"() => {view}.jobId") == fast_job
    assert page.locator("#job-select option").count() == 2
    assert page.input_value("#job-select") == str(fast_job)
    assert "実験: ボーカル＝元の曲−楽器" in page.inner_text("#job-select")
    assert "補正前の残差" in page.inner_text("#job-metric")

    # ドラムを OFF、1.5 秒から再生してから、分け方を切り替える
    page.click(".stem-btn[data-code='drums']")
    _wait_gains(page, {"drums": 0.0, "bass": 1.0})
    page.evaluate(f"() => {view}.seek(1.5)")
    page.click("#play-btn")
    page.wait_for_function(f"() => {view}.engine.playing")
    page.select_option("#job-select", str(exp_job))
    page.wait_for_function(
        f"() => {view}.jobId === {exp_job} && {view}.ready && {view}.engine.playing",
        timeout=60_000,
    )
    pos = page.evaluate(f"() => {view}.engine.position")
    assert 1.4 < pos < 3.0, pos  # 再生位置を保つ（読み込みの間は止まっている）
    _wait_gains(page, {"drums": 0.0, "bass": 1.0, "lead_vocal": 1.0})  # 選択を保つ
    assert page.input_value("#job-select") == str(exp_job)
    page.click("#play-btn")  # 止める

    # 数値の比較（stem ごとの RMS と残差）
    page.click(".levels-box summary")
    assert page.locator(".levels tbody tr").count() == 2
    assert page.locator(".levels tr.current").count() == 1
    _shot(page, "player_job_switch_pc.png")
    page.set_viewport_size(PHONE)
    _shot(page, "player_job_switch_phone.png")
    page.set_viewport_size(DESKTOP)

    # 選んだ分け方は覚えている（読み直しても同じ）
    page.reload()
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    assert page.evaluate(f"() => {view}.jobId") == exp_job
    assert not page.errors  # type: ignore[attr-defined]

    # 「この分け方を削除」: 確認して消し、残った方に切り替える（再生位置は保つ）
    page.evaluate(f"() => {view}.seek(2.0)")
    page.click("#delete-job-btn")
    page.wait_for_selector(".modal")
    page.click(".modal .btn:has-text('やめる')")
    assert page.evaluate(f"() => {view}.jobId") == exp_job
    page.click("#delete-job-btn")
    page.click(".modal .btn:has-text('削除する')")
    page.wait_for_function(f"() => {view}.jobId === {fast_job} && {view}.ready", timeout=60_000)
    assert page.evaluate(f"() => {view}.engine.position") == pytest.approx(2.0, abs=0.05)
    with server.session_factory() as s:
        assert s.get(SeparationJob, exp_job) is None
        assert s.get(SeparationJob, fast_job) is not None
    # 分け方が1つだけになったら、切り替えも削除も使えない（曲は残る）
    assert page.locator("#job-select[disabled]").count() == 1
    assert page.locator("#delete-job-btn").count() == 0
    assert not page.errors  # type: ignore[attr-defined]


def _second_job(server: LiveServer, track_id: int, preset: str) -> int:
    with httpx.Client(base_url=server.base_url, timeout=30) as c:
        res = c.post(f"/api/tracks/{track_id}/jobs", json={"preset": preset})
        assert res.status_code == 201, res.text
        job_id = int(res.json()["job"]["job_id"])
    _wait_jobs_done(server, track_id, 2)
    return job_id


def test_switch_job_while_loading(page: Any, server: LiveServer, tmp_path: Path) -> None:
    """読み込み中は切り替えられない。読み込みに失敗した後の切り替えは、前の状態を引き継ぐ。"""
    track_id, fast_job = _done_track(server, tmp_path)
    exp_job = _second_job(server, track_id, "exp_kara_mix")
    view = "window.__stemapp.view"
    page.goto(f"{server.base_url}/#/track/{track_id}")
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    assert page.evaluate(f"() => {view}.jobId") == fast_job
    page.evaluate(f"() => {view}.seek(1.5)")
    page.click(".stem-btn[data-code='drums']")
    _wait_gains(page, {"drums": 0.0})

    # 切り替えた直後（読み込み中）は選択肢を使えず、もう一度切り替えても無視される
    res = page.evaluate(
        f"""async ([a, b]) => {{
        const v = {view};
        v.switchJob(a);
        while (!v.loading) await new Promise((r) => setTimeout(r, 2));
        const disabled = [document.querySelector("#job-select").disabled,
                          document.querySelector("#delete-job-btn").disabled];
        v.switchJob(b);
        return [disabled, v.jobId];
    }}""",
        [exp_job, fast_job],
    )
    assert res == [[True, True], exp_job]
    page.wait_for_function(f"() => {view}.jobId === {exp_job} && {view}.ready", timeout=60_000)
    assert page.locator("#job-select:not([disabled])").count() == 1
    assert page.evaluate(f"() => {view}.engine.position") == pytest.approx(1.5, abs=0.05)
    _wait_gains(page, {"drums": 0.0, "bass": 1.0})

    # 読み込みに失敗 → 別の分け方へ切り替えると、失敗前の位置・選択を引き継ぐ
    def fail(route: Any) -> None:
        route.fulfill(status=500, content_type="application/json",
                      body='{"detail": "テスト用の失敗"}')

    page.route("**/api/files/renditions/*", fail)
    page.select_option("#job-select", str(fast_job))
    page.wait_for_selector("text=読み込めませんでした", timeout=30_000)
    assert page.locator("#job-select:not([disabled])").count() == 1
    page.unroute("**/api/files/renditions/*")
    page.select_option("#job-select", str(exp_job))
    page.wait_for_function(f"() => {view}.jobId === {exp_job} && {view}.ready", timeout=60_000)
    assert page.evaluate(f"() => {view}.engine.position") == pytest.approx(1.5, abs=0.05)
    _wait_gains(page, {"drums": 0.0, "bass": 1.0})
    assert not page.errors  # type: ignore[attr-defined]


def test_library_prefers_active_job(page: Any, server: LiveServer, tmp_path: Path) -> None:
    """同じ曲に複数のジョブがあるとき、一覧は分割待ち・分割中のものを優先して出す。"""
    track_id, _fast_job = _done_track(server, tmp_path)

    def tracks(route: Any) -> None:
        data = route.fetch().json()
        for t in data["tracks"]:
            if t["track_id"] == track_id:
                # 最新は完了済み、別の分け方が分割中（ほかに1件待ち）
                t["active_job"] = {
                    **t["latest_job"], "job_id": 99999, "status": "running",
                    "progress": 0.4, "stage": "分離中（1/3）", "preset": "exp_combo",
                    "preset_name": "テストの分け方", "preset_experimental": True,
                }
                t["active_count"] = 2
        route.fulfill(json=data)

    page.route("**/api/tracks", tracks)
    page.goto(server.base_url + "/#/library")
    row = f".track-row[data-track-id='{track_id}']"
    page.wait_for_selector(f"{row} .badge:has-text('分割中')")
    text = page.inner_text(row)
    assert "40%" in text and "実験: テストの分け方" in text and "ほか 1 件待ち" in text
    assert page.locator(f"{row} .btn:has-text('キャンセル')").count() == 1
    assert page.locator(f"{row} .btn:has-text('別の分け方で分割')").count() == 0
    assert page.locator(f"{row} .btn:has-text('再生')").count() == 1
