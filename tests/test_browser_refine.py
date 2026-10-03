"""「もっと分ける」の画面テスト（T07。PC の Microsoft Edge を Playwright で動かす）。

`uv run pytest -m browser` で実行する。分割・詳細分割は FakeSeparator（HPSS はテスト用の関数）、
配信用データは本物の ffmpeg。スクリーンショットは data/cache/screens/ に保存する（コミットしない）。
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from browser_helpers import SCREENS_DIR, LiveServer, run_server
from stemapp.config import Settings
from stemapp.models import SeparationJob
from stemapp.seed import DRUMSEP, HPSS, MEGA53
from test_browser import (  # noqa: F401  fixture を使う
    PHONE,
    _done_track,
    _gains,
    _shot,
    _wait_gains,
    browser,
    page,
    server,
)

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がありません"),
]

DRUM_KIDS = ["kick", "snare", "toms", "hihat", "ride", "crash", "drums_rest"]


def _open_player(pg: Any, srv: LiveServer, track_id: int) -> None:
    pg.goto(f"{srv.base_url}/#/track/{track_id}")
    pg.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)


def _refine(pg: Any, code: str, model_prefix: str) -> None:
    pg.click(f".stem-cell[data-code='{code}'] .refine-act.split")
    pg.wait_for_selector(".refine-modal")
    pg.click(f".refine-option[data-model^='{model_prefix}']")


def _wait_loaded_with(pg: Any, code: str, timeout: int = 60_000) -> None:
    pg.wait_for_selector(f".stem-btn[data-code='{code}']", timeout=timeout)
    pg.wait_for_selector("#play-btn:not([disabled])", timeout=timeout)


def test_refine_drums_flow(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    track_id, _job_id = _done_track(server, tmp_path)
    _open_player(page, server, track_id)

    # 分けられる stem にだけ「もっと分ける」が付く（vocals・bass には付かない）
    split = page.locator(".refine-act.split")
    codes = sorted(
        split.nth(i).evaluate("b => b.closest('.stem-cell').dataset.code")
        for i in range(split.count())
    )
    assert codes == ["backing_vocal", "drums", "lead_vocal", "other"]

    # 方法を選ぶダイアログ
    page.click(".stem-cell[data-code='drums'] .refine-act.split")
    page.wait_for_selector(".refine-modal")
    option = page.locator(".refine-option")
    assert option.count() == 1
    names = "キック・スネア・タム・ハイハット・ライド・クラッシュ・残り（ドラム）"
    assert names in option.inner_text()
    assert "GPU" in option.inner_text()
    _shot(page, "refine_menu_pc.png")
    # 開いている間はキー操作が再生に効かない
    page.keyboard.press("2")
    assert page.locator(".stem-btn[data-code='lead_vocal']").get_attribute("aria-pressed") == "true"
    page.click(".refine-option")
    page.wait_for_selector(".refine-modal", state="detached")

    # 分割中: 進み具合の帯と、STEM 欄の下の状態。再生は続けられる
    page.wait_for_selector(".stem-cell.refining .refine-bar")
    page.wait_for_selector("#refine-status .refine-row")
    assert "ドラム" in page.locator("#refine-status").inner_text()
    _shot(page, "refine_progress_pc.png")

    # 終わると読み直し、子のボタンが親と一緒の枠に出る
    _wait_loaded_with(page, "kick")
    family = page.locator(".stem-family[data-family='drums']")
    assert family.count() == 1
    fam_codes = family.locator(".stem-cell").evaluate_all("els => els.map(e => e.dataset.code)")
    assert fam_codes == ["drums", *DRUM_KIDS]
    assert page.locator(".stem-cell[data-code='drums'] .refine-act.undo").count() == 1
    assert page.locator(".stem-cell[data-code='drums'] .refine-act.split").count() == 0
    assert page.locator("#refine-status").is_hidden()

    # ドラムは鳴らしていたので子が全部 ON、親（drums）の GainNode は 0
    _wait_gains(page, {**dict.fromkeys(DRUM_KIDS, 1.0), "drums": 0.0, "bass": 1.0})
    assert page.locator(".stem-btn[data-code='drums']").get_attribute("aria-pressed") == "true"
    # 子を1つ切り替える → 親は「一部」
    page.click(".stem-btn[data-code='kick']")
    _wait_gains(page, {"kick": 0.0, "snare": 1.0})
    assert page.locator(".stem-btn[data-code='drums']").get_attribute("aria-pressed") == "mixed"
    # 親を押すと子をまとめて ON
    page.click(".stem-btn[data-code='drums']")
    _wait_gains(page, dict.fromkeys(DRUM_KIDS, 1.0))
    # ソロで子だけ
    page.click(".stem-btn[data-code='snare']", modifiers=["Shift"])
    g = _wait_gains(page, {"snare": 1.0, "kick": 0.0, "bass": 0.0, "lead_vocal": 0.0})
    assert g["drums"] == 0.0
    page.click("#play-btn")
    page.wait_for_function("() => window.__stemapp.view.engine.position > 0.2", timeout=10_000)
    _shot(page, "refine_children_pc.png")
    page.set_viewport_size(PHONE)
    _shot(page, "refine_children_phone.png")
    assert not page.errors  # type: ignore[attr-defined]


def test_refine_other_and_undo(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    track_id, _job_id = _done_track(server, tmp_path)
    _open_player(page, server, track_id)
    _refine(page, "other", "hpss")
    _wait_loaded_with(page, "sustained")
    kids = page.locator(".stem-family[data-family='other'] .stem-cell").evaluate_all(
        "els => els.map(e => e.dataset.code)"
    )
    assert kids == ["other", "sustained", "transient", "other_rest"]
    name = page.locator(".stem-btn[data-code='sustained'] .name").inner_text()
    assert name == "持続音（パッド等）"
    # 子を1つだけ鳴らしてから戻す → 親が鳴る
    page.click(".stem-btn[data-code='transient']", modifiers=["Shift"])
    _wait_gains(page, {"transient": 1.0, "sustained": 0.0})
    page.click(".stem-cell[data-code='other'] .refine-act.undo")
    page.wait_for_selector(".modal .danger-confirm")
    kids_text = "持続音（パッド等）・短い音（ヒット等）・残り（その他）"
    assert kids_text in page.locator(".modal").inner_text()
    page.click(".modal .danger-confirm")
    page.wait_for_function(
        "() => !document.querySelector(\".stem-btn[data-code='sustained']\")", timeout=30_000
    )
    page.wait_for_selector("#play-btn:not([disabled])", timeout=60_000)
    assert page.locator(".stem-family").count() == 0
    assert page.locator(".stem-cell[data-code='other'] .refine-act.split").count() == 1
    g = _wait_gains(page, {"other": 1.0, "bass": 0.0})
    assert "sustained" not in g
    assert not page.errors  # type: ignore[attr-defined]


def test_refine_other_with_mega53(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    """T07b: other の方法に Mega 53 が出る。無音の子（Fake では木管）はボタンにならない。"""
    track_id, _job_id = _done_track(server, tmp_path)
    _open_player(page, server, track_id)
    page.click(".stem-cell[data-code='other'] .refine-act.split")
    page.wait_for_selector(".refine-modal")
    models = page.locator(".refine-option").evaluate_all("els => els.map(e => e.dataset.model)")
    # T17: Mega 53 が先頭（既定で選ばれている）、HPSS は「実験」として末尾
    assert models == [MEGA53, HPSS]
    mega = page.locator(f".refine-option[data-model='{MEGA53}']")
    assert "Mega 53" in mega.inner_text() and "GPU" in mega.inner_text()
    assert mega.evaluate("e => e === document.activeElement")
    assert mega.locator(".refine-tag.exp").count() == 0
    hpss = page.locator(f".refine-option[data-model='{HPSS}']")
    assert hpss.locator(".refine-tag.exp").inner_text() == "実験"
    assert "ストリングス" in mega.inner_text() and "シンセ" in mega.inner_text()
    _shot(page, "refine_mega53_menu_pc.png")
    mega.click()
    page.wait_for_selector(".refine-modal", state="detached")
    _wait_loaded_with(page, "synth")
    kids = page.locator(".stem-family[data-family='other'] .stem-cell").evaluate_all(
        "els => els.map(e => e.dataset.code)"
    )
    assert kids == ["other", "brass", "strings", "synth", "percussion", "other_rest"]
    assert page.locator(".stem-btn[data-code='woodwind']").count() == 0
    _wait_gains(page, {"synth": 1.0, "other_rest": 1.0, "other": 0.0})
    page.click(".stem-btn[data-code='synth']", modifiers=["Shift"])
    _wait_gains(page, {"synth": 1.0, "strings": 0.0, "other_rest": 0.0, "bass": 0.0})
    _shot(page, "refine_mega53_children_pc.png")
    assert not page.errors  # type: ignore[attr-defined]


def test_refine_conflict_and_preset_fallback(
    page: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    """男女をメインボーカルに使うと、サブボーカルには使えない（理由を出す）。息は使える。
    子（キック）を指定した組み合わせを、子の無い曲で開くと親（ドラム）として扱う。"""
    track_id, _job_id = _done_track(server, tmp_path)
    _open_player(page, server, track_id)
    res = page.evaluate(
        """() => {
        const { selection: S } = window.__stemapp.modules;
        const v = window.__stemapp.view;
        const kick = v.stemTypes.find((t) => t.code === "kick");
        const item = { stem_type_code: "kick", stem_type_id: kick.stem_type_id, gain_db: -6 };
        const preset = { items: [item] };
        const r = S.presetToSelection(v.tree, preset, v.groups, v.stemTypes);
        const without = S.presetToSelection(v.tree, preset, v.groups);
        return [[...r.sel], r.gainsDb.get("drums"), [...without.sel]];
    }"""
    )
    assert res == [["drums"], -6, []]

    _refine(page, "lead_vocal", "bs_roformer_male_female")
    _wait_loaded_with(page, "male")
    page.click(".stem-cell[data-code='backing_vocal'] .refine-act.split")
    page.wait_for_selector(".refine-modal")
    mf = page.locator(".refine-option[data-model^='bs_roformer_male_female']")
    assert mf.is_disabled()
    assert "男声" in mf.inner_text()
    breath = page.locator(".refine-option[data-model^='aspiration']")
    assert breath.is_enabled()
    _shot(page, "refine_menu_conflict_pc.png")
    page.keyboard.press("Escape")
    page.wait_for_selector(".refine-modal", state="detached")
    assert not page.errors  # type: ignore[attr-defined]


@pytest.fixture
def slow_server(tmp_path: Path) -> Iterator[LiveServer]:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    with run_server(settings, fake_delay=4.0) as srv:
        yield srv


def test_refine_cancel(page: Any, slow_server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    track_id, _job_id = _done_track(slow_server, tmp_path)
    _open_player(page, slow_server, track_id)
    _refine(page, "drums", "MDX23C")
    page.wait_for_selector(".stem-cell[data-code='drums'] .refine-act.stop")
    page.wait_for_function(
        "() => (document.querySelector('#refine-status')?.textContent || '').includes('分離中')",
        timeout=20_000,
    )
    page.click(".stem-cell[data-code='drums'] .refine-act.stop")
    page.wait_for_selector(".stem-cell[data-code='drums'] .refine-act.split", timeout=30_000)
    assert page.locator("#refine-status").is_hidden()
    assert page.locator(".stem-btn[data-code='kick']").count() == 0
    time.sleep(0.2)
    assert "drums" in _gains(page)
    assert not page.errors  # type: ignore[attr-defined]


def test_refine_done_while_loading(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    """画面の読み込み中に分割が終わっても、読み込みが終わったら子を出す（読み直す）。"""
    track_id, job_id = _done_track(server, tmp_path)
    with httpx.Client(base_url=server.base_url, timeout=30) as c:
        stems = {s["code"]: s for s in c.get(f"/api/jobs/{job_id}/stems").json()["stems"]}
        res = c.post(f"/api/stems/{stems['drums']['stem_id']}/refine", json={"model": DRUMSEP})
        assert res.status_code == 201
        refine_id = res.json()["job"]["job_id"]

    # 再生用の音声の取得を止めておき（画面の読み込みが終わらない）
    held: list[Any] = []
    release = {"on": False}

    def hold(route: Any) -> None:
        if release["on"]:
            route.continue_()
        else:
            held.append(route)

    page.route("**/api/files/renditions/**", hold)
    page.goto(f"{server.base_url}/#/track/{track_id}")
    page.wait_for_selector("#loading:not([hidden])")
    # サーバーで詳細分割が終わるまで待つ（画面はまだ読み込み中）
    deadline = time.monotonic() + 60
    with httpx.Client(base_url=server.base_url, timeout=30) as c:
        while c.get(f"/api/jobs/{refine_id}").json()["status"] != "done":
            assert time.monotonic() < deadline, "詳細分割が終わりません"
            page.wait_for_timeout(200)
    # 画面の問い合わせ（1.2 秒おき）が、読み込み中に done を見た状態を確実に作る
    page.wait_for_timeout(2500)
    assert page.locator(".stem-btn[data-code='kick']").count() == 0
    waiting_text = page.locator("#refine-status").inner_text()
    # 音声の取得を再開する → 読み込みが終わったら読み直して、子が出る
    release["on"] = True
    for r in held:
        r.continue_()
    _wait_loaded_with(page, "kick", timeout=20_000)
    assert page.locator(".stem-family[data-family='drums']").count() == 1
    assert page.locator("#refine-status").is_hidden()
    assert not page.errors  # type: ignore[attr-defined]
    # （任意）読み込み中は「読み直し待ち」の表示が出ていた
    assert "読み込みが終わったら" in waiting_text


def test_refine_note_for_legacy_folder(page: Any, server: LiveServer, tmp_path: Path) -> None:  # noqa: F811
    """古い保存フォルダ（output_dir が NULL）のジョブは、分けられない理由を出す。"""
    track_id, job_id = _done_track(server, tmp_path)
    with server.session_factory() as s:
        job = s.get(SeparationJob, job_id)
        assert job is not None
        job.output_dir = None
        s.commit()
    _open_player(page, server, track_id)
    note = page.locator("#refine-note")
    assert note.is_visible()
    assert "migrate-folders" in note.inner_text()
    assert page.locator(".refine-act").count() == 0
    assert not page.errors  # type: ignore[attr-defined]


def test_refine_buttons_are_large_on_touch(
    browser: Any, server: LiveServer, tmp_path: Path  # noqa: F811
) -> None:
    """指で操作する画面では「分ける」を 32px 以上にし、名前と重ならない。"""
    track_id, _job_id = _done_track(server, tmp_path)
    ctx = browser.new_context(viewport=PHONE, locale="ja-JP", has_touch=True, is_mobile=True)
    try:
        pg = ctx.new_page()
        _open_player(pg, server, track_id)
        assert pg.evaluate("() => matchMedia('(pointer: coarse)').matches") is True
        act = pg.locator(".stem-cell[data-code='drums'] .refine-act.split")
        box = act.bounding_box()
        assert box is not None and box["width"] >= 32 and box["height"] >= 32
        # 名前の行はボタンより上で終わり、横幅いっぱいに出る（省略されない）
        name = pg.locator(".stem-cell[data-code='drums'] .stem-btn .name")
        nbox = name.bounding_box()
        assert nbox is not None and nbox["y"] + nbox["height"] <= box["y"] + 1
        assert name.evaluate("e => e.scrollWidth <= e.clientWidth")
        lead = pg.locator(".stem-cell[data-code='lead_vocal'] .stem-btn .name")
        assert lead.evaluate("e => e.scrollWidth <= e.clientWidth")
        # full_page の撮影は画面の大きさを変えてしまうので、見えている範囲だけ撮る
        pg.locator("#stems").scroll_into_view_if_needed()
        SCREENS_DIR.mkdir(parents=True, exist_ok=True)
        pg.screenshot(path=str(SCREENS_DIR / "refine_touch_phone.png"))
    finally:
        ctx.close()
