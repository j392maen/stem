// 拍の補正パネル（T10c）。普段は閉じておく（<details>）。
// 補正の計算と保存はサーバー（POST /api/tracks/{id}/beats/edit・undo・reset）。結果の「有効な拍」で
// 波形の拍の線・小節線と BPM 表示をすぐに差し替える。

import { api } from "./api.js";
import { BeatGrid, TapRecorder, formatBpm } from "./beats.js";
import { confirmDialog, el, formatTime, toast } from "./ui.js";

// タップ: 最後にたたいてからこの時間がたったら、4回以上なら拍を作る
export const TAP_COMMIT_MS = 1500;
export const MIN_TAPS = 4;
const METERS = [3, 4, 5, 6, 7];
const SHIFT_MS = 10;
const FINE_SHIFT_MS = 1;

export class BeatEditPanel {
  /** view: PlayerView（trackId, beatGrid, engine, cues, activeLoop(), setBeatGrid(), later()）。 */
  constructor(view) {
    this.view = view;
    this.busy = false;
    this.taps = new TapRecorder();
    this.tapTimer = 0;
    this.root = this.build();
  }

  get grid() {
    return this.view.beatGrid;
  }

  get open() {
    return this.root.open;
  }

  // --- 画面 --------------------------------------------------------------------

  build() {
    const btn = (id, text, title, onclick) => el("button", {
      class: "btn small", type: "button", id, text, title, onclick,
    });
    this.rangeSel = el("select", {
      class: "select small", id: "be-range", "aria-label": "補正の範囲",
    },
    el("option", { value: "segment", text: "再生位置の区間" }),
    el("option", { value: "all", text: "曲全体" }),
    el("option", { value: "loop", text: "ループ区間" }));
    this.meterSel = el("select", {
      class: "select small", id: "be-meter", "aria-label": "拍子（1小節の拍数）",
      onchange: (e) => this.edit("meter", { beats_per_bar: Number(e.target.value) }),
    }, METERS.map((n) => el("option", { value: String(n), text: `${n}/4` })));
    this.tapInfo = el("span", { class: "be-tap-info", id: "be-tap-info", text: "" });
    this.cueA = el("select", { class: "select small", id: "be-cue-a", "aria-label": "1つ目のキュー", onchange: () => this.suggestBars() });
    this.cueB = el("select", { class: "select small", id: "be-cue-b", "aria-label": "2つ目のキュー", onchange: () => this.suggestBars() });
    // キュー2点からの1小節の拍数（既定は曲の拍子。乱れた所の小節から推定しない）
    this.cueMeterSel = el("select", {
      class: "select small", id: "be-cue-meter", "aria-label": "キューの間の拍子（1小節の拍数）",
      onchange: () => this.suggestBars(),
    }, METERS.map((n) => el("option", { value: String(n), text: `${n}/4` })));
    this.barsInput = el("input", {
      class: "input small be-bars", id: "be-bars", type: "number", min: "1", max: "1024", value: "8",
      "aria-label": "キューの間の小節数",
    });
    this.cueRow = el("div", { class: "be-row" },
      el("span", { class: "be-label", text: "キュー2点から" }),
      this.cueA, el("span", { class: "muted", text: "〜" }), this.cueB,
      this.barsInput, el("span", { class: "muted", text: "小節" }), this.cueMeterSel,
      btn("be-cues", "適用", "2つのキューの間を、指定した小節数の一定テンポの拍で置き換えます（小節の頭はキューの位置）",
        () => this.applyCues()));
    this.cueNote = el("span", { class: "muted be-note", text: "キューを2つ打つと使えます。" });
    this.stateTag = el("span", { class: "be-state", id: "be-state" });
    this.undoBtn = btn("be-undo", "元に戻す", "直前の補正を取り消します（Ctrl+Z）", () => this.undo());
    this.resetBtn = btn("be-reset", "自動に戻す", "補正をすべて取り消して、自動解析の結果に戻します", () => this.reset());

    const shift = (sign) => (e) => this.edit("shift", {
      delta_sec: (sign * (e.shiftKey ? FINE_SHIFT_MS : SHIFT_MS)) / 1000,
    });
    const details = el("details", { class: "panel beat-edit", id: "beat-edit" },
      el("summary", {},
        el("span", { class: "be-title", text: "拍の補正" }), this.stateTag),
      el("div", { class: "be-body" },
        el("div", { class: "be-row" },
          el("span", { class: "be-label", text: "範囲" }), this.rangeSel,
          el("span", { class: "muted be-note", text: "キュー2点からは範囲によらず2つのキューの間" })),
        el("div", { class: "be-row" },
          el("span", { class: "be-label", text: "小節" }),
          btn("be-downbeat", "1小節目をここに", "再生位置にいちばん近い拍を小節の頭にします（拍の位置は動かしません）",
            () => this.edit("downbeat")),
          btn("be-double", "×2", "拍の数を倍にします（拍の間に拍を足す。BPM が倍）", () => this.edit("double")),
          btn("be-half", "÷2", "拍の数を半分にします（1つおきに間引く。BPM が半分）", () => this.edit("half")),
          el("label", { class: "be-inline" }, el("span", { class: "muted", text: "拍子" }), this.meterSel)),
        el("div", { class: "be-row" },
          el("span", { class: "be-label", text: "ずらす" }),
          btn("be-shift-minus", `−${SHIFT_MS}ms`, `拍を前へ ${SHIFT_MS}ms（Shift で ${FINE_SHIFT_MS}ms）`, shift(-1)),
          btn("be-shift-plus", `+${SHIFT_MS}ms`, `拍を後ろへ ${SHIFT_MS}ms（Shift で ${FINE_SHIFT_MS}ms）`, shift(1))),
        el("div", { class: "be-row" },
          el("span", { class: "be-label", text: "タップ" }),
          el("button", {
            class: "btn small be-tap", type: "button", id: "be-tap", text: "タップ（T）",
            title: `再生しながら拍に合わせて ${MIN_TAPS} 回以上たたくと、その間隔と位置から一定テンポの拍を作ります`,
            onpointerdown: (e) => { e.preventDefault(); this.tapNow(); },
            // Space・Enter はタップだけにする（プレイヤーの Space＝再生/停止に届けない）
            onkeydown: (e) => {
              if (e.key === "Enter" || e.key === " " || e.code === "Space") {
                e.preventDefault();
                e.stopPropagation();
                if (!e.repeat) this.tapNow();
              }
            },
          }),
          this.tapInfo),
        this.cueRow, this.cueNote,
        el("div", { class: "be-row be-foot" }, this.undoBtn, this.resetBtn)));
    details.addEventListener("toggle", () => this.refresh());
    return details;
  }

  /** 拍・キュー・ループが変わったときに表示を合わせる。 */
  refresh() {
    const grid = this.grid;
    this.root.hidden = !grid;
    if (!grid) return;
    this.stateTag.textContent = grid.edited ? "補正済み" : "自動";
    this.stateTag.classList.toggle("edited", grid.edited);
    this.undoBtn.disabled = this.busy || !grid.canUndo;
    this.resetBtn.disabled = this.busy || !grid.edited;
    const pos = this.view.engine ? this.view.engine.position : 0;
    const meter = grid.meterAt(pos);
    if (!METERS.includes(meter) && !this.meterSel.querySelector(`option[value="${meter}"]`)) {
      this.meterSel.append(el("option", { value: String(meter), text: `${meter}/4` }));
    }
    this.meterSel.value = String(meter);
    this.meterShown = meter;
    this.ensureMeterOption(this.cueMeterSel, grid.timeSignature);
    if (this.cueGridKey !== grid) {
      // 拍が変わったら、キュー2点からの拍子を曲の拍子に合わせ直す
      this.cueGridKey = grid;
      this.cueMeterSel.value = String(grid.timeSignature);
    }
    const loopOpt = this.rangeSel.querySelector("option[value='loop']");
    loopOpt.disabled = !this.view.activeLoop();
    if (loopOpt.disabled && this.rangeSel.value === "loop") this.rangeSel.value = "segment";
    this.refreshCues();
    for (const b of this.root.querySelectorAll(".be-body button, .be-body select")) {
      if (b !== this.undoBtn && b !== this.resetBtn) b.disabled = this.busy;
    }
    this.refreshCueButtons();
  }

  ensureMeterOption(sel, n) {
    if (!sel.querySelector(`option[value="${n}"]`)) {
      sel.append(el("option", { value: String(n), text: `${n}/4` }));
    }
  }

  /** 開いている間、拍子の選択欄を再生位置の小節の拍数に合わせる（選んでいる最中は変えない）。 */
  syncMeter(meter) {
    if (!this.open || !this.grid || this.busy || meter === this.meterShown) return;
    if (document.activeElement === this.meterSel) return;
    this.ensureMeterOption(this.meterSel, meter);
    this.meterSel.value = String(meter);
    this.meterShown = meter;
  }

  /** キュー2点からの1小節の拍数（選択欄。無ければ曲の拍子）。 */
  cueMeter() {
    const n = Number(this.cueMeterSel.value);
    return n >= 2 && n <= 12 ? n : (this.grid ? this.grid.timeSignature : 4);
  }

  refreshCues() {
    const cues = this.view.cues || [];
    const keep = [this.cueA.value, this.cueB.value];
    const opts = () => cues.map((c) => el("option", {
      value: String(c.cue_id), text: `${c.label || "キュー"} ${formatTime(c.position_sec, true)}`,
    }));
    this.cueA.replaceChildren(...opts());
    this.cueB.replaceChildren(...opts());
    const has = (v) => cues.some((c) => String(c.cue_id) === v);
    if (has(keep[0])) this.cueA.value = keep[0];
    else if (cues[0]) this.cueA.value = String(cues[0].cue_id);
    const other = cues.find((c) => String(c.cue_id) !== this.cueA.value);
    if (has(keep[1]) && keep[1] !== this.cueA.value) this.cueB.value = keep[1];
    else if (other) this.cueB.value = String(other.cue_id);
    if (keep[0] !== this.cueA.value || keep[1] !== this.cueB.value) this.suggestBars();
  }

  refreshCueButtons() {
    const enough = (this.view.cues || []).length >= 2;
    this.cueRow.hidden = !enough;
    this.cueNote.hidden = enough;
  }

  cuePair() {
    const find = (sel) => (this.view.cues || []).find((c) => String(c.cue_id) === sel.value);
    const a = find(this.cueA);
    const b = find(this.cueB);
    if (!a || !b) return null;
    return a.position_sec <= b.position_sec ? [a, b] : [b, a];
  }

  /** 2つのキューの間の小節数の目安（今の BPM と拍子から）を入れる。 */
  suggestBars() {
    const pair = this.cuePair();
    const grid = this.grid;
    if (!pair || !grid) return;
    const [a, b] = pair;
    const bpm = grid.bpmAt(a.position_sec);
    if (!bpm) return;
    const barSec = (this.cueMeter() * 60) / bpm;
    this.barsInput.value = String(Math.max(1, Math.round((b.position_sec - a.position_sec) / barSec)));
  }

  // --- 操作 --------------------------------------------------------------------

  /** 補正の操作を送り、返ってきた有効な拍で表示を差し替える。 */
  async edit(op, args = {}) {
    if (!this.grid || this.busy) return false;
    const view = this.view;
    const body = { op, position: Math.max(0, view.engine ? view.engine.position : 0), ...args };
    if (op !== "cues") {
      body.range = this.rangeSel.value;
      const loop = view.activeLoop();
      if (body.range === "loop") {
        if (!loop) { toast("ループ区間がありません。"); return false; }
        body.loop_start = loop.start;
        body.loop_end = loop.end;
      }
    }
    return this.send(`/api/tracks/${view.trackId}/beats/edit`, body);
  }

  async send(path, body) {
    this.busy = true;
    this.refresh();
    try {
      const res = await api(path, { method: "POST", body });
      if (!this.view.alive) return false;
      this.view.setBeatGrid(new BeatGrid(res));
      return true;
    } catch (e) {
      toast(e.message);
      return false;
    } finally {
      this.busy = false;
      if (this.view.alive) this.refresh();
    }
  }

  async undo() {
    if (!this.grid || !this.grid.canUndo || this.busy) return;
    if (await this.send(`/api/tracks/${this.view.trackId}/beats/undo`)) toast("元に戻しました。");
  }

  async reset() {
    if (!this.grid || !this.grid.edited || this.busy) return;
    const ok = await confirmDialog(
      "拍の補正をすべて取り消して、自動解析の結果に戻しますか？（「元に戻す」で取り消せます）",
      { ok: "自動に戻す", danger: true },
    );
    if (!ok || !this.view.alive) return;
    if (await this.send(`/api/tracks/${this.view.trackId}/beats/reset`)) toast("自動解析の結果に戻しました。");
  }

  async applyCues() {
    const pair = this.cuePair();
    if (!pair || pair[0].cue_id === pair[1].cue_id) { toast("別々のキューを2つ選んでください。"); return; }
    const bars = Math.round(Number(this.barsInput.value));
    if (!(bars >= 1)) { toast("小節数を入れてください。"); return; }
    await this.edit("cues", {
      cue_start: pair[0].position_sec, cue_end: pair[1].position_sec, bars,
      beats_per_bar: this.cueMeter(),
    });
  }

  /** 今たたいた（ボタン・T キー）。再生中の曲の時刻から出力の遅延を差し引く。 */
  tapNow() {
    const engine = this.view.engine;
    if (!engine || !engine.playing) { toast("再生しながら拍に合わせてたたいてください。"); return; }
    this.tap(engine.position - engine.outputLatency(), performance.now() / 1000);
  }

  /** たたいた時刻（曲の時刻）を記録する（テストでは時刻を直接渡す）。 */
  tap(songTime, wallSec = performance.now() / 1000) {
    if (!this.grid) return;
    const { count, bpm } = this.taps.add(songTime, wallSec);
    this.tapInfo.textContent = `${count} 回${bpm ? `・${formatBpm(bpm)} BPM` : ""}${count < MIN_TAPS ? `（あと ${MIN_TAPS - count} 回）` : ""}`;
    this.root.querySelector("#be-tap").classList.add("on");
    clearTimeout(this.tapTimer);
    this.tapTimer = setTimeout(() => this.commitTaps(), TAP_COMMIT_MS);
  }

  /** たたき終わった: 4回以上ならその拍で範囲を置き換える。 */
  async commitTaps() {
    clearTimeout(this.tapTimer);
    const taps = this.taps.taps.slice();
    this.taps.reset();
    const btn = this.root.querySelector("#be-tap");
    if (btn) btn.classList.remove("on");
    if (!this.view.alive) return false;
    if (taps.length < MIN_TAPS) {
      this.tapInfo.textContent = taps.length ? `${MIN_TAPS} 回以上たたいてください` : "";
      return false;
    }
    // 1小節の拍数は拍子の選択欄（再生位置の小節に追従する値）。区間ごとに拍子が違う曲のため
    const perBar = Number(this.meterSel.value);
    const args = { taps };
    if (perBar >= 2 && perBar <= 12) args.beats_per_bar = perBar;
    const ok = await this.edit("tap", args);
    this.tapInfo.textContent = ok ? "タップから拍を作りました" : "";
    return ok;
  }

  dispose() {
    clearTimeout(this.tapTimer);
  }
}
