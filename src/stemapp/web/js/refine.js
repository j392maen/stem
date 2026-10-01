// 「もっと分ける」（詳細分割、T07）。プレイヤーの stem ボタンに操作を付ける。
//
// - 分けられる stem（API の refine_methods がある葉）には「分ける」。押すと方法を選ぶダイアログ。
// - 分割待ち・分割中は、ボタンの下に進み具合の帯と「中止」。STEM の欄の下に段階を出す。
// - 終わったら画面を読み直す（再生位置・選択は保つ。親を鳴らしていたら子を全部鳴らす）。
//   子の stem ボタンは親と一緒に1つの枠にまとまって出る。
// - 詳細分割で分けた親には「戻す」（子を削除して分ける前に戻す）。
// 選択の規則（葉だけで持つ、親ボタンは子をまとめて切り替え）は selection.js のまま。

import { api } from "./api.js";
import { confirmDialog, el, toast } from "./ui.js";

const POLL_MS = 1200;
const ACTIVE = new Set(["queued", "running"]);

const PATHS = {
  // 1本が3本に分かれる形
  split: "M12 3v7M12 10L5 20M12 10l7 10M12 10v10",
  // 戻す（反時計回りの矢印）
  undo: "M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3",
  stop: "M7 7h10v10H7z",
};

function svgIcon(name) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", PATHS[name]);
  svg.append(path);
  return svg;
}

function percent(p) {
  return `${Math.round(Math.min(Math.max(Number(p) || 0, 0), 1) * 100)}%`;
}

/** プレイヤー1画面（1回の mount）につき1つ。 */
export class RefineUI {
  constructor(view) {
    this.view = view;
    // 分けた stem の stem_id → 分割待ち・分割中・失敗した詳細分割（model 付き）
    this.jobs = new Map();
    for (const j of view.job.refine_jobs || []) this.jobs.set(j.input_stem_id, j);
    // 分け終わって、画面の読み直しを待っているジョブ（読み込み中に終わったとき）
    this.pendingDone = [];
    this.timer = 0;
    this.back = null;
    this.disposed = false;
    this.onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); this.closeMenu(); }
    };
    if (this.activeJobs().length) this.schedule();
  }

  dispose() {
    this.disposed = true;
    this.closeMenu();
    clearTimeout(this.timer);
    this.timer = 0;
  }

  get stems() {
    return this.view.tree ? this.view.tree.order.map((c) => this.view.tree.byCode.get(c)) : [];
  }

  stemById(id) {
    return this.stems.find((s) => s.stem_id === id) || null;
  }

  activeJobs() {
    return [...this.jobs.values()].filter((j) => ACTIVE.has(j.status));
  }

  /** code の子孫（木の順）。 */
  descendants(code) {
    const tree = this.view.tree;
    const out = [];
    const walk = (c) => {
      for (const k of tree.children.get(c) || []) { out.push(k); walk(k); }
    };
    walk(code);
    return out;
  }

  // --- stem ボタンへの飾り付け（player.renderStems から呼ぶ） --------------------------

  decorate(box) {
    for (const b of [...box.querySelectorAll(":scope > .stem-btn")]) {
      const s = this.view.tree.byCode.get(b.dataset.code);
      const cell = el("div", { class: "stem-cell", dataset: { code: s.code } });
      b.replaceWith(cell);
      cell.append(b);
      this.fillCell(cell, s);
    }
    // 詳細分割した親と子を1つの枠にまとめる（深いものから。枠ごと外側の枠に入る）
    const parents = this.stems.filter((s) => s.refined_by).reverse();
    for (const p of parents) {
      const cell = box.querySelector(`.stem-cell[data-code="${CSS.escape(p.code)}"]`);
      if (!cell) continue;
      const family = el("div", { class: "stem-family", dataset: { family: p.code } });
      family.style.setProperty("--c", p.color);
      cell.before(family);
      family.append(cell);
      for (const code of this.descendants(p.code)) {
        const kid = box.querySelector(`.stem-cell[data-code="${CSS.escape(code)}"]`);
        // 内側の枠に入っていれば、その枠ごと移す
        const node = kid && kid.parentElement.classList.contains("stem-family")
          && kid.parentElement !== family && kid.parentElement.dataset.family === code
          ? kid.parentElement : kid;
        if (node && node.parentElement !== family) family.append(node);
      }
    }
    this.renderStatus();
  }

  fillCell(cell, s) {
    cell.querySelectorAll(".refine-act, .refine-bar").forEach((n) => n.remove());
    const job = this.jobs.get(s.stem_id);
    const active = job && ACTIVE.has(job.status);
    const waiting = !!job && job.status === "done"; // 読み直し待ち
    cell.classList.toggle("refining", !!active);
    cell.classList.toggle("refine-failed", !!job && job.status === "failed");
    let act = null;
    if (active) {
      act = el("button", {
        class: "refine-act stop", type: "button", "aria-label": `${s.display_name} の分割をやめる`,
        title: "分割をやめる", onclick: () => this.cancel(job),
      }, svgIcon("stop"));
      cell.append(el("div", { class: "refine-bar", role: "progressbar", "aria-label": `${s.display_name} を分割中` },
        el("span", { style: { width: percent(job.progress) } })));
    } else if (waiting) {
      act = null;
    } else if (s.refined_by) {
      act = el("button", {
        class: "refine-act undo", type: "button", "aria-label": `${s.display_name} を分ける前に戻す`,
        title: `分ける前に戻す（${s.refined_by.display_name || "詳細分割"} の子を削除）`,
        onclick: () => this.undo(s),
      }, svgIcon("undo"));
    } else if ((s.refine_methods || []).length) {
      act = el("button", {
        class: "refine-act split", type: "button", "aria-label": `${s.display_name} をもっと分ける`,
        title: "もっと分ける", onclick: () => this.openMenu(s),
      }, svgIcon("split"));
    }
    cell.classList.toggle("has-act", !!act);
    if (act) cell.append(act);
  }

  /** STEM の欄の下の、分割中・失敗の一覧。 */
  renderStatus() {
    const box = this.view.root.querySelector("#stems");
    if (!box) return;
    let status = this.view.root.querySelector("#refine-status");
    if (!status) {
      status = el("div", { class: "refine-status", id: "refine-status", "aria-live": "polite" });
      box.after(status);
    }
    const rows = [];
    if (this.view.job.refine_note) {
      rows.push(el("div", { class: "refine-row note", id: "refine-note" },
        el("span", { class: "muted", text: this.view.job.refine_note })));
    }
    for (const s of this.stems) {
      const w = s.refined_by && s.refined_by.warning;
      if (w) {
        rows.push(el("div", { class: "refine-row warn" },
          el("strong", { text: s.display_name }), el("span", { class: "muted", text: w })));
      }
    }
    for (const job of this.jobs.values()) {
      const s = this.stemById(job.input_stem_id);
      if (!s) continue;
      if (ACTIVE.has(job.status)) {
        rows.push(el("div", { class: "refine-row", dataset: { jobId: String(job.job_id) } },
          el("span", { class: "refine-dot" }),
          el("strong", { text: s.display_name }),
          el("span", { class: "muted", text: job.status === "queued" ? "分割待ち" : (job.stage || "分割中") }),
          el("span", { class: "refine-pct", text: job.status === "queued" ? "" : percent(job.progress) })));
      } else if (job.status === "done") {
        rows.push(el("div", { class: "refine-row", dataset: { jobId: String(job.job_id) } },
          el("strong", { text: s.display_name }),
          el("span", { class: "muted", text: "分け終わりました。読み込みが終わったら子の stem を出します。" })));
      } else if (job.status === "failed") {
        rows.push(el("div", { class: "refine-row failed", dataset: { jobId: String(job.job_id) } },
          el("strong", { text: s.display_name }),
          el("span", { class: "error-text", text: job.error_message || "分割に失敗しました。" }),
          el("button", {
            class: "btn small", type: "button", text: "閉じる",
            onclick: () => { this.jobs.delete(job.input_stem_id); this.refresh(); },
          })));
      }
    }
    status.replaceChildren(...rows);
    status.hidden = !rows.length;
  }

  /** ボタンの飾りと一覧を今の状態に合わせる（再生は止めない）。 */
  refresh() {
    const box = this.view.root.querySelector("#stems");
    if (!box) return;
    for (const cell of box.querySelectorAll(".stem-cell")) {
      const s = this.view.tree.byCode.get(cell.dataset.code);
      if (s) this.fillCell(cell, s);
    }
    this.renderStatus();
  }

  // --- 方法を選ぶ -------------------------------------------------------------------

  openMenu(s) {
    this.closeMenu();
    const methods = s.refine_methods || [];
    this.back = el("div", { class: "modal-back", onclick: (e) => { if (e.target === this.back) this.closeMenu(); } });
    document.addEventListener("keydown", this.onKey, true);
    const options = methods.map((m) => {
      const names = m.children.map((c) => c.display_name);
      return el("button", {
        class: "refine-option", type: "button", disabled: !m.available, dataset: { model: m.model },
        onclick: () => this.start(s, m),
      },
      el("span", { class: "refine-option-name", text: names.join("・") }),
      el("span", { class: "refine-option-sub" },
        el("span", { text: m.display_name }),
        el("span", { class: "refine-tag", text: m.gpu ? "GPU" : "CPU" })),
      m.available ? null : el("span", { class: "refine-option-why", text: m.reason || "今は分けられません。" }));
    });
    const modal = el("div", { class: "modal refine-modal", role: "dialog", "aria-modal": "true", "aria-labelledby": "refine-title" },
      el("div", { class: "refine-head" },
        el("h2", { id: "refine-title", text: "もっと分ける" }),
        el("span", { class: "refine-target" },
          el("span", { class: "refine-swatch", style: { background: s.color } }), s.display_name),
        el("button", { class: "btn small icon refine-close", type: "button", "aria-label": "閉じる", text: "×", onclick: () => this.closeMenu() })),
      el("p", { class: "muted refine-note", text: "分けた後も、子を全部鳴らせば元と同じ音になります（残りの stem を必ず作ります）。数十秒〜数分かかります。" }),
      el("div", { class: "refine-options" }, options));
    this.back.append(modal);
    document.body.append(this.back);
    const first = modal.querySelector(".refine-option:not([disabled])");
    if (first) first.focus();
  }

  closeMenu() {
    if (!this.back) return;
    document.removeEventListener("keydown", this.onKey, true);
    this.back.remove();
    this.back = null;
  }

  async start(s, m) {
    this.closeMenu();
    try {
      const res = await api(`/api/stems/${s.stem_id}/refine`, { method: "POST", body: { model: m.model } });
      if (this.disposed) return;
      toast(res.message);
      if (res.reason === "done") return;
      this.jobs.set(s.stem_id, { ...res.job, model: m.model });
      this.refresh();
      this.schedule();
    } catch (e) {
      toast(e.message);
    }
  }

  async cancel(job) {
    try {
      await api(`/api/jobs/${job.job_id}/cancel`, { method: "POST" });
      toast("分割をやめます。");
    } catch (e) {
      toast(e.message);
    }
  }

  /** 詳細分割の子を消して、分ける前に戻す。 */
  async undo(s) {
    const kids = this.descendants(s.code).map((c) => this.view.tree.byCode.get(c).display_name);
    const ok = await confirmDialog(
      `「${s.display_name}」を分ける前に戻しますか？ 子の stem（${kids.join("・")}）のファイルを削除します。`,
      { ok: "戻す", danger: true },
    );
    if (!ok || this.disposed) return;
    const kidCodes = this.descendants(s.code);
    try {
      await api(`/api/jobs/${s.refined_by.job_id}`, { method: "DELETE" });
    } catch (e) {
      toast(e.message);
      return;
    }
    toast(`「${s.display_name}」を分ける前に戻しました。`);
    this.reload((sel, gains) => {
      // 子を1つでも鳴らしていたら、親を鳴らす
      const on = kidCodes.filter((c) => sel.has(c));
      if (on.length) {
        sel.add(s.code);
        gains.set(s.code, gains.get(on[0]) || 0);
      }
    });
  }

  // --- 進み具合 ---------------------------------------------------------------------

  schedule() {
    clearTimeout(this.timer);
    if (this.disposed) return;
    if (!this.activeJobs().length && !this.pendingDone.length) return;
    this.timer = setTimeout(() => this.poll(), POLL_MS);
  }

  async poll() {
    this.timer = 0;
    if (this.disposed) return;
    const active = this.activeJobs();
    const done = [];
    for (const job of active) {
      let fresh;
      try {
        fresh = await api(`/api/jobs/${job.job_id}`);
      } catch (e) {
        if (e.status === 404) { this.jobs.delete(job.input_stem_id); continue; }
        continue; // 一時的な失敗は次で
      }
      if (this.disposed) return;
      const merged = { ...job, ...fresh, model: job.model };
      if (fresh.status === "done") done.push(merged);
      if (fresh.status === "canceled") {
        this.jobs.delete(job.input_stem_id);
        toast("分割をやめました。");
      } else {
        this.jobs.set(job.input_stem_id, merged);
        if (fresh.status === "failed") toast(fresh.error_message || "分割に失敗しました。");
      }
    }
    this.pendingDone.push(...done);
    if (this.pendingDone.length && !this.view.loading) {
      const ready = this.pendingDone;
      this.pendingDone = [];
      this.finish(ready);
      return;
    }
    // 読み込み中に終わったものは、読み込みが終わるまで待ってから読み直す
    this.refresh();
    this.schedule();
  }

  /** 詳細分割が終わった: 子を読み込むため画面を読み直す（読み込み中なら終わるのを待つ）。 */
  finish(done) {
    const expand = done.map((job) => {
      const s = this.stemById(job.input_stem_id);
      const method = s && (s.refine_methods || []).find((m) => m.model === job.model);
      return { code: s ? s.code : null, kids: method ? method.children.map((c) => c.code) : [] };
    });
    toast("分け終わりました。子の stem を読み込みます。");
    this.reload((sel, gains) => {
      // 親を鳴らしていたら、子を全部（同じ音量で）鳴らす
      for (const { code, kids } of expand) {
        if (!code || !sel.has(code)) continue;
        for (const k of kids) {
          sel.add(k);
          if (gains.has(code)) gains.set(k, gains.get(code));
        }
      }
    });
  }

  /** 再生位置・選択を保って読み直す。adjust(sel, gainsDb) で選択を直してから渡す。 */
  reload(adjust) {
    const view = this.view;
    const state = view.restore || view.captureState();
    adjust(state.sel, state.gainsDb);
    view.restore = state;
    view.remount();
  }
}
