// 書き出しメニュー（プレイヤーの「書き出し」ボタンから開く）。
// stem を1つ／全部を ZIP／今の組み合わせをミックス と、形式（WAV・FLAC・MP3）を選んで作成し、
// できたらダウンロードのリンクを出す。作成はサーバーで行い、ここでは進み具合を問い合わせるだけ。

import { api } from "./api.js";
import * as S from "./selection.js";
import { el, toast } from "./ui.js";

const POLL_MS = 700;

export const EXPORT_TYPES = [
  { value: "single", label: "stem を1つ" },
  { value: "all", label: "全部を ZIP" },
  { value: "mix", label: "今の組み合わせ" },
];

export const EXPORT_FORMATS = [
  { value: "wav", label: "WAV", sub: "24bit" },
  { value: "flac", label: "FLAC", sub: "24bit" },
  { value: "mp3", label: "MP3", sub: "320kbps" },
];

/** バイト数を「12.3 MB」のようにする。 */
export function formatBytes(n) {
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

/** iPhone・iPad（iPadOS はデスクトップの UA を名乗るので触れる画面かどうかでも見る）。 */
export function isIOS(nav = navigator) {
  const ua = nav.userAgent || "";
  return /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && (nav.maxTouchPoints || 0) > 1);
}

/**
 * 選択（葉の集合）を、書き出しに送る stem の並びにする。子が全部 ON で音量がそろっている親は
 * 親1つにまとめる（ファイル名が短くなる）。戻り値: [{ code, gain_db }]
 */
export function compactSelection(tree, sel, gainsDb = new Map()) {
  const out = [];
  const gainOf = (c) => Number(gainsDb.get(c)) || 0;
  const walk = (code) => {
    const state = S.stateOf(tree, sel, code);
    if (state === "off") return;
    const leaves = S.leavesOf(tree, code);
    const uniform = leaves.every((c) => gainOf(c) === gainOf(leaves[0]));
    if (state === "on" && uniform) {
      out.push({ code, gain_db: gainOf(leaves[0]) });
      return;
    }
    for (const k of tree.children.get(code)) walk(k);
  };
  const roots = tree.order.filter((c) => {
    const s = tree.byCode.get(c);
    return !s.parent_code || !tree.byCode.has(s.parent_code);
  });
  for (const r of roots) walk(r);
  return out;
}

/**
 * 今の組み合わせの書き出し指定。組み合わせプリセットを選んだままなら listen_preset_id、
 * それ以外は選択中の stem と音量。label は画面に出す説明。
 */
export function mixRequest(view) {
  const preset = view.activePresetId != null
    ? (view.presets || []).find((p) => p.listen_preset_id === view.activePresetId)
    : null;
  if (preset) {
    return { body: { listen_preset_id: preset.listen_preset_id }, label: `組み合わせ「${preset.name}」`, empty: false };
  }
  const stems = compactSelection(view.tree, view.sel, view.gainsDb);
  const names = stems.map((s) => {
    const st = view.tree.byCode.get(s.code);
    const db = s.gain_db ? `（${s.gain_db > 0 ? "+" : ""}${s.gain_db}dB）` : "";
    return `${st ? st.display_name : s.code}${db}`;
  });
  return { body: { stems }, label: names.length ? names.join("＋") : "（何も選ばれていません）", empty: !stems.length };
}

function segment(name, options, value, onChange) {
  return el("div", { class: "seg", role: "radiogroup", id: `export-${name}` },
    options.map((o) => el("button", {
      type: "button", role: "radio", class: `seg-btn${o.value === value ? " on" : ""}`,
      "aria-checked": String(o.value === value), dataset: { value: o.value },
      onclick: () => onChange(o.value),
    }, el("span", { text: o.label }), o.sub ? el("small", { text: o.sub }) : null)));
}

/** プレイヤー1画面につき1つ。閉じても作成中の書き出しは続き、開き直すと状態を出す。 */
export class Exporter {
  constructor(view) {
    this.view = view;
    this.type = "single";
    this.format = "wav";
    const leaves = view.tree.leaves;
    this.stemCode = leaves[0] || view.tree.order[0];
    this.parentsOnly = false;
    this.current = null; // 最後に作った書き出し（API の export）
    this.pollTimer = 0;
    this.back = null;
    this.onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); this.close(); }
    };
  }

  get isOpen() { return !!this.back; }

  open() {
    if (this.back) return;
    this.back = el("div", { class: "modal-back", onclick: (e) => { if (e.target === this.back) this.close(); } });
    document.addEventListener("keydown", this.onKey, true);
    document.body.append(this.back);
    this.render();
    const first = this.back.querySelector(".seg-btn.on");
    if (first) first.focus();
  }

  close() {
    if (!this.back) return;
    document.removeEventListener("keydown", this.onKey, true);
    this.back.remove();
    this.back = null;
  }

  /** 画面を離れるとき。問い合わせも止める（サーバーの作成は続く）。 */
  dispose() {
    this.close();
    clearTimeout(this.pollTimer);
    this.pollTimer = 0;
  }

  busy() {
    return !!this.current && (this.current.status === "queued" || this.current.status === "running");
  }

  render() {
    if (!this.back) return;
    const view = this.view;
    const modal = el("div", { class: "modal export-modal", role: "dialog", "aria-modal": "true", "aria-labelledby": "export-title" },
      el("div", { class: "export-head" },
        el("h2", { id: "export-title", text: "書き出し" }),
        el("span", { class: "muted export-track", text: view.track.title, title: view.track.title }),
        el("button", { class: "btn small icon export-close", type: "button", "aria-label": "閉じる", text: "×", onclick: () => this.close() })),
      el("div", { class: "export-field" },
        el("div", { class: "export-label", text: "書き出すもの" }),
        segment("type", EXPORT_TYPES, this.type, (v) => { this.type = v; this.render(); })),
      el("div", { class: "export-field", id: "export-target" }, this.targetEl()),
      el("div", { class: "export-field" },
        el("div", { class: "export-label", text: this.type === "all" ? "形式（ZIP の中身）" : "形式" }),
        segment("format", EXPORT_FORMATS, this.format, (v) => { this.format = v; this.render(); })),
      el("div", { class: "export-actions" },
        el("button", {
          class: "btn primary", type: "button", id: "export-start", text: "作成する",
          disabled: this.busy() || (this.type === "mix" && mixRequest(view).empty),
          onclick: () => this.start(),
        })),
      el("div", { class: "export-status", id: "export-status", "aria-live": "polite" }, this.statusEl()));
    this.back.replaceChildren(modal);
  }

  targetEl() {
    const view = this.view;
    const tree = view.tree;
    if (this.type === "single") {
      const select = el("select", {
        class: "select export-select", id: "export-stem", "aria-label": "書き出す stem",
        onchange: (e) => { this.stemCode = e.target.value; },
      }, tree.order.map((code) => {
        const s = tree.byCode.get(code);
        const child = s.parent_code && tree.byCode.has(s.parent_code);
        return el("option", { value: code, selected: code === this.stemCode, text: `${child ? "└ " : ""}${s.display_name}` });
      }));
      return [el("div", { class: "export-label", text: "stem" }), select];
    }
    if (this.type === "all") {
      const hasChildren = tree.order.some((c) => S.isParent(tree, c));
      const count = hasChildren && this.parentsOnly
        ? tree.order.filter((c) => { const s = tree.byCode.get(c); return !s.parent_code || !tree.byCode.has(s.parent_code); }).length
        : tree.leaves.length;
      return [
        el("div", { class: "export-label", text: "ZIP に入れる stem" }),
        hasChildren
          ? segment("level", [
            { value: "leaves", label: "分けた stem（子）" },
            { value: "parents", label: "親の stem だけ" },
          ], this.parentsOnly ? "parents" : "leaves", (v) => { this.parentsOnly = v === "parents"; this.render(); })
          : null,
        el("p", { class: "muted export-note", text: `${count} 個の stem を1つの ZIP にまとめます。` }),
      ];
    }
    const mix = mixRequest(view);
    return [
      el("div", { class: "export-label", text: "混ぜるもの（今の組み合わせ）" }),
      el("p", { class: `export-mix${mix.empty ? " error-text" : ""}`, id: "export-mix", text: mix.label }),
      el("p", { class: "muted export-note", text: "音量（dB）も反映します。音割れする場合は全体を同じだけ下げます。" }),
    ];
  }

  statusEl() {
    const x = this.current;
    if (!x) return null;
    if (x.status === "queued" || x.status === "running") {
      const pct = Math.round((x.progress || 0) * 100);
      return [
        el("div", { class: "row export-stage" },
          el("span", { class: "grow", text: x.stage || "書き出し中" }),
          el("span", { class: "muted", text: `${pct}%` })),
        el("div", { class: "progress" }, el("span", { style: { width: `${pct}%` } })),
      ];
    }
    if (x.status === "failed") {
      return el("p", { class: "error-text", text: x.error_message || "書き出しに失敗しました。" });
    }
    const link = el("a", {
      class: "btn primary export-download", id: "export-download", href: x.download_url,
      download: x.filename, text: "ダウンロード",
    });
    return [
      el("div", { class: "export-file" },
        el("div", { class: "export-filename", text: x.filename, title: x.filename }),
        el("div", { class: "muted", text: `${x.format_label}${x.zip ? "（ZIP）" : ""}・${formatBytes(x.bytes)}` })),
      x.mix_gain_db < 0
        ? el("p", { class: "export-gain", id: "export-gain", text: `音割れしないよう、全体を ${x.mix_gain_db.toFixed(1)} dB 下げました。` })
        : null,
      link,
      isIOS()
        ? el("p", { class: "muted export-note", id: "export-ios-hint", text: "iPhone: 確認が出たら「ダウンロード」を選ぶと、ファイル App の「ダウンロード」に保存されます。保存できないときはボタンを長押しして「リンク先のファイルをダウンロード」を選んでください。" })
        : null,
      el("p", { class: "muted export-note", text: "ファイルはサーバーに24時間残ります。" }),
    ];
  }

  async start() {
    const view = this.view;
    const body = { export_type: this.type, format: this.format };
    if (this.type === "single") body.stem_code = this.stemCode;
    if (this.type === "all") body.parents_only = this.parentsOnly;
    if (this.type === "mix") {
      const mix = mixRequest(view);
      if (mix.empty) { toast("ミックスする stem を選んでください。"); return; }
      Object.assign(body, mix.body);
    }
    // 前の結果（ダウンロードのリンク）は消してから作る
    this.current = { status: "queued", progress: 0, stage: "書き出しを登録中" };
    this.render();
    try {
      const res = await api(`/api/jobs/${view.job.job_id}/exports`, { method: "POST", body });
      this.current = res.export;
      this.render();
      this.schedulePoll();
    } catch (e) {
      this.current = { status: "failed", error_message: e.message };
      this.render();
    }
  }

  schedulePoll() {
    clearTimeout(this.pollTimer);
    this.pollTimer = setTimeout(() => this.poll(), POLL_MS);
  }

  async poll() {
    this.pollTimer = 0;
    if (!this.view.alive || !this.current) return;
    try {
      const res = await api(`/api/exports/${this.current.export_id}`);
      if (!this.view.alive) return;
      this.current = res.export;
    } catch (e) {
      if (!this.view.alive) return;
      if (e.status === 404) {
        this.current = { status: "failed", error_message: e.message };
      } else {
        this.schedulePoll(); // 一時的な通信の失敗は、もう一度問い合わせる
        return;
      }
    }
    if (this.busy()) this.schedulePoll();
    else if (!this.isOpen && this.current.status === "done") toast("書き出しができました。「書き出し」から保存できます。");
    this.render();
  }
}
