// ライブラリ画面: 曲の一覧、取り込み（ファイル・URL）、分割の進捗・キャンセル、削除。

import { api } from "./api.js";
import { confirmDialog, el, formatTime, toast } from "./ui.js";

const STATUS_LABELS = {
  queued: "分割待ち",
  running: "分割中",
  done: "分割済み",
  failed: "失敗",
  canceled: "キャンセル",
};
const FINISHED = new Set(["done", "failed", "canceled"]);
const IMPORT_POLL_MS = 1000;
const PRESET_KEY = "stemapp.preset";
// 聴き比べ用の実験プリセットを品質の選択肢に出すか
const SHOW_EXP_KEY = "stemapp.showExperimental";

// 取り込み中の一覧は画面を切り替えても残す（ページを読み直すと消える）
const imports = [];
let importSeq = 0;

function loadPresetChoice() {
  try { return localStorage.getItem(PRESET_KEY); } catch { return null; }
}

function savePresetChoice(code) {
  try { localStorage.setItem(PRESET_KEY, code); } catch { /* 保存できなくても動く */ }
}

function loadShowExperimental() {
  try { return localStorage.getItem(SHOW_EXP_KEY) === "1"; } catch { return false; }
}

function saveShowExperimental(on) {
  try { localStorage.setItem(SHOW_EXP_KEY, on ? "1" : "0"); } catch { /* 保存できなくても動く */ }
}

export class LibraryView {
  constructor(root) {
    this.root = root;
    this.tracks = [];
    this.presets = [];
    this.watchers = new Map(); // job_id → EventSource
    this.jobs = new Map(); // job_id → 最新のジョブ情報（SSE で更新）
    this.timers = new Set();
    this.alive = true;
    this.showExperimental = loadShowExperimental();
  }

  async mount() {
    this.root.replaceChildren(el("p", { class: "empty", text: "読み込み中…" }));
    try {
      const [presets] = await Promise.all([api("/api/presets"), this.refresh(false)]);
      this.presets = presets.presets;
    } catch (e) {
      this.root.replaceChildren(el("p", { class: "empty error-text", text: e.message }));
      return;
    }
    this.render();
    for (const task of imports) if (!task.done) this.pollImport(task);
  }

  unmount() {
    this.alive = false;
    for (const es of this.watchers.values()) es.close();
    this.watchers.clear();
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
  }

  later(fn, ms) {
    const t = setTimeout(() => { this.timers.delete(t); if (this.alive) fn(); }, ms);
    this.timers.add(t);
  }

  async refresh(render = true) {
    const data = await api("/api/tracks");
    this.tracks = data.tracks;
    for (const t of this.tracks) {
      if (t.latest_job) this.jobs.set(t.latest_job.job_id, t.latest_job);
    }
    if (render && this.alive) this.render();
    if (this.alive) this.syncWatchers();
  }

  // --- 進捗（SSE） -------------------------------------------------------------

  syncWatchers() {
    const active = new Set();
    for (const t of this.tracks) {
      const job = t.latest_job && this.jobs.get(t.latest_job.job_id);
      if (job && !FINISHED.has(job.status)) active.add(job.job_id);
    }
    for (const [id, es] of this.watchers) {
      if (!active.has(id)) { es.close(); this.watchers.delete(id); }
    }
    for (const id of active) if (!this.watchers.has(id)) this.watch(id);
  }

  watch(jobId) {
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    this.watchers.set(jobId, es);
    es.addEventListener("job", (ev) => {
      const job = JSON.parse(ev.data);
      this.jobs.set(jobId, job);
      this.updateTrackRow(job.track_id);
      if (FINISHED.has(job.status)) {
        // 終わった状態を受けたら閉じる（閉じないとブラウザが接続し直す）
        es.close();
        this.watchers.delete(jobId);
        if (job.status === "failed") toast(`分割に失敗しました: ${job.error_message || ""}`);
        this.refresh().catch(() => {});
      }
    });
    es.addEventListener("error", () => {
      // サーバーが止まった・ジョブが消えた等。閉じて、少し後に一覧を読み直す
      es.close();
      this.watchers.delete(jobId);
      this.later(() => this.refresh().catch(() => {}), 3000);
    });
  }

  // --- 取り込み ----------------------------------------------------------------

  selectedPreset() {
    const select = this.root.querySelector("#preset-select");
    return select ? select.value : "standard";
  }

  async startImport({ file, url }) {
    const preset = this.selectedPreset();
    const task = {
      key: ++importSeq, name: file ? file.name : url, sourceId: null,
      status: "uploading", message: file ? "送信中…" : "登録中…", error: null, done: false,
    };
    imports.unshift(task);
    this.renderImports();
    try {
      let res;
      if (file) {
        const form = new FormData();
        form.append("file", file);
        form.append("separate", "true");
        form.append("preset", preset);
        res = await api("/api/imports", { method: "POST", body: form });
      } else {
        res = await api("/api/imports", { method: "POST", body: { url, separate: true, preset } });
      }
      task.sourceId = res.source_id;
      task.status = res.status;
      task.message = res.message;
    } catch (e) {
      task.status = "failed";
      task.error = e.message;
      task.done = true;
    }
    this.renderImports();
    if (!task.done && this.alive) this.pollImport(task);
  }

  async pollImport(task) {
    if (!this.alive || task.done || task.sourceId === null) return;
    try {
      const res = await api(`/api/imports/${task.sourceId}`);
      task.status = res.status;
      task.message = res.message;
      if (res.status === "failed") {
        task.error = res.message;
        task.done = true;
      } else if (res.status === "done") {
        task.done = true;
        if (!res.job_created && res.job_id) toast("この曲は取り込み済み・分割済みです。");
        const i = imports.indexOf(task);
        if (i >= 0) imports.splice(i, 1);
        await this.refresh(false);
        if (this.alive) this.render();
        return;
      }
    } catch (e) {
      task.message = e.message;
    }
    if (this.alive) {
      this.renderImports();
      if (!task.done) this.later(() => this.pollImport(task), IMPORT_POLL_MS);
    }
  }

  // --- 操作 --------------------------------------------------------------------

  async separate(track) {
    try {
      const res = await api(`/api/tracks/${track.track_id}/jobs`, {
        method: "POST", body: { preset: this.selectedPreset() },
      });
      toast(res.message);
      await this.refresh();
    } catch (e) {
      toast(e.message);
    }
  }

  async cancel(job) {
    try {
      await api(`/api/jobs/${job.job_id}/cancel`, { method: "POST" });
      toast("キャンセルを受け付けました。");
      await this.refresh();
    } catch (e) {
      toast(e.message);
    }
  }

  async remove(track) {
    const ok = await confirmDialog(
      `「${track.title}」を削除しますか？ 分割した stem・キューも消えます。元に戻せません。`,
      { ok: "削除する", danger: true },
    );
    if (!ok) return;
    try {
      await api(`/api/tracks/${track.track_id}`, { method: "DELETE" });
      toast("削除しました。");
      await this.refresh();
    } catch (e) {
      toast(e.message);
    }
  }

  // --- 描画 --------------------------------------------------------------------

  presetSelectEl() {
    const normal = this.presets.filter((p) => !p.experimental);
    const exp = this.showExperimental ? this.presets.filter((p) => p.experimental) : [];
    const saved = loadPresetChoice();
    const visible = [...normal, ...exp].map((p) => p.code);
    const chosen = visible.includes(saved) ? saved
      : (normal.find((p) => p.is_default) || normal[0] || {}).code || "standard";
    const option = (p) => el("option", { value: p.code, text: p.display_name, selected: p.code === chosen });
    return el("select", {
      id: "preset-select", class: "select", "aria-label": "品質",
      onchange: (e) => savePresetChoice(e.target.value),
    },
    normal.map(option),
    exp.length ? el("optgroup", { label: "実験（聴き比べ用）" }, exp.map(option)) : null);
  }

  toggleExperimental(on) {
    this.showExperimental = on;
    saveShowExperimental(on);
    const old = this.root.querySelector("#preset-select");
    if (old) old.replaceWith(this.presetSelectEl());
  }

  render() {
    if (!this.alive) return;
    const presetSelect = this.presetSelectEl();
    const expToggle = this.presets.some((p) => p.experimental)
      ? el("label", { class: "exp-toggle", title: "同じ曲を別の分け方で分割して、プレイヤーで聴き比べるための選択肢を出します" },
        el("input", {
          type: "checkbox", id: "show-experimental", checked: this.showExperimental,
          onchange: (e) => this.toggleExperimental(e.target.checked),
        }), "実験を表示")
      : null;

    const fileInput = el("input", {
      type: "file", accept: "audio/*,video/*,.mp3,.m4a,.flac,.wav,.ogg,.opus,.aac,.webm,.mp4",
      multiple: true, hidden: true, id: "file-input",
      onchange: (e) => {
        for (const f of e.target.files) this.startImport({ file: f });
        e.target.value = "";
      },
    });
    const drop = el("div", {
      class: "dropzone", tabindex: "0", role: "button", id: "dropzone",
      onclick: () => fileInput.click(),
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } },
      ondragover: (e) => { e.preventDefault(); drop.classList.add("over"); },
      ondragleave: () => drop.classList.remove("over"),
      ondrop: (e) => {
        e.preventDefault();
        drop.classList.remove("over");
        for (const f of e.dataTransfer.files) this.startImport({ file: f });
      },
    }, el("strong", { text: "音声ファイルをここにドロップ" }), " または クリックして選択");

    const urlInput = el("input", {
      class: "input grow", type: "url", placeholder: "https://…（YouTube などの URL）",
      "aria-label": "URL", id: "url-input",
    });
    const urlForm = el("form", {
      class: "row",
      onsubmit: (e) => {
        e.preventDefault();
        const url = urlInput.value.trim();
        if (!/^https?:\/\//i.test(url)) { toast("http:// か https:// で始まる URL を入れてください。"); return; }
        urlInput.value = "";
        this.startImport({ url });
      },
    }, urlInput, el("button", { class: "btn primary", type: "submit", text: "取り込む" }));

    const importPanel = el("section", { class: "panel import-panel" },
      el("h2", { text: "取り込み・分割" }),
      drop, fileInput, urlForm,
      el("div", { class: "row" },
        el("label", { class: "muted", for: "preset-select", text: "品質" }), presetSelect, expToggle,
        el("span", { class: "muted", text: "取り込んだら自動で分割します。" })),
      el("div", { class: "jobs-list", id: "imports" }),
    );

    const trackList = el("div", { class: "tracks", id: "tracks" });
    const libPanel = el("section", { class: "panel" },
      el("h2", { text: `ライブラリ（${this.tracks.length} 曲）` }), trackList);

    this.root.replaceChildren(el("div", { class: "library" }, importPanel, libPanel));
    this.renderImports();
    this.renderTracks();
  }

  renderImports() {
    const box = this.root.querySelector("#imports");
    if (!box) return;
    box.replaceChildren(...imports.map((task) => {
      const failed = task.status === "failed";
      return el("div", { class: "job-card", dataset: { importKey: String(task.key) } },
        el("div", { class: "row" },
          el("span", { class: "title grow", text: task.name }),
          failed
            ? el("button", {
              class: "btn small", type: "button", text: "閉じる",
              onclick: () => { imports.splice(imports.indexOf(task), 1); this.renderImports(); },
            })
            : null),
        failed
          ? el("div", { class: "error-text", text: task.error || "取り込みに失敗しました。" })
          : [
            el("div", { class: "progress indeterminate" }, el("span")),
            el("div", { class: "muted", text: task.message || "" }),
          ],
      );
    }));
  }

  trackRow(t) {
    const job = t.latest_job ? (this.jobs.get(t.latest_job.job_id) || t.latest_job) : null;
    const status = job ? job.status : null;
    const active = status === "queued" || status === "running";
    const playable = t.playable_job_id !== null && t.playable_job_id !== undefined;
    const open = () => { location.hash = `#/track/${t.track_id}`; };

    let state;
    if (active) {
      const pct = Math.round((job.progress || 0) * 100);
      state = el("div", { class: "t-state" },
        el("div", { class: "row" },
          el("span", { class: "badge active", text: STATUS_LABELS[status] }),
          el("span", { class: "muted", text: `${pct}%` })),
        el("div", { class: "progress" }, el("span", { style: { width: `${pct}%` } })),
        el("div", { class: "stage", text: job.stage || "" }));
    } else {
      const cls = status === "done" ? "done" : status === "failed" ? "failed" : "";
      const label = status ? STATUS_LABELS[status] : "未分割";
      state = el("div", { class: "t-state" },
        el("span", { class: `badge ${cls}`, text: playable && status !== "done" ? `${label}（前回の分割あり）` : label }),
        status === "failed" && job.error_message
          ? el("div", { class: "stage error-text", text: job.error_message, title: job.error_message })
          : null);
    }

    const stop = (fn) => (e) => { e.stopPropagation(); fn(); };
    const actions = el("div", { class: "t-actions" },
      playable ? el("button", { class: "btn small primary", type: "button", text: "再生", onclick: stop(open) }) : null,
      active ? el("button", { class: "btn small", type: "button", text: "キャンセル", onclick: stop(() => this.cancel(job)) }) : null,
      !active && !playable
        ? el("button", { class: "btn small", type: "button", text: "分割", onclick: stop(() => this.separate(t)) })
        : null,
      !active && playable
        ? el("button", {
          class: "btn small", type: "button", text: "別の分け方で分割",
          title: "上の「品質」で選んだ分け方で、もう一度分割します（今の分け方も残ります。プレイヤーで切り替えて聴き比べられます）",
          onclick: stop(() => this.separate(t)),
        })
        : null,
      el("button", {
        class: "btn small danger", type: "button", text: "削除", "aria-label": `${t.title} を削除`,
        onclick: stop(() => this.remove(t)),
      }),
    );

    return el("div", {
      class: `track-row${playable ? " playable" : ""}`,
      dataset: { trackId: String(t.track_id) },
      onclick: playable ? open : null,
    },
    el("div", { style: { minWidth: "0" } },
      el("div", { class: "t-title", text: t.title, title: t.title }),
      el("div", { class: "t-artist", text: t.artist || "アーティスト不明" })),
    el("div", { class: "t-len", text: t.duration_sec ? formatTime(t.duration_sec) : "-" }),
    state,
    actions);
  }

  renderTracks() {
    const box = this.root.querySelector("#tracks");
    if (!box) return;
    if (!this.tracks.length) {
      box.replaceChildren(el("p", { class: "empty", text: "まだ曲がありません。上から取り込んでください。" }));
      return;
    }
    box.replaceChildren(...this.tracks.map((t) => this.trackRow(t)));
  }

  updateTrackRow(trackId) {
    const t = this.tracks.find((x) => x.track_id === trackId);
    const old = this.root.querySelector(`.track-row[data-track-id="${trackId}"]`);
    if (t && old) old.replaceWith(this.trackRow(t));
  }
}
