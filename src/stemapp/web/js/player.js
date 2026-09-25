// プレイヤー画面: 波形、再生・シーク、stem の ON/OFF、グループ、組み合わせプリセット、キュー・ループ。

import { api, fetchBinary } from "./api.js";
import { Engine, clampTime } from "./engine.js";
import { parsePeaks } from "./peaks.js";
import * as S from "./selection.js";
import { confirmDialog, el, formatTime, icon, ICONS, promptDialog, toast } from "./ui.js";
import { WaveformView, ZOOM_STEPS } from "./waveform.js";

const SEEK_STEP_SEC = 5;
const LOAD_CONCURRENCY = 3;
const POSTPROCESS_POLL_MS = 1500;
const VOLUME_KEY = "stemapp.volume";
// キューの色。stem・グループの色と重ならないよう、差し色の赤2色と白だけにする
export const CUE_COLORS = ["#FF3B4E", "#F5F5F4", "#FF8A95"];

function loadVolume() {
  try {
    const v = Number(localStorage.getItem(VOLUME_KEY));
    return Number.isFinite(v) && localStorage.getItem(VOLUME_KEY) !== null ? v : 0.9;
  } catch { return 0.9; }
}

/** items を最大 limit 個ずつ並行して処理する。signal が中断されたら新しい項目を始めない。 */
async function mapLimit(items, limit, fn, signal) {
  const out = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      if (signal && signal.aborted) throw new DOMException("中断しました", "AbortError");
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return out;
}

export class PlayerView {
  constructor(root, trackId) {
    this.root = root;
    this.trackId = trackId;
    this.alive = true;
    this.abort = new AbortController();
    this.timers = new Set();
    this.engine = null;
    this.wave = null;
    this.raf = 0;
    this.sel = new Set();
    this.gainsDb = new Map();
    this.soloMode = false;
    this.activePresetId = null;
    this.cues = [];
    this.loopCueId = null;
    this.loopOn = false;
    this.ready = false;
    this.onKey = (e) => this.handleKey(e);
  }

  later(fn, ms) {
    const t = setTimeout(() => { this.timers.delete(t); if (this.alive) fn(); }, ms);
    this.timers.add(t);
  }

  // --- 読み込み ---------------------------------------------------------------

  async mount() {
    this.root.replaceChildren(el("p", { class: "empty", text: "読み込み中…" }));
    try {
      this.track = await api(`/api/tracks/${this.trackId}`);
    } catch (e) {
      if (this.alive) this.showMessage(e.status === 404 ? "曲が見つかりません。" : e.message);
      return;
    }
    if (!this.alive) return;
    const jobId = this.track.playable_job_id;
    if (!jobId) {
      this.showMessage("この曲はまだ分割されていません。ライブラリで分割してください。");
      return;
    }
    try {
      const [stems, types, groups, presets, cues] = await Promise.all([
        api(`/api/jobs/${jobId}/stems`),
        api("/api/stem-types"),
        api("/api/stem-groups"),
        api("/api/listen-presets"),
        api(`/api/tracks/${this.trackId}/cues`),
      ]);
      this.job = stems;
      this.stemTypes = types.stem_types;
      this.groups = groups.stem_groups;
      this.presets = presets.listen_presets;
      this.cues = cues.cues;
    } catch (e) {
      if (this.alive) this.showMessage(e.message);
      return;
    }
    if (!this.alive) return;
    if (!this.job.delivery_ready) {
      this.showDeliveryMissing();
      return;
    }
    this.tree = S.buildTree(this.job.stems);
    this.sel = S.allOn(this.tree);
    this.render();
    document.addEventListener("keydown", this.onKey);
    await this.loadMedia();
    if (!this.alive) return;
  }

  showMessage(text) {
    this.root.replaceChildren(el("div", { class: "panel" },
      el("p", { class: "empty", text }),
      el("div", { class: "row", style: { justifyContent: "center" } },
        el("a", { class: "btn", href: "#/library", text: "ライブラリへ戻る" }))));
  }

  showDeliveryMissing() {
    const status = this.job.postprocess_status;
    const busy = status === "queued" || status === "running";
    const failed = status === "failed";
    const button = el("button", {
      class: "btn primary", type: "button", id: "rebuild-btn", text: "作り直す", disabled: busy,
      onclick: () => this.requestPostprocess(),
    });
    this.root.replaceChildren(el("div", { class: "player-main" },
      this.headEl(),
      el("div", { class: "notice", id: "delivery-notice" },
        el("strong", { text: "配信用データがありません" }),
        el("span", { class: "muted", text: busy
          ? "配信用データ（再生用の音声と波形）を作成しています。しばらくお待ちください。"
          : "この曲の再生用の音声・波形がまだ作られていません。作り直すと再生できるようになります。" }),
        failed ? el("span", { class: "error-text", text: "前回の作り直しは失敗しました。もう一度お試しください。" }) : null,
        busy ? el("div", { class: "progress indeterminate", style: { width: "240px" } }, el("span")) : null,
        button)));
    if (busy) this.later(() => this.pollPostprocess(), POSTPROCESS_POLL_MS);
  }

  async requestPostprocess() {
    try {
      const res = await api(`/api/jobs/${this.job.job_id}/postprocess`, { method: "POST" });
      if (!this.alive) return;
      toast(res.message);
      this.job.postprocess_status = res.job.postprocess_status;
      if (res.reason === "ready") { this.remount(); return; }
      this.showDeliveryMissing();
    } catch (e) {
      toast(e.message);
    }
  }

  async pollPostprocess() {
    try {
      const job = await api(`/api/jobs/${this.job.job_id}`);
      if (!this.alive) return;
      this.job.postprocess_status = job.postprocess_status;
      if (job.postprocess_status === "queued" || job.postprocess_status === "running") {
        this.later(() => this.pollPostprocess(), POSTPROCESS_POLL_MS);
        return;
      }
      if (job.postprocess_status === "done") { this.remount(); return; }
      this.showDeliveryMissing();
    } catch (e) {
      toast(e.message);
      this.later(() => this.pollPostprocess(), POSTPROCESS_POLL_MS * 2);
    }
  }

  remount() {
    this.unmount();
    this.alive = true;
    this.abort = new AbortController();
    this.mount();
  }

  async loadMedia() {
    const leaves = this.tree.leaves.map((c) => this.tree.byCode.get(c));
    const cover = this.root.querySelector("#loading");
    const label = cover.querySelector(".label");
    const bar = cover.querySelector(".progress > span");
    const total = leaves.length * 2;
    let done = 0;
    const step = () => {
      done++;
      label.textContent = `音声と波形を読み込み中… ${done}/${total}`;
      bar.style.width = `${Math.round((done / total) * 100)}%`;
    };
    // 1つでも失敗したら残りの取得を中断する（画面を離れたときの中断にも従う）
    const loading = new AbortController();
    const onLeave = () => loading.abort();
    this.abort.signal.addEventListener("abort", onLeave);
    try {
      this.engine = new Engine();
      const signal = loading.signal;
      const loaded = await mapLimit(leaves, LOAD_CONCURRENCY, async (stem) => {
        const stream = stem.renditions.find((r) => r.purpose === "stream");
        const peaks = new Map();
        const peakBufs = await Promise.all(stem.peaks.map((p) => fetchBinary(p.url, signal)));
        stem.peaks.forEach((p, i) => peaks.set(p.samples_per_px, parsePeaks(peakBufs[i])));
        step();
        const audio = await fetchBinary(stream.url, signal);
        if (signal.aborted) throw new DOMException("中断しました", "AbortError");
        const buffer = await this.engine.decode(audio);
        step();
        return { stem, buffer, peaks };
      }, signal).catch((e) => {
        loading.abort();
        throw e;
      });
      if (!this.alive) return;
      for (const code of this.tree.order) {
        const item = loaded.find((x) => x.stem.code === code);
        this.engine.addTrack(code, item ? item.buffer : null, 0);
      }
      this.engine.duration = Math.max(this.engine.duration, 0);
      this.engine.onEnded = () => this.updateTransport();
      this.engine.setVolume(loadVolume());
      const first = loaded[0].peaks.values().next().value;
      const sampleRate = first.sampleRate;
      const totalSamples = Math.round(
        Math.max(this.engine.duration, this.track.duration_sec || 0) * sampleRate);
      this.wave = new WaveformView({
        overview: this.root.querySelector("#wave-overview"),
        zoom: this.root.querySelector("#wave-zoom"),
        layers: loaded.map((x) => ({ code: x.stem.code, color: x.stem.color, peaks: x.peaks })),
        sampleRate,
        totalSamples,
        getState: () => ({
          position: this.engine.position,
          sel: this.sel,
          cues: this.cues,
          loop: this.loopOn ? this.activeLoop() : null,
        }),
        onSeek: (t) => this.seek(t),
      });
      this.applySelection(0);
      this.ready = true;
      cover.hidden = true;
      this.root.querySelector("#play-btn").disabled = false;
      this.frame();
    } catch (e) {
      if (!this.alive) return;
      if (e.name === "AbortError") return;
      label.textContent = `読み込めませんでした: ${e.message}`;
      label.classList.add("error-text");
      cover.querySelector(".progress").hidden = true;
    }
  }

  unmount() {
    this.alive = false;
    this.abort.abort();
    document.removeEventListener("keydown", this.onKey);
    cancelAnimationFrame(this.raf);
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
    if (this.engine) this.engine.close();
    this.engine = null;
    this.ready = false;
  }

  // --- 再生 -------------------------------------------------------------------

  frame() {
    if (!this.alive || !this.engine) return;
    this.engine.tick();
    this.wave.draw();
    const t = this.root.querySelector("#time-now");
    if (t) t.textContent = formatTime(this.wave.preview ?? this.engine.position, true);
    this.raf = requestAnimationFrame(() => this.frame());
  }

  async togglePlay() {
    if (!this.ready) return;
    if (this.engine.playing) this.engine.pause();
    else await this.engine.play();
    this.updateTransport();
  }

  seek(t) {
    if (!this.ready) return;
    const target = clampTime(t, this.engine.duration);
    const loop = this.loopOn ? this.activeLoop() : null;
    if (loop && (target < loop.start || target >= loop.end)) {
      // ループ区間の外へ動かしたらループを切る
      this.loopOn = false;
      this.engine.setLoop(null);
      this.renderCues();
      this.updateTransport();
    }
    this.engine.seek(target);
  }

  updateTransport() {
    const btn = this.root.querySelector("#play-btn");
    if (btn && this.engine) {
      btn.innerHTML = this.engine.playing ? ICONS.pause : ICONS.play;
      btn.setAttribute("aria-label", this.engine.playing ? "一時停止" : "再生");
    }
    const loopBtn = this.root.querySelector("#loop-btn");
    if (loopBtn) {
      loopBtn.classList.toggle("on", this.loopOn);
      loopBtn.setAttribute("aria-pressed", String(this.loopOn));
    }
  }

  // --- 選択 -------------------------------------------------------------------

  applySelection(ramp) {
    if (this.engine) {
      this.engine.setGains(S.targetGains(this.tree, this.sel, this.gainsDb), ramp);
    }
    this.renderSelectionState();
  }

  setSelection(sel, { gainsDb = null, presetId = null } = {}) {
    this.sel = sel;
    if (gainsDb) this.gainsDb = gainsDb;
    this.activePresetId = presetId;
    this.applySelection();
    this.renderPresets();
  }

  pressStem(code, forceSolo = false) {
    if (this.soloMode || forceSolo) this.setSelection(S.solo(this.tree, code));
    else this.setSelection(S.toggle(this.tree, this.sel, code));
  }

  pressGroup(group) {
    const leaves = S.groupLeaves(this.tree, group);
    if (!leaves.length) { toast("この曲にはグループの stem がありません。"); return; }
    if (this.soloMode) this.setSelection(new Set(leaves));
    else this.setSelection(S.toggleLeaves(this.sel, leaves));
  }

  pressAll() {
    this.setSelection(S.allOn(this.tree), { gainsDb: new Map() });
  }

  toggleSolo() {
    this.soloMode = !this.soloMode;
    this.renderSelectionState();
  }

  applyPreset(preset) {
    const { sel, gainsDb } = S.presetToSelection(this.tree, preset, this.groups);
    if (!sel.size) toast("この曲には、この組み合わせの stem がありません。");
    this.setSelection(sel, { gainsDb, presetId: preset.listen_preset_id });
  }

  // --- 組み合わせプリセット ------------------------------------------------------

  typeIdOf(code) {
    const t = this.stemTypes.find((x) => x.code === code);
    return t ? t.stem_type_id : undefined;
  }

  async reloadPresets() {
    const presets = (await api("/api/listen-presets")).listen_presets;
    if (!this.alive) return;
    this.presets = presets;
    this.renderPresets();
  }

  async savePreset() {
    const items = S.selectionToItems(this.tree, this.sel, (c) => this.typeIdOf(c), this.gainsDb);
    if (!items.length) { toast("鳴らす stem を1つ以上選んでください。"); return; }
    const name = await promptDialog("今の組み合わせに名前を付けて保存します。", "", { ok: "保存" });
    if (!name) return;
    try {
      const p = await api("/api/listen-presets", { method: "POST", body: { name, items } });
      this.activePresetId = p.listen_preset_id;
      await this.reloadPresets();
      toast(`「${p.name}」を保存しました。`);
    } catch (e) { toast(e.message); }
  }

  async renamePreset(p) {
    const name = await promptDialog("新しい名前", p.name, { ok: "変更" });
    if (!name || name === p.name) return;
    try {
      await api(`/api/listen-presets/${p.listen_preset_id}`, { method: "PUT", body: { name } });
      await this.reloadPresets();
    } catch (e) { toast(e.message); }
  }

  async deletePreset(p) {
    const ok = await confirmDialog(`組み合わせ「${p.name}」を削除しますか？`, { ok: "削除する", danger: true });
    if (!ok) return;
    try {
      await api(`/api/listen-presets/${p.listen_preset_id}`, { method: "DELETE" });
      if (this.activePresetId === p.listen_preset_id) this.activePresetId = null;
      await this.reloadPresets();
    } catch (e) { toast(e.message); }
  }

  async movePreset(p, delta) {
    const list = [...this.presets];
    const i = list.indexOf(p);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= list.length) return;
    [list[i], list[j]] = [list[j], list[i]];
    try {
      // 並び順を 10, 20, 30… に振り直し、変わったものだけ送る
      for (let k = 0; k < list.length; k++) {
        const want = (k + 1) * 10;
        if (list[k].sort_order !== want) {
          await api(`/api/listen-presets/${list[k].listen_preset_id}`, {
            method: "PUT", body: { sort_order: want },
          });
        }
      }
      await this.reloadPresets();
    } catch (e) { toast(e.message); }
  }

  // --- キュー・ループ -----------------------------------------------------------

  activeLoop() {
    const cue = this.cues.find((c) => c.cue_id === this.loopCueId);
    return cue && cue.loop_end_sec ? { start: cue.position_sec, end: cue.loop_end_sec } : null;
  }

  applyLoop() {
    if (this.engine) this.engine.setLoop(this.loopOn ? this.activeLoop() : null);
    this.renderCues();
    this.updateTransport();
  }

  toggleLoop() {
    if (!this.ready) return;
    if (!this.activeLoop()) {
      const pos = this.engine.position;
      const loops = this.cues.filter((c) => c.loop_end_sec);
      const inside = loops.find((c) => pos >= c.position_sec && pos < c.loop_end_sec);
      const pick = inside || loops[loops.length - 1];
      if (!pick) { toast("ループ区間がありません。キューに「終点」を付けてください。"); return; }
      this.loopCueId = pick.cue_id;
      this.loopOn = true;
    } else {
      this.loopOn = !this.loopOn;
    }
    this.applyLoop();
  }

  sortCues() {
    this.cues.sort((a, b) => a.position_sec - b.position_sec || a.cue_id - b.cue_id);
  }

  async addCue() {
    if (!this.ready) return;
    const pos = Math.round(this.engine.position * 1000) / 1000;
    const n = this.cues.length + 1;
    try {
      const cue = await api(`/api/tracks/${this.trackId}/cues`, {
        method: "POST",
        body: { position_sec: pos, label: `キュー ${n}`, color: CUE_COLORS[(n - 1) % CUE_COLORS.length] },
      });
      this.cues.push(cue);
      this.sortCues();
      this.renderCues();
    } catch (e) { toast(e.message); }
  }

  async updateCue(cue, body) {
    try {
      const updated = await api(`/api/cues/${cue.cue_id}`, { method: "PUT", body });
      Object.assign(cue, updated);
      this.sortCues();
      return true;
    } catch (e) {
      toast(e.message);
      return false;
    }
  }

  async setLoopEnd(cue) {
    const end = Math.round(this.engine.position * 1000) / 1000;
    if (end <= cue.position_sec + 0.05) {
      toast("終点はキューより後ろの位置で押してください。");
      return;
    }
    if (await this.updateCue(cue, { loop_end_sec: end })) {
      this.loopCueId = cue.cue_id;
      this.loopOn = true;
      this.applyLoop();
    }
  }

  async clearLoopEnd(cue) {
    if (await this.updateCue(cue, { loop_end_sec: null })) {
      if (this.loopCueId === cue.cue_id) { this.loopCueId = null; this.loopOn = false; }
      this.applyLoop();
    }
  }

  async cycleCueColor(cue) {
    const i = CUE_COLORS.indexOf((cue.color || "").toUpperCase());
    if (await this.updateCue(cue, { color: CUE_COLORS[(i + 1) % CUE_COLORS.length] })) this.renderCues();
  }

  async renameCue(cue) {
    const label = await promptDialog("キューの名前", cue.label || "", { ok: "変更" });
    if (label === null) return;
    if (await this.updateCue(cue, { label })) this.renderCues();
  }

  async deleteCue(cue) {
    try {
      await api(`/api/cues/${cue.cue_id}`, { method: "DELETE" });
      this.cues = this.cues.filter((c) => c !== cue);
      if (this.loopCueId === cue.cue_id) { this.loopCueId = null; this.loopOn = false; }
      this.applyLoop();
    } catch (e) { toast(e.message); }
  }

  jumpToCue(cue) {
    if (cue.loop_end_sec) {
      // ループ付きのキューを押したら、その区間をループの対象にする（ON/OFF は今のまま）
      this.loopCueId = cue.cue_id;
      if (this.engine) this.engine.setLoop(this.loopOn ? this.activeLoop() : null);
    }
    this.seek(cue.position_sec);
    this.renderCues();
  }

  // --- キーボード -------------------------------------------------------------

  handleKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = (e.target && e.target.tagName) || "";
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || document.querySelector(".modal-back")) return;
    if (e.code === "Space" || e.key === " ") {
      e.preventDefault();
      if (!e.repeat) this.togglePlay(); // 押しっぱなしの繰り返しは無視する
    } else if (/^[1-9]$/.test(e.key)) {
      const code = this.tree.order[Number(e.key) - 1];
      if (code) { e.preventDefault(); this.pressStem(code, e.shiftKey); }
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      if (this.ready) this.seek(this.engine.position - SEEK_STEP_SEC);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      if (this.ready) this.seek(this.engine.position + SEEK_STEP_SEC);
    } else if (e.key === "l" || e.key === "L") {
      e.preventDefault();
      this.toggleLoop();
    }
  }

  // --- 描画 -------------------------------------------------------------------

  headEl() {
    return el("div", { class: "track-head" },
      el("a", { class: "btn small", href: "#/library", text: "← ライブラリ" }),
      el("h1", { text: this.track.title, title: this.track.title }),
      el("span", { class: "muted", text: this.track.artist || "" }));
  }

  render() {
    const duration = this.track.duration_sec || 0;
    const wave = el("section", { class: "panel wave-wrap" },
      el("canvas", { class: "wave-overview", id: "wave-overview", "aria-label": "曲全体の波形（クリックで移動）" }),
      el("canvas", { class: "wave-zoom", id: "wave-zoom", "aria-label": "拡大した波形（ドラッグで前後に移動）" }),
      el("div", { class: "wave-tools" },
        el("button", { class: "btn small icon", type: "button", text: "−", "aria-label": "縮小", onclick: () => this.zoomBy(1) }),
        el("button", { class: "btn small icon", type: "button", text: "＋", "aria-label": "拡大", onclick: () => this.zoomBy(-1) })),
      el("div", { class: "loading-cover", id: "loading" },
        el("div", { class: "label", text: "音声と波形を読み込み中…" }),
        el("div", { class: "progress" }, el("span", { style: { width: "0%" } }))));

    const volume = el("input", {
      type: "range", min: "0", max: "100", value: String(Math.round(loadVolume() * 100)),
      "aria-label": "音量",
      oninput: (e) => {
        const v = Number(e.target.value) / 100;
        if (this.engine) this.engine.setVolume(v);
        try { localStorage.setItem(VOLUME_KEY, String(v)); } catch { /* 保存できなくても動く */ }
      },
    });
    const transport = el("section", { class: "panel transport" },
      el("button", {
        class: "play-btn", id: "play-btn", type: "button", disabled: true, "aria-label": "再生",
        onclick: () => this.togglePlay(),
      }),
      el("div", { class: "time" },
        el("span", { id: "time-now", text: formatTime(0, true) }),
        el("span", { class: "muted", text: ` / ${formatTime(duration, true)}` })),
      el("button", { class: "btn", type: "button", text: `−${SEEK_STEP_SEC}秒`, onclick: () => this.ready && this.seek(this.engine.position - SEEK_STEP_SEC) }),
      el("button", { class: "btn", type: "button", text: `+${SEEK_STEP_SEC}秒`, onclick: () => this.ready && this.seek(this.engine.position + SEEK_STEP_SEC) }),
      el("button", { class: "btn", id: "loop-btn", type: "button", text: "ループ", "aria-pressed": "false", onclick: () => this.toggleLoop() }),
      el("label", { class: "volume" }, el("span", { class: "muted", text: "音量" }), volume));

    const stems = el("section", { class: "panel" },
      el("h2", { text: "STEM" }),
      el("div", { class: "stems", id: "stems" }),
      el("div", { class: "mode-row", style: { marginTop: "10px" } },
        el("button", { class: "btn", id: "all-btn", type: "button", text: "全部（元の曲）", onclick: () => this.pressAll() }),
        el("button", { class: "btn", id: "solo-btn", type: "button", text: "ソロ", "aria-pressed": "false", title: "ON にすると、押した stem だけを鳴らします（Shift＋クリックでも同じ）", onclick: () => this.toggleSolo() }),
        el("span", { class: "muted", text: "グループ:" }),
        el("div", { class: "mode-row", id: "groups" })));

    const presets = el("section", { class: "panel" },
      el("h2", { text: "組み合わせ" }),
      el("ul", { class: "list", id: "presets" }),
      el("div", { class: "row", style: { marginTop: "8px" } },
        el("button", { class: "btn", id: "save-preset-btn", type: "button", text: "今の組み合わせを保存", onclick: () => this.savePreset() })));

    const cues = el("section", { class: "panel" },
      el("h2", { text: "キュー・ループ" }),
      el("ul", { class: "list", id: "cues" }),
      el("div", { class: "row", style: { marginTop: "8px" } },
        el("button", { class: "btn", id: "add-cue-btn", type: "button", text: "＋ 今の位置にキュー", onclick: () => this.addCue() })));

    const help = el("p", { class: "keys-help" },
      el("kbd", { text: "Space" }), " 再生/停止　", el("kbd", { text: "1" }), "〜", el("kbd", { text: "9" }),
      " stem の ON/OFF（Shift でソロ）　", el("kbd", { text: "←" }), el("kbd", { text: "→" }),
      ` ${SEEK_STEP_SEC}秒戻る/進む　`, el("kbd", { text: "L" }), " ループ");

    this.root.replaceChildren(el("div", { class: "player" },
      el("div", { class: "player-main" }, this.headEl(), wave, transport, stems),
      el("div", { class: "player-side" }, presets, cues, help)));
    this.updateTransport();
    this.renderStems();
    this.renderGroups();
    this.renderPresets();
    this.renderCues();
  }

  zoomBy(dir) {
    if (!this.wave) return;
    const i = ZOOM_STEPS.indexOf(this.wave.zoomSeconds);
    const j = Math.min(Math.max(0, (i < 0 ? 2 : i) + dir), ZOOM_STEPS.length - 1);
    this.wave.setZoom(ZOOM_STEPS[j]);
  }

  renderStems() {
    const box = this.root.querySelector("#stems");
    box.replaceChildren(...this.tree.order.map((code, i) => {
      const s = this.tree.byCode.get(code);
      const parent = S.isParent(this.tree, code);
      const sub = parent ? "子をまとめて" : "";
      return el("button", {
        class: `stem-btn${s.parent_code ? " child" : ""}${s.is_silent ? " silent" : ""}`,
        type: "button", dataset: { code },
        title: s.is_silent ? `${s.display_name}（ほぼ無音）` : s.display_name,
        onclick: (e) => this.pressStem(code, e.shiftKey),
      },
      el("span", { class: "name", text: s.display_name }),
      el("span", { class: "sub", text: sub }),
      i < 9 ? el("span", { class: "key", text: String(i + 1) }) : null);
    }));
    // style に CSS 変数を入れる（Object.assign では入らないため）
    for (const b of box.querySelectorAll(".stem-btn")) {
      b.style.setProperty("--c", this.tree.byCode.get(b.dataset.code).color);
    }
    this.renderSelectionState();
  }

  renderGroups() {
    const box = this.root.querySelector("#groups");
    box.replaceChildren(...this.groups.map((g) => {
      const chip = el("button", {
        class: "chip", type: "button", dataset: { group: g.code }, title: g.members.join(", "),
        onclick: () => this.pressGroup(g),
      }, el("span", { class: "dot" }), g.display_name);
      chip.style.setProperty("--c", g.color);
      return chip;
    }));
    this.renderSelectionState();
  }

  renderSelectionState() {
    for (const b of this.root.querySelectorAll(".stem-btn")) {
      const state = S.stateOf(this.tree, this.sel, b.dataset.code);
      b.classList.toggle("on", state === "on");
      b.classList.toggle("partial", state === "partial");
      b.setAttribute("aria-pressed", state === "on" ? "true" : state === "partial" ? "mixed" : "false");
    }
    for (const c of this.root.querySelectorAll(".chip[data-group]")) {
      const g = this.groups.find((x) => x.code === c.dataset.group);
      const state = S.stateOfLeaves(this.sel, S.groupLeaves(this.tree, g));
      c.classList.toggle("on", state === "on");
      c.classList.toggle("partial", state === "partial");
      c.setAttribute("aria-pressed", state === "on" ? "true" : state === "partial" ? "mixed" : "false");
    }
    const all = this.root.querySelector("#all-btn");
    if (all) all.classList.toggle("on", S.sameSelection(this.sel, S.allOn(this.tree)));
    const soloBtn = this.root.querySelector("#solo-btn");
    if (soloBtn) {
      soloBtn.classList.toggle("on", this.soloMode);
      soloBtn.setAttribute("aria-pressed", String(this.soloMode));
    }
  }

  renderPresets() {
    const box = this.root.querySelector("#presets");
    if (!box) return;
    if (!this.presets.length) {
      box.replaceChildren(el("li", {}, el("span", { class: "muted", text: "保存した組み合わせはありません。" })));
      return;
    }
    box.replaceChildren(...this.presets.map((p, i) => el("li", {
      class: p.listen_preset_id === this.activePresetId ? "active" : "",
      dataset: { presetId: String(p.listen_preset_id) },
    },
    el("span", {
      class: "name", text: p.name, title: `${p.name}（クリックで切り替え）`, role: "button", tabindex: "0",
      onclick: () => this.applyPreset(p),
      onkeydown: (e) => { if (e.key === "Enter") this.applyPreset(p); },
    }),
    el("button", { class: "btn small icon", type: "button", title: "上へ", "aria-label": `${p.name} を上へ`, disabled: i === 0, onclick: () => this.movePreset(p, -1) }, icon("up")),
    el("button", { class: "btn small icon", type: "button", title: "下へ", "aria-label": `${p.name} を下へ`, disabled: i === this.presets.length - 1, onclick: () => this.movePreset(p, 1) }, icon("down")),
    el("button", { class: "btn small icon", type: "button", title: "名前を変える", "aria-label": `${p.name} の名前を変える`, onclick: () => this.renamePreset(p) }, icon("edit")),
    el("button", { class: "btn small icon danger", type: "button", title: "削除", "aria-label": `${p.name} を削除`, onclick: () => this.deletePreset(p) }, icon("close")))));
  }

  renderCues() {
    const box = this.root.querySelector("#cues");
    if (!box) return;
    if (!this.cues.length) {
      box.replaceChildren(el("li", {}, el("span", { class: "muted", text: "キューはありません。" })));
      return;
    }
    box.replaceChildren(...this.cues.map((c) => {
      const isLoop = !!c.loop_end_sec;
      const active = isLoop && this.loopCueId === c.cue_id && this.loopOn;
      const swatch = el("button", {
        class: "cue-swatch", type: "button", "aria-label": "色を変える", onclick: () => this.cycleCueColor(c),
      });
      swatch.style.background = c.color || "#F5F5F4";
      return el("li", { class: active ? "active" : "", dataset: { cueId: String(c.cue_id) } },
        swatch,
        el("span", {
          class: "name", role: "button", tabindex: "0", title: "クリックでこの位置へ",
          onclick: () => this.jumpToCue(c),
          onkeydown: (e) => { if (e.key === "Enter") this.jumpToCue(c); },
        }, c.label || "キュー", " ", el("span", { class: "time", text: formatTime(c.position_sec, true) })),
        isLoop ? el("span", { class: "loop-tag", text: `〜${formatTime(c.loop_end_sec, true)}` }) : null,
        isLoop
          ? el("button", { class: "btn small", type: "button", text: "解除", title: "ループの終点を外す", onclick: () => this.clearLoopEnd(c) })
          : el("button", { class: "btn small", type: "button", text: "終点", title: "今の位置をループの終点にする（A-B ループ）", onclick: () => this.setLoopEnd(c) }),
        el("button", { class: "btn small icon", type: "button", title: "名前を変える", "aria-label": "名前を変える", onclick: () => this.renameCue(c) }, icon("edit")),
        el("button", { class: "btn small icon danger", type: "button", title: "削除", "aria-label": "削除", onclick: () => this.deleteCue(c) }, icon("close")));
    }));
  }
}
