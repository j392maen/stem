// 速度の変更（T11）。
//
// - ピッチも変わる方式（pitch）: 全 stem の playbackRate を同じ時刻で r にする（engine.setRate）。すぐ効く。
// - ピッチを保つ方式（keep）: サーバーで各 stem を r 倍に伸縮した音声を作ってもらい（POST
//   /api/jobs/{id}/tempo、進み具合は SSE）、できたら同じ曲の時刻から差し替える（engine.setBuffers）。
//   作っている間は「元の速度のまま（今の音のまま）」か「ピッチを変えて指定の速度で」鳴らす（設定）。
// - 拍・キュー・ループ・波形・再生位置はすべて元の曲の時刻のまま（engine が音声データ上の時刻に直す）。
// - 曲ごとの速度・方式・スライダーの幅はブラウザ（localStorage）に保存し、次に開いたときに戻す。
// - メモリ: 音声は「今鳴らしている組（元の音声 or ある倍率の伸縮済み）」だけを持つ。別の組に替えるときは
//   読み込み終わるまで今の組で鳴らし、差し替えたら前の組を捨てる（iPhone でメモリを倍に使わないため）。

import { api, fetchBinary } from "./api.js";
import { formatBpm } from "./beats.js";
import { el, toast } from "./ui.js";

export const MIN_RATIO = 0.5;
export const MAX_RATIO = 2.0;
export const FINE_STEP = 0.001; // ± ボタン（0.1%）
export const COARSE_STEP = 0.01; // Shift＋キー（1%）
export const RANGES = [8, 16, 50]; // スライダーの幅（±%）
export const MODES = ["pitch", "keep"];
export const PENDING = ["original", "pitch"]; // 作成中の鳴らし方
const KEY_PREFIX = "stemapp.tempo.";
const PENDING_KEY = "stemapp.tempo.pending";
const REQUEST_DELAY_MS = 500; // ピッチを保つ方式でスライダーを動かしている間は作成を頼まない
const LOAD_CONCURRENCY = 3;
const POLL_MS = 1500;
const ORIGINAL = "orig";

/** 倍率を小数3桁に丸めて MIN〜MAX に収める。 */
export function clampRatio(r) {
  const v = Number(r);
  if (!Number.isFinite(v)) return 1;
  return Math.round(Math.min(MAX_RATIO, Math.max(MIN_RATIO, v)) * 1000) / 1000;
}

/** 目標 BPM と区間の BPM から倍率（収まらなければ端に寄せる）。区間の BPM が無ければ null。 */
export function ratioFromBpm(target, base) {
  const t = Number(target);
  const b = Number(base);
  if (!(t > 0) || !(b > 0)) return null;
  return clampRatio(t / b);
}

/** 倍率を「+3.5%」「−8.0%」「±0.0%」で表す。 */
export function formatPercent(r) {
  const p = Math.round((r - 1) * 1000) / 10;
  if (p === 0) return "±0.0%";
  return `${p > 0 ? "+" : "−"}${Math.abs(p).toFixed(1)}%`;
}

export function ratioKey(r) {
  return clampRatio(r).toFixed(3);
}

/** スライダーの値（0.1% 単位の整数）→ 倍率。 */
export function sliderToRatio(value) {
  return clampRatio(1 + Number(value) / 1000);
}

/** 倍率 → スライダーの値（幅 ±range% に収める）。 */
export function ratioToSlider(r, range) {
  const lim = range * 10;
  return Math.max(-lim, Math.min(lim, Math.round((r - 1) * 1000)));
}

export function loadTempoState(trackId) {
  const def = { ratio: 1, mode: "pitch", range: 8 };
  try {
    const raw = JSON.parse(localStorage.getItem(KEY_PREFIX + trackId) || "null");
    if (!raw || typeof raw !== "object") return def;
    return {
      ratio: clampRatio(raw.ratio ?? 1),
      mode: MODES.includes(raw.mode) ? raw.mode : "pitch",
      range: RANGES.includes(raw.range) ? raw.range : 8,
    };
  } catch { return def; }
}

function saveTempoState(trackId, state) {
  try {
    localStorage.setItem(KEY_PREFIX + trackId, JSON.stringify(state));
  } catch { /* 保存できなくても動く */ }
}

function loadPending() {
  try {
    const v = localStorage.getItem(PENDING_KEY);
    return PENDING.includes(v) ? v : "original";
  } catch { return "original"; }
}

/** 項目を最大 limit 個ずつ並行して処理する。 */
async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  }));
  return out;
}

export class TempoPanel {
  /** view: PlayerView（trackId, job, engine, beatGrid, alive, later()）。 */
  constructor(view) {
    this.view = view;
    const saved = loadTempoState(view.trackId);
    this.ratio = saved.ratio;
    this.mode = saved.mode;
    this.range = saved.range;
    this.pending = loadPending();
    this.activeKey = ORIGINAL; // 今鳴らしている音声の組（元の音声 or 倍率）
    this.loadingKey = null; // 読み込み中の組
    this.failedKey = null; // 読み込みに失敗した組（速度・方式を変えるまで読み直さない）
    this.render = null; // 今の目標の倍率の作成（サーバーの TEMPO_RENDER）
    this.source = null; // EventSource
    this.requestTimer = 0;
    this.urls = null; // 元の音声の URL { code: url }
    this.disposed = false;
    this.root = this.build();
    this.refresh();
  }

  get engine() {
    return this.view.engine;
  }

  // --- 画面 --------------------------------------------------------------------

  build() {
    const seg = (name, items, onpick) => el("div", { class: "seg", role: "group", "aria-label": name },
      items.map(([value, text, title]) => el("button", {
        class: "seg-btn", type: "button", text, title, dataset: { value: String(value) },
        "aria-pressed": "false", onclick: () => onpick(value),
      })));
    this.modeSeg = seg("速度の方式", [
      ["pitch", "ピッチも変わる", "再生の速さをそのまま変えます（すぐ効く。音の高さも変わる）"],
      ["keep", "ピッチを保つ", "サーバーで音の高さを変えずに伸縮した音声を作ります（数秒〜十数秒かかる）"],
    ], (v) => this.setMode(v));
    this.modeSeg.id = "tp-mode";
    this.rangeSeg = seg("スライダーの幅", RANGES.map((n) => [n, `±${n}`, `スライダーの幅を ±${n}% にします`]),
      (v) => this.setRange(v));
    this.rangeSeg.id = "tp-range";
    this.slider = el("input", {
      type: "range", class: "tp-slider", id: "tp-slider", step: "1", value: "0",
      "aria-label": "速度",
      oninput: (e) => this.setRatio(sliderToRatio(e.target.value), { fromSlider: true }),
      ondblclick: () => this.setRatio(1),
    });
    this.readout = el("span", { class: "tp-readout", id: "tp-readout", text: "±0.0%" });
    this.ratioText = el("span", { class: "tp-ratio", id: "tp-ratio", text: "×1.000" });
    this.bpmInput = el("input", {
      class: "input small tp-bpm", id: "tp-bpm", type: "number", min: "20", max: "400", step: "0.1",
      "aria-label": "目標 BPM", placeholder: "BPM",
      title: "再生位置の区間の BPM を基準に、この BPM になる速度にします（Enter で決定）",
      onchange: (e) => this.setTargetBpm(e.target.value),
      onkeydown: (e) => { if (e.key === "Enter") { e.preventDefault(); e.target.blur(); } },
    });
    this.resetBtn = el("button", {
      class: "btn small", type: "button", id: "tp-reset", text: "元の速度に戻す",
      title: "速度を元に戻します（R キー）", onclick: () => this.setRatio(1),
    });
    const step = (sign) => (e) => this.nudge(sign * (e.shiftKey ? COARSE_STEP : FINE_STEP));
    this.status = el("span", { class: "tp-status", id: "tp-status", role: "status" });
    this.bar = el("div", { class: "progress tp-progress", id: "tp-progress", hidden: true },
      el("span", { style: { width: "0%" } }));
    this.retryBtn = el("button", {
      class: "btn small", type: "button", id: "tp-retry", text: "作り直す", hidden: true,
      title: "この速度の音声をもう一度作ります", onclick: () => this.retryRender(),
    });
    this.cancelBtn = el("button", {
      class: "btn small", type: "button", id: "tp-cancel", text: "やめる", hidden: true,
      title: "伸縮した音声の作成をやめます", onclick: () => this.cancelRender(),
    });
    this.pendingSel = el("select", {
      class: "select small", id: "tp-pending", "aria-label": "作成中の鳴らし方",
      title: "ピッチを保つ音声を作っている間の鳴らし方",
      onchange: (e) => this.setPending(e.target.value),
    },
    el("option", { value: "original", text: "作成中は今の音のまま" }),
    el("option", { value: "pitch", text: "作成中はピッチを変えて先に速度を変える" }));
    this.pendingSel.value = this.pending;
    this.keepRow = el("div", { class: "tp-row tp-keep", id: "tp-keep" },
      this.pendingSel, this.status, this.bar, this.cancelBtn, this.retryBtn);

    return el("section", { class: "panel tempo-panel", id: "tempo-panel" },
      el("div", { class: "tp-row" },
        el("span", { class: "tp-title", text: "SPEED" }),
        this.modeSeg,
        el("span", { class: "tp-gap" }),
        this.rangeSeg),
      el("div", { class: "tp-row tp-fader" },
        el("button", {
          class: "btn small icon", type: "button", id: "tp-minus", text: "−",
          "aria-label": "遅く（0.1%）", title: "遅く 0.1%（Shift で 1%。, キー）", onclick: step(-1),
        }),
        this.slider,
        el("button", {
          class: "btn small icon", type: "button", id: "tp-plus", text: "＋",
          "aria-label": "速く（0.1%）", title: "速く 0.1%（Shift で 1%。. キー）", onclick: step(1),
        }),
        el("span", { class: "tp-values" }, this.readout, this.ratioText)),
      el("div", { class: "tp-row" },
        el("label", { class: "tp-target" }, el("span", { class: "muted", text: "目標" }), this.bpmInput,
          el("span", { class: "muted", text: "BPM" })),
        this.resetBtn),
      this.keepRow);
  }

  /** 表示を今の状態に合わせる。 */
  refresh() {
    const lim = this.range * 10;
    this.slider.min = String(-lim);
    this.slider.max = String(lim);
    this.slider.value = String(ratioToSlider(this.ratio, this.range));
    this.readout.textContent = formatPercent(this.ratio);
    this.readout.classList.toggle("changed", this.ratio !== 1);
    this.ratioText.textContent = `×${this.ratio.toFixed(3)}`;
    for (const b of this.modeSeg.querySelectorAll(".seg-btn")) {
      const on = b.dataset.value === this.mode;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    }
    for (const b of this.rangeSeg.querySelectorAll(".seg-btn")) {
      const on = Number(b.dataset.value) === this.range;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    }
    this.resetBtn.disabled = this.ratio === 1;
    this.bpmInput.disabled = !this.view.beatGrid;
    this.refreshStatus();
  }

  refreshStatus() {
    const keep = this.mode === "keep" && this.ratio !== 1;
    this.keepRow.hidden = this.mode !== "keep";
    const r = this.render;
    const want = ratioKey(this.ratio);
    const mine = r && r.ratio_key === want;
    let text = "";
    let busy = false;
    let progress = 0;
    if (!keep) {
      text = this.mode === "keep" ? "元の速度です" : "";
    } else if (this.loadingKey === want) {
      text = "読み込み中…";
      busy = true;
      progress = 1;
    } else if (this.activeKey === want) {
      text = `ピッチを保って再生中（×${want}）`;
    } else if (mine && (r.status === "queued" || r.status === "running")) {
      progress = r.progress || 0;
      text = r.status === "queued" ? "作成待ち…" : `作成中 ${Math.round(progress * 100)}%`;
      busy = true;
    } else if (mine && r.status === "failed") {
      text = r.error_message || "作れませんでした";
    } else if (mine && r.status === "canceled") {
      text = "作成をやめました";
    } else {
      text = "準備中…";
      busy = true;
    }
    this.status.textContent = text;
    this.status.classList.toggle("error-text", !!(mine && r.status === "failed"));
    this.bar.hidden = !busy;
    this.bar.querySelector("span").style.width = `${Math.round(progress * 100)}%`;
    this.cancelBtn.hidden = !(keep && mine && (r.status === "queued" || r.status === "running"));
    this.retryBtn.hidden = !(keep && mine && (r.status === "failed" || r.status === "canceled"));
  }

  /** プレイヤーの BPM 表示の補助（目標 BPM の欄の目安）。 */
  syncBpm(bpm) {
    if (document.activeElement === this.bpmInput) return;
    const v = bpm ? formatBpm(bpm) : "";
    if (this.bpmInput.placeholder !== (v || "BPM")) this.bpmInput.placeholder = v || "BPM";
  }

  // --- 操作 --------------------------------------------------------------------

  save() {
    saveTempoState(this.view.trackId, { ratio: this.ratio, mode: this.mode, range: this.range });
  }

  setRatio(r, { fromSlider = false } = {}) {
    const v = clampRatio(r);
    if (v === this.ratio && !fromSlider) { this.refresh(); return; }
    this.ratio = v;
    this.failedKey = null;
    this.save();
    this.refresh();
    this.apply();
  }

  nudge(delta) {
    this.setRatio(Math.round((this.ratio + delta) * 1000) / 1000);
  }

  setMode(mode) {
    if (!MODES.includes(mode) || mode === this.mode) return;
    this.mode = mode;
    this.failedKey = null;
    this.save();
    this.refresh();
    this.apply();
  }

  setRange(range) {
    if (!RANGES.includes(range)) return;
    this.range = range;
    this.save();
    this.refresh();
  }

  setPending(value) {
    if (!PENDING.includes(value)) return;
    this.pending = value;
    try { localStorage.setItem(PENDING_KEY, value); } catch { /* 保存できなくても動く */ }
    this.apply();
  }

  /** 目標 BPM（再生位置の区間の BPM が基準）。 */
  setTargetBpm(value) {
    const grid = this.view.beatGrid;
    const pos = this.engine ? this.engine.position : 0;
    const base = grid ? grid.bpmAt(pos) : null;
    const r = ratioFromBpm(value, base);
    this.bpmInput.value = "";
    if (r === null) { toast("この位置の BPM が分からないため、目標 BPM は使えません。"); return; }
    const exact = Number(value) / base;
    if (exact < MIN_RATIO || exact > MAX_RATIO) {
      toast(`速度は ${MIN_RATIO.toFixed(2)}〜${MAX_RATIO.toFixed(2)} 倍までです。`);
    }
    this.setRatio(r);
  }

  // --- 音声の組と再生 -------------------------------------------------------------

  /** 読み込みが終わった元の音声を受け取り、保存していた速度を反映する。 */
  start(urls) {
    this.urls = urls;
    this.activeKey = ORIGINAL;
    this.refresh();
    this.apply();
  }

  /** 今の設定で鳴らすべき組 { key, scale, rate, urls }。まだ無い（作成中）なら null。 */
  desired() {
    const r = this.ratio;
    if (r === 1 || this.mode === "pitch") return { key: ORIGINAL, scale: 1, rate: r, urls: this.urls };
    const want = ratioKey(r);
    const rd = this.render;
    if (rd && rd.ratio_key === want && rd.status === "done") {
      return { key: want, scale: Number(want), rate: 1, urls: rd.files };
    }
    return null;
  }

  /** 状態の変化（速度・方式・作成の完了）を再生に反映する。 */
  apply() {
    const engine = this.engine;
    if (!engine || !this.urls || this.disposed) return;
    const d = this.desired();
    if (this.mode === "keep" && this.ratio !== 1) this.scheduleRequest();
    else this.dropRender();
    if (d && d.key === this.activeKey) {
      engine.setRate(d.rate);
      this.refreshStatus();
      return;
    }
    // 目標の組がまだ無い・読み込み中: 今の組で鳴らしておく
    const speedNow = this.mode === "pitch" || this.ratio === 1 || this.pending === "pitch";
    if (speedNow) engine.setRate(this.ratio / engine.bufScale);
    if (d && d.key !== this.failedKey) this.loadSet(d);
    this.refreshStatus();
  }

  async loadSet(d) {
    if (this.loadingKey === d.key) return;
    const codes = [...this.engine.tracks.keys()].filter((c) => this.engine.tracks.get(c).buffer);
    const missing = codes.filter((c) => !d.urls || !d.urls[c]);
    if (missing.length) {
      this.failedKey = d.key;
      toast(`速度を変えた音声がそろっていません（${missing.join(", ")}）。`);
      return;
    }
    this.loadingKey = d.key;
    this.refreshStatus();
    const engine = this.engine;
    try {
      const signal = this.view.abort.signal;
      const list = await mapLimit(codes, LOAD_CONCURRENCY, async (code) => {
        const buf = await fetchBinary(d.urls[code], signal);
        return [code, await engine.decode(buf)];
      });
      if (this.disposed || engine !== this.engine) return;
      const now = this.desired();
      if (!now || now.key !== d.key) return; // 読み込む間に設定が変わった
      engine.setBuffers(Object.fromEntries(list), d.scale, now.rate);
      this.activeKey = d.key;
    } catch (e) {
      if (e.status === 404 && this.render && this.render.ratio_key === d.key) {
        // キャッシュから消えていた: 作り直しを頼む
        this.render = null;
      } else if (e.name !== "AbortError" && !this.disposed) {
        this.failedKey = d.key; // 速度・方式を変えるまで読み直さない
        toast(`音声を読み込めませんでした: ${e.message}`);
      }
    } finally {
      if (this.loadingKey === d.key) this.loadingKey = null;
      if (!this.disposed) {
        this.refreshStatus();
        // 読み込む間に設定が変わっていたら、今の設定で合わせ直す
        const now = this.desired();
        if (!now || now.key !== this.activeKey) this.apply();
        else this.engine.setRate(now.rate);
      }
    }
  }

  // --- サーバーでの作成（ピッチを保つ方式） -----------------------------------------------

  /** 少し待ってから作成を頼む（スライダーを動かしている間は頼まない）。 */
  scheduleRequest() {
    const want = ratioKey(this.ratio);
    const r = this.render;
    if (r && r.ratio_key === want) {
      if (r.status === "done") return;
      if (r.status === "queued" || r.status === "running") {
        if (r.render_id) this.watch(r.render_id);
        return;
      }
      // 失敗・やめたものは自動では頼み直さない（「作り直す」ボタンか、速度を変えたとき）
      if (r.status === "failed" || r.status === "canceled") return;
    }
    clearTimeout(this.requestTimer);
    this.requestTimer = setTimeout(() => this.requestRender(want), REQUEST_DELAY_MS);
  }

  async requestRender(want) {
    if (this.disposed || ratioKey(this.ratio) !== want || this.mode !== "keep") return;
    const old = this.render;
    if (old && old.ratio_key !== want && (old.status === "queued" || old.status === "running")) {
      // 別の倍率に変えた: 前の作成はやめる（キャッシュの枠を無駄に使わない）
      api(`/api/tempo/${old.render_id}/cancel`, { method: "POST" }).catch(() => {});
    }
    this.closeSource();
    try {
      const res = await api(`/api/jobs/${this.view.job.job_id}/tempo`, {
        method: "POST", body: { ratio: Number(want) },
      });
      if (this.disposed || ratioKey(this.ratio) !== want) return;
      this.setRender(res.render);
    } catch (e) {
      if (!this.disposed) {
        toast(e.message);
        this.render = { ratio_key: want, status: "failed", error_message: e.message };
        this.refreshStatus();
      }
    }
  }

  setRender(render) {
    this.render = render;
    if (render.status === "queued" || render.status === "running") this.watch(render.render_id);
    else this.closeSource();
    if (render.status === "failed") toast(render.error_message || "速度を変えた音声を作れませんでした。");
    this.refreshStatus();
    if (render.status === "done") this.apply();
  }

  /** 作成の進み具合を SSE で受け取る（使えなければ一定間隔で問い合わせる）。 */
  watch(renderId) {
    if (this.source && this.sourceId === renderId) return;
    this.closeSource();
    if (typeof EventSource === "undefined") { this.poll(renderId); return; }
    const es = new EventSource(`/api/tempo/${renderId}/events`);
    this.source = es;
    this.sourceId = renderId;
    es.addEventListener("tempo", (ev) => {
      if (this.disposed || !this.render || this.render.render_id !== renderId) { es.close(); return; }
      const data = JSON.parse(ev.data);
      if (data.status === "queued" || data.status === "running") {
        this.render = data;
        this.refreshStatus();
      } else {
        this.closeSource();
        this.setRender(data);
      }
    });
    es.addEventListener("error", () => {
      // 接続が切れた・作成が消えた: 問い合わせに切り替える
      if (this.source !== es) return;
      this.closeSource();
      this.view.later(() => this.poll(renderId), POLL_MS);
    });
  }

  async poll(renderId) {
    if (this.disposed || !this.render || this.render.render_id !== renderId) return;
    try {
      const data = await api(`/api/tempo/${renderId}`);
      if (this.disposed || !this.render || this.render.render_id !== renderId) return;
      if (data.status === "queued" || data.status === "running") {
        this.render = data;
        this.refreshStatus();
        this.view.later(() => this.poll(renderId), POLL_MS);
      } else {
        this.setRender(data);
      }
    } catch (e) {
      if (e.status === 404) { this.render = null; this.refreshStatus(); return; }
      this.view.later(() => this.poll(renderId), POLL_MS * 2);
    }
  }

  async cancelRender() {
    const r = this.render;
    if (!r || !r.render_id) return;
    try {
      const data = await api(`/api/tempo/${r.render_id}/cancel`, { method: "POST" });
      // 作成中のものは、ワーカーが止めるまで（canceled になるまで）進み具合を見続ける
      if (this.render && this.render.render_id === r.render_id) this.setRender(data);
    } catch (e) { toast(e.message); }
    this.refreshStatus();
  }

  retryRender() {
    this.render = null;
    clearTimeout(this.requestTimer);
    this.requestRender(ratioKey(this.ratio));
  }

  /** 目標がピッチを保つ方式でなくなった: 作成中のものはそのまま（キャッシュに残る）見るのをやめる。 */
  dropRender() {
    clearTimeout(this.requestTimer);
    this.closeSource();
  }

  closeSource() {
    if (this.source) this.source.close();
    this.source = null;
    this.sourceId = null;
  }

  dispose() {
    this.disposed = true;
    clearTimeout(this.requestTimer);
    this.closeSource();
  }
}
