// 端末の診断ページ（#/diag）: この端末のブラウザで何が使えるかを調べ、結果をサーバーに保存する。
// iPhone 向けの再生の作り（T06b）を決めるための材料集め。プレイヤーの処理には関わらない。

import { api, fetchBinary } from "./api.js";
import { el, toast } from "./ui.js";

const DIAG_VERSION = 1;
const DEVICE_KEY = "stemapp.diagDevice";
const MAX_EVENTS = 200;

const CAN_PLAY_TYPES = [
  'audio/webm; codecs="opus"',
  'audio/ogg; codecs="opus"',
  'audio/mp4; codecs="mp4a.40.2"',
  "audio/mp4",
  "audio/x-m4a",
  "audio/aac",
  "audio/mpeg",
  "audio/flac",
  "audio/wav",
];

/** UA から「iPhone-Safari」のような端末の目安を作る（保存ファイル名に使う）。 */
export function guessDevice(ua = navigator.userAgent) {
  let os = "other";
  if (/iPhone/.test(ua)) os = "iPhone";
  else if (/iPad/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)) os = "iPad";
  else if (/Android/.test(ua)) os = "Android";
  else if (/Windows/.test(ua)) os = "Windows";
  else if (/Mac OS X|Macintosh/.test(ua)) os = "Mac";
  let browser = "browser";
  if (/Edg\//.test(ua)) browser = "Edge";
  else if (/CriOS|Chrome\//.test(ua)) browser = "Chrome";
  else if (/FxiOS|Firefox\//.test(ua)) browser = "Firefox";
  else if (/Safari\//.test(ua)) browser = "Safari";
  return `${os}-${browser}`;
}

function isStandalone() {
  try {
    if (navigator.standalone === true) return true;
    return window.matchMedia("(display-mode: standalone)").matches;
  } catch {
    return false;
  }
}

function round(v, digits = 4) {
  return typeof v === "number" && Number.isFinite(v) ? Number(v.toFixed(digits)) : v ?? null;
}

function errText(e) {
  if (!e) return "不明なエラー";
  return `${e.name || "Error"}: ${e.message || String(e)}`;
}

/** 画面・UA など（すぐ取れるもの）。 */
export function collectBasics() {
  return {
    ua: navigator.userAgent,
    platform: navigator.platform || null,
    language: navigator.language || null,
    screen: {
      width: screen.width, height: screen.height,
      inner_width: window.innerWidth, inner_height: window.innerHeight,
      device_pixel_ratio: window.devicePixelRatio,
      orientation: (screen.orientation && screen.orientation.type) || null,
    },
    standalone: isStandalone(),
    secure_context: window.isSecureContext,
    touch_points: navigator.maxTouchPoints ?? null,
  };
}

/** <audio> の canPlayType の結果（""・"maybe"・"probably"）。 */
export function collectCanPlay() {
  const audio = document.createElement("audio");
  const out = {};
  for (const type of CAN_PLAY_TYPES) out[type] = audio.canPlayType(type);
  return out;
}

export function collectMemory() {
  const mem = performance.memory;
  return {
    device_memory_gb: navigator.deviceMemory ?? null,
    hardware_concurrency: navigator.hardwareConcurrency ?? null,
    js_heap_limit_mb: mem ? Math.round(mem.jsHeapSizeLimit / 1048576) : null,
  };
}

function testIndexedDB() {
  return new Promise((resolve) => {
    if (!("indexedDB" in window)) { resolve({ available: false }); return; }
    let req;
    try {
      req = indexedDB.open("stemapp-diag-probe", 1);
    } catch (e) {
      resolve({ available: true, open_ok: false, error: errText(e) });
      return;
    }
    const timer = setTimeout(() => resolve({ available: true, open_ok: false, error: "応答なし" }), 3000);
    req.onsuccess = () => {
      clearTimeout(timer);
      req.result.close();
      try { indexedDB.deleteDatabase("stemapp-diag-probe"); } catch { /* 片付けだけ */ }
      resolve({ available: true, open_ok: true });
    };
    req.onerror = () => {
      clearTimeout(timer);
      resolve({ available: true, open_ok: false, error: errText(req.error) });
    };
  });
}

/** Media Session・Push・通知・Service Worker・保存領域など。 */
export async function collectFeatures() {
  const f = {
    media_session: "mediaSession" in navigator,
    push_manager: "PushManager" in window,
    notification: typeof Notification !== "undefined",
    notification_permission: typeof Notification !== "undefined" ? Notification.permission : null,
    service_worker: "serviceWorker" in navigator,
    service_worker_controlled: !!(navigator.serviceWorker && navigator.serviceWorker.controller),
    cache_storage: "caches" in window,
    media_source: "MediaSource" in window,
    managed_media_source: "ManagedMediaSource" in window,
    audio_worklet: typeof AudioWorkletNode !== "undefined",
    storage_estimate: null,
    storage_persisted: null,
  };
  f.indexed_db = await testIndexedDB();
  if (navigator.storage && navigator.storage.estimate) {
    try {
      const est = await navigator.storage.estimate();
      f.storage_estimate = {
        usage_mb: round((est.usage ?? 0) / 1048576, 1),
        quota_mb: round((est.quota ?? 0) / 1048576, 1),
      };
    } catch (e) {
      f.storage_estimate = { error: errText(e) };
    }
  }
  if (navigator.storage && navigator.storage.persisted) {
    try { f.storage_persisted = await navigator.storage.persisted(); } catch { /* 取れない */ }
  }
  return f;
}

function newAudioContext() {
  const Ctor = window.AudioContext || window.webkitAudioContext;
  return Ctor ? new Ctor() : null;
}

function describeContext(ctx) {
  return {
    sample_rate: ctx.sampleRate,
    base_latency: round(ctx.baseLatency),
    output_latency: round(ctx.outputLatency),
    state: ctx.state,
  };
}

function decode(ctx, buf) {
  // 古い Safari はコールバック形式だけのことがあるので両方に対応する
  return new Promise((resolve, reject) => {
    const p = ctx.decodeAudioData(buf, resolve, reject);
    if (p && typeof p.then === "function") p.then(resolve, reject);
  });
}

/** 形式ごとに、テスト音声を取ってきて decodeAudioData できるか調べる。 */
export async function testDecode(ctx, samples) {
  const out = {};
  for (const s of samples) {
    const r = { label: s.label, ok: false };
    const t0 = performance.now();
    try {
      const buf = await fetchBinary(s.url);
      r.bytes = buf.byteLength;
      const audio = await decode(ctx, buf);
      r.ok = true;
      r.duration = round(audio.duration);
      r.sample_rate = audio.sampleRate;
      r.channels = audio.numberOfChannels;
    } catch (e) {
      r.error = errText(e);
    }
    r.ms = Math.round(performance.now() - t0);
    out[s.code] = r;
  }
  return out;
}

// --- 画面 ---------------------------------------------------------------------------

function yesNo(v) {
  if (v === true) return el("span", { class: "diag-ok", text: "あり" });
  if (v === false) return el("span", { class: "diag-ng", text: "なし" });
  return el("span", { class: "muted", text: v === null || v === undefined ? "—" : String(v) });
}

function table(rows) {
  return el("table", { class: "diag-table" },
    el("tbody", {}, rows.map(([k, v]) => el("tr", {},
      el("th", { text: k }),
      el("td", {}, v instanceof Node ? v : String(v ?? "—"))))));
}

function section(title, ...body) {
  return el("section", { class: "panel diag-section" }, el("h2", { text: title }), ...body);
}

export class DiagView {
  constructor(root) {
    this.root = root;
    this.alive = true;
    this.result = null;
    this.ctx = null;
    this.lock = null; // 実行中の画面ロックの試験
    this.onVisibility = () => this.logLock("visibility", document.visibilityState);
  }

  async mount() {
    let device = guessDevice();
    try { device = localStorage.getItem(DEVICE_KEY) || device; } catch { /* 保存できなくても動く */ }
    this.deviceInput = el("input", {
      class: "input", id: "diag-device", value: device, maxlength: 32,
      "aria-label": "端末の名前",
    });
    this.runBtn = el("button", {
      class: "btn primary", type: "button", id: "diag-run", text: "診断を始める",
      onclick: () => this.run(),
    });
    this.status = el("p", { class: "muted", id: "diag-status", text: "ボタンを押すと調べて、結果をサーバーに保存します。" });
    this.output = el("div", { class: "diag-output", id: "diag-output" });
    this.lockBox = el("div", { class: "diag-lock", id: "diag-lock" });

    this.root.replaceChildren(el("div", { class: "diag" },
      section("端末の診断",
        el("p", { class: "muted", text: "この端末のブラウザで、音声の形式や機能が使えるかを調べます。曲やデータは変わりません。" }),
        el("div", { class: "row" },
          el("label", { class: "muted", for: "diag-device", text: "端末の名前" }), this.deviceInput, this.runBtn),
        this.status),
      this.output,
      section("画面ロック中の再生（任意）", this.lockBox),
      el("p", { class: "library-foot" }, el("a", { href: "#/library", text: "ライブラリへ戻る" })),
    ));
    this.renderLock();
    document.addEventListener("visibilitychange", this.onVisibility);
  }

  unmount() {
    this.alive = false;
    document.removeEventListener("visibilitychange", this.onVisibility);
    this.stopLock();
    if (this.ctx) this.ctx.close().catch(() => {});
    this.ctx = null;
  }

  deviceName() {
    const name = this.deviceInput.value.trim() || guessDevice();
    try { localStorage.setItem(DEVICE_KEY, name); } catch { /* 保存できなくても動く */ }
    return name;
  }

  async run() {
    this.runBtn.disabled = true;
    this.status.textContent = "調べています…";
    const result = {
      diag_version: DIAG_VERSION,
      device: this.deviceName(),
      collected_at: new Date().toISOString(),
      ...collectBasics(),
      can_play_type: collectCanPlay(),
      memory: collectMemory(),
    };
    try {
      if (!this.ctx) this.ctx = newAudioContext();
      if (this.ctx) {
        try { await this.ctx.resume(); } catch { /* 再生の許可が無くてもデコードは試す */ }
        // outputLatency は動き始めてから値が入ることがある
        await new Promise((r) => setTimeout(r, 300));
        result.audio_context = describeContext(this.ctx);
        const { samples } = await api("/api/diag/samples");
        result.decode = await testDecode(this.ctx, samples);
      } else {
        result.audio_context = null;
        result.decode = {};
      }
      result.features = await collectFeatures();
    } catch (e) {
      result.error = errText(e);
    }
    if (!this.alive) return;
    result.lock_tests = this.result ? this.result.lock_tests : [];
    this.result = result;
    this.renderResult();
    await this.send();
    this.runBtn.disabled = false;
    this.runBtn.textContent = "もう一度調べる";
  }

  async send() {
    if (!this.result) return;
    this.status.textContent = "サーバーに送っています…";
    try {
      const res = await api("/api/diag", { method: "POST", body: this.result });
      if (!this.alive) return;
      this.status.textContent = `サーバーに保存しました（${res.name}）。`;
      this.status.dataset.saved = res.name;
    } catch (e) {
      if (!this.alive) return;
      this.status.textContent = `送れませんでした: ${e.message}`;
      toast(e.message);
    }
  }

  renderResult() {
    const r = this.result;
    const s = r.screen;
    const parts = [];
    parts.push(section("この端末",
      table([
        ["端末の名前", r.device],
        ["UA", r.ua],
        ["画面", `${s.width}×${s.height}（表示 ${s.inner_width}×${s.inner_height}、倍率 ${s.device_pixel_ratio}）`],
        ["ホーム画面から開いた", yesNo(r.standalone)],
        ["HTTPS など安全な接続", yesNo(r.secure_context)],
      ])));
    const ac = r.audio_context;
    parts.push(section("音声の再生（Web Audio）",
      ac ? table([
        ["サンプルレート", `${ac.sample_rate} Hz`],
        ["baseLatency", ac.base_latency ?? "—"],
        ["outputLatency", ac.output_latency ?? "—"],
        ["状態", ac.state],
      ]) : el("p", { class: "diag-ng", text: "AudioContext が使えません。" })));
    const decodeRows = Object.entries(r.decode || {}).map(([code, d]) => [
      d.label || code,
      d.ok
        ? el("span", {}, el("span", { class: "diag-ok", text: "デコードできる" }),
          ` ${d.duration}秒・${d.sample_rate}Hz・${d.channels}ch・${d.ms}ms`)
        : el("span", {}, el("span", { class: "diag-ng", text: "できない" }), ` ${d.error || ""}`),
    ]);
    parts.push(section("decodeAudioData できる形式", table(decodeRows), el("p", { class: "muted small-note", text: "1秒のテスト音声を読み込んで試しています。" })));
    parts.push(section("<audio> の canPlayType",
      table(Object.entries(r.can_play_type).map(([t, v]) => [t, v || "（空＝再生できない）"]))));
    const m = r.memory;
    parts.push(section("メモリの目安",
      table([
        ["deviceMemory", m.device_memory_gb === null ? "—（取れない）" : `${m.device_memory_gb} GB`],
        ["CPU のコア数", m.hardware_concurrency ?? "—"],
        ["JS ヒープの上限", m.js_heap_limit_mb === null ? "—（取れない）" : `${m.js_heap_limit_mb} MB`],
      ])));
    const f = r.features || {};
    const est = f.storage_estimate;
    parts.push(section("機能",
      table([
        ["Media Session（ロック画面の操作）", yesNo(f.media_session)],
        ["Web Push（PushManager）", yesNo(f.push_manager)],
        ["通知（Notification）", f.notification ? el("span", {}, yesNo(true), ` 許可: ${f.notification_permission}`) : yesNo(false)],
        ["Service Worker", f.service_worker ? el("span", {}, yesNo(true), f.service_worker_controlled ? "（このページで動作中）" : "") : yesNo(false)],
        ["Cache Storage", yesNo(f.cache_storage)],
        ["IndexedDB", f.indexed_db ? el("span", {}, yesNo(f.indexed_db.available), f.indexed_db.open_ok === false ? ` 開けない: ${f.indexed_db.error || ""}` : "") : "—"],
        ["保存領域（storage.estimate）", est ? (est.error ? est.error : `使用 ${est.usage_mb} MB / 上限 ${est.quota_mb} MB`) : yesNo(false)],
        ["保存の永続化（persisted）", yesNo(f.storage_persisted)],
        ["MediaSource / ManagedMediaSource", el("span", {}, yesNo(f.media_source), " / ", yesNo(f.managed_media_source))],
        ["AudioWorklet", yesNo(f.audio_worklet)],
      ])));
    if (r.error) parts.push(el("p", { class: "error-text", text: `途中でエラー: ${r.error}` }));
    this.output.replaceChildren(...parts);
  }

  // --- 画面ロック中の再生の試験 ------------------------------------------------------

  renderLock() {
    const box = this.lockBox;
    const tests = (this.result && this.result.lock_tests) || [];
    const history = tests.length
      ? el("ul", { class: "diag-lock-list" }, tests.map((t) => el("li", {
        text: `${t.mode === "webaudio" ? "Web Audio" : "<audio> 要素"}: ${t.answer_label}`
          + `（画面が隠れていた時間 ${Math.round(t.hidden_ms / 1000)}秒${t.clock_advance_hidden_sec !== null ? `、その間に進んだ音の時間 ${t.clock_advance_hidden_sec}秒` : ""}）`,
      })))
      : null;
    if (!this.lock) {
      box.replaceChildren(
        el("ol", { class: "muted diag-steps" },
          el("li", { text: "下のボタンで試験の音を鳴らす（1秒ごとに小さく「ピッ」と鳴ります）" }),
          el("li", { text: "画面をロックして 10 秒ほど待つ（音が続くか聞く）" }),
          el("li", { text: "ロックを解除して、結果を選ぶ" })),
        el("div", { class: "row" },
          el("button", { class: "btn", type: "button", id: "lock-webaudio", text: "Web Audio で鳴らす", onclick: () => this.startLock("webaudio") }),
          el("button", { class: "btn", type: "button", id: "lock-element", text: "<audio> 要素で鳴らす", onclick: () => this.startLock("element") })),
        history,
      );
      return;
    }
    const answer = (value, label) => el("button", {
      class: "btn", type: "button", "data-answer": value, text: label,
      onclick: () => this.finishLock(value, label),
    });
    box.replaceChildren(
      el("p", { text: "鳴っています。画面をロックして 10 秒ほど待ち、解除してから選んでください。" }),
      el("div", { class: "row" },
        answer("continued", "ずっと鳴っていた"),
        answer("stopped", "ロックすると止まった"),
        answer("stopped_later", "しばらくして止まった"),
        answer("unknown", "わからない・やめる")),
      history,
    );
  }

  logLock(type, detail) {
    const lock = this.lock;
    if (!lock) return;
    const now = performance.now();
    const clock = lock.clock();
    if (type === "visibility") {
      if (detail === "hidden") { lock.hiddenAt = now; lock.clockAtHidden = clock; }
      if (detail === "visible" && lock.hiddenAt !== null) {
        lock.hiddenMs += now - lock.hiddenAt;
        if (clock !== null && lock.clockAtHidden !== null) lock.clockHidden += clock - lock.clockAtHidden;
        lock.hiddenAt = null;
      }
    }
    if (lock.events.length < MAX_EVENTS) {
      lock.events.push({ t_ms: Math.round(now - lock.t0), type, detail, clock: round(clock, 2) });
    }
  }

  async startLock(mode) {
    this.stopLock();
    const lock = {
      mode, t0: performance.now(), started_at: new Date().toISOString(), events: [],
      hiddenAt: null, hiddenMs: 0, clockAtHidden: null, clockHidden: 0,
      clock: () => null, stop: () => {},
    };
    try {
      if (mode === "webaudio") {
        const ctx = newAudioContext();
        if (!ctx) throw new Error("AudioContext が使えません。");
        await ctx.resume();
        // 1秒ごとに 0.12 秒の小さな音が鳴るバッファをくり返す
        const rate = ctx.sampleRate;
        const buf = ctx.createBuffer(1, rate, rate);
        const data = buf.getChannelData(0);
        const beep = Math.floor(rate * 0.12);
        for (let i = 0; i < beep; i++) {
          const env = Math.min(1, i / 200, (beep - i) / 200);
          data[i] = 0.12 * env * Math.sin(2 * Math.PI * 880 * i / rate);
        }
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.loop = true;
        src.connect(ctx.destination);
        src.start();
        ctx.onstatechange = () => this.logLock("context", ctx.state);
        lock.clock = () => ctx.currentTime;
        lock.stop = () => { try { src.stop(); } catch { /* 止まっている */ } ctx.close().catch(() => {}); };
      } else {
        const audio = new Audio("/api/diag/samples/m4a");
        audio.loop = true;
        audio.volume = 0.3;
        for (const ev of ["play", "pause", "stalled", "suspend", "ended", "error"]) {
          audio.addEventListener(ev, () => this.logLock("audio", ev));
        }
        lock.clock = () => lock.elementTime;
        lock.elementTime = 0;
        let last = null;
        audio.addEventListener("timeupdate", () => {
          const t = audio.currentTime;
          if (last !== null) lock.elementTime += t >= last ? t - last : t; // くり返しで 0 に戻る分を足す
          last = t;
        });
        await audio.play();
        lock.stop = () => { audio.pause(); audio.removeAttribute("src"); audio.load(); };
      }
    } catch (e) {
      toast(`鳴らせませんでした: ${e.message || e}`);
      lock.stop();
      return;
    }
    this.lock = lock;
    this.logLock("start", mode);
    this.renderLock();
  }

  stopLock() {
    if (!this.lock) return;
    this.lock.stop();
    this.lock = null;
  }

  async finishLock(answer, label) {
    const lock = this.lock;
    if (!lock) return;
    this.logLock("finish", answer);
    if (lock.hiddenAt !== null) this.logLock("visibility", "visible");
    const entry = {
      mode: lock.mode,
      started_at: lock.started_at,
      duration_sec: round((performance.now() - lock.t0) / 1000, 1),
      hidden_ms: Math.round(lock.hiddenMs),
      clock_advance_hidden_sec: lock.hiddenMs > 0 ? round(lock.clockHidden, 2) : null,
      answer,
      answer_label: label,
      events: lock.events,
    };
    this.stopLock();
    if (!this.result) {
      this.result = {
        diag_version: DIAG_VERSION, device: this.deviceName(),
        collected_at: new Date().toISOString(), ...collectBasics(), lock_only: true, lock_tests: [],
      };
    }
    this.result.lock_tests.push(entry);
    this.renderLock();
    await this.send();
  }
}
