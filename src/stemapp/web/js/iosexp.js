// 診断ページ（#/diag）の「iPhone 再生の実験」欄（T06b-0）。
//
// iPhone では Web Audio が消音モード（本体の消音スイッチ）で鳴らず、画面ロックで止まる。
// T06b（iPhone 向けの再生）の作り方を決めるため、次の 4 つの鳴らし方を実機で比べる。
//   A: navigator.audioSession.type = "playback" にしてから Web Audio で鳴らす
//   B: 無音の <audio> をループで鳴らしながら Web Audio で鳴らす
//   C: Web Audio の出力（createMediaStreamDestination）を <audio>.srcObject で鳴らす
//   D: 参考。<audio> 要素だけで鳴らす
// どれも同じテストのメロディ（8 秒をくり返す）を鳴らし、Media Session（ロック画面の曲名と操作）も設定する。
// ユーザーが「消音モードで鳴ったか」「ロック中も鳴り続けたか」「ロック画面に出たか」を選び、
// 自動で取れる値（AudioContext の状態、時計の進み、画面が隠れていた時間など）と一緒に記録する。
// プレイヤー（engine.js）には関わらない。

import { el, toast } from "./ui.js";

export const EXP_VERSION = 1;
const MAX_EVENTS = 150;
const TICK_MS = 1000;

export const EXPERIMENTS = [
  {
    id: "A", key: "audio_session",
    title: "A: audioSession を「再生」にして Web Audio",
    desc: "navigator.audioSession.type を \"playback\" にしてから、今の再生と同じ Web Audio で鳴らします。",
  },
  {
    id: "B", key: "silent_element",
    title: "B: 無音の <audio> と一緒に Web Audio",
    desc: "聞こえない無音の <audio> をくり返し鳴らしながら、Web Audio で鳴らします。",
  },
  {
    id: "C", key: "stream_element",
    title: "C: Web Audio の音を <audio> から出す",
    desc: "Web Audio の出力を MediaStream にして、<audio> 要素から鳴らします。",
  },
  {
    id: "D", key: "element_only",
    title: "D: 参考 <audio> 要素だけ",
    desc: "比べるための基準。<audio> 要素だけで同じメロディを鳴らします。",
  },
];

const QUESTIONS = [
  {
    key: "silent_mode", label: "消音モードで",
    options: [["played", "鳴った"], ["silent", "鳴らなかった"], ["not_tested", "試していない"]],
  },
  {
    key: "lock", label: "ロック中",
    options: [
      ["continued", "ずっと鳴っていた"], ["stopped", "ロックすると止まった"],
      ["stopped_later", "しばらくして止まった"], ["unknown", "わからない"],
    ],
  },
  {
    key: "lock_screen", label: "ロック画面",
    options: [
      ["title_and_controls", "曲名と操作が出た"], ["title_only", "曲名だけ出た"],
      ["none", "出なかった"], ["not_checked", "見ていない"],
    ],
  },
];

function round(v, digits = 3) {
  return typeof v === "number" && Number.isFinite(v) ? Number(v.toFixed(digits)) : v ?? null;
}

function errText(e) {
  if (!e) return "不明なエラー";
  return `${e.name || "Error"}: ${e.message || String(e)}`;
}

// --- テスト音 ------------------------------------------------------------------------

const MELODY = [
  // [周波数 Hz, 長さ（拍）]。1 拍 0.5 秒、16 拍＝8 秒でくり返す。iPhone のスピーカーで聞きやすい高さ
  [523.25, 1], [659.25, 1], [783.99, 1], [1046.5, 1],
  [783.99, 1], [659.25, 1], [587.33, 2],
  [587.33, 1], [698.46, 1], [880.0, 1], [1174.66, 1],
  [987.77, 1], [783.99, 1], [523.25, 2],
];
const BEAT_SEC = 0.5;

/** くり返し用のメロディ（モノラル、8 秒）。各拍の頭に小さなクリックも入れる（音の高さが分かりにくくても拍が分かる）。 */
export function makeMelody(rate) {
  const beats = MELODY.reduce((n, [, b]) => n + b, 0);
  const out = new Float32Array(Math.round(beats * BEAT_SEC * rate));
  let pos = 0;
  for (const [freq, b] of MELODY) {
    const len = Math.round(b * BEAT_SEC * rate);
    const tone = Math.round(len * 0.85);
    const fade = Math.round(rate * 0.01);
    for (let i = 0; i < tone && pos + i < out.length; i++) {
      const env = Math.min(1, i / fade, (tone - i) / fade) * Math.exp(-i / (rate * 0.6));
      const t = i / rate;
      out[pos + i] = 0.22 * env
        * (Math.sin(2 * Math.PI * freq * t) + 0.3 * Math.sin(4 * Math.PI * freq * t));
    }
    for (let k = 0; k < b; k++) {
      const start = pos + Math.round(k * BEAT_SEC * rate);
      const click = Math.round(rate * 0.004);
      for (let i = 0; i < click && start + i < out.length; i++) {
        out[start + i] += 0.15 * (1 - i / click) * (i % 2 ? 1 : -1);
      }
    }
    pos += len;
  }
  return out;
}

/** モノラル 16bit の WAV を作る（<audio> 用。サーバーには頼らない）。 */
export function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, Math.round(s * 32767), true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function newAudioContext() {
  const Ctor = window.AudioContext || window.webkitAudioContext;
  return Ctor ? new Ctor() : null;
}

// --- 1 回の実験 -----------------------------------------------------------------------

class Run {
  constructor(exp) {
    this.exp = exp;
    this.t0 = performance.now();
    this.startedAt = new Date().toISOString();
    this.events = [];
    this.hiddenAt = null;
    this.hiddenMs = 0;
    this.clockAtHidden = null;
    this.clockHidden = 0;
    this.clock = () => null; // 音の時計（秒）。取れなければ null
    this.cleanups = [];
    this.auto = {};
    this.ticks = 0;
    this.ticksHidden = 0;
    this.maxGap = 0;
    this.lastTick = performance.now();
    this.ctx = null;
    this.audio = null;
    this.elementTime = 0;
  }

  log(type, detail) {
    const now = performance.now();
    const clock = this.clock();
    if (type === "visibility") {
      if (detail === "hidden" && this.hiddenAt === null) {
        this.hiddenAt = now; this.clockAtHidden = clock;
      }
      if (detail === "visible" && this.hiddenAt !== null) {
        this.hiddenMs += now - this.hiddenAt;
        if (clock !== null && this.clockAtHidden !== null) this.clockHidden += clock - this.clockAtHidden;
        this.hiddenAt = null;
      }
    }
    if (this.events.length < MAX_EVENTS) {
      this.events.push({ t_ms: Math.round(now - this.t0), type, detail, clock: round(clock, 2) });
    }
  }

  tick() {
    const now = performance.now();
    this.maxGap = Math.max(this.maxGap, now - this.lastTick);
    this.lastTick = now;
    this.ticks += 1;
    if (document.visibilityState === "hidden") this.ticksHidden += 1;
  }

  /** <audio> のイベントを記録し、進んだ時間を数える（くり返しで 0 に戻る分も足す）。 */
  watchElement(audio, name) {
    for (const ev of ["play", "playing", "pause", "waiting", "stalled", "suspend", "ended", "error"]) {
      const fn = () => this.log(name, ev);
      audio.addEventListener(ev, fn);
    }
    let last = null;
    audio.addEventListener("timeupdate", () => {
      const t = audio.currentTime;
      if (last !== null) this.elementTime += t >= last ? t - last : t;
      last = t;
    });
  }

  watchContext(ctx) {
    this.auto.context_state_start = ctx.state;
    this.auto.sample_rate = ctx.sampleRate;
    ctx.onstatechange = () => this.log("context", ctx.state);
  }

  stop() {
    for (const fn of this.cleanups.splice(0).reverse()) {
      try { fn(); } catch { /* 片付けだけ */ }
    }
  }
}

function playMelodyBuffer(ctx, out) {
  const data = makeMelody(ctx.sampleRate);
  const buf = ctx.createBuffer(1, data.length, ctx.sampleRate);
  buf.getChannelData(0).set(data);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.loop = true;
  src.connect(out);
  src.start();
  return src;
}

/** ロック画面の曲名と操作（Media Session）。設定できたかを返す。 */
function setupMediaSession(run, { play, pause }) {
  const ms = navigator.mediaSession;
  if (!ms) return { available: false };
  const info = { available: true, metadata_set: false, handlers: [] };
  try {
    const Meta = window.MediaMetadata;
    if (Meta) {
      ms.metadata = new Meta({
        title: `stemapp 実験 ${run.exp.id}`,
        artist: "stemapp 端末の診断",
        album: run.exp.title,
        artwork: [
          { src: "icons/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icons/icon-512.png", sizes: "512x512", type: "image/png" },
        ],
      });
      info.metadata_set = true;
    }
  } catch (e) {
    info.error = errText(e);
  }
  const actions = {
    play: async () => { run.log("media_session", "play"); await play(); ms.playbackState = "playing"; },
    pause: async () => { run.log("media_session", "pause"); await pause(); ms.playbackState = "paused"; },
    stop: async () => { run.log("media_session", "stop"); await pause(); ms.playbackState = "paused"; },
  };
  for (const [action, fn] of Object.entries(actions)) {
    try {
      ms.setActionHandler(action, () => { fn().catch((e) => run.log("media_session_error", errText(e))); });
      info.handlers.push(action);
    } catch { /* その操作は使えない */ }
  }
  try { ms.playbackState = "playing"; } catch { /* 古い端末 */ }
  run.cleanups.push(() => {
    try { ms.metadata = null; } catch { /* 片付けだけ */ }
    for (const action of info.handlers) {
      try { ms.setActionHandler(action, null); } catch { /* 片付けだけ */ }
    }
    try { ms.playbackState = "none"; } catch { /* 片付けだけ */ }
  });
  return info;
}

function makeElement(run, url, name, volume = 1) {
  const audio = new Audio();
  audio.loop = true;
  audio.volume = volume;
  audio.setAttribute("playsinline", "");
  if (url) audio.src = url;
  run.watchElement(audio, name);
  run.cleanups.push(() => {
    audio.pause();
    audio.srcObject = null;
    audio.removeAttribute("src");
    audio.load();
  });
  return audio;
}

function blobUrl(run, blob) {
  const url = URL.createObjectURL(blob);
  run.cleanups.push(() => URL.revokeObjectURL(url));
  return url;
}

/**
 * 実験の音を鳴らし始める（ボタンを押した処理の中で、await の前に再生の命令を出す。
 * iPhone は「ユーザーの操作の中」でないと鳴らせないため）。
 * 戻り値: { play, pause }（Media Session の操作用）。
 */
async function startSound(run) {
  const id = run.exp.id;
  if (id === "A") {
    const session = navigator.audioSession;
    run.auto.audio_session_available = !!session;
    run.auto.audio_session_type_before = session.type ?? null;
    try {
      session.type = "playback";
    } catch (e) {
      run.auto.audio_session_error = errText(e);
    }
    run.auto.audio_session_type_after = session.type ?? null;
    run.auto.audio_session_state = session.state ?? null;
    const onState = () => run.log("audio_session", `${session.state}`);
    session.addEventListener?.("statechange", onState);
    run.cleanups.push(() => {
      session.removeEventListener?.("statechange", onState);
      try { session.type = run.auto.audio_session_type_before || "auto"; } catch { /* 片付けだけ */ }
    });
  }

  if (id === "D") {
    const rate = 44100;
    const url = blobUrl(run, encodeWav(makeMelody(rate), rate));
    const audio = makeElement(run, url, "audio");
    run.audio = audio;
    run.clock = () => run.elementTime;
    await audio.play();
    return {
      play: () => audio.play(),
      pause: async () => audio.pause(),
    };
  }

  // A・B・C は Web Audio で鳴らす
  let silent = null;
  let silentPlay = null;
  if (id === "B") {
    // 先に無音の <audio> を鳴らす（1 秒の無音をくり返す）
    const rate = 22050;
    const url = blobUrl(run, encodeWav(new Float32Array(rate), rate));
    silent = makeElement(run, url, "silent_audio");
    run.audio = silent;
    silentPlay = silent.play();
  }
  const ctx = newAudioContext();
  if (!ctx) throw new Error("AudioContext が使えません。");
  run.ctx = ctx;
  run.watchContext(ctx);
  run.cleanups.push(() => { ctx.close().catch(() => {}); });
  const resumed = ctx.resume();
  let out = ctx.destination;
  let streamAudio = null;
  if (id === "C") {
    if (typeof ctx.createMediaStreamDestination !== "function") {
      run.auto.media_stream_destination = false;
      throw new Error("この端末には createMediaStreamDestination がありません。");
    }
    run.auto.media_stream_destination = true;
    const dest = ctx.createMediaStreamDestination();
    out = dest;
    streamAudio = makeElement(run, null, "stream_audio");
    streamAudio.srcObject = dest.stream;
    run.audio = streamAudio;
  }
  const src = playMelodyBuffer(ctx, out);
  run.cleanups.push(() => { try { src.stop(); } catch { /* 止まっている */ } });
  run.clock = () => ctx.currentTime;
  const waits = [resumed];
  if (silentPlay) waits.push(silentPlay);
  if (streamAudio) waits.push(streamAudio.play());
  await Promise.all(waits);
  run.auto.context_state_after_resume = ctx.state;
  const elem = silent || streamAudio;
  return {
    play: async () => { if (elem) await elem.play(); await ctx.resume(); },
    pause: async () => { if (elem) elem.pause(); await ctx.suspend(); },
  };
}

// --- 画面 -----------------------------------------------------------------------------

function labelOf(qkey, value) {
  const q = QUESTIONS.find((x) => x.key === qkey);
  const opt = q && q.options.find(([v]) => v === value);
  return opt ? opt[1] : "未回答";
}

/** 記録 1 件を 1 行の文にする（画面の履歴用）。 */
export function describeEntry(e) {
  if (e.answer === "unavailable") return `${e.experiment}: この端末には無い（${e.note || ""}）`;
  const a = e.answers || {};
  const parts = QUESTIONS.map((q) => `${q.label}「${labelOf(q.key, a[q.key])}」`);
  let extra = `画面が隠れていた時間 ${Math.round((e.hidden_ms || 0) / 1000)}秒`;
  if (e.clock_advance_hidden_sec !== null && e.clock_advance_hidden_sec !== undefined) {
    extra += `、その間に進んだ音の時間 ${e.clock_advance_hidden_sec}秒`;
  }
  return `${e.experiment}: ${parts.join("・")}（${extra}）`;
}

export class IosExperiments {
  /**
   * @param {HTMLElement} box 欄を描く場所
   * @param {(entry: object) => Promise<void>|void} onRecord 記録 1 件ができたときに呼ぶ
   * @param {() => object[]} history これまでの記録（表示用）
   */
  constructor(box, { onRecord, history }) {
    this.box = box;
    this.onRecord = onRecord;
    this.history = history;
    this.run = null;
    this.answers = {};
    this.onVisibility = () => this.run && this.run.log("visibility", document.visibilityState);
    this.onPage = (e) => this.run && this.run.log("page", e.type);
    document.addEventListener("visibilitychange", this.onVisibility);
    for (const ev of ["pagehide", "pageshow", "freeze", "resume", "blur", "focus"]) {
      window.addEventListener(ev, this.onPage);
    }
    document.addEventListener("freeze", this.onPage);
    document.addEventListener("resume", this.onPage);
    this.render();
  }

  destroy() {
    document.removeEventListener("visibilitychange", this.onVisibility);
    for (const ev of ["pagehide", "pageshow", "freeze", "resume", "blur", "focus"]) {
      window.removeEventListener(ev, this.onPage);
    }
    document.removeEventListener("freeze", this.onPage);
    document.removeEventListener("resume", this.onPage);
    this.stopRun();
  }

  render() {
    const entries = this.history();
    const historyList = entries.length
      ? el("ul", { class: "diag-lock-list ios-exp-history" },
        entries.map((e) => el("li", { text: describeEntry(e) })))
      : null;
    if (!this.run) {
      this.box.replaceChildren(
        el("ol", { class: "muted diag-steps" },
          el("li", { text: "本体の消音スイッチを ON（消音モード）にし、音量を上げておく" }),
          el("li", { text: "実験のボタンを押す（8 秒のメロディがくり返し鳴ります）" }),
          el("li", { text: "鳴ったか聞く。次に画面をロックして 10 秒待ち、ロック画面に曲名と操作が出るかも見る" }),
          el("li", { text: "ロックを解除して、結果を選んで「記録する」" })),
        el("div", { class: "ios-exp-grid" }, EXPERIMENTS.map((exp) => el("div", { class: "ios-exp" },
          el("div", { class: "ios-exp-title", text: exp.title }),
          el("p", { class: "muted small-note", text: exp.desc }),
          el("button", {
            class: "btn", type: "button", id: `exp-${exp.id}`, text: "鳴らす",
            onclick: () => this.start(exp),
          })))),
        ...(historyList ? [historyList] : []),
      );
      return;
    }
    const run = this.run;
    const groups = QUESTIONS.map((q) => el("div", { class: "ios-exp-q", role: "group", "aria-label": q.label },
      el("span", { class: "ios-exp-qlabel", text: q.label }),
      el("div", { class: "row" }, q.options.map(([value, label]) => el("button", {
        class: this.answers[q.key] === value ? "btn small on" : "btn small",
        type: "button", "data-q": q.key, "data-v": value, text: label,
        "aria-pressed": this.answers[q.key] === value ? "true" : "false",
        onclick: () => { this.answers[q.key] = value; this.render(); },
      })))));
    this.box.replaceChildren(
      el("div", { class: "ios-exp-running", id: "exp-running", "data-exp": run.exp.id },
        el("div", { class: "ios-exp-title", text: `鳴らしています — ${run.exp.title}` }),
        el("p", { class: "muted small-note", text: "画面をロックして 10 秒待ち、ロック画面の表示も見てから解除して選んでください。" }),
        ...groups,
        el("div", { class: "row" },
          el("button", { class: "btn primary", type: "button", id: "exp-record", text: "記録する", onclick: () => this.finish() }),
          el("button", { class: "btn", type: "button", id: "exp-cancel", text: "やめる（記録しない）", onclick: () => { this.stopRun(); this.render(); } }))),
      ...(historyList ? [historyList] : []),
    );
  }

  async start(exp) {
    this.stopRun();
    this.answers = {};
    if (exp.id === "A" && !navigator.audioSession) {
      const entry = {
        exp_version: EXP_VERSION, experiment: "A", mode: exp.key, started_at: new Date().toISOString(),
        answer: "unavailable", answer_label: "この端末には無い",
        note: "navigator.audioSession がありません",
        auto: { audio_session_available: false },
      };
      toast("この端末には navigator.audioSession がありません（記録しました）。");
      await this.onRecord(entry);
      this.render();
      return;
    }
    const run = new Run(exp);
    run.log("start", exp.key);
    let controls;
    try {
      controls = await startSound(run);
    } catch (e) {
      run.log("error", errText(e));
      run.stop();
      toast(`鳴らせませんでした: ${e.message || e}`);
      return;
    }
    run.auto.media_session = setupMediaSession(run, controls);
    const timer = setInterval(() => run.tick(), TICK_MS);
    run.cleanups.push(() => clearInterval(timer));
    this.run = run;
    run.log("playing", exp.key);
    this.render();
  }

  stopRun() {
    if (!this.run) return;
    this.run.stop();
    this.run = null;
  }

  async finish() {
    const run = this.run;
    if (!run) return;
    run.log("finish", "record");
    if (run.hiddenAt !== null) run.log("visibility", "visible");
    const answers = {};
    const labels = {};
    for (const q of QUESTIONS) {
      answers[q.key] = this.answers[q.key] ?? null;
      labels[q.key] = labelOf(q.key, answers[q.key]);
    }
    const auto = { ...run.auto };
    if (run.ctx) {
      auto.context_state_end = run.ctx.state;
      auto.context_time_end = round(run.ctx.currentTime, 2);
    }
    if (run.audio) {
      auto.element_paused_end = run.audio.paused;
      auto.element_time_total = round(run.elementTime, 2);
    }
    auto.timer_ticks = run.ticks;
    auto.timer_ticks_hidden = run.ticksHidden;
    auto.max_timer_gap_ms = Math.round(run.maxGap);
    const entry = {
      exp_version: EXP_VERSION,
      experiment: run.exp.id,
      mode: run.exp.key,
      started_at: run.startedAt,
      duration_sec: round((performance.now() - run.t0) / 1000, 1),
      hidden_ms: Math.round(run.hiddenMs),
      clock_advance_hidden_sec: run.hiddenMs > 0 ? round(run.clockHidden, 2) : null,
      answer: answers.lock ?? "unanswered",
      answer_label: labels.lock,
      answers,
      answer_labels: labels,
      auto,
      events: run.events,
    };
    this.stopRun();
    await this.onRecord(entry);
    this.render();
  }
}
