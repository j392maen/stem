// プレイヤー画面: 波形、再生・シーク、stem の ON/OFF、グループ、組み合わせプリセット、キュー・ループ、
// 拍・小節線と再生位置の BPM、拍の補正（T10c）、小節単位のループ、キューの拍へのスナップ、
// 速度の変更（T11。ピッチも変わる方式・ピッチを保つ方式）、
// iPhone 向けの再生（T06b。音声セッション、ロック画面の表示と操作、選択中の stem だけ読み込む、
// 続きから再生）。

import { api, fetchBinary } from "./api.js";
import {
  AudioRoute, ROUTE_SHORT, isCoarsePointer, loadRouteChoice, routeForThisDevice, saveRouteChoice,
} from "./audioroute.js";
import { BeatEditPanel } from "./beatedit.js";
import { BeatGrid, barLoop, formatBpm } from "./beats.js";
import { renameDevice } from "./device.js";
import { Engine, clampTime } from "./engine.js";
import { Exporter } from "./export.js";
import { MediaSessionControl } from "./mediasession.js";
import { parsePeaks } from "./peaks.js";
import { RefineUI } from "./refine.js";
import { PlaybackSync, describeOther, restoreFromState, resumePosition } from "./resume.js";
import * as S from "./selection.js";
import { formatMB, loadLazySetting, pickEvictions, saveLazySetting } from "./stemload.js";
import { COARSE_STEP, FINE_STEP, TempoPanel } from "./tempo.js";
import { confirmDialog, el, formatTime, icon, ICONS, promptDialog, toast } from "./ui.js";
import { WaveformView, ZOOM_STEPS } from "./waveform.js";

const SEEK_STEP_SEC = 5;
const LOAD_CONCURRENCY = 3;
const POSTPROCESS_POLL_MS = 1500;
const HOUSEKEEP_MS = 5000; // 使わない stem の音声を捨てる確認の間隔（スマホ）
const VOLUME_KEY = "stemapp.volume";
// 曲ごとに最後に選んだ分け方（ジョブ）。無い・消えたときは新しい完了済みジョブ
const JOB_KEY_PREFIX = "stemapp.job.";
// キューを拍に合わせる（スナップ）の設定。既定は ON
const SNAP_KEY = "stemapp.snap";
// 小節ループの長さ（小節）。½・×2 はこの範囲で変える
export const BAR_LOOPS = [1, 2, 4, 8, 16];
const MIN_LOOP_BARS = 0.25;
const MAX_LOOP_BARS = 64;
// キューの色。stem・グループの色と重ならないよう、差し色の赤2色と白だけにする
export const CUE_COLORS = ["#FF3B4E", "#F5F5F4", "#FF8A95"];

function loadVolume() {
  try {
    const v = Number(localStorage.getItem(VOLUME_KEY));
    return Number.isFinite(v) && localStorage.getItem(VOLUME_KEY) !== null ? v : 0.9;
  } catch { return 0.9; }
}

function loadSnap() {
  try { return localStorage.getItem(SNAP_KEY) !== "0"; } catch { return true; }
}

function loadJobChoice(trackId) {
  try { return Number(localStorage.getItem(JOB_KEY_PREFIX + trackId)) || null; } catch { return null; }
}

function saveJobChoice(trackId, jobId) {
  try { localStorage.setItem(JOB_KEY_PREFIX + trackId, String(jobId)); } catch { /* 保存できなくても動く */ }
}

function formatDb(v) {
  return v === null || v === undefined ? "-" : v.toFixed(1).replace("-", "−");
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
    // 「全部」にする直前の組み合わせ（0 キーで戻る）
    this.beforeAll = null;
    this.canOpenFolder = false;
    // 音声と波形を読み込み中（分け方の切り替え・削除はできない）
    this.loading = false;
    // 再生する分け方（ジョブ）。切り替えたときは restore に再生位置・選択を持って読み直す
    this.jobId = null;
    this.restore = null;
    this.beatGrid = null; // 拍が未解析なら null
    this.beatBusy = false; // 拍の解析を依頼して待っている
    // 拍の解析を受け持つジョブ（拍は曲ごと。同じ曲に分け方が複数あると、表示中とは限らない）
    this.beatJobId = null;
    this.tempoKey = "";
    this.beatEdit = null; // 拍の補正パネル
    this.tempo = null; // 速度の変更パネル
    // 小節ループ（キューに保存しない一時的なループ。{start, end, bars}）。キューのループより優先
    this.barLoop = null;
    this.loopBars = 4; // B キーで作る小節ループの長さ（最後に選んだもの）
    this.snap = loadSnap();
    this.onKey = (e) => this.handleKey(e);
    // --- T06b: iPhone 向けの再生 ---
    this.lazy = loadLazySetting(); // 選択中の stem だけ読み込む（スマホの既定）
    this.stemLoads = new Map(); // 読み込み中の stem: code → { key, abort }
    this.stemFailed = new Set(); // 読み込めなかった「組:code」（押し直すまで読み直さない）
    this.offSince = new Map(); // OFF にした時刻（ms）: code → 時刻
    this.route = null; // 音の出し方（AudioRoute）
    this.media = null; // ロック画面の表示と操作（MediaSessionControl）
    this.sync = null; // 続きから再生の保存（PlaybackSync）
    this.device = null; // この端末（DEVICE）
    this.otherState = null; // ほかの端末で最後に聴いていた状態
    this.resumed = false; // 開いたときの「続きから」を済ませた（読み直しでは使わない）
    this.resumeTempo = null; // 続きから戻す速度と方式（TempoPanel が使う）
    this.memKey = "";
  }

  later(fn, ms) {
    const t = setTimeout(() => { this.timers.delete(t); if (this.alive) fn(); }, ms);
    this.timers.add(t);
  }

  // --- 読み込み ---------------------------------------------------------------

  async mount() {
    // 読み直し（remount）の後に、前の mount の続きが画面を触らないようにする
    const signal = this.abort.signal;
    const live = () => this.alive && !signal.aborted;
    this.root.replaceChildren(el("p", { class: "empty", text: "読み込み中…" }));
    // 端末の登録と、この曲の再生の状態（続きから再生）。失敗しても再生はできる
    this.sync = new PlaybackSync(this);
    const syncLoad = this.sync.load();
    try {
      this.track = await api(`/api/tracks/${this.trackId}`);
    } catch (e) {
      if (live()) this.showMessage(e.status === 404 ? "曲が見つかりません。" : e.message);
      return;
    }
    const { device, mine, other } = await syncLoad;
    if (!live()) return;
    this.device = device;
    this.otherState = other;
    // 開いたときだけ、自分の端末の状態に戻す（分け方の切り替えなどの読み直しでは、今の状態を保つ）
    let resumeJob = null;
    if (!this.resumed) {
      this.resumed = true;
      if (mine && !this.restore) {
        this.restore = restoreFromState(mine);
        resumeJob = mine.job_id;
        if (mine.tempo_ratio) this.resumeTempo = { ratio: mine.tempo_ratio, mode: mine.tempo_mode };
      }
    }
    this.doneJobs = (this.track.jobs || []).filter((j) => j.job_kind === "full" && j.status === "done");
    const isDone = (id) => this.doneJobs.some((j) => j.job_id === id);
    const saved = loadJobChoice(this.trackId);
    const jobId = [this.jobId, resumeJob, saved].find((id) => id && isDone(id))
      || this.track.playable_job_id;
    this.jobId = jobId;
    if (!jobId) {
      this.showMessage("この曲はまだ分割されていません。ライブラリで分割してください。");
      return;
    }
    try {
      const [stems, types, groups, presets, cues, me, beats] = await Promise.all([
        api(`/api/jobs/${jobId}/stems`),
        api("/api/stem-types"),
        api("/api/stem-groups"),
        api("/api/listen-presets"),
        api(`/api/tracks/${this.trackId}/cues`),
        api("/api/me").catch(() => ({})),
        // 拍がまだ無い曲は 404（1秒目盛りのまま再生できる）
        api(`/api/tracks/${this.trackId}/beats`).catch(() => null),
      ]);
      this.beatGrid = beats ? new BeatGrid(beats) : null;
      if (this.beatGrid && this.beatGrid.empty) this.beatGrid = null;
      this.canOpenFolder = !!me.can_open_folder;
      this.job = stems;
      this.tree = null; // 前のジョブの木を残さない（見出しの「書き出し」は木ができてから出す）
      this.stemTypes = types.stem_types;
      this.groups = groups.stem_groups;
      this.presets = presets.listen_presets;
      this.cues = cues.cues;
    } catch (e) {
      if (live()) this.showMessage(e.message);
      return;
    }
    if (!live()) return;
    if (!this.job.delivery_ready) {
      this.showDeliveryMissing();
      return;
    }
    this.tree = S.buildTree(this.job.stems);
    this.sel = S.allOn(this.tree);
    if (this.restore) {
      // 戻す選択を先に決めておく（スマホでは選択中の stem だけ読み込むため）
      const leaves = new Set(this.tree.leaves);
      const sel = new Set([...this.restore.sel].filter((c) => leaves.has(c)));
      if (sel.size) this.sel = sel;
    }
    this.refine = new RefineUI(this); // 「もっと分ける」（web/js/refine.js）
    this.loading = true;
    this.render();
    document.addEventListener("keydown", this.onKey);
    // 拍の解析を待っている間に分け方を切り替えた（読み直した）ときは、待つのを続ける
    if (this.beatBusy) this.later(() => this.pollBeats(), POSTPROCESS_POLL_MS);
    try {
      await this.loadMedia();
    } finally {
      if (live()) {
        this.loading = false;
        this.updateJobBar();
      }
    }
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
      this.jobBarEl(),
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
    // スマホ（iPhone の音声セッション・<audio> の経路）では、読み直した後の再生がユーザーの操作の外に
    // なって鳴らないことがあるので、止めてから読み直し、▶ を押してもらう（PC は今までどおり続ける）
    const r = this.restore;
    if (r && r.playing && this.needsGestureToPlay()) {
      if (this.engine) this.engine.pause();
      r.playing = false;
      r.askPlay = true;
    }
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
    // スマホ: 音声は選択中の stem だけ（波形は全部）
    const wanted = this.lazy ? new Set(leaves.filter((s) => this.sel.has(s.code)).map((s) => s.code)) : null;
    const total = leaves.length + (wanted ? wanted.size : leaves.length);
    let done = 0;
    const step = () => {
      done++;
      label.textContent = `音声と波形を読み込み中… ${done}/${total}`;
      bar.style.width = `${Math.round((done / total) * 100)}%`;
    };
    // 1つでも失敗したら残りの取得を中断する（画面を離れたときの中断にも従う）
    const loading = new AbortController();
    const onLeave = () => loading.abort();
    const mountSignal = this.abort.signal;
    const live = () => this.alive && !mountSignal.aborted;
    mountSignal.addEventListener("abort", onLeave);
    try {
      this.engine = new Engine();
      this.route = new AudioRoute(this.engine, routeForThisDevice(this.engine.ctx));
      this.engine.setRoute(this.route);
      const signal = loading.signal;
      const loaded = await mapLimit(leaves, LOAD_CONCURRENCY, async (stem) => {
        const stream = stem.renditions.find((r) => r.purpose === "stream");
        const peaks = new Map();
        const peakBufs = await Promise.all(stem.peaks.map((p) => fetchBinary(p.url, signal)));
        stem.peaks.forEach((p, i) => peaks.set(p.samples_per_px, parsePeaks(peakBufs[i])));
        step();
        if (wanted && !wanted.has(stem.code)) return { stem, buffer: null, peaks };
        const audio = await fetchBinary(stream.url, signal);
        if (signal.aborted) throw new DOMException("中断しました", "AbortError");
        const buffer = await this.engine.decode(audio);
        step();
        return { stem, buffer, peaks };
      }, signal).catch((e) => {
        loading.abort();
        throw e;
      });
      if (!live()) return;
      for (const code of this.tree.order) {
        const item = loaded.find((x) => x.stem.code === code);
        this.engine.addTrack(code, item ? item.buffer : null, 0);
      }
      this.engine.duration = Math.max(this.engine.duration, 0);
      // 読み込んでいない stem があるときは曲の長さを使う（どれも読み込んでいなくても位置が進む）
      if (wanted) this.engine.duration = Math.max(this.engine.duration, this.track.duration_sec || 0);
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
        beatGrid: this.beatGrid,
      });
      this.applySelection(0);
      this.ready = true;
      cover.hidden = true;
      this.root.querySelector("#play-btn").disabled = false;
      this.tempo.start(Object.fromEntries(loaded.map((x) => [
        x.stem.code, x.stem.renditions.find((r) => r.purpose === "stream").url,
      ])));
      this.setupMediaSession();
      this.renderPlayInfo();
      this.frame();
      await this.applyRestore();
      this.ensureStemAudio();
      if (this.sync) this.sync.start();
      this.later(() => this.housekeeping(), HOUSEKEEP_MS);
    } catch (e) {
      if (!live()) return;
      if (e.name === "AbortError") return;
      label.textContent = `読み込めませんでした: ${e.message}`;
      label.classList.add("error-text");
      cover.querySelector(".progress").hidden = true;
    }
  }

  // --- 分け方（ジョブ）の切り替え ------------------------------------------------------

  /** 別の分け方に切り替える。再生位置・stem の選択・ループをなるべく保つ。 */
  switchJob(jobId) {
    if (!jobId || jobId === this.jobId || this.loading) return;
    // 前の切り替えがまだ反映されていない（読み込みに失敗した等）なら、その状態を引き継ぐ
    if (!this.restore) this.restore = this.captureState();
    saveJobChoice(this.trackId, jobId);
    this.jobId = jobId;
    this.remount();
  }

  /** 再生位置・stem の選択・ループなど、分け方を変えても保ちたい状態。 */
  captureState() {
    const engine = this.ready ? this.engine : null;
    return {
      position: engine ? engine.position : 0,
      playing: !!(engine && engine.playing),
      sel: new Set(this.sel),
      gainsDb: new Map(this.gainsDb),
      presetId: this.activePresetId,
      beforeAll: this.beforeAll,
      soloMode: this.soloMode,
      loopCueId: this.loopCueId,
      loopOn: this.loopOn,
      barLoop: this.barLoop,
      zoom: this.wave ? this.wave.zoomSeconds : null,
    };
  }

  /** 今の分け方（ジョブ）を消す。ほかに完了した分け方があるときだけ（残った方に切り替える）。 */
  async deleteCurrentJob() {
    const job = (this.doneJobs || []).find((j) => j.job_id === this.jobId);
    if (!job || this.doneJobs.length < 2 || this.loading) return;
    const ok = await confirmDialog(
      `分け方「${this.jobLabel(job)}」を削除しますか？ この分け方の stem のファイルも消えます。元に戻せません。`
        + "（曲とほかの分け方、キューは残ります）",
      { ok: "削除する", danger: true },
    );
    if (!ok || !this.alive) return;
    try {
      await api(`/api/jobs/${job.job_id}`, { method: "DELETE" });
    } catch (e) {
      toast(e.message);
      return;
    }
    toast("分け方を削除しました。");
    if (!this.alive) return;
    try { localStorage.removeItem(JOB_KEY_PREFIX + this.trackId); } catch { /* 保存できなくても動く */ }
    if (!this.restore) this.restore = this.captureState();
    this.jobId = null; // 既定の分け方に戻る
    this.remount();
  }

  /** 読み込み中かどうかに合わせて、分け方の切り替え・削除を使えるようにする。 */
  updateJobBar() {
    const n = this.doneJobs ? this.doneJobs.length : 0;
    const select = this.root.querySelector("#job-select");
    if (select) select.disabled = n < 2 || this.loading;
    const del = this.root.querySelector("#delete-job-btn");
    if (del) del.disabled = n < 2 || this.loading;
  }

  async applyRestore() {
    const r = this.restore;
    this.restore = null;
    if (!r || !this.alive || !this.engine) return;
    // 新しいジョブに無い stem は外す（全部外れたら全部 ON）
    const leaves = new Set(this.tree.leaves);
    const sel = new Set([...r.sel].filter((c) => leaves.has(c)));
    this.soloMode = r.soloMode;
    this.beforeAll = r.beforeAll;
    this.sel = sel.size ? sel : S.allOn(this.tree);
    this.gainsDb = r.gainsDb;
    this.activePresetId = sel.size ? r.presetId : null;
    this.applySelection(0);
    this.renderPresets();
    this.loopCueId = r.loopCueId;
    this.barLoop = r.barLoop;
    this.loopOn = r.loopOn && !!this.activeLoop();
    this.engine.setLoop(this.loopOn ? this.activeLoop() : null);
    this.renderCues();
    this.renderBarLoop();
    if (r.zoom && this.wave) this.wave.setZoom(r.zoom);
    // 続きから: 曲の終わり近く（5 秒より後）で止めていたら最初から
    const pos = r.fromServer ? resumePosition(r.position, this.engine.duration) : r.position;
    this.engine.seek(clampTime(pos, this.engine.duration));
    if (r.fromServer && pos >= 1) toast(`前回の続き（${formatTime(pos)}）から再生します。`);
    if (r.askPlay) toast("▶ を押して再生してください。");
    if (r.playing) await this.engine.play();
    this.updateTransport();
  }

  jobLabel(j) {
    const name = j.preset_name || j.preset || "不明";
    return `${j.preset_experimental ? "実験: " : ""}${name}（#${j.job_id}）`;
  }

  /** 分け方の切り替えと、聴き比べ用の数値（stem ごとの RMS、補正前の残差）。 */
  jobBarEl() {
    if (!this.doneJobs || !this.doneJobs.length) return null;
    const current = this.doneJobs.find((j) => j.job_id === this.jobId);
    const select = el("select", {
      class: "select", id: "job-select", "aria-label": "分け方",
      disabled: this.doneJobs.length < 2 || this.loading,
      onchange: (e) => this.switchJob(Number(e.target.value)),
    }, this.doneJobs.map((j) => el("option", {
      value: String(j.job_id), text: this.jobLabel(j), selected: j.job_id === this.jobId,
    })));
    const resid = current && current.residual_rms_db !== null && current.residual_rms_db !== undefined
      ? `補正前の残差 ${formatDb(current.residual_rms_db)} dB（元の曲 ${formatDb(current.mixture_rms_db)} dB）`
      : "";
    // 分け方が1つしか無いときは消せない（曲の削除はライブラリで）
    const del = this.doneJobs.length > 1 && current
      ? el("button", {
        class: "btn small danger", id: "delete-job-btn", type: "button", text: "この分け方を削除",
        title: "今の分け方（ジョブ）と、その stem のファイルを消します。曲とほかの分け方は残ります。",
        disabled: this.loading, onclick: () => this.deleteCurrentJob(),
      })
      : null;
    return el("div", { class: "job-bar", id: "job-bar" },
      el("label", { class: "muted", for: "job-select", text: "分け方" }), select, del,
      el("span", { class: "job-metric", id: "job-metric", text: resid }),
      this.doneJobs.length > 1 || current ? this.levelsEl() : null);
  }

  /** 完了したジョブを行、stem を列にした RMS（dBFS）の表（開いたときだけ見える）。 */
  levelsEl() {
    const codes = [];
    for (const j of this.doneJobs) {
      for (const c of Object.keys(j.stem_rms_db || {})) if (!codes.includes(c)) codes.push(c);
    }
    const nameOf = (c) => {
      const t = (this.stemTypes || []).find((x) => x.code === c);
      return t ? t.display_name : c;
    };
    const table = el("table", { class: "levels" },
      el("thead", {}, el("tr", {},
        el("th", { text: "分け方" }),
        ...codes.map((c) => el("th", { text: nameOf(c) })),
        el("th", { text: "残差" }))),
      el("tbody", {}, ...this.doneJobs.map((j) => el("tr", { class: j.job_id === this.jobId ? "current" : "" },
        el("th", { text: this.jobLabel(j) }),
        ...codes.map((c) => el("td", { text: formatDb((j.stem_rms_db || {})[c]) })),
        el("td", { text: formatDb(j.residual_rms_db) })))));
    return el("details", { class: "levels-box" },
      el("summary", { text: "数値で比較（RMS dB）" }),
      el("div", { class: "levels-scroll" }, table));
  }

  unmount() {
    // 続きから再生: 最後の状態を保存する（エンジンを閉じる前に）
    if (this.sync) this.sync.dispose();
    this.sync = null;
    if (this.media) this.media.dispose();
    this.media = null;
    this.alive = false;
    this.abort.abort();
    this.stemLoads.clear();
    this.offSince.clear();
    document.removeEventListener("keydown", this.onKey);
    cancelAnimationFrame(this.raf);
    if (this.beatEdit) this.beatEdit.dispose();
    if (this.tempo) this.tempo.dispose();
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
    if (this.engine) this.engine.close();
    this.engine = null;
    this.route = null;
    this.ready = false;
    this.loading = false;
    // 書き出しメニューは表示中のジョブに結びつくので、分け方の切り替え（remount）でも作り直す
    if (this.exporter) this.exporter.dispose();
    this.exporter = null;
    if (this.refine) this.refine.dispose();
    this.refine = null;
  }

  // --- 再生 -------------------------------------------------------------------

  frame() {
    if (!this.alive || !this.engine) return;
    this.engine.tick();
    this.wave.draw();
    const pos = this.wave.preview ?? this.engine.position;
    const t = this.root.querySelector("#time-now");
    if (t) t.textContent = formatTime(pos, true);
    this.updateTempo(pos);
    this.renderMemory();
    if (this.media && this.engine.playing) {
      this.media.setPosition(this.engine.duration, this.engine.position, this.engine.speed);
    }
    this.raf = requestAnimationFrame(() => this.frame());
  }

  /** 聞こえている速さ（曲の時刻の進む速さ。元の速度なら 1）。 */
  playSpeed() {
    return this.engine ? this.engine.speed : 1;
  }

  /**
   * 再生位置の区間の BPM（× 速さ）と拍子を表示する（変わったときだけ書き換える）。
   * 速度を変えているときは元の BPM も小さく出す。
   */
  updateTempo(pos) {
    const grid = this.beatGrid;
    const base = grid ? grid.bpmAt(pos) : null;
    const speed = this.playSpeed();
    const bpm = base ? base * speed : null;
    const meter = grid ? grid.meterAt(pos) : null;
    const changed = Math.abs(speed - 1) > 1e-9;
    const key = `${formatBpm(bpm)}|${formatBpm(base)}|${changed}|${meter}`;
    if (key === this.tempoKey) return;
    this.tempoKey = key;
    const v = this.root.querySelector("#bpm-value");
    const m = this.root.querySelector("#meter");
    const orig = this.root.querySelector("#bpm-orig");
    if (v) v.textContent = formatBpm(bpm);
    if (orig) {
      orig.hidden = !changed || !grid;
      orig.textContent = `元 ${formatBpm(base)}`;
    }
    const box = this.root.querySelector("#tempo");
    if (box) box.classList.toggle("shifted", changed);
    if (this.tempo) this.tempo.syncBpm(bpm);
    if (grid && this.beatEdit) this.beatEdit.syncMeter(meter);
    if (m) {
      m.textContent = grid ? `${meter}/4` : "";
      m.hidden = !grid;
    }
  }

  /** 拍の結果を差し替える（波形・BPM 表示・ボタン）。再生は止めない。 */
  setBeatGrid(grid) {
    this.beatGrid = grid && !grid.empty ? grid : null;
    if (this.wave) this.wave.setBeatGrid(this.beatGrid);
    const box = this.root.querySelector("#tempo");
    if (box) {
      box.classList.toggle("none", !this.beatGrid);
      box.title = this.tempoTitle();
    }
    this.tempoKey = "";
    this.updateTempo(this.engine ? this.engine.position : 0);
    this.renderBeatButton();
    this.renderBarLoop();
    if (this.tempo) this.tempo.refresh();
  }

  renderBeatButton() {
    const btn = this.root.querySelector("#beats-btn");
    if (!btn) return;
    btn.disabled = this.beatBusy;
    btn.textContent = this.beatBusy ? "拍を解析中…" : this.beatGrid ? "拍を再解析" : "拍を解析";
    btn.title = this.beatBusy
      ? "拍・小節の頭を解析しています"
      : "拍・小節の頭を自動で解析します（GPU で数秒）";
  }

  /** 拍の解析（再解析）を依頼し、終わるまで待って表示を差し替える。 */
  async requestBeats() {
    if (this.beatBusy) return;
    if (this.beatGrid) {
      const note = this.beatGrid.edited
        ? "（手で直した拍は、解析の後に「拍の補正」の「元に戻す」で戻せます）" : "";
      const ok = await confirmDialog(`今の拍・小節線を消して、解析し直しますか？${note}`, { ok: "解析し直す" });
      if (!ok || !this.alive) return;
    }
    try {
      // 表示中のジョブで解析する（失敗の警告が表示中のジョブに付くように）
      const res = await api(`/api/tracks/${this.trackId}/beats`, { method: "POST", body: { job_id: this.job.job_id } });
      if (!this.alive) return;
      toast(res.message);
      this.beatJobId = res.job.job_id;
      this.beatBusy = true;
      this.setBeatGrid(null);
      this.later(() => this.pollBeats(), POSTPROCESS_POLL_MS);
    } catch (e) {
      toast(e.message);
    }
  }

  async pollBeats() {
    try {
      const job = await api(`/api/jobs/${this.beatJobId || this.job.job_id}`);
      if (!this.alive) return;
      if (job.postprocess_status === "queued" || job.postprocess_status === "running") {
        this.later(() => this.pollBeats(), POSTPROCESS_POLL_MS);
        return;
      }
      // 警告はジョブごと。画面には表示中のジョブの警告を出す（成功すれば曲の全ジョブで消える）
      const cur = job.job_id === this.job.job_id
        ? job : await api(`/api/jobs/${this.job.job_id}`).catch(() => null);
      if (!this.alive) return;
      if (cur) this.job.beat_warning = cur.beat_warning;
      const beats = await api(`/api/tracks/${this.trackId}/beats`).catch(() => null);
      if (!this.alive) return;
      this.beatBusy = false;
      this.setBeatGrid(beats ? new BeatGrid(beats) : null);
      toast(this.beatGrid ? "拍を解析しました。" : (job.beat_warning || "拍を解析できませんでした。"));
    } catch (e) {
      toast(e.message);
      this.later(() => this.pollBeats(), POSTPROCESS_POLL_MS * 2);
    }
  }

  tempoTitle() {
    if (this.beatGrid) {
      const how = this.beatGrid.edited ? "手動で補正済み" : "自動解析";
      return `再生位置の BPM と拍子（${how}: ${this.beatGrid.analyzer}）`;
    }
    if (this.job.beat_warning) return this.job.beat_warning;
    return "拍はまだ解析されていません";
  }

  async togglePlay() {
    if (!this.ready) return;
    // 省メモリの読み込み中に押した: 読み込みの後に勝手に鳴らさない（押した結果を優先する）
    if (this.tempo) this.tempo.resumeAfterLoad = false;
    if (this.engine.playing) {
      this.engine.pause();
      if (this.sync) this.sync.save();
    } else {
      await this.engine.play();
    }
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
      this.renderBarLoop();
      this.updateTransport();
    }
    this.engine.seek(target);
    if (this.media) this.media.setPosition(this.engine.duration, target, this.engine.speed, true);
  }

  updateTransport() {
    const btn = this.root.querySelector("#play-btn");
    if (btn) {
      const playing = !!(this.engine && this.engine.playing);
      btn.innerHTML = playing ? ICONS.pause : ICONS.play;
      btn.setAttribute("aria-label", playing ? "一時停止" : "再生");
      btn.classList.toggle("playing", playing);
    }
    const loopBtn = this.root.querySelector("#loop-btn");
    if (loopBtn) {
      loopBtn.classList.toggle("on", this.loopOn);
      loopBtn.setAttribute("aria-pressed", String(this.loopOn));
    }
    this.updateMediaSession();
  }

  // --- 選択 -------------------------------------------------------------------

  applySelection(ramp) {
    if (this.engine) {
      this.engine.setGains(S.targetGains(this.tree, this.sel, this.gainsDb), ramp);
    }
    this.trackOffTimes();
    this.ensureStemAudio();
    this.renderSelectionState();
  }

  isAll(sel = this.sel) {
    return S.sameSelection(sel, S.allOn(this.tree));
  }

  setSelection(sel, { gainsDb = null, presetId = null } = {}) {
    if (this.isAll(sel) && !this.isAll()) {
      // 「全部」にする前の組み合わせを覚えておく（0 キーで戻る）
      this.beforeAll = { sel: this.sel, gainsDb: this.gainsDb, presetId: this.activePresetId };
    }
    this.sel = sel;
    if (gainsDb) this.gainsDb = gainsDb;
    this.activePresetId = presetId;
    // ON にし直した stem は、前に読み込めなかったものも読み直す
    for (const k of [...this.stemFailed]) {
      if (sel.has(k.slice(k.indexOf(":") + 1))) this.stemFailed.delete(k);
    }
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

  /** 0 キー: 「全部（元の曲）」と、その直前の組み合わせを行き来する。 */
  toggleAll() {
    if (this.isAll() && this.beforeAll) {
      const { sel, gainsDb, presetId } = this.beforeAll;
      this.beforeAll = null;
      this.setSelection(new Set(sel), { gainsDb: new Map(gainsDb), presetId });
    } else if (!this.isAll()) {
      this.pressAll();
    }
  }

  toggleSolo() {
    this.soloMode = !this.soloMode;
    this.renderSelectionState();
  }

  applyPreset(preset) {
    const { sel, gainsDb } = S.presetToSelection(this.tree, preset, this.groups, this.stemTypes);
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
    if (this.barLoop) return { start: this.barLoop.start, end: this.barLoop.end };
    const cue = this.cues.find((c) => c.cue_id === this.loopCueId);
    return cue && cue.loop_end_sec ? { start: cue.position_sec, end: cue.loop_end_sec } : null;
  }

  applyLoop() {
    if (this.engine) this.engine.setLoop(this.loopOn ? this.activeLoop() : null);
    this.renderCues();
    this.renderBarLoop();
    this.updateTransport();
  }

  /** 再生位置の小節の頭から bars 小節のループを作って鳴らす（キューには保存しない）。 */
  setBarLoop(bars) {
    if (!this.ready) return;
    if (!this.beatGrid) { toast("拍が解析されていないため、小節ループは使えません。"); return; }
    const loop = barLoop(this.beatGrid, this.engine.position, bars, this.engine.duration);
    if (!loop) { toast("ここでは小節ループを作れません。"); return; }
    if (bars >= 1) this.loopBars = bars;
    this.barLoop = loop;
    this.loopOn = true;
    this.applyLoop();
  }

  /** ループの長さを factor 倍（2 または 0.5）にする。小節ループは小節で、キューのループは時間で。 */
  async scaleLoop(factor) {
    if (!this.ready) return;
    if (this.barLoop) {
      const bars = Math.min(MAX_LOOP_BARS, Math.max(MIN_LOOP_BARS, this.barLoop.bars * factor));
      if (bars === this.barLoop.bars || !this.beatGrid) return;
      const loop = barLoop(this.beatGrid, this.barLoop.start, bars, this.engine.duration);
      if (!loop) return;
      this.barLoop = loop;
      this.applyLoop();
      return;
    }
    const cue = this.cues.find((c) => c.cue_id === this.loopCueId && c.loop_end_sec);
    if (!cue) { toast("ループがありません。小節ループのボタンか、キューの「終点」で作ってください。"); return; }
    const len = (cue.loop_end_sec - cue.position_sec) * factor;
    const end = Math.round(Math.min(cue.position_sec + len, this.engine.duration) * 1000) / 1000;
    if (end <= cue.position_sec + 0.05) return;
    if (await this.updateCue(cue, { loop_end_sec: end })) {
      this.applyLoop();
      toast(`キュー「${cue.label || "キュー"}」のループの終点を ${formatTime(end, true)} に変えて保存しました。`);
    }
  }

  toggleSnap(on) {
    this.snap = on;
    try { localStorage.setItem(SNAP_KEY, on ? "1" : "0"); } catch { /* 保存できなくても動く */ }
  }

  /** キューの位置を拍に合わせる（設定が ON で拍があるとき）。 */
  snapTime(t) {
    return this.snap && this.beatGrid ? this.beatGrid.snap(t) : t;
  }

  renderBarLoop() {
    const box = this.root.querySelector("#bar-loops");
    if (box) {
      const cur = this.loopOn && this.barLoop ? this.barLoop.bars : null;
      for (const b of box.querySelectorAll("button[data-bars]")) {
        const on = cur !== null && Number(b.dataset.bars) === cur;
        b.classList.toggle("on", on);
        b.setAttribute("aria-pressed", String(on));
        b.disabled = !this.beatGrid;
      }
      const label = this.root.querySelector("#bar-loop-state");
      const bl = this.barLoop;
      if (label) {
        const bars = bl ? (bl.bars < 1 ? `1/${Math.round(1 / bl.bars)}` : String(bl.bars)) : "";
        label.textContent = bl
          ? `${bars} 小節 ${formatTime(bl.start, true)}〜${formatTime(bl.end, true)}${this.loopOn ? "" : "（停止中）"}`
          : "";
      }
    }
    if (this.beatEdit) this.beatEdit.refresh();
  }

  toggleLoop() {
    if (!this.ready) return;
    if (!this.activeLoop()) {
      const pos = this.engine.position;
      const loops = this.cues.filter((c) => c.loop_end_sec);
      const inside = loops.find((c) => pos >= c.position_sec && pos < c.loop_end_sec);
      const pick = inside || loops[loops.length - 1];
      if (!pick) { toast("ループ区間がありません。小節ループのボタンか、キューの「終点」で作ってください。"); return; }
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
    const pos = Math.round(this.snapTime(this.engine.position) * 1000) / 1000;
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
    const end = Math.round(this.snapTime(this.engine.position) * 1000) / 1000;
    if (end <= cue.position_sec + 0.05) {
      toast("終点はキューより後ろの位置で押してください。");
      return;
    }
    if (await this.updateCue(cue, { loop_end_sec: end })) {
      this.barLoop = null;
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
      this.barLoop = null;
      this.loopCueId = cue.cue_id;
      if (this.engine) this.engine.setLoop(this.loopOn ? this.activeLoop() : null);
    }
    this.seek(cue.position_sec);
    this.renderCues();
    this.renderBarLoop();
  }

  // --- キーボード -------------------------------------------------------------

  handleKey(e) {
    if (e.defaultPrevented) return; // ボタンなどが自分で処理したキー（タップの Space など）
    const tag = (e.target && e.target.tagName) || "";
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || document.querySelector(".modal-back")) return;
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && (e.key === "z" || e.key === "Z")) {
      // 拍の補正を元に戻す（補正パネルを開いているときだけ）
      if (this.beatEdit && this.beatEdit.open && this.beatGrid && this.beatGrid.canUndo) {
        e.preventDefault();
        this.beatEdit.undo();
      }
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.code === "Space" || e.key === " ") {
      e.preventDefault();
      if (!e.repeat) this.togglePlay(); // 押しっぱなしの繰り返しは無視する
    } else if (e.key === "0") {
      e.preventDefault();
      this.toggleAll();
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
    } else if (e.key === "b" || e.key === "B") {
      e.preventDefault();
      this.setBarLoop(this.loopBars);
    } else if (e.key === "[") {
      e.preventDefault();
      this.scaleLoop(0.5);
    } else if (e.key === "]") {
      e.preventDefault();
      this.scaleLoop(2);
    } else if (e.key === "," || e.key === "<") {
      // 速度 −0.1%（Shift で −1%）
      e.preventDefault();
      if (this.tempo) this.tempo.nudge(e.key === "<" ? -COARSE_STEP : -FINE_STEP);
    } else if (e.key === "." || e.key === ">") {
      e.preventDefault();
      if (this.tempo) this.tempo.nudge(e.key === ">" ? COARSE_STEP : FINE_STEP);
    } else if ((e.key === "r" || e.key === "R") && !e.shiftKey) {
      e.preventDefault();
      if (this.tempo) this.tempo.setRatio(1);
    } else if ((e.key === "t" || e.key === "T") && this.beatEdit && this.beatEdit.open && this.beatGrid) {
      // タップは補正パネルを開いているときだけ（うっかり押して拍を置き換えないように）
      e.preventDefault();
      if (!e.repeat) this.beatEdit.tapNow();
    }
  }

  // --- 描画 -------------------------------------------------------------------

  headEl() {
    return el("div", { class: "track-head" },
      el("a", { class: "btn small", href: "#/library", text: "← ライブラリ" }),
      el("h1", { text: this.track.title, title: this.track.title }),
      el("span", { class: "muted", text: this.track.artist || "" }),
      this.tree ? el("button", {
        class: "btn small export-btn", id: "export-btn", type: "button", text: "書き出し",
        title: "stem・全部（ZIP）・今の組み合わせを WAV / FLAC / MP3 で保存します",
        onclick: () => this.openExport(),
      }) : null,
      this.canOpenFolder ? this.folderEl() : null);
  }

  /** 書き出しメニュー（web/js/export.js）。 */
  openExport() {
    if (!this.exporter) this.exporter = new Exporter(this);
    this.exporter.open();
  }

  /** 保存フォルダを開くボタン（サーバーと同じ PC のブラウザのときだけ出す）。 */
  folderEl() {
    const note = "stem は FLAC（24bit）で保存されています";
    return el("div", { class: "folder-box" },
      el("button", {
        class: "btn small", id: "open-folder-btn", type: "button", text: "保存フォルダを開く",
        title: `${note}（ファイル名は stem の名前.flac）。エクスプローラーで開きます。`,
        onclick: () => this.openFolder(),
      }),
      el("span", { class: "folder-note", text: "FLAC 24bit" }));
  }

  async openFolder() {
    try {
      await api(`/api/jobs/${this.job.job_id}/open-folder`, { method: "POST" });
      toast("エクスプローラーで保存フォルダを開きました。");
    } catch (e) { toast(e.message); }
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
      // 動かし終わったらフォーカスを外す（Space・数字キーなどをプレイヤーに戻す）
      onchange: (e) => e.target.blur(),
    });
    const transport = el("section", { class: "panel transport" },
      el("button", {
        class: "play-btn", id: "play-btn", type: "button", disabled: true, "aria-label": "再生",
        onclick: () => this.togglePlay(),
      }),
      el("div", { class: "time" },
        el("span", { id: "time-now", text: formatTime(0, true) }),
        el("span", { class: "muted", text: ` / ${formatTime(duration, true)}` })),
      el("button", {
        class: "btn small restart-btn", id: "restart-btn", type: "button", text: "最初から",
        title: "再生位置を曲の最初に戻します", onclick: (e) => { e.currentTarget.blur(); this.seek(0); },
      }),
      el("div", { class: `tempo${this.beatGrid ? "" : " none"}`, id: "tempo", title: this.tempoTitle() },
        el("span", { class: "bpm", id: "bpm-value", text: "—" }),
        el("span", { class: "unit", text: "BPM" }),
        el("span", { class: "bpm-orig", id: "bpm-orig", hidden: true }),
        el("span", { class: "meter", id: "meter", hidden: true })),
      el("button", {
        class: "btn small", id: "beats-btn", type: "button", onclick: () => this.requestBeats(),
      }),
      el("button", { class: "btn", type: "button", text: `−${SEEK_STEP_SEC}秒`, onclick: () => this.ready && this.seek(this.engine.position - SEEK_STEP_SEC) }),
      el("button", { class: "btn", type: "button", text: `+${SEEK_STEP_SEC}秒`, onclick: () => this.ready && this.seek(this.engine.position + SEEK_STEP_SEC) }),
      el("button", { class: "btn", id: "loop-btn", type: "button", text: "ループ", "aria-pressed": "false", onclick: () => this.toggleLoop() }),
      el("label", { class: "volume" }, el("span", { class: "muted", text: "音量" }), volume),
      this.playInfoEl());

    const stems = el("section", { class: "panel" },
      el("h2", { text: "STEM" }),
      el("div", { class: "stems", id: "stems" }),
      el("div", { class: "mode-row", style: { marginTop: "10px" } },
        el("button", { class: "btn", id: "all-btn", type: "button", text: "全部（元の曲）", title: "全部の stem を鳴らします（0 キーで直前の組み合わせと切り替え）", onclick: () => this.pressAll() }),
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
        el("button", { class: "btn", id: "add-cue-btn", type: "button", text: "＋ 今の位置にキュー", onclick: () => this.addCue() }),
        el("label", { class: "snap-toggle", title: "キューを打つとき・ループの終点を決めるとき、いちばん近い拍に合わせます" },
          el("input", {
            type: "checkbox", id: "snap-toggle", checked: this.snap,
            onchange: (e) => this.toggleSnap(e.target.checked),
          }),
          el("span", { text: "拍に合わせる" }))),
      el("div", { class: "bar-loops", id: "bar-loops" },
        el("span", { class: "muted", text: "小節ループ" }),
        ...BAR_LOOPS.map((n) => el("button", {
          class: "btn small", type: "button", text: String(n), dataset: { bars: String(n) },
          title: `再生位置の小節の頭から ${n} 小節をループします`, "aria-pressed": "false",
          onclick: () => this.setBarLoop(n),
        })),
        el("button", { class: "btn small", type: "button", id: "loop-half", text: "½", title: "ループを半分の長さに（[ キー）。キューのループは保存した終点を書き換えます", onclick: () => this.scaleLoop(0.5) }),
        el("button", { class: "btn small", type: "button", id: "loop-double", text: "×2", title: "ループを倍の長さに（] キー）。キューのループは保存した終点を書き換えます", onclick: () => this.scaleLoop(2) }),
        el("span", { class: "bar-loop-state", id: "bar-loop-state" })));

    const help = el("p", { class: "keys-help" },
      el("kbd", { text: "Space" }), " 再生/停止　", el("kbd", { text: "0" }),
      " 全部（元の曲）⇔ 直前の組み合わせ　", el("kbd", { text: "1" }), "〜", el("kbd", { text: "9" }),
      " stem の ON/OFF（Shift でソロ）　", el("kbd", { text: "←" }), el("kbd", { text: "→" }),
      ` ${SEEK_STEP_SEC}秒戻る/進む　`, el("kbd", { text: "L" }), " ループ　",
      el("kbd", { text: "B" }), " 小節ループ　", el("kbd", { text: "[" }), el("kbd", { text: "]" }),
      " ループ ½/×2　", el("kbd", { text: "T" }), " タップ（拍の補正を開いているとき）　",
      el("kbd", { text: "Ctrl+Z" }), " 拍の補正を元に戻す（拍の補正を開いているとき）　",
      el("kbd", { text: "," }), el("kbd", { text: "." }), " 速度 −/+0.1%（Shift で 1%）　",
      el("kbd", { text: "R" }), " 元の速度に戻す");

    if (this.beatEdit) this.beatEdit.dispose();
    this.beatEdit = new BeatEditPanel(this);
    if (this.tempo) this.tempo.dispose();
    this.tempo = new TempoPanel(this);
    this.root.replaceChildren(el("div", { class: "player" },
      el("div", { class: "player-main" }, this.headEl(), this.jobBarEl(), wave, transport, this.tempo.root,
        this.beatEdit.root, stems),
      el("div", { class: "player-side" }, presets, cues, help)));
    this.tempoKey = "";
    this.updateTempo(0);
    this.renderBeatButton();
    this.updateTransport();
    this.renderStems();
    this.renderGroups();
    this.renderPresets();
    this.renderCues();
    this.renderBarLoop();
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
    if (this.refine) this.refine.decorate(box);
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
      const loadingNow = S.leavesOf(this.tree, b.dataset.code).some((c) => this.stemLoads.has(c));
      b.classList.toggle("loading", loadingNow);
      if (loadingNow) b.setAttribute("aria-busy", "true");
      else b.removeAttribute("aria-busy");
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
    this.updateMediaSession();
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
    if (this.beatEdit) this.beatEdit.refresh();
    if (!this.cues.length) {
      box.replaceChildren(el("li", {}, el("span", { class: "muted", text: "キューはありません。" })));
      return;
    }
    box.replaceChildren(...this.cues.map((c) => {
      const isLoop = !!c.loop_end_sec;
      const active = isLoop && this.loopCueId === c.cue_id && this.loopOn && !this.barLoop;
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

  // --- iPhone 向けの再生（T06b） ----------------------------------------------------------

  /** 続きから再生で保存する今の状態（読み込みが終わる前は null）。 */
  playbackSnapshot() {
    if (!this.ready || !this.engine) return null;
    const gains = {};
    for (const [code, db] of this.gainsDb) {
      const v = Number(db);
      if (Number.isFinite(v) && v !== 0) gains[code] = Math.max(-60, Math.min(24, v));
    }
    return {
      position_sec: Math.round(Math.max(0, this.engine.position) * 1000) / 1000,
      job_id: this.jobId,
      selected: [...this.sel],
      gains_db: gains,
      listen_preset_id: this.activePresetId,
      tempo_ratio: this.tempo ? this.tempo.ratio : null,
      tempo_mode: this.tempo ? this.tempo.mode : null,
    };
  }

  /** ロック画面の曲名と操作。 */
  setupMediaSession() {
    if (this.media) this.media.dispose();
    this.media = new MediaSessionControl({
      play: () => { if (this.engine && !this.engine.playing) this.togglePlay(); },
      pause: () => { if (this.engine && this.engine.playing) this.togglePlay(); },
      seekBy: (d) => { if (this.ready) this.seek(this.engine.position + d); },
      seekTo: (t) => { if (this.ready) this.seek(t); },
    });
    this.updateMediaSession();
  }

  /** 曲名・アーティスト・組み合わせの名前（アルバム欄）と、再生中か・位置を伝える。 */
  updateMediaSession() {
    const m = this.media;
    if (!m || !this.track) return;
    const preset = (this.presets || []).find((p) => p.listen_preset_id === this.activePresetId);
    m.setMetadata({ title: this.track.title, artist: this.track.artist || "", album: preset ? preset.name : "" });
    const e = this.engine;
    if (!e) return;
    m.setPlaying(e.playing);
    m.setPosition(e.duration, e.position, e.speed, true);
  }

  /** OFF にした時刻を覚える（スマホで、しばらくたった stem の音声を捨てるため）。 */
  trackOffTimes() {
    if (!this.lazy || !this.tree) return;
    const now = Date.now();
    for (const c of this.tree.leaves) {
      if (this.sel.has(c)) this.offSince.delete(c);
      else if (!this.offSince.has(c)) this.offSince.set(c, now);
    }
  }

  /** 今鳴らしている音声の組 { key, urls }（速度の変更で作った音声のことがある）。無ければ null。 */
  audioSet() {
    return this.tempo ? this.tempo.audioSet() : null;
  }

  /** スマホ: 選択中で未読み込みの stem を読み込む。 */
  ensureStemAudio() {
    if (!this.lazy || !this.ready || !this.engine || !this.alive) return;
    const set = this.audioSet();
    if (!set || !set.urls) return;
    for (const code of this.sel) {
      const t = this.engine.tracks.get(code);
      if (!t || t.buffer || this.stemLoads.has(code) || !set.urls[code]) continue;
      if (this.stemFailed.has(`${set.key}:${code}`)) continue;
      this.loadStem(code, set);
    }
  }

  /** 読み込み中の stem をやめる（音声の組を切り替えるとき）。 */
  abortStemLoads() {
    for (const entry of this.stemLoads.values()) entry.abort.abort();
    this.stemLoads.clear();
    this.renderSelectionState();
  }

  /** 1 つの stem の音声を読み込み、鳴っているほかの stem と同じ曲の時刻から鳴らし始める。 */
  async loadStem(code, set) {
    const engine = this.engine;
    const abort = new AbortController();
    const onLeave = () => abort.abort();
    this.abort.signal.addEventListener("abort", onLeave);
    const entry = { key: set.key, abort };
    this.stemLoads.set(code, entry);
    this.renderSelectionState();
    try {
      const buf = await fetchBinary(set.urls[code], abort.signal);
      if (abort.signal.aborted) return;
      const decoded = await engine.decode(buf);
      const now = this.audioSet();
      if (abort.signal.aborted || !this.alive || engine !== this.engine || !now || now.key !== set.key) return;
      engine.addBuffer(code, decoded);
    } catch (e) {
      if (e.name !== "AbortError" && this.alive && !abort.signal.aborted) {
        this.stemFailed.add(`${set.key}:${code}`);
        const s = this.tree && this.tree.byCode.get(code);
        toast(`${s ? s.display_name : code} の音声を読み込めませんでした: ${e.message}`);
      }
    } finally {
      this.abort.signal.removeEventListener("abort", onLeave);
      if (this.stemLoads.get(code) === entry) this.stemLoads.delete(code);
      if (this.alive) {
        this.renderSelectionState();
        this.ensureStemAudio(); // 読み込む間に選択や音声の組が変わっていたら合わせる
      }
    }
  }

  /** スマホ: OFF にしてしばらくたった stem（合計が上限を超えたら古いものから）の音声を捨てる。 */
  housekeeping() {
    if (!this.alive) return;
    if (this.lazy && this.engine && this.ready) {
      const now = Date.now();
      const entries = [];
      for (const [code, t] of this.engine.tracks) {
        if (!t.buffer) continue;
        const off = this.sel.has(code) ? null : (this.offSince.get(code) ?? now);
        entries.push({ code, bytes: t.buffer.length * t.buffer.numberOfChannels * 4, offSince: off });
      }
      for (const code of pickEvictions(entries, now)) this.engine.dropBuffer(code);
    }
    this.later(() => this.housekeeping(), HOUSEKEEP_MS);
  }

  /** 鳴らし方・メモリ・端末・ほかの端末の続き（トランスポートの下の小さな行）。 */
  playInfoEl() {
    const routeSel = el("select", {
      class: "select small", id: "route-select", "aria-label": "音の出し方",
      onchange: (e) => { e.target.blur(); saveRouteChoice(e.target.value); this.reloadKeepingState(); },
    },
    el("option", { value: "auto", text: "出し方: 自動" }),
    el("option", { value: "direct", text: "出し方: 通常（Web Audio）" }),
    el("option", { value: "stream", text: "出し方: <audio> 経由" }));
    routeSel.value = loadRouteChoice();
    const lazy = el("input", {
      type: "checkbox", id: "lazy-toggle", checked: this.lazy,
      onchange: (e) => {
        e.target.blur();
        saveLazySetting(e.target.checked);
        this.lazy = e.target.checked;
        this.reloadKeepingState();
      },
    });
    return el("div", { class: "play-info", id: "play-info" },
      el("span", { class: "pi-item pi-route", id: "route-info" }),
      el("span", { class: "pi-item", id: "mem-info", title: "デコードした音声の大きさ（読み込んだ stem の数 / 全部）" }),
      el("button", {
        class: "pi-item pi-btn", type: "button", id: "device-btn", title: "この端末の名前（押すと変えられます）",
        onclick: () => this.renameThisDevice(),
      }),
      el("button", {
        class: "pi-item pi-btn pi-resume", type: "button", id: "resume-other", hidden: true,
        onclick: () => this.resumeFromOther(),
      }),
      el("details", { class: "pi-more" },
        el("summary", { text: "再生の設定" }),
        el("div", { class: "pi-opts" },
          routeSel,
          el("label", {
            class: "snap-toggle",
            title: "選んでいる stem の音声だけを読み込みます（スマホの既定）。メモリを節約できます",
          }, lazy, el("span", { text: "選択中の stem だけ読み込む" })))));
  }

  renderPlayInfo() {
    const r = this.root.querySelector("#route-info");
    if (r && this.route) {
      r.textContent = `出力 ${ROUTE_SHORT[this.route.mode] || this.route.mode}`;
      r.title = this.route.label + (this.route.error ? `（${this.route.error}）` : "");
      r.dataset.route = this.route.mode;
    }
    const d = this.root.querySelector("#device-btn");
    if (d) {
      d.textContent = this.device ? `端末: ${this.device.name}` : "端末: 未登録";
      d.disabled = !this.device;
    }
    const o = this.root.querySelector("#resume-other");
    if (o) {
      o.hidden = !this.otherState;
      if (this.otherState) {
        o.textContent = `${describeOther(this.otherState)} ▸ そこから`;
        o.title = "その位置へ移ります";
      }
    }
    this.memKey = "";
    this.renderMemory();
  }

  /** デコードした音声の大きさ（変わったときだけ書き換える）。 */
  renderMemory() {
    const m = this.root.querySelector("#mem-info");
    if (!m || !this.engine || !this.tree) return;
    const bytes = this.engine.memoryBytes();
    let n = 0;
    for (const c of this.tree.leaves) {
      const t = this.engine.tracks.get(c);
      if (t && t.buffer) n++;
    }
    const key = `${bytes}|${n}|${this.lazy}`;
    if (key === this.memKey) return;
    this.memKey = key;
    m.textContent = `音声 ${formatMB(bytes)}（${n}/${this.tree.leaves.length}${this.lazy ? "・選択中だけ" : ""}）`;
    m.dataset.bytes = String(bytes);
  }

  async renameThisDevice() {
    if (!this.device) return;
    const name = await promptDialog(
      "この端末の名前（ほかの端末で「〇〇 で 1:23 まで聴いた」と出ます）", this.device.name, { ok: "変更" });
    if (!name || name === this.device.name) return;
    try {
      this.device = await renameDevice(name);
      if (this.sync) this.sync.device = this.device;
      this.renderPlayInfo();
    } catch (e) { toast(e.message); }
  }

  /** ほかの端末で最後に聴いていた位置へ移る。 */
  resumeFromOther() {
    const st = this.otherState;
    if (!st || !this.ready) return;
    this.seek(st.position_sec);
    toast(`${st.device_name} で聴いていた位置（${formatTime(st.position_sec)}）へ移りました。`);
  }

  /** 読み直した後の再生にユーザーの操作が要る端末か（スマホ、または iPhone 用の経路）。 */
  needsGestureToPlay() {
    return this.lazy || isCoarsePointer() || !!(this.route && this.route.mode !== "direct");
  }

  /** 設定を変えたとき: 再生位置・選択を保って読み直す（エンジンを作り直す）。 */
  reloadKeepingState() {
    if (this.loading) return;
    if (!this.restore) this.restore = this.captureState();
    this.remount();
  }
}
