// 続きから再生（PLAYBACK_STATE。T06b）。
//
// - 曲ごと・端末ごとに、再生位置・選択中の stem・音量・組み合わせプリセット・分け方（ジョブ）・
//   速度と方式をサーバーに保存する（5 秒ごと〈変わったときだけ〉、一時停止・画面を離れる・
//   アプリを隠すとき）。画面を離れるときは fetch の keepalive で送る（ページが閉じても届くように）。
// - 開いたときに自分の端末の状態に戻す（位置・選択・組み合わせ・分け方・速度。再生は始めない）。
// - ほかの端末で最後に聴いていた位置は「PC で 1:23 まで聴いた」と出し、押すとその位置へ移る。
// - 自分の端末の状態は、送るたびにブラウザ（localStorage）にも写しを残し、開いたときは写しを優先する。
//   画面を閉じる間際の保存（keepalive）が、開き直したときの読み出しより後にサーバーに届くことがあり、
//   サーバーの値が一つ前のことがあるため（自分の端末の行は自分しか書かないので、写しが常に最新）。

import { api } from "./api.js";
import { ensureDevice } from "./device.js";
import { formatTime } from "./ui.js";

export const SAVE_MS = 5000;
const MIN_OTHER_SEC = 1; // これより前ならほかの端末の位置は出さない
const LOCAL_PREFIX = "stemapp.resume.";

function readLocal(trackId) {
  try {
    const v = JSON.parse(localStorage.getItem(LOCAL_PREFIX + trackId) || "null");
    return v && typeof v === "object" && Number.isFinite(v.position_sec) ? v : null;
  } catch { return null; }
}

function writeLocal(trackId, snap) {
  try { localStorage.setItem(LOCAL_PREFIX + trackId, JSON.stringify(snap)); } catch { /* 無くても動く */ }
}

/** 自分の端末の状態と、ほかの端末の最新の状態（位置が MIN_OTHER_SEC 以上）を選ぶ純粋関数。 */
export function pickStates(states, myDeviceId) {
  const list = states || [];
  const mine = list.find((s) => s.device_id === myDeviceId) || null;
  const other = list.find((s) => s.device_id !== myDeviceId && s.position_sec >= MIN_OTHER_SEC)
    || null;
  return { mine, other };
}

/** 「PC で 1:23 まで聴いた」。 */
export function describeOther(st) {
  return `${st.device_name} で ${formatTime(st.position_sec)} まで聴いた`;
}

/** 保存した状態から、プレイヤーの restore（applyRestore に渡す形）を作る。 */
export function restoreFromState(st) {
  const gains = new Map(Object.entries(st.gains_db || {}).map(([k, v]) => [k, Number(v)]));
  return {
    position: Number(st.position_sec) || 0,
    playing: false,
    sel: new Set(Array.isArray(st.selected) ? st.selected : []),
    // 選択が無い（古い保存・全部 OFF）ときは applyRestore が全部 ON にする
    gainsDb: gains,
    presetId: st.listen_preset_id ?? null,
    beforeAll: null,
    soloMode: false,
    loopCueId: null,
    loopOn: false,
    barLoop: null,
    zoom: null,
    fromServer: true,
  };
}

export class PlaybackSync {
  /** view: PlayerView（trackId, playbackSnapshot()）。 */
  constructor(view) {
    this.view = view;
    this.device = null;
    this.lastKey = "";
    this.timer = 0;
    this.disposed = false;
    this.onHide = () => {
      if (document.visibilityState === "hidden") this.save({ keepalive: true });
    };
    this.onPageHide = () => this.save({ keepalive: true });
  }

  /** この端末を登録し、この曲の状態を読む。{ device, mine, other }（失敗したら null の項目）。 */
  async load() {
    try {
      this.device = await ensureDevice();
    } catch {
      this.device = null;
      return { device: null, mine: null, other: null };
    }
    let states = [];
    try {
      states = (await api(`/api/tracks/${this.view.trackId}/playback`)).states;
    } catch { /* 読めなくても再生はできる */ }
    const picked = pickStates(states, this.device.device_id);
    const local = readLocal(this.view.trackId);
    if (local) picked.mine = { ...(picked.mine || {}), ...local, device_id: this.device.device_id };
    return { device: this.device, ...picked };
  }

  /** 保存を始める（一定間隔と、画面を隠す・離れるとき）。 */
  start() {
    if (this.timer || this.disposed) return;
    this.timer = setInterval(() => this.save(), SAVE_MS);
    document.addEventListener("visibilitychange", this.onHide);
    window.addEventListener("pagehide", this.onPageHide);
  }

  /** 今の状態を保存する（前回と同じなら送らない）。 */
  save({ keepalive = false, force = false } = {}) {
    if (!this.device) return null;
    const snap = this.view.playbackSnapshot();
    if (!snap) return null;
    const key = JSON.stringify({ ...snap, position_sec: Math.round(snap.position_sec * 10) / 10 });
    if (key === this.lastKey && !force) return null;
    this.lastKey = key;
    writeLocal(this.view.trackId, snap);
    const url = `/api/tracks/${this.view.trackId}/playback/${this.device.device_id}`;
    return fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(snap),
      keepalive,
    }).then((res) => {
      if (!res.ok) this.lastKey = ""; // 次の機会にもう一度送る
      return res.ok;
    }).catch(() => {
      this.lastKey = "";
      return false;
    });
  }

  /** 止める（最後に一度保存する）。 */
  dispose() {
    if (this.disposed) return;
    this.save({ keepalive: true });
    this.disposed = true;
    clearInterval(this.timer);
    this.timer = 0;
    document.removeEventListener("visibilitychange", this.onHide);
    window.removeEventListener("pagehide", this.onPageHide);
  }
}
