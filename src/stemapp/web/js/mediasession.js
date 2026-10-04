// ロック画面・通知の曲名と操作（Media Session。T06b）。
//
// - 曲名・アーティスト・アートワーク（アプリのアイコン）。組み合わせプリセットを選んでいるときは、
//   その名前をアルバム欄に出す（ロック画面では曲名の下に出る）。
// - 操作: 再生・一時停止・10 秒戻る／進む・シーク（seekto）。
// - setPositionState（長さ・位置・速度）は、再生・一時停止・シーク・速度の変更のときと、再生中は
//   1 秒ごとに更新する（速度変更の倍率は playbackRate に入れる）。
// navigator.mediaSession が無いブラウザでは何もしない。

export const SEEK_OFFSET_SEC = 10;
const ARTWORK = [
  { src: "icons/icon-192.png", sizes: "192x192", type: "image/png" },
  { src: "icons/icon-512.png", sizes: "512x512", type: "image/png" },
];
const ACTIONS = ["play", "pause", "stop", "seekbackward", "seekforward", "seekto"];

/** setPositionState に渡す値（範囲の外の値は例外になるので収める）。 */
export function positionStateOf(duration, position, rate) {
  const d = Number.isFinite(duration) && duration > 0 ? duration : 0;
  const p = Math.min(Math.max(0, Number(position) || 0), d);
  const r = Number(rate) > 0 ? Number(rate) : 1;
  return { duration: d, position: p, playbackRate: r };
}

export class MediaSessionControl {
  /**
   * handlers: { play(), pause(), seekBy(deltaSec), seekTo(sec) }。
   * ms: navigator.mediaSession（テストで差し替え可）。
   */
  constructor(handlers, ms = typeof navigator !== "undefined" ? navigator.mediaSession : null) {
    this.ms = ms || null;
    this.handlers = handlers;
    this.installed = [];
    this.meta = null;
    this.lastPosAt = 0;
    if (!this.ms) return;
    const h = handlers;
    const map = {
      play: () => h.play(),
      pause: () => h.pause(),
      stop: () => h.pause(),
      seekbackward: (d) => h.seekBy(-((d && d.seekOffset) || SEEK_OFFSET_SEC)),
      seekforward: (d) => h.seekBy((d && d.seekOffset) || SEEK_OFFSET_SEC),
      seekto: (d) => { if (d && Number.isFinite(d.seekTime)) h.seekTo(d.seekTime); },
    };
    for (const action of ACTIONS) {
      try {
        this.ms.setActionHandler(action, (details) => {
          try { map[action](details); } catch { /* 操作に失敗しても止めない */ }
        });
        this.installed.push(action);
      } catch { /* その操作は使えない */ }
    }
  }

  get available() {
    return !!this.ms;
  }

  /** 曲名・アーティスト・アルバム（組み合わせの名前）。変わったときだけ設定し直す。 */
  setMetadata({ title, artist, album }) {
    if (!this.ms) return;
    const key = JSON.stringify([title, artist, album]);
    if (key === this.meta) return;
    this.meta = key;
    try {
      const Meta = window.MediaMetadata;
      if (!Meta) return;
      const base = typeof location !== "undefined" ? location.href : undefined;
      this.ms.metadata = new Meta({
        title: title || "",
        artist: artist || "",
        album: album || "",
        artwork: ARTWORK.map((a) => ({ ...a, src: base ? new URL(a.src, base).href : a.src })),
      });
    } catch { /* 古い端末 */ }
  }

  setPlaying(playing) {
    if (!this.ms) return;
    try { this.ms.playbackState = playing ? "playing" : "paused"; } catch { /* 古い端末 */ }
  }

  /** 長さ・位置・速度を伝える（force でなければ 1 秒に 1 回まで）。 */
  setPosition(duration, position, rate, force = false) {
    if (!this.ms || typeof this.ms.setPositionState !== "function") return;
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    if (!force && now - this.lastPosAt < 1000) return;
    this.lastPosAt = now;
    const st = positionStateOf(duration, position, rate);
    if (!(st.duration > 0)) return;
    try { this.ms.setPositionState(st); } catch { /* 範囲の外など */ }
  }

  dispose() {
    if (!this.ms) return;
    for (const action of this.installed) {
      try { this.ms.setActionHandler(action, null); } catch { /* 片付けだけ */ }
    }
    this.installed = [];
    try { this.ms.metadata = null; } catch { /* 片付けだけ */ }
    try { this.ms.playbackState = "none"; } catch { /* 片付けだけ */ }
    this.meta = null;
  }
}
