// 音の出し方（音声セッション。T06b）。
//
// iPhone では Web Audio が消音モード（本体の消音スイッチ）で鳴らず、画面ロックで止まる。
// T06b-0 の実機の実験（iPhone の Chrome、iOS 26.6.2）で次の 2 つが効いた:
//   A（session）: navigator.audioSession.type = "playback" にしてから Web Audio で鳴らす
//   C（stream） : Web Audio の最終出力を createMediaStreamDestination() に流し、<audio>.srcObject で鳴らす
// そこで、audioSession がある端末は A、無いスマホ・タブレット（ポインタが粗い）は C、PC はそのまま
// （direct）にする。A は「再生」の種類を宣言するだけなので、PC のブラウザに有っても付けて害は無い
// （音の経路は direct と同じ）。設定（localStorage）で direct / stream に固定もできる（確認用）。
//
// どちらも「ユーザーの操作の中」で始める必要がある（iPhone）: Engine.play() の最初（await の前）で
// beforePlay() を呼ぶ。C の <audio> は一時停止で止め（ロック画面の表示を「停止中」にする）、
// 再生で鳴らし直す。C は MediaStream を通る分だけ音が遅れて聞こえるので、その遅れ（routeLatency）を
// 再生位置の計算（聞こえている位置）に入れる。

export const ROUTE_KEY = "stemapp.audioRoute"; // auto / direct / stream
export const ROUTE_CHOICES = ["auto", "direct", "stream"];

// C の経路で増える遅れ（秒）。MediaStream を通して同じ AudioContext に戻した音の遅れを Edge で測った値
// （約 20.7ms。tests/test_browser_iphone.py の measureStreamLoopback）。<audio> 要素自身の出力の遅れは
// ブラウザの中からは測れないため、MediaStream の受け渡しの遅れで代わりにする（iPhone では違うことがある）。
export const STREAM_ROUTE_LATENCY_SEC = 0.021;

export const ROUTE_LABELS = {
  session: "Web Audio＋audioSession（消音モード・ロック中も再生）",
  stream: "Web Audio → <audio>（消音モード・ロック中も再生）",
  direct: "Web Audio（そのまま）",
};
export const ROUTE_SHORT = { session: "A: audioSession", stream: "C: <audio> 経由", direct: "通常" };

function mq(q) {
  return typeof window !== "undefined" && !!window.matchMedia && window.matchMedia(q).matches;
}

/** ポインタが粗い（指で触る）画面か。 */
export function isCoarsePointer() {
  return mq("(pointer: coarse)");
}

/** スマホ（指で触り、幅 640px 以下）か。 */
export function isPhoneScreen() {
  return isCoarsePointer() && mq("(max-width: 640px)");
}

export function loadRouteChoice() {
  try {
    const v = localStorage.getItem(ROUTE_KEY);
    return ROUTE_CHOICES.includes(v) ? v : "auto";
  } catch { return "auto"; }
}

export function saveRouteChoice(value) {
  try {
    if (value === "auto") localStorage.removeItem(ROUTE_KEY);
    else localStorage.setItem(ROUTE_KEY, value);
  } catch { /* 保存できなくても動く */ }
}

/**
 * 鳴らし方を決める純粋関数。choice: auto / direct / stream、hasSession: navigator.audioSession がある、
 * hasStream: createMediaStreamDestination がある、coarse: 指で触る端末。
 */
export function chooseRoute({ choice = "auto", hasSession = false, hasStream = false, coarse = false } = {}) {
  if (choice === "direct") return "direct";
  if (choice === "stream") return hasStream ? "stream" : (hasSession ? "session" : "direct");
  if (hasSession) return "session";
  if (coarse && hasStream) return "stream";
  return "direct";
}

/** 今の端末の鳴らし方。 */
export function routeForThisDevice(ctx, choice = loadRouteChoice()) {
  return chooseRoute({
    choice,
    hasSession: typeof navigator !== "undefined" && !!navigator.audioSession,
    hasStream: !!ctx && typeof ctx.createMediaStreamDestination === "function",
    coarse: isCoarsePointer(),
  });
}

/** エンジンの最終出力のつなぎ方と、再生・一時停止のときの処理。 */
export class AudioRoute {
  /** engine: Engine（ctx・master を持つ）。mode: session / stream / direct。 */
  constructor(engine, mode) {
    this.engine = engine;
    this.mode = mode;
    this.audio = null;
    this.dest = null;
    this.error = null; // 失敗したときの理由（表示用）
    this.sessionType = null;
    this.latency = 0;
    if (mode === "stream") this._attachStream();
  }

  _attachStream() {
    const { ctx, master } = this.engine;
    try {
      this.dest = ctx.createMediaStreamDestination();
      const audio = new Audio();
      audio.setAttribute("playsinline", "");
      audio.autoplay = false;
      audio.srcObject = this.dest.stream;
      master.disconnect();
      master.connect(this.dest);
      this.audio = audio;
      this.latency = STREAM_ROUTE_LATENCY_SEC;
    } catch (e) {
      // 使えなければそのまま鳴らす
      try { master.disconnect(); } catch { /* つないでいない */ }
      master.connect(ctx.destination);
      this.mode = "direct";
      this.dest = null;
      this.audio = null;
      this.error = `${e.name || "Error"}: ${e.message || e}`;
    }
  }

  get label() {
    return ROUTE_LABELS[this.mode] || this.mode;
  }

  /** 再生を始める直前（ユーザーの操作の中で、await の前に呼ぶ）。 */
  beforePlay() {
    if (this.mode === "session") {
      try {
        const s = navigator.audioSession;
        if (s && s.type !== "playback") s.type = "playback";
        this.sessionType = s ? s.type : null;
      } catch (e) {
        this.error = `${e.name || "Error"}: ${e.message || e}`;
      }
    } else if (this.mode === "stream" && this.audio) {
      const p = this.audio.play();
      if (p && p.catch) {
        p.catch((e) => { if (e && e.name !== "AbortError") this.error = `${e.name}: ${e.message}`; });
      }
    }
  }

  /** 一時停止した後（フェードアウトが終わる時刻 untilCtx の後に <audio> を止める）。 */
  afterPause(untilCtx = 0) {
    if (this.mode !== "stream" || !this.audio) return;
    const ms = Math.max(0, (untilCtx - this.engine.ctx.currentTime) * 1000) + 30;
    const audio = this.audio;
    setTimeout(() => {
      if (!this.engine.playing && !this.engine._wantPlay) audio.pause();
    }, ms);
  }

  /** <audio> が鳴っているか（確認用）。direct・session は null。 */
  get elementPlaying() {
    return this.audio ? !this.audio.paused : null;
  }

  close() {
    if (this.audio) {
      try { this.audio.pause(); } catch { /* 片付けだけ */ }
      this.audio.srcObject = null;
    }
    this.audio = null;
  }
}

/**
 * MediaStream を通った音の遅れ（秒）を測る（テスト・確認用）。ctx の中でクリックを鳴らし、
 * そのままの音と、createMediaStreamDestination → createMediaStreamSource を通した音を
 * 左右に分けて録り、立ち上がりの差を数える。スピーカーには出さない。
 */
export async function measureStreamLoopback(ctx, { seconds = 1.5 } = {}) {
  const sr = ctx.sampleRate;
  const dest = ctx.createMediaStreamDestination();
  const back = ctx.createMediaStreamSource(dest.stream);
  const merger = ctx.createChannelMerger(2);
  const proc = ctx.createScriptProcessor(1024, 2, 2);
  const mute = ctx.createGain();
  mute.gain.value = 0;
  const click = ctx.createBuffer(1, Math.round(sr * 0.05), sr);
  click.getChannelData(0).fill(0.9);
  const src = ctx.createBufferSource();
  src.buffer = click;
  src.connect(merger, 0, 0);
  src.connect(dest);
  back.connect(merger, 0, 1);
  merger.connect(proc);
  proc.connect(mute);
  mute.connect(ctx.destination);
  const left = [];
  const right = [];
  proc.onaudioprocess = (ev) => {
    left.push(Float32Array.from(ev.inputBuffer.getChannelData(0)));
    right.push(Float32Array.from(ev.inputBuffer.getChannelData(1)));
  };
  if (ctx.state !== "running") await ctx.resume();
  src.start(ctx.currentTime + 0.3);
  await new Promise((r) => setTimeout(r, seconds * 1000));
  proc.onaudioprocess = null;
  for (const n of [src, back, merger, proc, mute]) { try { n.disconnect(); } catch { /* 片付け */ } }
  const onset = (chunks) => {
    let i = 0;
    for (const c of chunks) {
      for (let k = 0; k < c.length; k++, i++) if (Math.abs(c[k]) > 0.1) return i;
    }
    return -1;
  };
  const a = onset(left);
  const b = onset(right);
  if (a < 0 || b < 0) return null;
  return (b - a) / sr;
}
