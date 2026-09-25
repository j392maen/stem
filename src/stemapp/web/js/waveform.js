// DJ 風の波形表示（Canvas）。
// - 概観: 曲全体。選択中の stem を stem 色で重ね、非選択は暗い灰色。再生済みの部分は暗くする。
// - 拡大: 再生位置を中央に固定して流れる。表示の秒数は変えられる（zoomSeconds）。
// 解像度（samples_per_px）は表示の幅に合わせて選ぶ（peaks.js）。

import { chooseOverviewLevel, chooseZoomLevel, rangeMinMax } from "./peaks.js";

const COLORS = {
  bg: "#0d0d11",
  grid: "#1d1d24",
  dim: "#34343e",
  played: "rgba(11, 11, 14, 0.55)",
  playhead: "#ff3b4e",
  loop: "rgba(158, 27, 44, 0.32)",
  loopEdge: "#9e1b2c",
  center: "rgba(255, 255, 255, 0.06)",
};

export const ZOOM_STEPS = [2, 4, 8, 16, 32];

function hexToRgba(hex, alpha) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return `rgba(160,160,170,${alpha})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

/**
 * 表示の縦の倍率。いちばん大きい stem の振れ幅が高さの 95% になるようにする（最大 8 倍）。
 * すべての stem に同じ倍率をかけるので、stem 同士の大きさの比は変わらない。
 */
export function displayScale(layers, spp) {
  let peak = 0;
  for (const layer of layers) {
    const pk = layer.peaks.get(spp);
    if (!pk) continue;
    for (let i = 0; i < pk.data.length; i++) peak = Math.max(peak, Math.abs(pk.data[i]));
  }
  if (peak <= 0) return 1;
  return Math.min(8, 0.95 / (peak / 127));
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

export class WaveformView {
  /**
   * layers: [{ code, color, peaks: Map(spp → parsed) }]（葉の stem だけ。表示順）
   * getState(): { position, duration, sel: Set, cues: [], loop: {start,end}|null, preview: 秒|null }
   * onSeek(sec): シークの確定
   */
  constructor({ overview, zoom, layers, sampleRate, totalSamples, getState, onSeek }) {
    this.overview = overview;
    this.zoom = zoom;
    this.layers = layers;
    this.sampleRate = sampleRate;
    this.totalSamples = totalSamples;
    this.getState = getState;
    this.onSeek = onSeek;
    this.zoomSeconds = 8;
    this.overviewCache = document.createElement("canvas");
    this.overviewKey = "";
    this.preview = null; // ドラッグ中の位置（秒）
    this.levels = [...new Set(layers.flatMap((l) => [...l.peaks.keys()]))].sort((a, b) => a - b);
    this.scale = displayScale(layers, this.levels[this.levels.length - 1]);
    this._bindPointer();
  }

  get duration() {
    return this.totalSamples / this.sampleRate;
  }

  setZoom(seconds) {
    this.zoomSeconds = seconds;
  }

  // --- 描画 ------------------------------------------------------------------

  _drawLayers(ctx, width, height, sel, timeAt, spp) {
    const mid = height / 2;
    const amp = (height / 2 - 2) * this.scale;
    const draw = (layer, style) => {
      const pk = layer.peaks.get(spp);
      if (!pk) return;
      ctx.fillStyle = style;
      const ptsPerSec = this.sampleRate / pk.samplesPerPx;
      for (let x = 0; x < width; x++) {
        const t0 = timeAt(x);
        const t1 = timeAt(x + 1);
        if (t1 <= 0 || t0 >= this.duration) continue;
        const [mn, mx] = rangeMinMax(pk, t0 * ptsPerSec, t1 * ptsPerSec);
        const y0 = Math.max(0, mid - mx * amp);
        const y1 = Math.min(height, mid - mn * amp);
        ctx.fillRect(x, y0, 1, Math.max(1, y1 - y0));
      }
    };
    for (const layer of this.layers) if (!sel.has(layer.code)) draw(layer, COLORS.dim);
    for (const layer of this.layers) if (sel.has(layer.code)) draw(layer, hexToRgba(layer.color, 0.78));
  }

  _renderOverviewCache(sel) {
    const c = this.overviewCache;
    c.width = this.overview.width;
    c.height = this.overview.height;
    const ctx = c.getContext("2d");
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, c.width, c.height);
    const spp = chooseOverviewLevel(this.levels, this.totalSamples, c.width);
    const secPerPx = this.duration / c.width;
    this._drawLayers(ctx, c.width, c.height, sel, (x) => x * secPerPx, spp);
  }

  _drawMarkers(ctx, width, height, xOf, state) {
    const { loop, cues } = state;
    if (loop) {
      const x0 = xOf(loop.start);
      const x1 = xOf(loop.end);
      ctx.fillStyle = COLORS.loop;
      ctx.fillRect(x0, 0, x1 - x0, height);
      ctx.fillStyle = COLORS.loopEdge;
      ctx.fillRect(x0, 0, 2, height);
      ctx.fillRect(x1 - 2, 0, 2, height);
    }
    const dpr = window.devicePixelRatio || 1;
    for (const cue of cues || []) {
      const x = xOf(cue.position_sec);
      if (x < -10 || x > width + 10) continue;
      ctx.fillStyle = cue.color || "#f5f5f4";
      ctx.fillRect(x - dpr / 2, 0, dpr, height);
      ctx.beginPath();
      ctx.moveTo(x - 5 * dpr, 0);
      ctx.lineTo(x + 5 * dpr, 0);
      ctx.lineTo(x, 7 * dpr);
      ctx.closePath();
      ctx.fill();
    }
  }

  draw() {
    const state = this.getState();
    const pos = this.preview ?? state.position;
    const resized = fitCanvas(this.overview) | fitCanvas(this.zoom);
    const key = `${this.overview.width}x${this.overview.height}|${[...state.sel].sort().join(",")}`;
    if (resized || key !== this.overviewKey) {
      this._renderOverviewCache(state.sel);
      this.overviewKey = key;
    }

    // 概観
    {
      const ctx = this.overview.getContext("2d");
      const w = this.overview.width;
      const h = this.overview.height;
      ctx.drawImage(this.overviewCache, 0, 0);
      const xOf = (t) => (t / this.duration) * w;
      ctx.fillStyle = COLORS.played;
      ctx.fillRect(0, 0, xOf(pos), h);
      this._drawMarkers(ctx, w, h, xOf, state);
      ctx.fillStyle = COLORS.playhead;
      const dpr = window.devicePixelRatio || 1;
      ctx.fillRect(Math.round(xOf(pos) - dpr), 0, 2 * dpr, h);
    }

    // 拡大（再生位置を中央に固定）
    {
      const ctx = this.zoom.getContext("2d");
      const w = this.zoom.width;
      const h = this.zoom.height;
      ctx.fillStyle = COLORS.bg;
      ctx.fillRect(0, 0, w, h);
      const secPerPx = this.zoomSeconds / w;
      const t0 = pos - (w / 2) * secPerPx;
      const timeAt = (x) => t0 + x * secPerPx;
      const xOf = (t) => (t - t0) / secPerPx;
      // 1秒ごとの目盛り
      ctx.fillStyle = COLORS.grid;
      const step = this.zoomSeconds > 16 ? 5 : 1;
      for (let s = Math.ceil(t0 / step) * step; s < timeAt(w); s += step) {
        if (s >= 0 && s <= this.duration) ctx.fillRect(Math.round(xOf(s)), 0, 1, h);
      }
      ctx.fillStyle = COLORS.center;
      ctx.fillRect(0, h / 2, w, 1);
      this._drawMarkers(ctx, w, h, xOf, state);
      const spp = chooseZoomLevel(this.levels, secPerPx * this.sampleRate);
      this._drawLayers(ctx, w, h, state.sel, timeAt, spp);
      const dpr = window.devicePixelRatio || 1;
      ctx.fillStyle = COLORS.playhead;
      ctx.fillRect(Math.round(w / 2 - dpr), 0, 2 * dpr, h);
    }
  }

  // --- 操作（クリック・ドラッグでシーク） ---------------------------------------

  _bindPointer() {
    const cssX = (canvas, e) => {
      const rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, w: rect.width };
    };
    // 概観: 押した位置・ドラッグした位置へ。離したときに確定する
    this.overview.addEventListener("pointerdown", (e) => {
      const canvas = this.overview;
      canvas.setPointerCapture(e.pointerId);
      const at = (ev) => {
        const { x, w } = cssX(canvas, ev);
        return Math.min(Math.max(0, x / w), 1) * this.duration;
      };
      this.preview = at(e);
      const move = (ev) => { this.preview = at(ev); };
      const up = (ev) => {
        canvas.removeEventListener("pointermove", move);
        canvas.removeEventListener("pointerup", up);
        canvas.removeEventListener("pointercancel", up);
        const t = at(ev);
        this.preview = null;
        this.onSeek(t);
      };
      canvas.addEventListener("pointermove", move);
      canvas.addEventListener("pointerup", up);
      canvas.addEventListener("pointercancel", up);
    });
    // 拡大: ドラッグで前後に動かす（DJ のジョグのように）。動かさずに離したらその位置へ
    this.zoom.addEventListener("pointerdown", (e) => {
      const canvas = this.zoom;
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add("dragging");
      const start = cssX(canvas, e);
      const base = this.getState().position;
      let moved = false;
      const secPerCss = () => this.zoomSeconds / cssX(canvas, e).w;
      const move = (ev) => {
        const dx = cssX(canvas, ev).x - start.x;
        if (Math.abs(dx) > 3) moved = true;
        if (moved) this.preview = Math.min(Math.max(0, base - dx * secPerCss()), this.duration);
      };
      const up = (ev) => {
        canvas.removeEventListener("pointermove", move);
        canvas.removeEventListener("pointerup", up);
        canvas.removeEventListener("pointercancel", up);
        canvas.classList.remove("dragging");
        let t;
        if (moved) t = this.preview ?? base;
        else {
          const { x, w } = cssX(canvas, ev);
          t = base + (x - w / 2) * secPerCss();
        }
        this.preview = null;
        this.onSeek(Math.min(Math.max(0, t), this.duration));
      };
      canvas.addEventListener("pointermove", move);
      canvas.addEventListener("pointerup", up);
      canvas.addEventListener("pointercancel", up);
    });
  }
}
