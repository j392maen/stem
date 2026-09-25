// DJ 風の波形表示（Canvas）。
// - 概観: 曲全体。選択中の stem を stem 色で重ね、非選択は暗い灰色。再生済みの部分は暗くする。
// - 拡大: 再生位置を中央に固定して流れる。表示の秒数は変えられる（zoomSeconds）。
// 解像度（samples_per_px）は表示の幅に合わせて選ぶ（peaks.js）。
// 拍が解析済みなら、拡大の目盛りを拍の線（細く暗め）と小節線（やや太く明るめ＋小節番号）にし、
// 概観にも小節線を間引いて描く。拍が無ければ拡大は1秒目盛りのまま。
// 線は stem の波形より先に描き（波形が上）、差し色の赤（再生位置・ループ）は使わない。

import { thinStep } from "./beats.js";
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
  beat: "#202029",
  bar: "rgba(196, 198, 214, 0.30)",
  barNumber: "rgba(196, 198, 214, 0.62)",
  overviewBar: "rgba(196, 198, 214, 0.16)",
};

// 線どうしの最小の間隔（CSS px）。これより狭くなるなら間引く
const MIN_BEAT_GAP_PX = 6;
const MIN_BAR_NUMBER_GAP_PX = 34;
const MIN_OVERVIEW_BAR_GAP_PX = 22;
// 概観の小節線の間引きの単位（小節）
const OVERVIEW_BAR_STEPS = [8, 16, 32, 64, 128];

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
   * beatGrid: BeatGrid（beats.js）。無ければ null（1秒目盛り）
   */
  constructor({ overview, zoom, layers, sampleRate, totalSamples, getState, onSeek, beatGrid = null }) {
    this.overview = overview;
    this.zoom = zoom;
    this.layers = layers;
    this.sampleRate = sampleRate;
    this.totalSamples = totalSamples;
    this.getState = getState;
    this.onSeek = onSeek;
    this.beatGrid = beatGrid && !beatGrid.empty ? beatGrid : null;
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

  setBeatGrid(grid) {
    this.beatGrid = grid && !grid.empty ? grid : null;
    this.overviewKey = ""; // 概観を描き直す
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
    this.overviewBarStep = this.beatGrid ? this._drawOverviewBars(ctx, c.width, c.height, secPerPx) : 0;
    this._drawLayers(ctx, c.width, c.height, sel, (x) => x * secPerPx, spp);
  }

  /** 概観の小節線（8・16・32… 小節ごと。表示幅に応じて間引く）。間引きの単位を返す。 */
  _drawOverviewBars(ctx, width, height, secPerPx) {
    const grid = this.beatGrid;
    const dpr = window.devicePixelRatio || 1;
    const step = thinStep(grid.meanBarSec / secPerPx, MIN_OVERVIEW_BAR_GAP_PX * dpr, OVERVIEW_BAR_STEPS);
    ctx.fillStyle = COLORS.overviewBar;
    const w = Math.max(1, Math.round(dpr));
    for (let i = 0; i < grid.downbeats.length; i += step) {
      ctx.fillRect(Math.round(grid.downbeats[i] / secPerPx), 0, w, height);
    }
    return step;
  }

  /** 拡大の拍の線と小節線（波形の下に描く）。小節番号は波形の後に描くので位置を返す。 */
  _drawBeatGrid(ctx, width, height, t0, t1, xOf, secPerPx) {
    const grid = this.beatGrid;
    const dpr = window.devicePixelRatio || 1;
    // 拍の線: 間隔が狭すぎるときは描かない（小節線だけにする）
    if (grid.meanBeatSec / secPerPx >= MIN_BEAT_GAP_PX * dpr) {
      ctx.fillStyle = COLORS.beat;
      const [b0, b1] = grid.beatRange(t0, t1);
      for (let i = b0; i < b1; i++) ctx.fillRect(Math.round(xOf(grid.beats[i])), 0, Math.max(1, Math.round(dpr)), height);
    }
    const [d0, d1] = grid.barRange(t0, t1);
    const barW = Math.max(2, Math.round(1.5 * dpr));
    const numberStep = thinStep(grid.meanBarSec / secPerPx, MIN_BAR_NUMBER_GAP_PX * dpr);
    ctx.fillStyle = COLORS.bar;
    const labels = [];
    for (let i = d0; i < d1; i++) {
      const x = Math.round(xOf(grid.downbeats[i]) - barW / 2);
      ctx.fillRect(x, 0, barW, height);
      if (i % numberStep === 0) labels.push([x + barW + 3 * dpr, i + 1]);
    }
    this.zoom.dataset.bars = String(d1 - d0);
    return labels;
  }

  _drawBarNumbers(ctx, labels) {
    if (!labels.length) return;
    const dpr = window.devicePixelRatio || 1;
    ctx.fillStyle = COLORS.barNumber;
    ctx.font = `600 ${10 * dpr}px system-ui, "Segoe UI", sans-serif`;
    ctx.textBaseline = "top";
    for (const [x, n] of labels) ctx.fillText(String(n), x, 3 * dpr);
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
      let labels = [];
      if (this.beatGrid) {
        // 拍の線と小節線
        labels = this._drawBeatGrid(ctx, w, h, t0, timeAt(w), xOf, secPerPx);
        this.zoom.dataset.grid = "beats";
      } else {
        // 1秒ごとの目盛り
        ctx.fillStyle = COLORS.grid;
        const step = this.zoomSeconds > 16 ? 5 : 1;
        for (let s = Math.ceil(t0 / step) * step; s < timeAt(w); s += step) {
          if (s >= 0 && s <= this.duration) ctx.fillRect(Math.round(xOf(s)), 0, 1, h);
        }
        this.zoom.dataset.grid = "seconds";
      }
      ctx.fillStyle = COLORS.center;
      ctx.fillRect(0, h / 2, w, 1);
      this._drawMarkers(ctx, w, h, xOf, state);
      const spp = chooseZoomLevel(this.levels, secPerPx * this.sampleRate);
      this._drawLayers(ctx, w, h, state.sel, timeAt, spp);
      this._drawBarNumbers(ctx, labels);
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
