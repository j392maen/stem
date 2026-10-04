// スマホでは選んでいる stem だけ読み込む（T06b。SPEC 7 章）。
//
// 1 stem・4 分でデコード後に約 85MB（44.1kHz・2ch・32bit float）になるため、iPhone で全 stem を
// 読み込むとメモリが足りなくなる。スマホ（指で触り、幅 640px 以下）か設定で選んだときは:
// - 選択中の葉 stem の音声だけを読み込む（波形の peaks は小さいので全部読む）。
// - ON にした stem が未読み込みなら、読み込んでから同じ曲の時刻で鳴らし始める（Engine.addBuffer）。
// - OFF にした stem の音声はしばらく（KEEP_MS）残してから捨てる。合計が上限（LIMIT_BYTES）を超えたら、
//   OFF にした時刻の古い順に、上限を下回るまで捨てる（選択中のものは捨てない）。
// ここには DOM・音声に触れない純粋な処理と設定だけを置く（読み込みは player.js）。

import { isPhoneScreen } from "./audioroute.js";

export const LAZY_KEY = "stemapp.lazyStems"; // "1" / "0"（無ければ画面から決める）
export const KEEP_MS = 2 * 60 * 1000;
export const LIMIT_BYTES = 400 * 1024 * 1024;

/** 選択中の stem だけ読み込むか（設定があればそれ、無ければスマホなら ON）。 */
export function loadLazySetting() {
  try {
    const v = localStorage.getItem(LAZY_KEY);
    if (v === "1" || v === "0") return v === "1";
  } catch { /* 読めなくても動く */ }
  return isPhoneScreen();
}

export function saveLazySetting(on) {
  try { localStorage.setItem(LAZY_KEY, on ? "1" : "0"); } catch { /* 保存できなくても動く */ }
}

/**
 * 捨てる stem を選ぶ純粋関数。entries: [{ code, bytes, offSince }]（offSince: OFF にした時刻 ms。
 * 選択中なら null）。OFF にしてから keepMs 以上たったものと、合計が limitBytes を超える間は
 * OFF の古いものから順に。
 */
export function pickEvictions(entries, now, { keepMs = KEEP_MS, limitBytes = LIMIT_BYTES } = {}) {
  const out = [];
  let total = entries.reduce((n, e) => n + (e.bytes || 0), 0);
  const off = entries.filter((e) => e.offSince !== null && e.offSince !== undefined)
    .sort((a, b) => a.offSince - b.offSince);
  for (const e of off) {
    if (now - e.offSince >= keepMs || total > limitBytes) {
      out.push(e.code);
      total -= e.bytes || 0;
    }
  }
  return out;
}

/** バイト数を「85 MB」にする。 */
export function formatMB(bytes) {
  const mb = (Number(bytes) || 0) / (1024 * 1024);
  return `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`;
}
