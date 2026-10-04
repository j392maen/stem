// この端末（ブラウザ）の登録（DEVICE。T06b）。
//
// 端末ごとの ID はブラウザが作るランダムな文字列で、localStorage に保存する（サイトのデータを消すと
// 別の端末として登録し直す）。名前は種類から自動で付け（「iPhone」「PC」など）、プレイヤーで変えられる。

import { api } from "./api.js";

const KEY = "stemapp.device.key";

function randomKey() {
  try {
    if (crypto && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  } catch { /* 古いブラウザ */ }
  let s = "";
  for (let i = 0; i < 32; i++) s += Math.floor(Math.random() * 16).toString(16);
  return s;
}

/** この端末の ID（無ければ作って保存する。保存できないときはこの画面の間だけの ID）。 */
let memoryKey = null;
export function deviceKey() {
  try {
    let k = localStorage.getItem(KEY);
    if (!k || !/^[A-Za-z0-9_-]{8,64}$/.test(k)) {
      k = randomKey();
      localStorage.setItem(KEY, k);
    }
    return k;
  } catch {
    if (!memoryKey) memoryKey = randomKey();
    return memoryKey;
  }
}

/** 端末の種類を UA などから推定する（pc / iphone / ipad / other）。 */
export function guessKind(ua = navigator.userAgent || "", touchPoints = navigator.maxTouchPoints || 0) {
  if (/iPhone|iPod/.test(ua)) return "iphone";
  if (/iPad/.test(ua) || (/Macintosh/.test(ua) && touchPoints > 1)) return "ipad";
  if (/Android|Mobile/.test(ua)) return "other";
  if (/Windows|Macintosh|Linux|CrOS/.test(ua)) return "pc";
  return "other";
}

let registering = null;

/** この端末を登録して { device_id, name, kind } を返す（1 回だけ。失敗したら次に呼んだとき再試行）。 */
export function ensureDevice() {
  if (!registering) {
    registering = api("/api/devices", {
      method: "POST", body: { device_key: deviceKey(), kind: guessKind() },
    }).catch((e) => {
      registering = null;
      throw e;
    });
  }
  return registering;
}

/** 端末の名前を変える。 */
export async function renameDevice(name) {
  const me = await ensureDevice();
  const res = await api(`/api/devices/${me.device_id}`, { method: "PUT", body: { name } });
  registering = Promise.resolve(res);
  return res;
}
