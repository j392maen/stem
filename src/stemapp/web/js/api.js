// API の呼び出し。401 を受けたらログイン画面を出す（onUnauthorized に登録した関数を呼ぶ）。

let unauthorizedHandler = null;

export function onUnauthorized(fn) {
  unauthorizedHandler = fn;
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function errorMessage(res) {
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    // 本文が JSON でない
  }
  return `エラーが起きました（${res.status}）。`;
}

/** JSON API を呼ぶ。body がオブジェクトなら JSON、FormData ならそのまま送る。 */
export async function api(path, { method = "GET", body, signal } = {}) {
  const opts = { method, headers: {}, credentials: "same-origin", signal };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError(0, "サーバーに接続できません。stemapp が起動しているか確認してください。");
  }
  if (res.status === 401 && path !== "/api/login") {
    if (unauthorizedHandler) unauthorizedHandler();
    throw new ApiError(401, "ログインしてください。");
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
  if (res.status === 204) return null;
  const type = res.headers.get("content-type") || "";
  return type.includes("application/json") ? res.json() : res;
}

/** バイナリ（音声・peaks）を ArrayBuffer で取る。 */
export async function fetchBinary(url, signal) {
  const res = await fetch(url, { credentials: "same-origin", signal });
  if (res.status === 401) {
    if (unauthorizedHandler) unauthorizedHandler();
    throw new ApiError(401, "ログインしてください。");
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
  return res.arrayBuffer();
}
