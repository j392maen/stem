// 画面の入口: ログインの確認と、ハッシュでの画面切り替え（#/library, #/track/<id>）。

import { api, onUnauthorized } from "./api.js";
import * as beatsMod from "./beats.js";
import { DiagView } from "./diag.js";
import * as engineMod from "./engine.js";
import { LibraryView } from "./library.js";
import * as peaksMod from "./peaks.js";
import { PlayerView } from "./player.js";
import * as selectionMod from "./selection.js";
import { el, toast } from "./ui.js";

const root = document.getElementById("view");
const nav = document.getElementById("topnav");
let current = null;
let passcodeRequired = false;

// テスト・動作確認用（ブラウザのテストが状態を読む）
window.__stemapp = {
  get view() { return current; },
  modules: { peaks: peaksMod, selection: selectionMod, engine: engineMod, beats: beatsMod },
};

function renderNav() {
  const items = [el("a", { class: "btn small", href: "#/library", text: "ライブラリ" })];
  if (passcodeRequired) {
    items.push(el("button", {
      class: "btn small", type: "button", text: "ログアウト",
      onclick: async () => {
        try { await api("/api/logout", { method: "POST" }); } catch { /* 401 でもログイン画面へ */ }
        showLogin();
      },
    }));
  }
  nav.replaceChildren(...items);
}

function unmountCurrent() {
  if (current && current.unmount) current.unmount();
  current = null;
}

function showLogin() {
  unmountCurrent();
  nav.replaceChildren();
  const input = el("input", {
    class: "input", type: "password", autocomplete: "current-password",
    placeholder: "パスコード", "aria-label": "パスコード", id: "passcode",
  });
  const error = el("p", { class: "error-text", hidden: true });
  const form = el("form", {
    onsubmit: async (e) => {
      e.preventDefault();
      error.hidden = true;
      try {
        await api("/api/login", { method: "POST", body: { passcode: input.value } });
        passcodeRequired = true;
        route();
      } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
        input.select();
      }
    },
  }, input, el("button", { class: "btn primary", type: "submit", text: "ログイン" }), error);
  root.replaceChildren(el("section", { class: "panel login" },
    el("h2", { text: "ログイン" }),
    el("p", { class: "muted", text: "パスコードを入力してください。" }),
    form));
  current = { unmount() {}, login: true };
  input.focus();
}

onUnauthorized(() => {
  if (!current || !current.login) showLogin();
});

async function route() {
  unmountCurrent();
  renderNav();
  const hash = location.hash || "#/library";
  const m = /^#\/track\/(\d+)$/.exec(hash);
  let title = "ライブラリ";
  if (m) {
    current = new PlayerView(root, Number(m[1]));
    title = "プレイヤー";
  } else if (hash === "#/diag") {
    current = new DiagView(root);
    title = "端末の診断";
  } else {
    if (hash !== "#/library") history.replaceState(null, "", "#/library");
    current = new LibraryView(root);
  }
  document.title = `stemapp - ${title}`;
  try {
    await current.mount();
  } catch (e) {
    if (e.status !== 401) toast(e.message || String(e));
  }
}

const NOTICE_KEY = "stemapp.hidePasscodeNotice";

/** 外（Tailscale 経由など）から開いたのにパスコードが無いとき、設定を勧める（使えなくはしない）。 */
function showPasscodeNotice(show) {
  const box = document.getElementById("notice");
  if (!box) return;
  let hidden = false;
  try { hidden = sessionStorage.getItem(NOTICE_KEY) === "1"; } catch { /* 保存できなくても動く */ }
  if (!show || hidden) { box.hidden = true; return; }
  box.replaceChildren(
    el("span", { class: "grow", id: "passcode-notice",
      text: "外から開いていますが、パスコードが設定されていません。PC の .env に STEMAPP_PASSCODE を設定し、stemapp を起動し直すことをおすすめします。" }),
    el("button", {
      class: "btn small", type: "button", text: "閉じる",
      onclick: () => {
        box.hidden = true;
        try { sessionStorage.setItem(NOTICE_KEY, "1"); } catch { /* 保存できなくても動く */ }
      },
    }),
  );
  box.hidden = false;
}

/** ホーム画面に追加したときのための Service Worker（画面ファイルだけ。音声・API は扱わない）。 */
function registerServiceWorker() {
  if (!("serviceWorker" in navigator) || !window.isSecureContext) return;
  navigator.serviceWorker.register("sw.js").catch(() => { /* 無くても動く */ });
}

async function boot() {
  registerServiceWorker();
  try {
    const me = await api("/api/me");
    passcodeRequired = !!me.passcode_required;
    showPasscodeNotice(!!me.passcode_recommended);
  } catch (e) {
    if (e.status === 401) return; // ログイン画面は onUnauthorized が出す
    root.replaceChildren(el("p", { class: "empty error-text", text: e.message }));
    return;
  }
  route();
}

// ログイン画面の間はハッシュが変わっても切り替えない（ログイン後に route() する）
window.addEventListener("hashchange", () => {
  if (current && current.login) return;
  route();
});

boot();
