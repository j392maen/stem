// 画面の入口: ログインの確認と、ハッシュでの画面切り替え（#/library, #/track/<id>）。

import { api, onUnauthorized } from "./api.js";
import * as beatsMod from "./beats.js";
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
  if (m) {
    current = new PlayerView(root, Number(m[1]));
  } else {
    if (hash !== "#/library") history.replaceState(null, "", "#/library");
    current = new LibraryView(root);
  }
  document.title = m ? "stemapp - プレイヤー" : "stemapp - ライブラリ";
  try {
    await current.mount();
  } catch (e) {
    if (e.status !== 401) toast(e.message || String(e));
  }
}

async function boot() {
  try {
    const me = await api("/api/me");
    passcodeRequired = !!me.passcode_required;
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
