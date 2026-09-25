// 画面部品の小さな道具（要素の作成、時刻の表示、確認・入力ダイアログ、通知）。

/** 要素を作る。attrs の on* はイベント、class/text/html/style は特別扱い。 */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "dataset") Object.assign(node.dataset, value);
    else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** 秒を「m:ss」または「m:ss.d」にする。 */
export function formatTime(sec, withTenths = false) {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  const total = withTenths ? Math.floor(sec * 10) / 10 : Math.floor(sec);
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  const ss = withTenths ? s.toFixed(1).padStart(4, "0") : String(Math.floor(s)).padStart(2, "0");
  return `${m}:${ss}`;
}

let toastTimer = null;

export function toast(message, ms = 3500) {
  const box = document.getElementById("toast");
  if (!box) return;
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.hidden = true; }, ms);
}

function openModal(build) {
  return new Promise((resolve) => {
    const back = el("div", { class: "modal-back" });
    const close = (value) => {
      document.removeEventListener("keydown", onKey, true);
      back.remove();
      resolve(value);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); close(null); }
    };
    document.addEventListener("keydown", onKey, true);
    back.addEventListener("click", (e) => { if (e.target === back) close(null); });
    back.append(build(close));
    document.body.append(back);
    const focus = back.querySelector("input, .primary, .danger-confirm");
    if (focus) focus.focus();
  });
}

/** 確認ダイアログ。OK なら true。 */
export function confirmDialog(message, { ok = "OK", danger = false } = {}) {
  return openModal((close) => el("div", { class: "modal", role: "dialog", "aria-modal": "true" },
    el("p", { text: message }),
    el("div", { class: "row" },
      el("button", { class: "btn", type: "button", text: "やめる", onclick: () => close(false) }),
      el("button", {
        class: danger ? "btn primary danger-confirm" : "btn primary",
        type: "button", text: ok, onclick: () => close(true),
      }),
    ),
  )).then((v) => v === true);
}

/** 文字の入力ダイアログ。やめたら null。 */
export function promptDialog(message, initial = "", { ok = "OK", maxLength = 100 } = {}) {
  return openModal((close) => {
    const input = el("input", {
      class: "input", type: "text", value: initial, maxlength: maxLength, "aria-label": message,
    });
    const submit = (e) => {
      e.preventDefault();
      const value = input.value.trim();
      if (value) close(value);
    };
    return el("form", { class: "modal", role: "dialog", "aria-modal": "true", onsubmit: submit },
      el("p", { text: message }),
      input,
      el("div", { class: "row" },
        el("button", { class: "btn", type: "button", text: "やめる", onclick: () => close(null) }),
        el("button", { class: "btn primary", type: "submit", text: ok }),
      ),
    );
  });
}

/** 線で描く小さなアイコン（絵文字にならないよう SVG にする）。 */
export function icon(name) {
  const paths = {
    up: "M6 15l6-6 6 6",
    down: "M6 9l6 6 6-6",
    edit: "M4 20h4L19 9l-4-4L4 16v4zM13 7l4 4",
    close: "M6 6l12 12M18 6L6 18",
  };
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", paths[name]);
  svg.append(path);
  return svg;
}

export const ICONS = {
  play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.5v15l13-7.5z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 4h4.5v16H6zM13.5 4H18v16h-4.5z"/></svg>',
};
