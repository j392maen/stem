// Service Worker（最小限）: アプリの画面ファイル（HTML・CSS・JS・アイコン）だけをキャッシュする。
//
// - /api/ 以下（音声・stem・peaks を含む）は一切扱わない（ブラウザにそのまま任せる）。
// - 画面ファイルは「ネットワーク優先」: つながるときは毎回サーバーの最新を使い、キャッシュを更新する。
//   つながらないときだけキャッシュを使う。なので画面の更新はすぐ反映される（版を上げ忘れても古い画面に
//   固定されない）。
// - VERSION はキャッシュの入れ物の名前。キャッシュの持ち方を変えたときに上げると、古い入れ物を消す。
// - 曲のオフライン保存は T06b で扱う。

const VERSION = "stemapp-shell-v1";

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      if (key.startsWith("stemapp-shell-") && key !== VERSION) await caches.delete(key);
    }
    await self.clients.claim();
  })());
});

function isShellRequest(request) {
  if (request.method !== "GET") return false;
  if (request.headers.has("range")) return false;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return false;
  if (url.pathname.startsWith("/api/") || url.pathname === "/api") return false;
  if (url.pathname.startsWith("/docs") || url.pathname.startsWith("/redoc")) return false;
  if (url.pathname === "/openapi.json") return false;
  return true;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (!isShellRequest(request)) return; // ブラウザの普段どおりの処理
  event.respondWith((async () => {
    const cache = await caches.open(VERSION);
    try {
      const response = await fetch(request);
      if (response.ok && response.type === "basic") {
        cache.put(request, response.clone()).catch(() => {});
      }
      return response;
    } catch (err) {
      const cached = await cache.match(request, { ignoreSearch: true });
      if (cached) return cached;
      if (request.mode === "navigate") {
        const index = await cache.match("/");
        if (index) return index;
      }
      throw err;
    }
  })());
});
