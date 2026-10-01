# signalsmith-stretch（同梱）

ブラウザ内で音の高さを保ったまま速度を変える（T11c「ピッチを保つ・すぐ（PC）」）ためのライブラリ。
npm・CDN は使わず、公式の Web 版の配布ファイルをそのまま置いている（変更なし）。

| 項目 | 内容 |
| --- | --- |
| 名前 | Signalsmith Stretch（Web 版。WASM ＋ AudioWorklet） |
| 作者 | Geraint Luff / Signalsmith Audio Ltd. |
| ライセンス | MIT（`LICENSE.txt`。リポジトリ直下の LICENSE.txt をそのまま取得） |
| 版 | git タグ `1.4.0`（同じタグの `web/release/package.json` の version は `1.3.2`） |
| 取得日 | 2026-10-01 |
| 取得元 | https://raw.githubusercontent.com/Signalsmith-Audio/signalsmith-stretch/1.4.0/web/release/SignalsmithStretch.mjs |
| ライセンスの取得元 | https://raw.githubusercontent.com/Signalsmith-Audio/signalsmith-stretch/1.4.0/LICENSE.txt |
| リポジトリ | https://github.com/Signalsmith-Audio/signalsmith-stretch |
| SHA-256（SignalsmithStretch.mjs） | `97530b11d5bc01015af4cde40d6aa55ff10c40aa1294ca4c8c5762027d517a46` |

## 使い方（このアプリでの）

- `src/stemapp/web/js/engine.js` の `Engine.ensureStretch()` が `import()` で読み込み、
  `SignalsmithStretch(audioContext)` で AudioWorkletNode を 1 つ作る。
- 「ライブ入力」モードで使う: 全 stem を音量をかけて混ぜた音を入力し、`schedule({semitones})` で
  音の高さだけを戻す（音源は playbackRate = r で鳴らす）。詳しくは engine.js の先頭のコメント。

## 更新するとき

1. 上の取得元のタグを新しいものに替えてファイルを取り直す。
2. この README の版・取得日・SHA-256 を書き換える。
3. `uv run pytest -m browser tests/test_browser_stretch.py` で時刻のずれ・音の高さを確かめる。
