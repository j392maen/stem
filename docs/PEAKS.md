# 波形 peaks ファイルの形式（STPK 版 1）

波形表示用に事前計算したデータ。stem ごと・解像度ごとに1ファイル。
作成: `src/stemapp/peaks.py`（`compute_peaks` / `encode_peaks`）。

- 置き場所: `data/stems/<job_id>/peaks/<stem code>_<samples_per_px>.stpk`
- DB: WAVEFORM（stem_id, samples_per_px, peaks_path）
- 配信: `GET /api/files/peaks/{stem_id}/{samples_per_px}`（`application/octet-stream`）
- 解像度（samples_per_px）: 256, 1024, 4096, 16384

## 中身

すべてリトルエンディアン。先頭 20 バイトがヘッダ、続いて点の数 × 2 バイト。

| 位置 | 大きさ | 型 | 内容 |
| --- | --- | --- | --- |
| 0 | 4 | バイト列 | マジック `STPK`（ASCII） |
| 4 | 2 | uint16 | 版（今は 1） |
| 6 | 2 | uint16 | 予約（0） |
| 8 | 4 | uint32 | samples_per_px（1点あたりのサンプル数） |
| 12 | 4 | uint32 | サンプルレート（今は 44100） |
| 16 | 4 | uint32 | 点の数 N |
| 20 | 2N | int8 の並び | min0, max0, min1, max1, …（min と max を交互に） |

- 点 i は、サンプル `i * samples_per_px` から `samples_per_px` 個（最後の点は残り全部）を表す。
  N = ceil(サンプル数 / samples_per_px)。
- 値は左右を平均したモノラルの最小値・最大値を 127 倍した int8（-127〜127）。
  min は切り下げ、max は切り上げで量子化するので、[min/127, max/127] は元の波形を必ず含み、
  min ≤ max が保証される。±1 を超える値は ±127 に切り詰める。
- 点 i の時刻（秒）は `i * samples_per_px / サンプルレート`。

## 読み方の例（JavaScript）

```js
const buf = await (await fetch(url)).arrayBuffer();
const v = new DataView(buf);
const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4)); // "STPK"
const version = v.getUint16(4, true);
const samplesPerPx = v.getUint32(8, true);
const sampleRate = v.getUint32(12, true);
const n = v.getUint32(16, true);
const data = new Int8Array(buf, 20, n * 2); // data[2*i] = min, data[2*i+1] = max
```

版を上げるとき（形式を変えるとき）は、読み手が版を見て判断できるよう `版` を増やす。
