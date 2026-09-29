# stemapp

個人用の楽曲 stem 分割・再生アプリ。楽曲を stem（ボーカル・ドラム・ベースなどのパート）に分け、
組み合わせを切り替えながら聴く。仕様は `docs/SPEC.md`、データモデルは `docs/ER.md`。

## 動作環境

- Windows 11（NVIDIA GPU 推奨。無くても CPU で動く）
- Python 3.12（uv が自動で用意する）
- 外部コマンド: ffmpeg、Deno（YouTube 取得用）、yt-dlp.exe（`C:\mine\yt-dlp.exe`）

## セットアップ

PowerShell でリポジトリ直下（`C:\mine\stem`）に移動して実行する。

```powershell
.\scripts\setup.ps1
```

スクリプトが行うこと（何度実行しても大丈夫。既にあるものはスキップ）:

1. uv・ffmpeg・Deno が無ければ winget で導入（`astral-sh.uv`、`Gyan.FFmpeg`、`DenoLand.Deno`）
2. `uv sync --extra gpu --extra url`（CUDA 12.8 版 torch と audio-separator を含む）
3. `.env` が無ければ `.env.example` からコピー
4. `uv run stemapp init-db`（DB 作成と初期データ投入）
5. `uv run stemapp doctor`（環境診断）

実行ポリシーで止められる場合は、次のように一時的に許可して実行する。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

## 設定（`.env`）

| 変数 | 既定値 | 説明 |
| --- | --- | --- |
| `STEMAPP_DATA_DIR` | リポジトリ直下の `data` | データ置き場（DB・音源・stem）。git 管理外 |
| `STEMAPP_YTDLP_PATH` | `C:\mine\yt-dlp.exe` | yt-dlp.exe の場所 |
| `STEMAPP_HOST` | `127.0.0.1` | 待ち受けアドレス（外部公開しない） |
| `STEMAPP_PORT` | `8000` | ポート |
| `STEMAPP_PASSCODE` | なし | 簡易パスコード。設定するとログインが必要になる（外から使うなら設定を推奨） |
| `STEMAPP_ALLOWED_HOSTS` | なし | 127.0.0.1・localhost・[::1] 以外に受け付ける名前（カンマ区切り）。Tailscale で開くなら `unagi.tail8b25a2.ts.net` |
| `STEMAPP_TEMPO_CACHE_MAX_MB` | `2048` | 速度変更（ピッチを保つ）の音声のキャッシュ全体の上限（MB） |
| `STEMAPP_TEMPO_CACHE_PER_TRACK` | `3` | 1曲あたりに残す速度（倍率）の数 |
| `STEMAPP_TEMPO_WORKERS` | `0` | 伸縮を同時に何 stem 行うか（0 = CPU の論理コア数の半分、最大 8） |

## 起動

```powershell
uv run stemapp serve
```

またはエクスプローラーで `scripts\start.bat` をダブルクリック。
動作確認: ブラウザで <http://127.0.0.1:8000/api/health> を開き `{"status":"ok",...}` が出れば OK。
止めるときはウィンドウで Ctrl+C。

## iPhone・外出先から使う（Tailscale Serve）

PC と iPhone を同じ Tailscale のネットワーク（tailnet）に入れておき、PC のアプリを
tailnet の中だけに HTTPS で公開する。インターネット全体に公開する Funnel は使わない。

1. `.env` に次を足す（この名前以外の Host は安全のため 400 で断る）。
   `uv run stemapp doctor` の「Tailscale」の行にも、この PC の名前が案内される。

   ```
   STEMAPP_ALLOWED_HOSTS=unagi.tail8b25a2.ts.net
   STEMAPP_PASSCODE=（好きなパスコード）
   ```

   パスコードの設定を強く勧める（無いと、外から開いたときに画面に注意が出る）。
2. stemapp を起動する（`scripts\start.bat`）。
3. 公開を始める（PC の設定を変える。止めるまで続く。PC を再起動しても残る）。

   ```powershell
   .\scripts\tailscale-serve.ps1 start    # https://unagi.tail8b25a2.ts.net/ → http://127.0.0.1:8000
   .\scripts\tailscale-serve.ps1 status   # 状態の確認（Funnel を使っていないことも表示）
   .\scripts\tailscale-serve.ps1 stop     # 公開をやめる
   ```

4. iPhone の Safari で <https://unagi.tail8b25a2.ts.net/> を開く（iPhone でも Tailscale を接続しておく）。
5. ホーム画面に追加する: Safari の共有ボタン →「ホーム画面に追加」。以後はアイコンから全画面で開ける。

注意:
- `tailscale serve --tcp`（TCP 転送）や SSH のポート転送では開かないこと。これらは中継のしるし
  （X-Forwarded-For など）を付けないため、外からの操作を「この PC のブラウザ」と取り違え、
  「保存フォルダを開く」が使えてしまう。HTTPS の serve（上のスクリプト）だけを使う。
- 端末の診断: ライブラリの一番下の小さな「端末の診断」から `#/diag` を開き「診断を始める」を押すと、
  その端末で使える音声の形式や機能を調べて PC の `data\diag\` に保存する（`GET /api/diag` で一覧）。

## 拍の補正（プレイヤーの「拍の補正」）

- 自動の拍がずれた・倍や半分に取られた・小節の頭が違うときに、1小節目をここに・×2/÷2・拍子・ずらす・
  タップ・キュー2点からで直せる。直した結果は自動の結果とは別に保存され、「元に戻す」（Ctrl+Z）と
  「自動に戻す」ができる。再解析すると新しい自動の結果になり、直した結果は「元に戻す」で戻せる。
- 既知の制限: テンポの区間の境目の拍を大きく（±50ms 以上）ずらすと、境目の前後の拍の間隔が変わるため、
  境目付近の BPM 表示がずれたり、短い区間ができたりする（区間は拍の間隔から毎回求めるため）。
  大きく直したいときは、区間ではなく「キュー2点から」やループ区間でのタップを使う。

## 速度の変更（プレイヤーの「SPEED」）

- スライダー（幅 ±8 / ±16 / ±50%）、−/＋（0.1%。Shift で 1%）、目標 BPM（再生位置の区間の BPM が基準）、
  「元の速度に戻す」で速度（0.50〜2.00 倍）を決める。キーは `,` `.`（Shift で 1%）、`R`（元に戻す）。
  曲ごとの速度と方式はブラウザに保存され、次に開いたときに戻る。
- 「ピッチも変わる」: すぐ変わる（全 stem の再生の速さを同じ時刻に変える）。
- 「ピッチを保つ」: サーバー（ワーカー）が ffmpeg の rubberband で全 stem を伸縮した音声を作る
  （4 分の曲で 10 秒前後。分割の処理中はその後になる）。できたら同じ位置から差し替える。作っている間は
  「今の音のまま」か「ピッチを変えて先に速度を変える」を選べる。作った音声は
  `data\cache\tempo\<job_id>\<倍率>\` にキャッシュされ（1曲 3 つ・全体 2GB まで。古く使ったものから消える）、
  同じ倍率はすぐ切り替わる。「省メモリ」（スマホ幅・8 分を超える曲では既定で ON）にすると、
  音声を切り替えるときに前の音声を捨ててから読み込む（読み込む間は止まり、終わると同じ位置から続く）。
- rubberband は左右をまとめて処理し（定位を保つ）、倍率に比例して生じる時刻のずれ（2 倍で約 36ms）を
  補正している。伸縮した音は「元の時刻 ÷ 倍率」と数 ms 以内で一致する。
- 拍・キュー・ループ・波形はいつも元の曲の時刻のまま。BPM 表示は「区間の BPM × 速度」（元の BPM も小さく出る）。

## コマンド

| コマンド | 内容 |
| --- | --- |
| `uv run stemapp init-db` | DB 作成と初期データ投入（何度実行しても同じ結果） |
| `uv run stemapp doctor` | 環境診断。項目ごとに OK / 注意 / NG。NG があると終了コード 1 |
| `uv run stemapp serve` | Web サーバー起動 |
| `uv run stemapp separate <ファイル> [--preset fast\|standard\|best] [--force] [--cpu]` | 1曲を stem に分割し `data\stems\<job_id>\` に FLAC で保存、DB に登録。分割済みの曲は `--force` が無ければ分割しない |
| `uv run stemapp bench <ファイル> [--presets fast,standard]` | プリセットごとの処理時間と GPU メモリ最大使用量を測り、`data\cache\bench\<日時>.json` に保存（DB には登録しない） |

分離モデルは初回に `data\models\` へ自動でダウンロードされる（fast と standard で約 3.5GB）。

## テスト

```powershell
uv run pytest -q          # 通常テスト（GPU 不要）
uv run pytest -q -m gpu   # 実 GPU を使うテスト
uv run pytest -q -m ffmpeg   # 本物の ffmpeg を使うテスト
uv run pytest -q -m browser  # PC の Edge で画面を動かすテスト
uv run ruff check         # lint
```

通常テストは一時フォルダの DB を使うので、`data` フォルダは汚れない。
