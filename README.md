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
