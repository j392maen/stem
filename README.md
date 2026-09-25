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
| `STEMAPP_PASSCODE` | なし | 簡易パスコード（後のタスクで使用） |

## 起動

```powershell
uv run stemapp serve
```

またはエクスプローラーで `scripts\start.bat` をダブルクリック。
動作確認: ブラウザで <http://127.0.0.1:8000/api/health> を開き `{"status":"ok",...}` が出れば OK。
止めるときはウィンドウで Ctrl+C。

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
uv run ruff check         # lint
```

通常テストは一時フォルダの DB を使うので、`data` フォルダは汚れない。
