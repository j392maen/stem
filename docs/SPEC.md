# stemapp 仕様書（開発用・要約版）

元の設計書: https://claude.ai/code/artifact/95c8e4a3-9564-442e-bd65-b48a57547cec
この文書はサブエージェント向けの要約。矛盾したら設計書を優先し、監督役に報告すること。

## 1. 目的
個人（1人）用。楽曲（音声ファイル / URL）を stem に分割し、DJ風の波形を見ながら、
鳴らす stem の組み合わせを再生中に即時切り替えて聴く。保存もできる。
分割は1曲につき1回だけ。以後は保存済み stem を使う。

## 2. 実行環境
- 処理PC: Windows 11 ノートPC、Core i7-14650HX、GeForce RTX 4060 Laptop（VRAM 8GB）。
- 操作・再生: PC のブラウザ、および外出先の iPhone（Chrome。中身は WebKit）。
- 外部アクセス: Tailscale Serve（`tailscale serve --bg 8000`）経由の HTTPS。アプリ自体は 127.0.0.1:8000 のみで待ち受ける。
- 開発・テストは Linux のクラウド環境（GPU なし）。GPU が必要な処理は必ず差し替え可能にし、テストは偽物（Fake）で通す。
- Windows と Linux の両方で動くコードにする（パス区切り、改行、シェル依存に注意。`pathlib` を使う）。

## 3. 技術スタック（決定事項）
- Python 3.12、パッケージ管理 uv（`pyproject.toml`）。
- Web: FastAPI + uvicorn。静的フロントは FastAPI から配信。
- DB: SQLite（SQLAlchemy 2.0 ORM）。起動時に自動作成。
- 分離: `audio-separator`（python-audio-separator）を optional 依存 `gpu` として利用。コアは抽象インターフェース越しに呼ぶ。
- 音声処理: ffmpeg（外部コマンド）、numpy、soundfile。
- URL取得: yt-dlp（Python API）。YouTube には Deno が必要。
- フロント: ビルド不要の素の JavaScript（ES modules）+ Canvas + Web Audio API。npm は使わない。
- テスト: pytest。CI 相当として `uv run pytest` が Linux で通ること。
- 設定: pydantic-settings。データ置き場の既定は `%USERPROFILE%\stemapp-data`（Linux は `~/stemapp-data`）。環境変数 `STEMAPP_DATA_DIR` で上書き可。

## 4. stem
基本7種（分割時に必ず作る）: lead_vocal, backing_vocal, drums, bass, guitar, piano, other
詳細stem（必要時のみ追加分割。親stemを持つ木構造、子の合計＝親になるよう「残り(residual)」を必ず作る）:
- drums → kick, snare, toms, hihat, ride, crash
- guitar → acoustic_guitar, electric_guitar
- other → wind, saxophone, brass, woodwind, strings, organ, keys, synth, percussion(+ congas, tambourine, triangle, bells, glockenspiel)
- vocals系 → male, female, breath
種類はデータ（STEM_TYPE 行）で追加できること。コード変更なしで増やせる設計にする。

既定グループ: ボーカル=lead+backing, コード=guitar+piano+other, リズム=drums+bass, 伴奏=ボーカル以外。

## 5. 品質プリセット（ユーザーに見せるのは名前だけ）
- fast: BS-Roformer-SW（overlap 少なめ）→ vocals に karaoke（Mel-RoFormer 系）で lead/backing
- standard（既定）: SW + ボーカル専用モデル（Kim 系 Mel-RoFormer）の平均 → BS-RoFormer karaoke
- best: SW + ボーカル専用2種の平均 → karaoke 2種の平均、TTA
- 仕上げ: backing = vocals − lead（残差）。最後に mixture − Σstems を other に足し戻し、全 stem の合計が元の曲に一致するようにする。
- VRAM 8GB: モデルは1つずつロード→解放。fp16。OOM 時はチャンクを縮めて再試行、最後は CPU。

audio-separator のモデル名（確認済み）:
- `BS-Roformer-SW.ckpt`（6stem: vocals, drums, bass, guitar, piano, other）
- `vocals_mel_band_roformer.ckpt`（Kim）, `mel_band_roformer_kim_ft_unwa.ckpt`, `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
- karaoke: `mel_band_roformer_karaoke_becruily.ckpt`, `bs_roformer_karaoke_frazer_becruily.ckpt`, `bs_roformer_karaoke_anvuew.ckpt`, `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`
  （karaoke は vocals を入力すると "(Vocals)"=lead, "(Instrumental)"=それ以外 を出力）
- 詳細: `MDX23C-DrumSep-aufr33-jarredou.ckpt`（kick, snare, toms, hh, ride, crash）, `bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt`, `aspiration_mel_band_roformer_sdr_18.9845.ckpt`
- Mega 53 stems は audio-separator 外（ZFTurbo MSST 形式）。後回し。

## 6. データモデル（ER 図の実体）
TRACK, INPUT_SOURCE, DEVICE, SEPARATION_PRESET, PRESET_STEP, MODEL, SEPARATION_JOB,
STEM_TYPE, STEM, STEM_RENDITION, WAVEFORM, STEM_GROUP, STEM_GROUP_MEMBER,
LISTEN_PRESET, LISTEN_PRESET_ITEM, PLAYBACK_STATE, CUE_POINT, EXPORT, EXPORT_ITEM, OFFLINE_CACHE
主な属性は docs/ER.md を参照。

## 7. 再生（フロント）要件の要点
- 全 stem を同じ AudioContext 時刻で start。選択は GainNode の 0/1（10〜20ms ランプ）。シーク時は全 stem を止めて同時再開。
- iPhone は選択中 stem だけ読み込む（1stem 4分で約85MB になるため）。
- 波形: 事前計算 peaks（min/max int8、複数解像度）。概観＋拡大（再生位置中央固定）。選択 stem を stem 色で重ね描き。
- 配信用 rendition: Opus 128kbps。保存用: FLAC。

## 8. 非機能
- URL 取得失敗は理由コードで返す: unsupported_site, login_required, geo_blocked, private_or_removed, drm, network, needs_update, unknown。
- 外部公開しない（127.0.0.1 bind）。簡易パスコード認証。
- 日本語 UI。エラーはユーザーに分かる日本語で。

## 9. 開発ルール
- 1タスク＝1ブランチ（`task/Txx-...`）。完了時に main へマージするのは監督役。
- テストを書く。GPU・ネットワーク・ffmpeg が無い環境でもテストが skip ではなく通るよう、Fake と依存注入を使う（ffmpeg 必須のテストは `@pytest.mark.ffmpeg` を付け、無ければ skip 可）。
- 型ヒント必須。ruff で lint（`uv run ruff check`）。
- 秘密情報をコミットしない。
