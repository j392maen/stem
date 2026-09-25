# 開発計画

進め方: 監督役（指示・次タスク決定・統合）→ 実装サブエージェント → レビューサブエージェント → 監督役、の繰り返し。
開発はユーザーの Windows PC（C:\mine\stem）上の Claude Code で行う。監督役＝メインセッション、実装＝coder、レビュー＝reviewer サブエージェント。

| ID | タスク | 主な成果物 | 完了条件 | 状態 |
| --- | --- | --- | --- | --- |
| T01 | 基盤 | リポジトリ骨格、設定、DB モデル全実体、初期データ投入、`stemapp doctor`、Windows 用 setup/start スクリプト | Linux で `uv run pytest` と `uv run ruff check` が通る。PC で doctor が GPU を認識 | 未着手 |
| T02 | 分離パイプライン（CLI） | 音声正規化、Separator 抽象、audio-separator 実装、Fake 実装、fast/standard/best、残差補正、OOM 再試行、`stemapp separate` / `stemapp bench` | Fake で合計一致テスト。PC で4分の曲が standard で完走し時間を記録 | 未着手 |
| T03 | 取り込み | ファイル取り込み（重複検出）、URL 取得（yt-dlp）と失敗理由の分類 | 分類ロジックのテスト。PC で URL 取得を確認 | 未着手 |
| T04 | ジョブと API | ワーカープロセス、進捗配信、キャンセル、後処理（Opus・波形 peaks）、REST API、パスコード認証 | API テスト。PC で分割→API で stem 取得 | 未着手 |
| T05 | 再生 UI | ライブラリ、マルチトラック再生、組み合わせ切替、グループ・組み合わせプリセット、DJ 風波形、キュー・ループ | ブラウザで同期再生と即時切替 | 未着手 |
| T06 | iPhone・外出先 | PWA、Tailscale 手順、通知、続きから再生、オフライン保存、ロック画面モード | iPhone 実機で外出先から再生 | 未着手 |
| T07 | 詳細分割 | refine ジョブ（DrumSep、男女、息、Mega 53）、「もっと分ける」UI | 子 stem の合計一致。PC で実行 | 未着手 |
| T08 | 保存 | 個別・一括 ZIP・組み合わせミックス（WAV/FLAC/MP3） | 書き出しテスト | 未着手 |
| T09 | 実機調整 | 処理時間・VRAM 計測、聴き比べ、プリセット調整 | 設計書の目標値と比較 | 未着手 |
