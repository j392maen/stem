# 開発計画

進め方: 監督役（指示・次タスク決定・統合）→ 実装サブエージェント → レビューサブエージェント → 監督役、の繰り返し。
開発はユーザーの Windows PC（C:\mine\stem）上の Claude Code で行う。監督役＝メインセッション、実装＝coder、レビュー＝reviewer サブエージェント。

| ID | タスク | 主な成果物 | 完了条件 | 状態 |
| --- | --- | --- | --- | --- |
| T01 | 基盤 | リポジトリ骨格、設定、DB モデル全実体、初期データ投入、`stemapp doctor`、Windows 用 setup/start スクリプト | `uv run pytest`（GPU テスト含む）と `uv run ruff check` が通る。doctor で GPU・ffmpeg・yt-dlp.exe・deno が OK | 完了（2026-09-25） |
| T02 | 分離パイプライン（CLI） | 音声正規化、Separator 抽象、audio-separator 実装、Fake 実装、fast/standard/best、残差補正、OOM 再試行、`stemapp separate` / `stemapp bench` | Fake で合計一致テスト。PC で4分の曲が standard で完走し時間を記録 | 完了（2026-09-25） |
| T03 | 取り込み | ファイル取り込み（重複検出）、URL 取得（yt-dlp）と失敗理由の分類 | 分類ロジックのテスト。PC で URL 取得を確認 | 完了（2026-09-25） |
| T04 | ジョブと API | ワーカープロセス、進捗配信、キャンセル、後処理（Opus・波形 peaks）、REST API、パスコード認証 | API テスト。PC で分割→API で stem 取得 | 完了（2026-09-25） |
| T05 | 再生 UI | ライブラリ、マルチトラック再生、組み合わせ切替、グループ・組み合わせプリセット、DJ 風波形、キュー・ループ | ブラウザで同期再生と即時切替 | 完了（2026-09-25） |
| T06 | iPhone・外出先 | PWA、Tailscale 手順、通知、続きから再生、オフライン保存、ロック画面モード | iPhone 実機で外出先から再生 | 未着手 |
| T07 | 詳細分割 | refine ジョブ（DrumSep、男女、息、Mega 53）、「もっと分ける」UI | 子 stem の合計一致。PC で実行 | 未着手 |
| T08 | 保存 | 個別・一括 ZIP・組み合わせミックス（WAV/FLAC/MP3） | 書き出しテスト | 実装中 |
| T09 | 実機調整 | 処理時間・VRAM 計測、聴き比べ、プリセット調整 | 設計書の目標値と比較 | 未着手 |

## 持ち越し事項（レビューで出たもの）
- T04: `STEMAPP_HOST` に 127.0.0.1 以外を設定したとき、doctor で注意を出すか設定で拒否する（SPEC 8章では外部公開しない前提）。
- T04: Starlette が httpx の非推奨警告を出している。テスト用クライアントをどうするか決める。
- T05: stem・グループの色が似ている（黄色系、オレンジ系）。グループの色が代表 stem の色と同じ。UI で見分けやすく調整する。
- T07: 詳細分割の「残り」stem にどの STEM_TYPE を使うか決める。（LISTEN_PRESET の組み込み判定は T05 で seed_code により解決済み）
- T09: プリセットの overlap 値は仮の値。Linux 側の torch 版は Windows と違う（Windows は 2.11+cu128）。
- T04（T02 レビューより）: 分割中の中断（Ctrl+C、ワーカー停止）で JOB が running のまま残る。canceled にして stems フォルダを片付けるか、起動時に残った JOB を片付ける。
- T04（T02 レビューより）: mp3/m4a では元の曲自体が ±1 を超えることがあり、FLAC 保存時の切り詰めで合計が一致しなくなる。クリップ件数を DB に残すか、master を float 形式にするか決める。
- T04 以降: onnxruntime-gpu（CUDA 13 版）と torch（cu128）の不一致警告が出る。実害はないが、ログが見づらい。
- T07/T09: best 用モデル（kim_ft_unwa 等）は未ダウンロード。初回実行時に出力名の対応表を確かめる。分割済みかの判定はプリセットを区別しない（再分割は --force）。
- T02 計測（240秒の合成音）: fast 80秒・GPU最大 3.4GB、standard 210秒・2.2GB。
- T04（T03 レビューより）: 同じ音を同時に取り込むと audio_hash の一意制約違反になる。IntegrityError なら探し直して既存扱いにする。
- T04/T05: 同じ URL の失敗が INPUT_SOURCE に溜まる。UI での見せ方（まとめるか）を決める。曲の削除機能では tracks/<id> フォルダも消す。
- 随時: URL 失敗の理由分類で誤りが見つかったら規則を足す（"is not available" は広めの規則）。
- T05（T04 より）: Opus の先頭の無音（pre-skip）で stem 同士がずれないか確かめる。SSE は終わった状態を受けたら EventSource.close()。/api/me の 401 でログイン画面を出す。
- T05 以降（T04 より）: CLI の `stemapp separate` は配信用データ（Opus・peaks）を作らない。CLI もジョブ登録に揃えるか決める（CLI とワーカーの同時実行は前提にしない）。
- T05 以降（T04 より）: 曲・ジョブ削除後に SQLite が番号を再利用する（stems/<id> の混在の恐れ）。AUTOINCREMENT 化を検討。
- 随時（T04 より）: ワーカーが落ちても serve は動き続ける。/api/health にワーカー状態を出すか、serve が再起動する。外部コマンドは必ず proc.run_bound / popen_bound で起動する。
- 随時（T04 より）: ログアウトは Cookie を消すだけ（トークンは30日有効。パスコード変更で全無効）。
- 任意（T05 より）: グループ色「伴奏」と「ボーカル」がややに似る。キュー色の赤が再生位置の線と同じ。PC では 4 分の曲で約 700MB のメモリを使う（iPhone は T06 で選択中 stem のみ読み込み）。配信用データの作り直しは途中キャンセル不可。
- T05b: 再生ボタンの見た目、0 キー、保存フォルダを開く（完了 2026-09-25）。
- T06（T05b レビューより）: Host ヘッダーの許可リスト（TrustedHostMiddleware。localhost・127.0.0.1・Tailscale のホスト名）で DNS リバインディングを防ぐ。open-folder は explorer を完全パスで起動、中継ヘッダーに x-forwarded-proto・x-real-ip・via も追加。Tailscale の TCP 転送では使わない旨を README に書く。

## 追加タスク（2026-09-25、ユーザー確認と調査 R01 を受けて）
| ID | タスク | 状態 |
| --- | --- | --- |
| T10 | 拍・小節の解析（beat_this）と表示、区間 BPM | 完了（2026-09-26） |
| T10c | 拍の手動補正 UI（1小節目、×2/÷2、拍子、タップ、キューから計算） | 未着手 |
| T11 | 速度変更: playbackRate（ピッチも変わる）、サーバー変換（rubberband、ピッチ保持）、PC 即時（signalsmith-stretch） | 未着手 |
| T12 | 分離品質の見直し: 残差の行き先、karaoke の入力と組み合わせ、実験プリセット、同じ曲の分け方の切り替え | 実装中 |
| T07 | （追加）other → 持続音／短い音／残り（HPSS）、Mega 53 を 8GB で試す | 未着手 |
- T10c（T10 レビューより）: 短い区間を吸収したとき、吸収先が短い区間だと誤った BPM が長く表示される（水槽 123〜134 秒の 165.64）。手動補正で直せるようにする。再解析と作り直し完了の間のわずかな競合（押し直せば直る）。
