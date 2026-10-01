# データモデル（ER）

設計書3章の ER 図をテキスト化したもの。PK=主キー、FK=外部キー、UK=一意。

| 実体 | 属性 |
| --- | --- |
| TRACK | track_id PK, title, artist, duration_sec, audio_hash UK, normalized_path, detected_instruments_json, created_at |
| INPUT_SOURCE | source_id PK, track_id FK, source_type(file/url), original_name, url, fetch_status, error_code, error_detail, fetched_at |
| DEVICE | device_id PK, name, kind(pc/iphone/ipad/other), push_subscription_json, last_seen_at |
| SEPARATION_PRESET | preset_id PK, code UK(fast/standard/best、実験は exp_*), display_name, is_default, is_experimental(bool、聴き比べ用の実験), options_json(パイプライン全体の選択肢。例 {"residual_to": "other/vocals/split"}) |
| PRESET_STEP | preset_id PK FK, step_order PK, model_id FK, input(mixture/vocals。karaoke のみ vocals も可、1プリセット内で同じ), role(multistem/vocals/karaoke), ensemble_weight, options_json |
| MODEL | model_id PK, filename UK, display_name, architecture, output_stems_json, min_vram_mb, checkpoint_sha256, license, source_url |
| SEPARATION_JOB | job_id PK, track_id FK, job_kind(full/refine), preset_id FK(full のみ), input_stem_id FK(refine のみ), refine_model_id FK→MODEL(refine のみ。詳細分割の方法。HPSS も architecture=hpss の MODEL 行), requested_by FK→DEVICE, status(queued/running/done/failed/canceled), run_on(gpu/cpu/nightly), progress(0-1), stage, created_at, started_at, finished_at, error_message, cancel_requested(bool、キャンセル依頼), output_gain_db(float、既定0。保存前に全 stem にかけた倍率), postprocess_status(NULL/queued/running/done/failed、配信用データ・拍の作り直し), beat_warning(拍の解析に失敗したときの警告。NULL=なし), residual_rms_db / mixture_rms_db(float、補正前の残差と元の曲の RMS dBFS。聴き比べの参考), output_dir(stem の保存フォルダ。データフォルダからの相対パス `stems/<元のファイル名>/<分け方>`。保存を始めるときに一度決める。NULL は T13 より前の `stems/<job_id>`。refine は分けた stem のジョブのフォルダの下 `…/<分け方>/<親の code>`), warning(処理は終わったが注意がある。例: 詳細分割で残りが 24bit の範囲を超えた。NULL=なし) |
| STEM_TYPE | stem_type_id PK, code UK, display_name(日本語), parent_id FK→STEM_TYPE, tier(base/detail), refine_model_id FK→MODEL, experimental, color(#RRGGBB), display_order |
| STEM | stem_id PK, job_id FK, stem_type_id FK, parent_stem_id FK→STEM, is_residual, rms_db, is_silent |
| STEM_RENDITION | rendition_id PK, stem_id FK, purpose(master/stream), codec(flac/opus/wav), bitrate_kbps, file_path, bytes |
| WAVEFORM | stem_id PK FK, samples_per_px PK, peaks_path |
| STEM_GROUP | group_id PK, code UK, display_name, color, is_builtin |
| STEM_GROUP_MEMBER | group_id PK FK, stem_type_id PK FK |
| LISTEN_PRESET | listen_preset_id PK, name, sort_order, seed_code UK(組み込みの識別子、ユーザー作成は NULL), hidden(bool、組み込みを削除したとき) |
| LISTEN_PRESET_ITEM | item_id PK, listen_preset_id FK, stem_type_id FK（どちらか一方）, group_id FK（どちらか一方）, gain_db |
| PLAYBACK_STATE | device_id PK FK, track_id PK FK, listen_preset_id FK, channel_gains_json, position_sec, updated_at |
| CUE_POINT | cue_id PK, track_id FK, position_sec, loop_end_sec(null可), label, color |
| BEAT_GRID | track_id PK FK, analyzer(解析器の名前と版。例 beat_this 1.1.0 final0), beats_json(拍の時刻・秒の配列), downbeats_json(小節の頭の時刻・秒の配列), time_signature(推定の拍子。1小節の拍数), created_at, edited_beats_json(null可。ユーザーが直した拍), edited_downbeats_json(null可), edited_time_signature(null可) |
| BEAT_EDIT | edit_id PK, track_id FK, op(downbeat/double/half/meter/shift/tap/cues/reset/reanalyze), params_json(操作の引数), before_json(操作の前の直した結果 {beats, downbeats, time_signature}。null=自動の結果のままだった), created_at |
| BEAT_ANCHOR | anchor_id PK, track_id FK, position_sec, kind(downbeat/beat), bar_number(null可), bpm(null可), created_at（T10c では未使用。将来のワープマーカー用） |
| EXPORT | export_id PK, job_id FK, listen_preset_id FK(mix のみ), export_type(single/all/mix), format(wav/flac/mp3。all は ZIP にまとめ中身がこの形式), output_path, created_at, status(queued/running/done/failed), progress, stage, error_message, filename, bytes, mix_gain_db(mix で ±1 を超えたとき全体にかけた dB), finished_at |
| EXPORT_ITEM | export_id PK FK, stem_id PK FK, gain_db(mix の音量) |
| OFFLINE_CACHE | device_id PK FK, track_id PK FK, cached_at, bytes |
| TEMPO_RENDER | render_id PK, job_id FK, ratio(速度の倍率。小数3桁。job_id と組で UK), pitch_mode(keep＝ピッチを保つ), status(queued/running/done/failed/canceled), progress(0-1), stage, error_message, cancel_requested(bool), dir_path(データフォルダからの相対。`cache/tempo/<job_id>/<倍率>`), bytes(全 stem の合計), frames(伸縮後の長さ。44.1kHz のサンプル数), created_at, started_at, finished_at, last_used_at(最後に使った時刻。キャッシュの片付けの順) |
| TEMPO_RENDITION | render_id PK FK, stem_id PK FK, codec(opus), bitrate_kbps, file_path(データフォルダからの相対), bytes |

制約:
- LISTEN_PRESET_ITEM は stem_type_id と group_id のどちらか一方だけが非NULL（CHECK 制約）。
- STEM の子（parent_stem_id が同じ）は合計すると親に一致するよう、is_residual=true の stem を1つ含む。
- 詳細分割（refine）の子 STEM は refine ジョブの行（job_id=refine ジョブ）。残りの STEM_TYPE は親ごとに `<親の code>_rest`（表示名「残り（親の表示名）」）。1つの stem を分けた結果は1組だけ（別の方法で分け直すと置き換え）。1つの分け方（full ジョブ）の木の中で STEM_TYPE の code は重ならない。
- TRACK.audio_hash は正規化後PCMの SHA-256（同じ曲の再分割防止）。
- BEAT_GRID の beats_json など（自動の結果）は解析だけが書き、補正では書き換えない。ユーザーが直した結果は edited_*（3つとも NULL か、3つとも値あり）。画面と API は「有効な拍」（edited_* があればそれ、無ければ自動の結果）を使う。区間ごとの BPM は保存せず、有効な拍から毎回計算する（`stemapp.beats.tempo`）。拍の時刻はすべて元の曲の時刻（速度変更の影響を受けない）。
- TEMPO_RENDER / TEMPO_RENDITION（T11）は「ピッチを保つ速度変更」のためにサーバーで伸縮した配信用の音声（stream と同じ Opus 128kbps / WebM）のキャッシュ。STEM_RENDITION には入れない（stem の保存フォルダとは別の `data/cache/tempo/` に置き、作成の状態・最後に使った時刻を持つため専用の表にした）。伸縮するのは子に分かれていない stem（画面で鳴らす stem）だけ。全 stem の長さは同じ（master の長さ ÷ 倍率）。上限（1曲あたりの倍率の数、全体の容量。設定で変更可）を超えたら last_used_at の古いものから行とフォルダを消す。ジョブ・曲を消すと行は CASCADE で消え、フォルダも消す。ピッチも変わる方式（playbackRate）はブラウザの中だけで完結し、DB には何も持たない。
- BEAT_EDIT は補正の履歴（元に戻す用。新しいものから取り消す。曲ごとに最大 100 件、古いものから消す）。「自動に戻す」も履歴に残し、元に戻せる。再解析すると新しい自動の結果を使い、それまでの直した結果は op=reanalyze の履歴に移す（解析の後に「元に戻す」で戻せる）。
