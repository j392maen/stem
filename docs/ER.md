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
| SEPARATION_JOB | job_id PK, track_id FK, job_kind(full/refine), preset_id FK(full のみ), input_stem_id FK(refine のみ), requested_by FK→DEVICE, status(queued/running/done/failed/canceled), run_on(gpu/cpu/nightly), progress(0-1), stage, created_at, started_at, finished_at, error_message, cancel_requested(bool、キャンセル依頼), output_gain_db(float、既定0。保存前に全 stem にかけた倍率), postprocess_status(NULL/queued/running/done/failed、配信用データ・拍の作り直し), beat_warning(拍の解析に失敗したときの警告。NULL=なし), residual_rms_db / mixture_rms_db(float、補正前の残差と元の曲の RMS dBFS。聴き比べの参考) |
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
| BEAT_GRID | track_id PK FK, analyzer(解析器の名前と版。例 beat_this 1.1.0 final0), beats_json(拍の時刻・秒の配列), downbeats_json(小節の頭の時刻・秒の配列), time_signature(推定の拍子。1小節の拍数), created_at |
| BEAT_ANCHOR | anchor_id PK, track_id FK, position_sec, kind(downbeat/beat), bar_number(null可), bpm(null可), created_at |
| EXPORT | export_id PK, job_id FK, listen_preset_id FK(mix のみ), export_type(single/all/mix), format(wav/flac/mp3/zip), output_path, created_at |
| EXPORT_ITEM | export_id PK FK, stem_id PK FK |
| OFFLINE_CACHE | device_id PK FK, track_id PK FK, cached_at, bytes |

制約:
- LISTEN_PRESET_ITEM は stem_type_id と group_id のどちらか一方だけが非NULL（CHECK 制約）。
- STEM の子（parent_stem_id が同じ）は合計すると親に一致するよう、is_residual=true の stem を1つ含む。
- TRACK.audio_hash は正規化後PCMの SHA-256（同じ曲の再分割防止）。
- BEAT_GRID は自動解析の結果（再解析で上書き）。区間ごとの BPM は保存せず beats_json から計算する（`stemapp.beats.tempo`）。ユーザーの補正は BEAT_ANCHOR に別に持ち、再解析で消さない（T10c で使う）。拍の時刻はすべて元の曲の時刻（速度変更の影響を受けない）。
