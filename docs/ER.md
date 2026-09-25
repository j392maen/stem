# データモデル（ER）

設計書3章の ER 図をテキスト化したもの。PK=主キー、FK=外部キー、UK=一意。

| 実体 | 属性 |
| --- | --- |
| TRACK | track_id PK, title, artist, duration_sec, audio_hash UK, normalized_path, detected_instruments_json, created_at |
| INPUT_SOURCE | source_id PK, track_id FK, source_type(file/url), original_name, url, fetch_status, error_code, error_detail, fetched_at |
| DEVICE | device_id PK, name, kind(pc/iphone/ipad/other), push_subscription_json, last_seen_at |
| SEPARATION_PRESET | preset_id PK, code UK(fast/standard/best), display_name, is_default |
| PRESET_STEP | preset_id PK FK, step_order PK, model_id FK, input(mixture/vocals), role(multistem/vocals/karaoke), ensemble_weight, options_json |
| MODEL | model_id PK, filename UK, display_name, architecture, output_stems_json, min_vram_mb, checkpoint_sha256, license, source_url |
| SEPARATION_JOB | job_id PK, track_id FK, job_kind(full/refine), preset_id FK(full のみ), input_stem_id FK(refine のみ), requested_by FK→DEVICE, status(queued/running/done/failed/canceled), run_on(gpu/cpu/nightly), progress(0-1), stage, created_at, started_at, finished_at, error_message, cancel_requested(bool、キャンセル依頼), output_gain_db(float、既定0。保存前に全 stem にかけた倍率) |
| STEM_TYPE | stem_type_id PK, code UK, display_name(日本語), parent_id FK→STEM_TYPE, tier(base/detail), refine_model_id FK→MODEL, experimental, color(#RRGGBB), display_order |
| STEM | stem_id PK, job_id FK, stem_type_id FK, parent_stem_id FK→STEM, is_residual, rms_db, is_silent |
| STEM_RENDITION | rendition_id PK, stem_id FK, purpose(master/stream), codec(flac/opus/wav), bitrate_kbps, file_path, bytes |
| WAVEFORM | stem_id PK FK, samples_per_px PK, peaks_path |
| STEM_GROUP | group_id PK, code UK, display_name, color, is_builtin |
| STEM_GROUP_MEMBER | group_id PK FK, stem_type_id PK FK |
| LISTEN_PRESET | listen_preset_id PK, name, sort_order |
| LISTEN_PRESET_ITEM | item_id PK, listen_preset_id FK, stem_type_id FK（どちらか一方）, group_id FK（どちらか一方）, gain_db |
| PLAYBACK_STATE | device_id PK FK, track_id PK FK, listen_preset_id FK, channel_gains_json, position_sec, updated_at |
| CUE_POINT | cue_id PK, track_id FK, position_sec, loop_end_sec(null可), label, color |
| EXPORT | export_id PK, job_id FK, listen_preset_id FK(mix のみ), export_type(single/all/mix), format(wav/flac/mp3/zip), output_path, created_at |
| EXPORT_ITEM | export_id PK FK, stem_id PK FK |
| OFFLINE_CACHE | device_id PK FK, track_id PK FK, cached_at, bytes |

制約:
- LISTEN_PRESET_ITEM は stem_type_id と group_id のどちらか一方だけが非NULL（CHECK 制約）。
- STEM の子（parent_stem_id が同じ）は合計すると親に一致するよう、is_residual=true の stem を1つ含む。
- TRACK.audio_hash は正規化後PCMの SHA-256（同じ曲の再分割防止）。
