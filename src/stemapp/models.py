"""DB モデル（docs/ER.md の全20実体）。

列挙的な値（status 等）は文字列で保存し、取りうる値はコメントに書く。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from stemapp.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


# --- 曲と入力 ---------------------------------------------------------------


class Track(Base):
    __tablename__ = "track"

    track_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    artist: Mapped[str | None] = mapped_column(String(500))
    duration_sec: Mapped[float | None] = mapped_column(Float)
    # 正規化後 PCM の SHA-256（同じ曲の再分割防止）
    audio_hash: Mapped[str] = mapped_column(String(64), unique=True)
    normalized_path: Mapped[str | None] = mapped_column(Text)
    detected_instruments_json: Mapped[Any | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class InputSource(Base):
    __tablename__ = "input_source"

    source_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    track_id: Mapped[int | None] = mapped_column(
        ForeignKey("track.track_id", ondelete="CASCADE")
    )
    source_type: Mapped[str] = mapped_column(String(10))  # file / url
    original_name: Mapped[str | None] = mapped_column(String(500))
    url: Mapped[str | None] = mapped_column(Text)
    fetch_status: Mapped[str | None] = mapped_column(String(20))
    # unsupported_site / login_required / geo_blocked / private_or_removed / drm /
    # network / needs_update / unknown
    error_code: Mapped[str | None] = mapped_column(String(40))
    error_detail: Mapped[str | None] = mapped_column(Text)
    # 成功時は取得（取り込み）した時刻、失敗時は取得を試みた時刻
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Device(Base):
    __tablename__ = "device"

    device_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(10), default="other")  # pc/iphone/ipad/other
    push_subscription_json: Mapped[Any | None] = mapped_column(JSON)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --- 分離の設定 ---------------------------------------------------------------


class Model(Base):
    """分離モデル（audio-separator のチェックポイント）。"""

    __tablename__ = "model"

    model_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(300), unique=True)
    display_name: Mapped[str] = mapped_column(String(200))
    architecture: Mapped[str | None] = mapped_column(String(50))
    output_stems_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    min_vram_mb: Mapped[int | None] = mapped_column(Integer)
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64))
    license: Mapped[str] = mapped_column(String(100), default="unknown")
    source_url: Mapped[str | None] = mapped_column(Text)


class SeparationPreset(Base):
    __tablename__ = "separation_preset"

    preset_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True)  # fast / standard / best / exp_*
    display_name: Mapped[str] = mapped_column(String(100))
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    # 聴き比べ用の実験プリセット（画面では通常隠す）
    is_experimental: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0")
    )
    # パイプライン全体の選択肢（例: {"residual_to": "vocals"}）。NULL は {} と同じ
    options_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class PresetStep(Base):
    __tablename__ = "preset_step"

    preset_id: Mapped[int] = mapped_column(
        ForeignKey("separation_preset.preset_id", ondelete="CASCADE"), primary_key=True
    )
    step_order: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model.model_id"))
    input: Mapped[str] = mapped_column(String(20))  # mixture / vocals
    role: Mapped[str] = mapped_column(String(20))  # multistem / vocals / karaoke
    ensemble_weight: Mapped[float] = mapped_column(Float, default=1.0)
    options_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class StemType(Base):
    __tablename__ = "stem_type"

    stem_type_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True)
    display_name: Mapped[str] = mapped_column(String(100))
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("stem_type.stem_type_id"))
    tier: Mapped[str] = mapped_column(String(10))  # base / detail
    refine_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("model.model_id", ondelete="SET NULL")
    )
    experimental: Mapped[bool] = mapped_column(Boolean, default=False)
    color: Mapped[str] = mapped_column(String(7))  # #RRGGBB
    display_order: Mapped[int] = mapped_column(Integer, default=0)


# --- 分離ジョブと成果物 ---------------------------------------------------------


class SeparationJob(Base):
    __tablename__ = "separation_job"

    job_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("track.track_id", ondelete="CASCADE"))
    job_kind: Mapped[str] = mapped_column(String(10))  # full / refine
    preset_id: Mapped[int | None] = mapped_column(
        ForeignKey("separation_preset.preset_id")
    )  # full のみ
    input_stem_id: Mapped[int | None] = mapped_column(
        ForeignKey("stem.stem_id", ondelete="SET NULL", use_alter=True)
    )  # refine のみ
    requested_by: Mapped[int | None] = mapped_column(
        ForeignKey("device.device_id", ondelete="SET NULL")
    )
    # queued / running / done / failed / canceled
    status: Mapped[str] = mapped_column(String(10), default="queued")
    run_on: Mapped[str] = mapped_column(String(10), default="gpu")  # gpu / cpu / nightly
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0-1
    stage: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    # キャンセルの依頼（running のジョブ。ワーカーが検知して子プロセスを終了させる）
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0")
    )
    # 保存前に全 stem にかけた倍率（dB）。±1 を超えないよう下げたときだけ負の値
    output_gain_db: Mapped[float] = mapped_column(
        Float, default=0.0, server_default=text("0.0")
    )
    # 配信用データ（stream rendition・peaks）の作り直し（done のジョブのみ）。
    # NULL=依頼なし / queued / running / done / failed。ワーカーが1件ずつ処理する
    postprocess_status: Mapped[str | None] = mapped_column(String(10))
    # 補正前の残差（mixture − 上位 stem の生出力の合計）と mixture の RMS（dBFS）。聴き比べの参考
    residual_rms_db: Mapped[float | None] = mapped_column(Float)
    mixture_rms_db: Mapped[float | None] = mapped_column(Float)


class Stem(Base):
    __tablename__ = "stem"

    stem_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("separation_job.job_id", ondelete="CASCADE")
    )
    stem_type_id: Mapped[int] = mapped_column(ForeignKey("stem_type.stem_type_id"))
    parent_stem_id: Mapped[int | None] = mapped_column(
        ForeignKey("stem.stem_id", ondelete="CASCADE")
    )
    is_residual: Mapped[bool] = mapped_column(Boolean, default=False)
    rms_db: Mapped[float | None] = mapped_column(Float)
    is_silent: Mapped[bool] = mapped_column(Boolean, default=False)


class StemRendition(Base):
    __tablename__ = "stem_rendition"

    rendition_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stem_id: Mapped[int] = mapped_column(ForeignKey("stem.stem_id", ondelete="CASCADE"))
    purpose: Mapped[str] = mapped_column(String(10))  # master / stream
    codec: Mapped[str] = mapped_column(String(10))  # flac / opus / wav
    bitrate_kbps: Mapped[int | None] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(Text)
    bytes: Mapped[int | None] = mapped_column(Integer)


class Waveform(Base):
    __tablename__ = "waveform"

    stem_id: Mapped[int] = mapped_column(
        ForeignKey("stem.stem_id", ondelete="CASCADE"), primary_key=True
    )
    samples_per_px: Mapped[int] = mapped_column(Integer, primary_key=True)
    peaks_path: Mapped[str] = mapped_column(Text)


# --- グループと組み合わせプリセット ---------------------------------------------------


class StemGroup(Base):
    __tablename__ = "stem_group"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True)
    display_name: Mapped[str] = mapped_column(String(100))
    color: Mapped[str] = mapped_column(String(7))
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)


class StemGroupMember(Base):
    __tablename__ = "stem_group_member"

    group_id: Mapped[int] = mapped_column(
        ForeignKey("stem_group.group_id", ondelete="CASCADE"), primary_key=True
    )
    stem_type_id: Mapped[int] = mapped_column(
        ForeignKey("stem_type.stem_type_id", ondelete="CASCADE"), primary_key=True
    )


class ListenPreset(Base):
    __tablename__ = "listen_preset"

    __table_args__ = (
        Index("ux_listen_preset_seed_code", "seed_code", unique=True),
    )

    listen_preset_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    # 組み込み（seed で作る）の識別子。ユーザーが作ったものは NULL
    seed_code: Mapped[str | None] = mapped_column(String(50))
    # 組み込みをユーザーが削除したときは行を残して隠す（次の起動で seed が作り直さないように）
    hidden: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("0"))


class ListenPresetItem(Base):
    __tablename__ = "listen_preset_item"
    __table_args__ = (
        CheckConstraint(
            "(stem_type_id IS NULL) <> (group_id IS NULL)",
            name="ck_listen_preset_item_one_target",
        ),
    )

    item_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    listen_preset_id: Mapped[int] = mapped_column(
        ForeignKey("listen_preset.listen_preset_id", ondelete="CASCADE")
    )
    stem_type_id: Mapped[int | None] = mapped_column(
        ForeignKey("stem_type.stem_type_id", ondelete="CASCADE")
    )
    group_id: Mapped[int | None] = mapped_column(
        ForeignKey("stem_group.group_id", ondelete="CASCADE")
    )
    gain_db: Mapped[float] = mapped_column(Float, default=0.0)


# --- 再生・端末 ------------------------------------------------------------------


class PlaybackState(Base):
    __tablename__ = "playback_state"

    device_id: Mapped[int] = mapped_column(
        ForeignKey("device.device_id", ondelete="CASCADE"), primary_key=True
    )
    track_id: Mapped[int] = mapped_column(
        ForeignKey("track.track_id", ondelete="CASCADE"), primary_key=True
    )
    listen_preset_id: Mapped[int | None] = mapped_column(
        ForeignKey("listen_preset.listen_preset_id", ondelete="SET NULL")
    )
    channel_gains_json: Mapped[Any | None] = mapped_column(JSON)
    position_sec: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class CuePoint(Base):
    __tablename__ = "cue_point"

    cue_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("track.track_id", ondelete="CASCADE"))
    position_sec: Mapped[float] = mapped_column(Float)
    loop_end_sec: Mapped[float | None] = mapped_column(Float)
    label: Mapped[str | None] = mapped_column(String(100))
    color: Mapped[str | None] = mapped_column(String(7))


class OfflineCache(Base):
    __tablename__ = "offline_cache"

    device_id: Mapped[int] = mapped_column(
        ForeignKey("device.device_id", ondelete="CASCADE"), primary_key=True
    )
    track_id: Mapped[int] = mapped_column(
        ForeignKey("track.track_id", ondelete="CASCADE"), primary_key=True
    )
    cached_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    bytes: Mapped[int | None] = mapped_column(Integer)


# --- 書き出し --------------------------------------------------------------------


class Export(Base):
    __tablename__ = "export"

    export_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("separation_job.job_id", ondelete="CASCADE")
    )
    listen_preset_id: Mapped[int | None] = mapped_column(
        ForeignKey("listen_preset.listen_preset_id", ondelete="SET NULL")
    )  # mix のみ
    export_type: Mapped[str] = mapped_column(String(10))  # single / all / mix
    format: Mapped[str] = mapped_column(String(10))  # wav / flac / mp3 / zip
    output_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExportItem(Base):
    __tablename__ = "export_item"

    export_id: Mapped[int] = mapped_column(
        ForeignKey("export.export_id", ondelete="CASCADE"), primary_key=True
    )
    stem_id: Mapped[int] = mapped_column(
        ForeignKey("stem.stem_id", ondelete="CASCADE"), primary_key=True
    )


ALL_MODELS: tuple[type[Base], ...] = (
    Track,
    InputSource,
    Device,
    SeparationPreset,
    PresetStep,
    Model,
    SeparationJob,
    StemType,
    Stem,
    StemRendition,
    Waveform,
    StemGroup,
    StemGroupMember,
    ListenPreset,
    ListenPresetItem,
    PlaybackState,
    CuePoint,
    Export,
    ExportItem,
    OfflineCache,
)
