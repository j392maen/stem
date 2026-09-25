"""初期データの投入。何度実行しても同じ結果になる（冪等）。

- STEM_TYPE / MODEL / SEPARATION_PRESET / STEM_GROUP は一意キー（code, filename）で
  作成または更新する。
- PRESET_STEP と STEM_GROUP_MEMBER（組み込みのみ）は定義どおりに揃える。
- LISTEN_PRESET は名前で探し、無いときだけ作る（ユーザーが編集した中身は上書きしない）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from stemapp.models import (
    ListenPreset,
    ListenPresetItem,
    Model,
    PresetStep,
    SeparationPreset,
    StemGroup,
    StemGroupMember,
    StemType,
)

# --- MODEL -------------------------------------------------------------------


@dataclass(frozen=True)
class ModelDef:
    filename: str
    display_name: str
    architecture: str
    output_stems: list[str]
    license: str = "unknown"


SW = "BS-Roformer-SW.ckpt"
KIM_VOCALS = "vocals_mel_band_roformer.ckpt"
KIM_FT_UNWA = "mel_band_roformer_kim_ft_unwa.ckpt"
BS_317 = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
KARAOKE_MEL_BECRUILY = "mel_band_roformer_karaoke_becruily.ckpt"
KARAOKE_BS_FRAZER = "bs_roformer_karaoke_frazer_becruily.ckpt"
KARAOKE_BS_ANVUEW = "bs_roformer_karaoke_anvuew.ckpt"
KARAOKE_MEL_AUFR33 = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
DRUMSEP = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
MALE_FEMALE = "bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt"
ASPIRATION = "aspiration_mel_band_roformer_sdr_18.9845.ckpt"

KARAOKE_OUT = ["lead_vocal", "backing_vocal"]  # "(Vocals)"=lead, "(Instrumental)"=それ以外

MODELS: list[ModelDef] = [
    ModelDef(SW, "BS-Roformer SW（6 stem）", "bs_roformer",
             ["vocals", "drums", "bass", "guitar", "piano", "other"]),
    ModelDef(KIM_VOCALS, "Mel-RoFormer ボーカル（Kim）", "mel_band_roformer",
             ["vocals", "instrumental"]),
    ModelDef(KIM_FT_UNWA, "Mel-RoFormer ボーカル（Kim FT unwa）", "mel_band_roformer",
             ["vocals", "instrumental"]),
    ModelDef(BS_317, "BS-RoFormer ボーカル（ep317）", "bs_roformer",
             ["vocals", "instrumental"]),
    ModelDef(KARAOKE_MEL_BECRUILY, "Mel-RoFormer カラオケ（becruily）", "mel_band_roformer",
             KARAOKE_OUT),
    ModelDef(KARAOKE_BS_FRAZER, "BS-RoFormer カラオケ（frazer/becruily）", "bs_roformer",
             KARAOKE_OUT),
    ModelDef(KARAOKE_BS_ANVUEW, "BS-RoFormer カラオケ（anvuew）", "bs_roformer",
             KARAOKE_OUT),
    ModelDef(KARAOKE_MEL_AUFR33, "Mel-RoFormer カラオケ（aufr33/viperx）", "mel_band_roformer",
             KARAOKE_OUT),
    ModelDef(DRUMSEP, "MDX23C ドラム分割", "mdx23c",
             ["kick", "snare", "toms", "hihat", "ride", "crash"]),
    ModelDef(MALE_FEMALE, "BS-RoFormer 男声/女声", "bs_roformer", ["male", "female"]),
    ModelDef(ASPIRATION, "Mel-RoFormer 息", "mel_band_roformer", ["breath", "no_breath"]),
]

# --- STEM_TYPE ---------------------------------------------------------------


@dataclass(frozen=True)
class StemTypeDef:
    code: str
    display_name: str
    parent: str | None
    tier: str
    color: str
    experimental: bool = False
    refine_model: str | None = None


BASE_STEM_CODES: tuple[str, ...] = (
    "vocals", "lead_vocal", "backing_vocal", "drums", "bass", "guitar", "piano", "other",
)

# 並び順＝表示順。親は必ず子より前に置く。
STEM_TYPES: list[StemTypeDef] = [
    # 基本（分割時に必ず作る）
    StemTypeDef("vocals", "ボーカル", None, "base", "#E6194B"),
    StemTypeDef("lead_vocal", "メインボーカル", "vocals", "base", "#FF4D6D"),
    StemTypeDef("backing_vocal", "サブボーカル", "vocals", "base", "#FF9EB5"),
    StemTypeDef("drums", "ドラム", None, "base", "#F58231"),
    StemTypeDef("bass", "ベース", None, "base", "#3CB44B"),
    StemTypeDef("guitar", "ギター", None, "base", "#FFE119"),
    StemTypeDef("piano", "ピアノ", None, "base", "#4363D8"),
    StemTypeDef("other", "その他", None, "base", "#911EB4"),
    # ボーカルの詳細
    StemTypeDef("male", "男声", "vocals", "detail", "#C2185B", refine_model=MALE_FEMALE),
    StemTypeDef("female", "女声", "vocals", "detail", "#F48FB1", refine_model=MALE_FEMALE),
    StemTypeDef("breath", "息", "vocals", "detail", "#FFCDD2", refine_model=ASPIRATION),
    # ドラムの詳細
    StemTypeDef("kick", "キック", "drums", "detail", "#E65100", refine_model=DRUMSEP),
    StemTypeDef("snare", "スネア", "drums", "detail", "#FB8C00", refine_model=DRUMSEP),
    StemTypeDef("toms", "タム", "drums", "detail", "#FFA726", refine_model=DRUMSEP),
    StemTypeDef("hihat", "ハイハット", "drums", "detail", "#FFCC80", experimental=True,
                refine_model=DRUMSEP),
    StemTypeDef("ride", "ライド", "drums", "detail", "#FFE0B2", experimental=True,
                refine_model=DRUMSEP),
    StemTypeDef("crash", "クラッシュ", "drums", "detail", "#BF360C", experimental=True,
                refine_model=DRUMSEP),
    # ギターの詳細
    StemTypeDef("acoustic_guitar", "アコースティックギター", "guitar", "detail", "#FDD835"),
    StemTypeDef("electric_guitar", "エレキギター", "guitar", "detail", "#F9A825"),
    # その他の詳細
    StemTypeDef("wind", "管楽器", "other", "detail", "#46F0F0", experimental=True),
    StemTypeDef("saxophone", "サックス", "other", "detail", "#00ACC1"),
    StemTypeDef("brass", "金管", "other", "detail", "#FFD700"),
    StemTypeDef("woodwind", "木管", "other", "detail", "#80CBC4", experimental=True),
    StemTypeDef("strings", "ストリングス", "other", "detail", "#AA6E28"),
    StemTypeDef("organ", "オルガン", "other", "detail", "#6D4C41"),
    StemTypeDef("keys", "キーボード", "other", "detail", "#7986CB"),
    StemTypeDef("synth", "シンセ", "other", "detail", "#F032E6", experimental=True),
    StemTypeDef("percussion", "パーカッション", "other", "detail", "#808000",
                experimental=True),
    # パーカッションの詳細
    StemTypeDef("congas", "コンガ", "percussion", "detail", "#9E9D24", experimental=True),
    StemTypeDef("tambourine", "タンバリン", "percussion", "detail", "#C0CA33",
                experimental=True),
    StemTypeDef("triangle", "トライアングル", "percussion", "detail", "#D4E157",
                experimental=True),
    StemTypeDef("bells", "ベル", "percussion", "detail", "#E6EE9C", experimental=True),
    StemTypeDef("glockenspiel", "グロッケンシュピール", "percussion", "detail", "#AFB42B",
                experimental=True),
]

# --- SEPARATION_PRESET / PRESET_STEP -------------------------------------------


@dataclass(frozen=True)
class StepDef:
    model: str
    input: str  # mixture / vocals
    role: str  # multistem / vocals / karaoke
    weight: float = 1.0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PresetDef:
    code: str
    display_name: str
    is_default: bool
    steps: list[StepDef]


# 数値（overlap 等）は仮の値。T02/T09 で実測して調整する。
PRESETS: list[PresetDef] = [
    PresetDef("fast", "速い", False, [
        StepDef(SW, "mixture", "multistem", options={"overlap": 2}),
        StepDef(KARAOKE_MEL_BECRUILY, "vocals", "karaoke", options={"overlap": 2}),
    ]),
    PresetDef("standard", "標準", True, [
        StepDef(SW, "mixture", "multistem", options={"overlap": 4}),
        StepDef(KIM_VOCALS, "mixture", "vocals", options={"overlap": 4}),
        StepDef(KARAOKE_BS_FRAZER, "vocals", "karaoke", options={"overlap": 4}),
    ]),
    PresetDef("best", "高品質", False, [
        StepDef(SW, "mixture", "multistem", options={"overlap": 4, "tta": True}),
        StepDef(KIM_VOCALS, "mixture", "vocals", options={"overlap": 4, "tta": True}),
        StepDef(KIM_FT_UNWA, "mixture", "vocals", options={"overlap": 4, "tta": True}),
        StepDef(KARAOKE_MEL_BECRUILY, "vocals", "karaoke", options={"overlap": 4, "tta": True}),
        StepDef(KARAOKE_BS_FRAZER, "vocals", "karaoke", options={"overlap": 4, "tta": True}),
    ]),
]

# --- STEM_GROUP --------------------------------------------------------------


@dataclass(frozen=True)
class GroupDef:
    code: str
    display_name: str
    color: str
    members: list[str]


GROUPS: list[GroupDef] = [
    GroupDef("vocals_all", "ボーカル", "#E6194B", ["lead_vocal", "backing_vocal"]),
    GroupDef("chords", "コード", "#4363D8", ["guitar", "piano", "other"]),
    GroupDef("rhythm", "リズム", "#F58231", ["drums", "bass"]),
    GroupDef("accompaniment", "伴奏", "#3CB44B", ["drums", "bass", "guitar", "piano", "other"]),
]

# --- LISTEN_PRESET -----------------------------------------------------------


@dataclass(frozen=True)
class ListenDef:
    name: str
    stem_types: list[str]
    groups: list[str]


LISTEN_PRESETS: list[ListenDef] = [
    ListenDef("ベース＋コード", ["bass"], ["chords"]),
    ListenDef("ドラム＋ボーカル＋コード", ["drums"], ["vocals_all", "chords"]),
    ListenDef("カラオケ（伴奏）", [], ["accompaniment"]),
    ListenDef("サブボーカルのみ", ["backing_vocal"], []),
]


# --- 投入処理 -------------------------------------------------------------------


def _seed_models(session: Session) -> dict[str, Model]:
    existing = {m.filename: m for m in session.scalars(select(Model))}
    for d in MODELS:
        m = existing.get(d.filename)
        if m is None:
            m = Model(filename=d.filename)
            session.add(m)
            existing[d.filename] = m
        m.display_name = d.display_name
        m.architecture = d.architecture
        m.output_stems_json = list(d.output_stems)
        m.license = d.license
    session.flush()
    return existing


def _seed_stem_types(session: Session, models: dict[str, Model]) -> dict[str, StemType]:
    existing = {t.code: t for t in session.scalars(select(StemType))}
    for order, d in enumerate(STEM_TYPES, start=1):
        t = existing.get(d.code)
        if t is None:
            t = StemType(code=d.code)
            session.add(t)
            existing[d.code] = t
        t.display_name = d.display_name
        t.parent_id = existing[d.parent].stem_type_id if d.parent else None
        t.tier = d.tier
        t.color = d.color
        t.experimental = d.experimental
        t.refine_model_id = models[d.refine_model].model_id if d.refine_model else None
        t.display_order = order * 10
        session.flush()  # 子が親の ID を参照できるように
    return existing


def _seed_presets(session: Session, models: dict[str, Model]) -> None:
    existing = {p.code: p for p in session.scalars(select(SeparationPreset))}
    for d in PRESETS:
        p = existing.get(d.code)
        if p is None:
            p = SeparationPreset(code=d.code)
            session.add(p)
        p.display_name = d.display_name
        p.is_default = d.is_default
        session.flush()

        steps = {
            s.step_order: s
            for s in session.scalars(select(PresetStep).where(PresetStep.preset_id == p.preset_id))
        }
        for order, sd in enumerate(d.steps, start=1):
            s = steps.pop(order, None)
            if s is None:
                s = PresetStep(preset_id=p.preset_id, step_order=order)
                session.add(s)
            s.model_id = models[sd.model].model_id
            s.input = sd.input
            s.role = sd.role
            s.ensemble_weight = sd.weight
            s.options_json = dict(sd.options)
        for extra in steps.values():
            session.delete(extra)
    session.flush()


def _seed_groups(session: Session, types: dict[str, StemType]) -> dict[str, StemGroup]:
    existing = {g.code: g for g in session.scalars(select(StemGroup))}
    for d in GROUPS:
        g = existing.get(d.code)
        if g is None:
            g = StemGroup(code=d.code)
            session.add(g)
            existing[d.code] = g
        g.display_name = d.display_name
        g.color = d.color
        g.is_builtin = True
        session.flush()

        want = {types[c].stem_type_id for c in d.members}
        have = set(
            session.scalars(
                select(StemGroupMember.stem_type_id).where(StemGroupMember.group_id == g.group_id)
            )
        )
        for type_id in sorted(want - have):
            session.add(StemGroupMember(group_id=g.group_id, stem_type_id=type_id))
        if have - want:
            session.execute(
                delete(StemGroupMember).where(
                    StemGroupMember.group_id == g.group_id,
                    StemGroupMember.stem_type_id.in_(have - want),
                )
            )
    session.flush()
    return existing


def _seed_listen_presets(
    session: Session, types: dict[str, StemType], groups: dict[str, StemGroup]
) -> None:
    existing = {p.name: p for p in session.scalars(select(ListenPreset))}
    for order, d in enumerate(LISTEN_PRESETS, start=1):
        if d.name in existing:
            continue
        p = ListenPreset(name=d.name, sort_order=order * 10)
        session.add(p)
        session.flush()
        for code in d.stem_types:
            session.add(
                ListenPresetItem(
                    listen_preset_id=p.listen_preset_id,
                    stem_type_id=types[code].stem_type_id,
                    gain_db=0.0,
                )
            )
        for code in d.groups:
            session.add(
                ListenPresetItem(
                    listen_preset_id=p.listen_preset_id,
                    group_id=groups[code].group_id,
                    gain_db=0.0,
                )
            )
    session.flush()


def seed(session: Session) -> None:
    """初期データを投入する（コミットまで行う）。"""
    models = _seed_models(session)
    types = _seed_stem_types(session, models)
    _seed_presets(session, models)
    groups = _seed_groups(session, types)
    _seed_listen_presets(session, types, groups)
    session.commit()
