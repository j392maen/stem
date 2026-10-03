"""初期データの投入。何度実行しても同じ結果になる（冪等）。

- STEM_TYPE / MODEL / SEPARATION_PRESET / STEM_GROUP は一意キー（code, filename）で
  作成または更新する。
- PRESET_STEP と STEM_GROUP_MEMBER（組み込みのみ）は定義どおりに揃える。
- LISTEN_PRESET は seed_code（組み込みの識別子）で探し、無いときだけ作る（ユーザーが編集した
  名前・中身は上書きしない。ユーザーが削除した組み込みは hidden で残っているので作り直さない）。
  seed_code を持たない古い DB では、同じ名前の行に seed_code を付けて組み込みとみなす。
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
    # 詳細分割の方法の並び順（小さいほど先。同じ親の stem で先頭のものが既定）。None は後ろ
    refine_order: int | None = None
    # 詳細分割の方法として「実験」扱い（画面に「実験」と出す）
    experimental: bool = False


SW = "BS-Roformer-SW.ckpt"
KIM_VOCALS = "vocals_mel_band_roformer.ckpt"
KIM_FT_UNWA = "mel_band_roformer_kim_ft_unwa.ckpt"
BS_317 = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
KARAOKE_MEL_BECRUILY = "mel_band_roformer_karaoke_becruily.ckpt"
KARAOKE_BS_FRAZER = "bs_roformer_karaoke_frazer_becruily.ckpt"
KARAOKE_BS_ANVUEW = "bs_roformer_karaoke_anvuew.ckpt"
KARAOKE_MEL_AUFR33 = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
KARAOKE_MEL_GABOX = "mel_band_roformer_karaoke_gabox.ckpt"
KARAOKE_MEL_GABOX_V2 = "mel_band_roformer_karaoke_gabox_v2.ckpt"
DRUMSEP = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
MALE_FEMALE = "bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt"
ASPIRATION = "aspiration_mel_band_roformer_sdr_18.9845.ckpt"
# 信号処理（librosa の HPSS）。モデルファイルは無い。詳細分割の「方法」を MODEL の行でそろえて
# 持つため、architecture="hpss" の行として登録する（stemapp.separation.refine が特別に扱う）
HPSS = "hpss"
# MVSep Mega 53 stems（MSST 形式の BS-Roformer。53 stem のうち other の子に使うものだけ動かす。
# stemapp.separation.msst、docs/research/R02-mega53.md）。重みのライセンスは明示なし
MEGA53 = "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt"
# Mega 53 から子にする stem（STEM_TYPE の code）。互いに重なりの少ない組（R02 2章）。
# keys は synth・organ と、organ は synth と重なるので入れない（重なると同じ音が2回入り、
# 「残り」がそれを打ち消す逆相の音になる）
MEGA53_CHILDREN = ["strings", "brass", "woodwind", "synth", "percussion"]

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
             KARAOKE_OUT, license="GPL-3.0"),
    ModelDef(KARAOKE_MEL_AUFR33, "Mel-RoFormer カラオケ（aufr33/viperx）", "mel_band_roformer",
             KARAOKE_OUT),
    ModelDef(KARAOKE_MEL_GABOX, "Mel-RoFormer カラオケ（gabox）", "mel_band_roformer",
             KARAOKE_OUT),
    ModelDef(KARAOKE_MEL_GABOX_V2, "Mel-RoFormer カラオケ（gabox v2）", "mel_band_roformer",
             KARAOKE_OUT),
    # 詳細分割の方法（refine_order の小さいものが先頭・既定）。
    # 「その他」は Mega 53 が既定。HPSS は短い音の側にノイズしか残らないことが多いので実験扱いで
    # 末尾（ユーザーの確認、2026-10-03。T17）
    ModelDef(DRUMSEP, "MDX23C ドラム分割", "mdx23c",
             ["kick", "snare", "toms", "hihat", "ride", "crash"], refine_order=10),
    ModelDef(MALE_FEMALE, "BS-RoFormer 男声/女声", "bs_roformer", ["male", "female"],
             refine_order=20),
    ModelDef(ASPIRATION, "Mel-RoFormer 息", "mel_band_roformer", ["breath", "no_breath"],
             refine_order=30),
    ModelDef(MEGA53, "Mega 53（楽器別）", "msst_bs_roformer", MEGA53_CHILDREN,
             license="不明（明示なし。コードは MIT）", refine_order=40),
    ModelDef(HPSS, "HPSS（持続音／短い音）", "hpss", ["sustained", "transient"],
             license="ISC（librosa）", refine_order=90, experimental=True),
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
# 色は暗い背景で見分けやすく、画面の差し色（赤系統）と紛れないよう、赤〜ピンクの色相を避ける。
# 基本 stem は色相を離し（紫・黄・緑・シアン・青・赤紫）、詳細 stem は親と同じ系統で明るさを変える。
STEM_TYPES: list[StemTypeDef] = [
    # 基本（分割時に必ず作る）
    StemTypeDef("vocals", "ボーカル", None, "base", "#A78BFA"),
    StemTypeDef("lead_vocal", "メインボーカル", "vocals", "base", "#E9D5FF"),
    StemTypeDef("backing_vocal", "サブボーカル", "vocals", "base", "#6366F1"),
    StemTypeDef("drums", "ドラム", None, "base", "#FACC15"),
    StemTypeDef("bass", "ベース", None, "base", "#4ADE80"),
    StemTypeDef("guitar", "ギター", None, "base", "#22D3EE"),
    StemTypeDef("piano", "ピアノ", None, "base", "#60A5FA"),
    StemTypeDef("other", "その他", None, "base", "#E879F9"),
    # ボーカルの詳細（紫の系統）
    StemTypeDef("male", "男声", "vocals", "detail", "#7C3AED", refine_model=MALE_FEMALE),
    StemTypeDef("female", "女声", "vocals", "detail", "#C4B5FD", refine_model=MALE_FEMALE),
    StemTypeDef("breath", "息", "vocals", "detail", "#DDD6FE", refine_model=ASPIRATION),
    # 詳細分割の「残り」（親 − 名前の付いた子の合計）。親の stem ごとに1つ（<親>_rest）。
    # 色は親の系統の彩度を落としたもの
    StemTypeDef("lead_vocal_rest", "残り（メインボーカル）", "lead_vocal", "detail", "#B7AFC9"),
    StemTypeDef("backing_vocal_rest", "残り（サブボーカル）", "backing_vocal", "detail",
                "#7E7FA6"),
    # ドラムの詳細（黄の系統）
    StemTypeDef("kick", "キック", "drums", "detail", "#EAB308", refine_model=DRUMSEP),
    StemTypeDef("snare", "スネア", "drums", "detail", "#FDE047", refine_model=DRUMSEP),
    StemTypeDef("toms", "タム", "drums", "detail", "#CA8A04", refine_model=DRUMSEP),
    StemTypeDef("hihat", "ハイハット", "drums", "detail", "#FEF08A", experimental=True,
                refine_model=DRUMSEP),
    StemTypeDef("ride", "ライド", "drums", "detail", "#A16207", experimental=True,
                refine_model=DRUMSEP),
    StemTypeDef("crash", "クラッシュ", "drums", "detail", "#FEF9C3", experimental=True,
                refine_model=DRUMSEP),
    StemTypeDef("drums_rest", "残り（ドラム）", "drums", "detail", "#A8996A"),
    # ギターの詳細（シアンの系統）
    StemTypeDef("acoustic_guitar", "アコースティックギター", "guitar", "detail", "#A5F3FC"),
    StemTypeDef("electric_guitar", "エレキギター", "guitar", "detail", "#0891B2"),
    # その他の詳細（赤紫・水色・黄緑・灰色など、基本 stem と重ならないもの）
    # 持続音・短い音は楽器の種類ではなく音の「形」で分ける（HPSS。docs/research/R01 D-3）
    StemTypeDef("sustained", "持続音（パッド等）", "other", "detail", "#F5D0FE",
                experimental=True, refine_model=HPSS),
    StemTypeDef("transient", "短い音（ヒット等）", "other", "detail", "#A21CAF",
                experimental=True, refine_model=HPSS),
    StemTypeDef("wind", "管楽器", "other", "detail", "#67E8F9", experimental=True),
    StemTypeDef("saxophone", "サックス", "other", "detail", "#06B6D4"),
    StemTypeDef("brass", "金管", "other", "detail", "#BEF264", refine_model=MEGA53),
    StemTypeDef("woodwind", "木管", "other", "detail", "#86EFAC", experimental=True,
                refine_model=MEGA53),
    StemTypeDef("strings", "ストリングス", "other", "detail", "#F0ABFC", refine_model=MEGA53),
    StemTypeDef("organ", "オルガン", "other", "detail", "#C026D3"),
    StemTypeDef("keys", "キーボード", "other", "detail", "#93C5FD"),
    StemTypeDef("synth", "シンセ", "other", "detail", "#D946EF", experimental=True,
                refine_model=MEGA53),
    StemTypeDef("percussion", "パーカッション", "other", "detail", "#A3A3A3",
                experimental=True, refine_model=MEGA53),
    # パーカッションの詳細（灰色の系統）
    StemTypeDef("congas", "コンガ", "percussion", "detail", "#D4D4D4", experimental=True),
    StemTypeDef("tambourine", "タンバリン", "percussion", "detail", "#737373",
                experimental=True),
    StemTypeDef("triangle", "トライアングル", "percussion", "detail", "#E5E5E5",
                experimental=True),
    StemTypeDef("bells", "ベル", "percussion", "detail", "#BDB76B", experimental=True),
    StemTypeDef("glockenspiel", "グロッケンシュピール", "percussion", "detail", "#8B8B6B",
                experimental=True),
    # 「残り」は other の子の最後に並べる（HPSS・Mega 53 のどちらで分けても子の後に出る）
    StemTypeDef("other_rest", "残り（その他）", "other", "detail", "#A58AAB"),
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
    # 聴き比べ用の実験プリセット（画面では通常隠す）
    experimental: bool = False
    # パイプライン全体の選択肢（pipeline.PRESET_OPTION_KEYS）。例: {"residual_to": "vocals"}
    options: dict[str, Any] = field(default_factory=dict)


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

# 聴き比べ用（T12、docs/research/R01 の D 章）。どれも standard（SW＋Kim の平均）を元にする。
# 既定は変えない。ユーザーが聴いて良いものを選んだら、standard / best に取り込む。
_SW4 = StepDef(SW, "mixture", "multistem", options={"overlap": 4})
_KIM4 = StepDef(KIM_VOCALS, "mixture", "vocals", options={"overlap": 4})


def _kara(model: str, source: str) -> StepDef:
    return StepDef(model, source, "karaoke", options={"overlap": 4})


RESID_VOCALS = {"residual_to": "vocals"}
EXPERIMENTAL_PRESETS: list[PresetDef] = [
    # 残差を vocals に足す。vocals = 元の曲 − SW の楽器 stem の合計 になり、ボーカル専用モデル
    # （Kim）の出力は結果に効かないので使わない（validate_plan も弾く）
    PresetDef("exp_resid_vocals", "ボーカル＝元の曲−楽器", False,
              [_SW4, _kara(KARAOKE_BS_FRAZER, "vocals")],
              experimental=True, options=RESID_VOCALS),
    # 対照: SW だけ（Kim の平均なし）。standard と比べて Kim が役立っているかを確かめる。
    # （残差の split はボーカル専用モデルが無いと other と同じなので、選択肢は既定のまま）
    PresetDef("exp_resid_split", "SW のみ（対照）", False,
              [_SW4, _kara(KARAOKE_BS_FRAZER, "vocals")],
              experimental=True),
    # karaoke を元の曲にかける（lead = karaoke の出力、backing = vocals − lead）
    PresetDef("exp_kara_mix", "カラオケを元の曲に", False,
              [_SW4, _KIM4, _kara(KARAOKE_BS_FRAZER, "mixture")],
              experimental=True),
    # karaoke を anvuew ＋ frazer の平均に（vocals にかける）
    PresetDef("exp_kara_anvuew", "カラオケ2種（anvuew＋frazer）", False,
              [_SW4, _KIM4, _kara(KARAOKE_BS_ANVUEW, "vocals"),
               _kara(KARAOKE_BS_FRAZER, "vocals")],
              experimental=True),
    # vocals = 元の曲 − 楽器 ＋ karaoke（anvuew ＋ frazer）を元の曲に
    PresetDef("exp_combo", "ボーカル＝元の曲−楽器＋カラオケ2種を元の曲に", False,
              [_SW4, _kara(KARAOKE_BS_ANVUEW, "mixture"), _kara(KARAOKE_BS_FRAZER, "mixture")],
              experimental=True, options=RESID_VOCALS),
    # exp_combo に gabox v2（Mel-RoFormer）を足した 3 種の平均（系統の違うモデルを混ぜる）
    PresetDef("exp_combo_gabox", "ボーカル＝元の曲−楽器＋カラオケ3種を元の曲に", False,
              [_SW4, _kara(KARAOKE_BS_ANVUEW, "mixture"),
               _kara(KARAOKE_BS_FRAZER, "mixture"), _kara(KARAOKE_MEL_GABOX_V2, "mixture")],
              experimental=True, options=RESID_VOCALS),
]

# --- STEM_GROUP --------------------------------------------------------------


@dataclass(frozen=True)
class GroupDef:
    code: str
    display_name: str
    color: str
    members: list[str]


# グループの色は、メンバーの stem のどの色とも違うものにする（赤系統も避ける）。
GROUPS: list[GroupDef] = [
    GroupDef("vocals_all", "ボーカル", "#FB923C", ["lead_vocal", "backing_vocal"]),
    GroupDef("chords", "コード", "#A3E635", ["guitar", "piano", "other"]),
    GroupDef("rhythm", "リズム", "#94A3B8", ["drums", "bass"]),
    GroupDef("accompaniment", "伴奏", "#D4A373", ["drums", "bass", "guitar", "piano", "other"]),
]

# --- LISTEN_PRESET -----------------------------------------------------------


@dataclass(frozen=True)
class ListenDef:
    code: str  # LISTEN_PRESET.seed_code
    name: str
    stem_types: list[str]
    groups: list[str]


LISTEN_PRESETS: list[ListenDef] = [
    ListenDef("bass_chords", "ベース＋コード", ["bass"], ["chords"]),
    ListenDef(
        "drums_vocals_chords", "ドラム＋ボーカル＋コード", ["drums"], ["vocals_all", "chords"]
    ),
    ListenDef("karaoke", "カラオケ（伴奏）", [], ["accompaniment"]),
    ListenDef("backing_only", "サブボーカルのみ", ["backing_vocal"], []),
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
        m.refine_order = d.refine_order
        m.is_experimental = d.experimental
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
    for d in [*PRESETS, *EXPERIMENTAL_PRESETS]:
        p = existing.get(d.code)
        if p is None:
            p = SeparationPreset(code=d.code)
            session.add(p)
        p.display_name = d.display_name
        p.is_default = d.is_default
        p.is_experimental = d.experimental
        p.options_json = dict(d.options)
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
    rows = session.scalars(select(ListenPreset).order_by(ListenPreset.listen_preset_id)).all()
    by_code = {p.seed_code: p for p in rows if p.seed_code}
    for order, d in enumerate(LISTEN_PRESETS, start=1):
        if d.code in by_code:
            continue
        # 移行: seed_code を持たない同じ名前の行（以前の seed が作ったもの）を組み込みとみなす
        legacy = next((p for p in rows if p.seed_code is None and p.name == d.name), None)
        if legacy is not None:
            legacy.seed_code = d.code
            by_code[d.code] = legacy
            continue
        p = ListenPreset(name=d.name, sort_order=order * 10, seed_code=d.code)
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
