from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from stemapp import seed as seed_mod
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
from stemapp.seed import BASE_STEM_CODES, seed


def _snapshot(session: Session) -> dict[str, list[Any]]:
    """全初期データ表の中身を比較可能な形にする。"""
    def rows(model: type, *order: Any) -> list[tuple[Any, ...]]:
        cols = [c.key for c in model.__table__.columns]  # type: ignore[attr-defined]
        return [
            tuple(getattr(r, c) for c in cols)
            for r in session.scalars(select(model).order_by(*order))
        ]

    return {
        "model": rows(Model, Model.model_id),
        "stem_type": rows(StemType, StemType.stem_type_id),
        "preset": rows(SeparationPreset, SeparationPreset.preset_id),
        "step": rows(PresetStep, PresetStep.preset_id, PresetStep.step_order),
        "group": rows(StemGroup, StemGroup.group_id),
        "member": rows(StemGroupMember, StemGroupMember.group_id, StemGroupMember.stem_type_id),
        "listen": rows(ListenPreset, ListenPreset.listen_preset_id),
        "listen_item": rows(ListenPresetItem, ListenPresetItem.item_id),
    }


def test_seed_idempotent(session: Session) -> None:
    seed(session)
    first = _snapshot(session)
    seed(session)
    seed(session)
    session.expire_all()
    assert _snapshot(session) == first
    assert all(first.values())  # どの表も空ではない


def test_seed_restores_builtin_definitions(session: Session) -> None:
    seed(session)
    first = _snapshot(session)
    # 組み込み定義を壊してから再投入すると元に戻る
    bass = session.scalar(select(StemType).where(StemType.code == "bass"))
    assert bass is not None
    bass.color = "#000000"
    std = session.scalar(select(SeparationPreset).where(SeparationPreset.code == "standard"))
    assert std is not None
    session.add(PresetStep(preset_id=std.preset_id, step_order=99, model_id=1,
                           input="mixture", role="vocals"))
    session.commit()
    seed(session)
    session.expire_all()
    assert _snapshot(session) == first


def test_stem_tree(session: Session) -> None:
    seed(session)
    types = {t.stem_type_id: t for t in session.scalars(select(StemType))}
    by_code = {t.code: t for t in types.values()}
    assert len(types) == len(seed_mod.STEM_TYPES)

    for t in types.values():
        # 親が存在し、循環しない
        seen = {t.stem_type_id}
        cur = t
        while cur.parent_id is not None:
            assert cur.parent_id in types
            cur = types[cur.parent_id]
            assert cur.stem_type_id not in seen, f"循環: {t.code}"
            seen.add(cur.stem_type_id)
        assert t.tier in ("base", "detail")
        assert t.color.startswith("#") and len(t.color) == 7

    assert {c for c, t in by_code.items() if t.tier == "base"} == set(BASE_STEM_CODES)
    vocals = by_code["vocals"].stem_type_id
    for code in ("lead_vocal", "backing_vocal", "male", "female", "breath"):
        assert by_code[code].parent_id == vocals
    for code in ("kick", "snare", "toms", "hihat", "ride", "crash"):
        assert by_code[code].parent_id == by_code["drums"].stem_type_id
    for code in ("acoustic_guitar", "electric_guitar"):
        assert by_code[code].parent_id == by_code["guitar"].stem_type_id
    for code in ("synth", "woodwind", "percussion", "tambourine", "hihat"):
        assert by_code[code].experimental
    assert not by_code["bass"].experimental
    # 色は見分けやすいよう重複させない
    colors = [t.color.upper() for t in types.values()]
    assert len(colors) == len(set(colors))


def test_models_and_presets(session: Session) -> None:
    seed(session)
    models = {m.model_id: m for m in session.scalars(select(Model))}
    assert "BS-Roformer-SW.ckpt" in {m.filename for m in models.values()}
    assert all(m.license for m in models.values())

    presets = {p.code: p for p in session.scalars(select(SeparationPreset))}
    assert set(presets) == {"fast", "standard", "best"}
    assert [c for c, p in presets.items() if p.is_default] == ["standard"]

    sw = "BS-Roformer-SW.ckpt"
    kim = "vocals_mel_band_roformer.ckpt"
    kim_unwa = "mel_band_roformer_kim_ft_unwa.ckpt"
    kara_mel = "mel_band_roformer_karaoke_becruily.ckpt"
    kara_bs = "bs_roformer_karaoke_frazer_becruily.ckpt"
    expected = {
        "fast": [
            (sw, "mixture", "multistem"),
            (kara_mel, "vocals", "karaoke"),
        ],
        "standard": [
            (sw, "mixture", "multistem"),
            (kim, "mixture", "vocals"),
            (kara_bs, "vocals", "karaoke"),
        ],
        "best": [
            (sw, "mixture", "multistem"),
            (kim, "mixture", "vocals"),
            (kim_unwa, "mixture", "vocals"),
            (kara_mel, "vocals", "karaoke"),
            (kara_bs, "vocals", "karaoke"),
        ],
    }

    steps: dict[str, list[PresetStep]] = {}
    for code, p in presets.items():
        steps[code] = list(
            session.scalars(
                select(PresetStep)
                .where(PresetStep.preset_id == p.preset_id)
                .order_by(PresetStep.step_order)
            )
        )
        actual = [(models[s.model_id].filename, s.input, s.role) for s in steps[code]]
        assert actual == expected[code], code
        assert [s.step_order for s in steps[code]] == list(range(1, len(actual) + 1))

    # best は全ステップ TTA、fast は standard より overlap が小さい（以下）
    assert all(s.options_json.get("tta") is True for s in steps["best"])
    assert not any(s.options_json.get("tta") for s in steps["fast"] + steps["standard"])
    fast_overlap = max(s.options_json["overlap"] for s in steps["fast"])
    std_overlap = min(s.options_json["overlap"] for s in steps["standard"])
    assert fast_overlap <= std_overlap


def test_aspiration_outputs_not_confused_with_other(session: Session) -> None:
    seed(session)
    m = session.scalar(
        select(Model).where(Model.filename == "aspiration_mel_band_roformer_sdr_18.9845.ckpt")
    )
    assert m is not None
    assert m.output_stems_json == ["breath", "no_breath"]


def _members(session: Session, code: str) -> set[str]:
    return set(
        session.scalars(
            select(StemType.code)
            .join(StemGroupMember, StemGroupMember.stem_type_id == StemType.stem_type_id)
            .join(StemGroup, StemGroup.group_id == StemGroupMember.group_id)
            .where(StemGroup.code == code)
        )
    )


def test_groups(session: Session) -> None:
    seed(session)
    assert _members(session, "vocals_all") == {"lead_vocal", "backing_vocal"}
    assert _members(session, "chords") == {"guitar", "piano", "other"}
    assert _members(session, "rhythm") == {"drums", "bass"}
    assert _members(session, "accompaniment") == {"drums", "bass", "guitar", "piano", "other"}
    assert all(g.is_builtin for g in session.scalars(select(StemGroup)))


def test_listen_presets(session: Session) -> None:
    seed(session)
    result: dict[str, set[str]] = {}
    for p in session.scalars(select(ListenPreset).order_by(ListenPreset.sort_order)):
        items = session.scalars(
            select(ListenPresetItem).where(ListenPresetItem.listen_preset_id == p.listen_preset_id)
        )
        names: set[str] = set()
        for it in items:
            if it.stem_type_id is not None:
                t = session.get(StemType, it.stem_type_id)
                assert t is not None
                names.add(f"type:{t.code}")
            else:
                g = session.get(StemGroup, it.group_id)
                assert g is not None
                names.add(f"group:{g.code}")
        result[p.name] = names
    assert result == {
        "ベース＋コード": {"type:bass", "group:chords"},
        "ドラム＋ボーカル＋コード": {"type:drums", "group:vocals_all", "group:chords"},
        "カラオケ（伴奏）": {"group:accompaniment"},
        "サブボーカルのみ": {"type:backing_vocal"},
    }


def test_listen_preset_user_edits_kept(session: Session) -> None:
    seed(session)
    p = session.scalar(select(ListenPreset).where(ListenPreset.name == "サブボーカルのみ"))
    assert p is not None
    item = session.scalar(
        select(ListenPresetItem).where(ListenPresetItem.listen_preset_id == p.listen_preset_id)
    )
    assert item is not None
    item.gain_db = -6.0
    session.commit()
    seed(session)
    session.expire_all()
    assert session.get(ListenPresetItem, item.item_id).gain_db == -6.0  # type: ignore[union-attr]


def _visible_presets(session: Session) -> list[ListenPreset]:
    return list(
        session.scalars(
            select(ListenPreset)
            .where(ListenPreset.hidden.is_(False))
            .order_by(ListenPreset.listen_preset_id)
        )
    )


def test_builtin_listen_presets_have_seed_code(session: Session) -> None:
    seed(session)
    codes = {p.seed_code: p.name for p in _visible_presets(session)}
    assert codes == {d.code: d.name for d in seed_mod.LISTEN_PRESETS}


def test_renamed_builtin_is_not_duplicated(session: Session) -> None:
    """組み込みの名前を変えても、次の起動（seed）で同じものを作り直さない。"""
    seed(session)
    p = session.scalar(select(ListenPreset).where(ListenPreset.seed_code == "karaoke"))
    assert p is not None
    p.name = "わたしのカラオケ"
    session.commit()
    seed(session)
    session.expire_all()
    names = [x.name for x in _visible_presets(session)]
    assert "わたしのカラオケ" in names and "カラオケ（伴奏）" not in names
    assert len(names) == len(seed_mod.LISTEN_PRESETS)


def test_hidden_builtin_is_not_restored(session: Session) -> None:
    """削除（hidden）した組み込みは、次の起動で復活しない。"""
    seed(session)
    p = session.scalar(select(ListenPreset).where(ListenPreset.seed_code == "backing_only"))
    assert p is not None
    p.hidden = True
    session.commit()
    seed(session)
    session.expire_all()
    visible = [x.seed_code for x in _visible_presets(session)]
    assert "backing_only" not in visible
    count = session.scalar(
        select(func.count())
        .select_from(ListenPreset)
        .where(ListenPreset.seed_code == "backing_only")
    )
    assert count == 1


def test_legacy_presets_get_seed_code(session: Session) -> None:
    """seed_code の無い古い DB: 同じ名前の行に seed_code を付け、重複して作らない。"""
    seed(session)
    for p in session.scalars(select(ListenPreset)):
        p.seed_code = None
    # 古い DB でユーザーが消していた組み込み（名前の行が無い）は、この移行で1度だけ作り直される
    karaoke = session.scalar(select(ListenPreset).where(ListenPreset.name == "カラオケ（伴奏）"))
    session.delete(karaoke)
    user = ListenPreset(name="ユーザーの組み合わせ", sort_order=99)
    session.add(user)
    session.commit()
    seed(session)
    session.expire_all()
    rows = _visible_presets(session)
    assert sorted(p.seed_code or "" for p in rows) == sorted(
        [d.code for d in seed_mod.LISTEN_PRESETS] + [""]
    )
    assert session.get(ListenPreset, user.listen_preset_id).seed_code is None  # type: ignore[union-attr]


def _hue(color: str) -> float:
    import colorsys

    r, g, b = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360


def _sat(color: str) -> float:
    import colorsys

    r, g, b = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[2]


def test_colors_avoid_accent_red_and_groups_differ(session: Session) -> None:
    """stem・グループの色は差し色（赤系統）と紛れず、グループはメンバーの stem と別の色。"""
    seed(session)
    types = {t.code: t for t in session.scalars(select(StemType))}
    groups = session.scalars(select(StemGroup)).all()
    all_colors = [t.color for t in types.values()] + [g.color for g in groups]
    for color in all_colors:
        h = _hue(color)
        # 彩度のある色は赤〜ピンク（330°〜15°）を避ける
        if _sat(color) > 0.25:
            assert 15 <= h <= 330, color
    base = [types[c].color.upper() for c in BASE_STEM_CODES]
    assert len(set(base)) == len(base)
    members = {
        g.group_id: {
            types_by_id.color.upper()
            for m in session.scalars(
                select(StemGroupMember).where(StemGroupMember.group_id == g.group_id)
            )
            for types_by_id in [session.get(StemType, m.stem_type_id)]
            if types_by_id is not None
        }
        for g in groups
    }
    for g in groups:
        assert g.color.upper() not in members[g.group_id], g.code
        assert g.color.upper() not in base, g.code
    group_colors = [g.color.upper() for g in groups]
    assert len(set(group_colors)) == len(group_colors)


def test_seed_updates_colors_of_existing_db(session: Session) -> None:
    """既存 DB の古い色（T01 の色）も seed で新しい色になる。"""
    seed(session)
    drums = session.scalar(select(StemType).where(StemType.code == "drums"))
    rhythm = session.scalar(select(StemGroup).where(StemGroup.code == "rhythm"))
    assert drums is not None and rhythm is not None
    drums.color = "#F58231"
    rhythm.color = "#F58231"
    session.commit()
    seed(session)
    session.expire_all()
    want_type = next(d.color for d in seed_mod.STEM_TYPES if d.code == "drums")
    want_group = next(g.color for g in seed_mod.GROUPS if g.code == "rhythm")
    assert session.get(StemType, drums.stem_type_id).color == want_type  # type: ignore[union-attr]
    assert session.get(StemGroup, rhythm.group_id).color == want_group  # type: ignore[union-attr]
