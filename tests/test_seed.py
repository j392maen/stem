from __future__ import annotations

from typing import Any

from sqlalchemy import select
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

    for code, p in presets.items():
        steps = list(
            session.scalars(
                select(PresetStep)
                .where(PresetStep.preset_id == p.preset_id)
                .order_by(PresetStep.step_order)
            )
        )
        roles = [s.role for s in steps]
        assert roles[0] == "multistem", code
        assert models[steps[0].model_id].filename == "BS-Roformer-SW.ckpt"
        assert roles[-1] == "karaoke", code
        assert all(s.input == "vocals" for s in steps if s.role == "karaoke")
        n_vocal = roles.count("vocals")
        n_karaoke = roles.count("karaoke")
        assert (n_vocal, n_karaoke) == {"fast": (0, 1), "standard": (1, 1), "best": (2, 2)}[code]


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
