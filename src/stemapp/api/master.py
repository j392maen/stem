"""マスタ（stem の種類、グループ、組み合わせプリセット、分割プリセット）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import select
from sqlalchemy.orm import Session

from stemapp.api.common import SessionDep
from stemapp.models import (
    ListenPreset,
    ListenPresetItem,
    SeparationPreset,
    StemGroup,
    StemGroupMember,
    StemType,
)

router = APIRouter(prefix="/api", tags=["master"])


def _type_codes(session: Session) -> dict[int, str]:
    return {t.stem_type_id: t.code for t in session.scalars(select(StemType))}


@router.get("/stem-types")
def stem_types(session: SessionDep) -> dict[str, Any]:
    types = session.scalars(select(StemType).order_by(StemType.display_order)).all()
    codes = {t.stem_type_id: t.code for t in types}
    return {
        "stem_types": [
            {
                "stem_type_id": t.stem_type_id,
                "code": t.code,
                "display_name": t.display_name,
                "parent_code": codes.get(t.parent_id) if t.parent_id else None,
                "tier": t.tier,
                "experimental": t.experimental,
                "color": t.color,
                "display_order": t.display_order,
            }
            for t in types
        ]
    }


@router.get("/stem-groups")
def stem_groups(session: SessionDep) -> dict[str, Any]:
    codes = _type_codes(session)
    members: dict[int, list[str]] = {}
    for m in session.scalars(select(StemGroupMember)):
        members.setdefault(m.group_id, []).append(codes[m.stem_type_id])
    groups = session.scalars(select(StemGroup).order_by(StemGroup.group_id)).all()
    return {
        "stem_groups": [
            {
                "group_id": g.group_id,
                "code": g.code,
                "display_name": g.display_name,
                "color": g.color,
                "is_builtin": g.is_builtin,
                "members": sorted(members.get(g.group_id, [])),
            }
            for g in groups
        ]
    }


@router.get("/listen-presets")
def listen_presets(session: SessionDep) -> dict[str, Any]:
    codes = _type_codes(session)
    group_codes = {g.group_id: g.code for g in session.scalars(select(StemGroup))}
    items: dict[int, list[dict[str, Any]]] = {}
    for it in session.scalars(select(ListenPresetItem).order_by(ListenPresetItem.item_id)):
        items.setdefault(it.listen_preset_id, []).append(
            {
                "item_id": it.item_id,
                "stem_type_code": codes.get(it.stem_type_id) if it.stem_type_id else None,
                "group_code": group_codes.get(it.group_id) if it.group_id else None,
                "gain_db": it.gain_db,
            }
        )
    presets = session.scalars(
        select(ListenPreset).order_by(ListenPreset.sort_order, ListenPreset.listen_preset_id)
    ).all()
    return {
        "listen_presets": [
            {
                "listen_preset_id": p.listen_preset_id,
                "name": p.name,
                "sort_order": p.sort_order,
                "items": items.get(p.listen_preset_id, []),
            }
            for p in presets
        ]
    }


@router.get("/presets")
def separation_presets(session: SessionDep) -> dict[str, Any]:
    presets = session.scalars(select(SeparationPreset).order_by(SeparationPreset.preset_id)).all()
    return {
        "presets": [
            {"code": p.code, "display_name": p.display_name, "is_default": p.is_default}
            for p in presets
        ]
    }
