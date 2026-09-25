"""マスタ（stem の種類、グループ、組み合わせプリセット、分割プリセット）。"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, StringConstraints, model_validator
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from stemapp.api.common import SessionDep, not_found
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
        items.setdefault(it.listen_preset_id, []).append(_item_dict(it, codes, group_codes))
    presets = session.scalars(
        select(ListenPreset)
        .where(ListenPreset.hidden.is_(False))
        .order_by(ListenPreset.sort_order, ListenPreset.listen_preset_id)
    ).all()
    return {
        "listen_presets": [
            _preset_dict(p, items.get(p.listen_preset_id, [])) for p in presets
        ]
    }


def _item_dict(
    it: ListenPresetItem, codes: dict[int, str], group_codes: dict[int, str]
) -> dict[str, Any]:
    return {
        "item_id": it.item_id,
        "stem_type_id": it.stem_type_id,
        "stem_type_code": codes.get(it.stem_type_id) if it.stem_type_id else None,
        "group_id": it.group_id,
        "group_code": group_codes.get(it.group_id) if it.group_id else None,
        "gain_db": it.gain_db,
    }


def _preset_dict(p: ListenPreset, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "listen_preset_id": p.listen_preset_id,
        "name": p.name,
        "sort_order": p.sort_order,
        "builtin": p.seed_code is not None,
        "items": items,
    }


# --- 組み合わせプリセットの作成・変更・削除 ----------------------------------------------

PresetName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class ListenItemIn(BaseModel):
    """stem_type_id と group_id のどちらか一方だけを指定する。"""

    stem_type_id: int | None = None
    group_id: int | None = None
    gain_db: float = Field(default=0.0, ge=-60.0, le=12.0)

    @model_validator(mode="after")
    def _one_target(self) -> ListenItemIn:
        if (self.stem_type_id is None) == (self.group_id is None):
            raise ValueError("stem_type_id と group_id のどちらか一方だけを指定してください。")
        return self


class ListenPresetCreate(BaseModel):
    name: PresetName
    items: list[ListenItemIn] = Field(default_factory=list, max_length=200)


class ListenPresetUpdate(BaseModel):
    name: PresetName | None = None
    items: list[ListenItemIn] | None = Field(default=None, max_length=200)
    sort_order: int | None = None


def _check_items(session: Session, items: list[ListenItemIn]) -> None:
    type_ids = set(session.scalars(select(StemType.stem_type_id)))
    group_ids = set(session.scalars(select(StemGroup.group_id)))
    for it in items:
        if it.stem_type_id is not None and it.stem_type_id not in type_ids:
            raise HTTPException(
                status_code=400, detail=f"stem の種類が見つかりません（{it.stem_type_id}）。"
            )
        if it.group_id is not None and it.group_id not in group_ids:
            raise HTTPException(
                status_code=400, detail=f"グループが見つかりません（{it.group_id}）。"
            )


def _replace_items(session: Session, preset_id: int, items: list[ListenItemIn]) -> None:
    session.execute(delete(ListenPresetItem).where(ListenPresetItem.listen_preset_id == preset_id))
    for it in items:
        session.add(
            ListenPresetItem(
                listen_preset_id=preset_id,
                stem_type_id=it.stem_type_id,
                group_id=it.group_id,
                gain_db=it.gain_db,
            )
        )


def _visible_preset(session: Session, preset_id: int) -> ListenPreset:
    """隠した（削除した組み込みの）組み合わせは無いものとして扱う。"""
    p = session.get(ListenPreset, preset_id)
    if p is None or p.hidden:
        raise not_found("組み合わせ")
    return p


def _load_preset(session: Session, preset_id: int) -> dict[str, Any]:
    p = _visible_preset(session, preset_id)
    codes = _type_codes(session)
    group_codes = {g.group_id: g.code for g in session.scalars(select(StemGroup))}
    rows = session.scalars(
        select(ListenPresetItem)
        .where(ListenPresetItem.listen_preset_id == preset_id)
        .order_by(ListenPresetItem.item_id)
    ).all()
    return _preset_dict(p, [_item_dict(it, codes, group_codes) for it in rows])


@router.post("/listen-presets", status_code=201)
def create_listen_preset(body: ListenPresetCreate, session: SessionDep) -> dict[str, Any]:
    _check_items(session, body.items)
    last = session.scalar(select(func.max(ListenPreset.sort_order))) or 0
    p = ListenPreset(name=body.name, sort_order=int(last) + 10)
    session.add(p)
    session.flush()
    _replace_items(session, p.listen_preset_id, body.items)
    session.commit()
    return _load_preset(session, p.listen_preset_id)


@router.put("/listen-presets/{preset_id}")
def update_listen_preset(
    preset_id: int, body: ListenPresetUpdate, session: SessionDep
) -> dict[str, Any]:
    p = _visible_preset(session, preset_id)
    if body.name is not None:
        p.name = body.name
    if body.sort_order is not None:
        p.sort_order = body.sort_order
    if body.items is not None:
        _check_items(session, body.items)
        _replace_items(session, preset_id, body.items)
    session.commit()
    return _load_preset(session, preset_id)


@router.delete("/listen-presets/{preset_id}", status_code=204)
def delete_listen_preset(preset_id: int, session: SessionDep) -> Response:
    p = _visible_preset(session, preset_id)
    if p.seed_code is not None:
        # 組み込みは行を残して隠す（起動時の seed が作り直さないように）
        p.hidden = True
    else:
        session.delete(p)  # 中身（LISTEN_PRESET_ITEM）は外部キーの CASCADE で消える
    session.commit()
    return Response(status_code=204)


@router.get("/presets")
def separation_presets(session: SessionDep) -> dict[str, Any]:
    presets = session.scalars(select(SeparationPreset).order_by(SeparationPreset.preset_id)).all()
    return {
        "presets": [
            {"code": p.code, "display_name": p.display_name, "is_default": p.is_default}
            for p in presets
        ]
    }
