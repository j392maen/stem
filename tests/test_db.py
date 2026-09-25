from __future__ import annotations

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stemapp.models import (
    ALL_MODELS,
    InputSource,
    ListenPreset,
    ListenPresetItem,
    StemGroup,
    StemType,
    Track,
)

EXPECTED_TABLES = {
    "track", "input_source", "device", "separation_preset", "preset_step", "model",
    "separation_job", "stem_type", "stem", "stem_rendition", "waveform", "stem_group",
    "stem_group_member", "listen_preset", "listen_preset_item", "playback_state",
    "cue_point", "export", "export_item", "offline_cache",
}


def test_all_tables_created(engine: Engine) -> None:
    assert len(ALL_MODELS) == 20
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES


def test_init_db_is_repeatable(engine: Engine) -> None:
    from stemapp.db import init_db

    init_db(engine)  # 2回目でも壊れない
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES


def test_foreign_keys_enabled(engine: Engine) -> None:
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_foreign_key_violation_rejected(session: Session) -> None:
    session.add(InputSource(track_id=9999, source_type="file"))
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        session.flush()


def test_unique_audio_hash(session: Session) -> None:
    session.add(Track(title="a", audio_hash="x" * 64))
    session.flush()
    session.add(Track(title="b", audio_hash="x" * 64))
    with pytest.raises(IntegrityError):
        session.flush()


@pytest.fixture
def targets(session: Session) -> tuple[int, int, int]:
    p = ListenPreset(name="t")
    t = StemType(code="bass", display_name="ベース", tier="base", color="#000000")
    g = StemGroup(code="g", display_name="G", color="#000000")
    session.add_all([p, t, g])
    session.flush()
    return p.listen_preset_id, t.stem_type_id, g.group_id


def test_listen_item_check_both_rejected(
    session: Session, targets: tuple[int, int, int]
) -> None:
    pid, tid, gid = targets
    session.add(ListenPresetItem(listen_preset_id=pid, stem_type_id=tid, group_id=gid))
    with pytest.raises(IntegrityError, match="ck_listen_preset_item_one_target"):
        session.flush()


def test_listen_item_check_neither_rejected(
    session: Session, targets: tuple[int, int, int]
) -> None:
    pid, _, _ = targets
    session.add(ListenPresetItem(listen_preset_id=pid))
    with pytest.raises(IntegrityError, match="ck_listen_preset_item_one_target"):
        session.flush()


def test_listen_item_check_one_accepted(
    session: Session, targets: tuple[int, int, int]
) -> None:
    pid, tid, gid = targets
    session.add(ListenPresetItem(listen_preset_id=pid, stem_type_id=tid))
    session.add(ListenPresetItem(listen_preset_id=pid, group_id=gid))
    session.flush()
