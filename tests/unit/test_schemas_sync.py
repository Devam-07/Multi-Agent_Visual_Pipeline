"""Tests for formsync.schemas.sync."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from formsync.schemas.sync import (
    DancerSectionSync,
    HitEvent,
    MotionGroup,
    SectionSync,
    SyncResult,
    WindowOffset,
)


def test_motion_group_seed_must_be_member() -> None:
    """A discovered group's seed must be one of its slots; time range must be ordered."""
    g = MotionGroup(
        group_id="S2.g0",
        slots=[0, 1, 2, 5],
        start_s=60.0,
        end_s=75.0,
        origin="discovered",
        seed_slot=1,
        mean_internal_corr=0.82,
    )
    assert MotionGroup.model_validate_json(g.model_dump_json()) == g
    with pytest.raises(ValidationError):
        MotionGroup(
            group_id="x", slots=[0, 1], start_s=0, end_s=1, origin="discovered", seed_slot=9
        )
    with pytest.raises(ValidationError):
        MotionGroup(group_id="x", slots=[0], start_s=5, end_s=1, origin="declared")


def test_window_offset_ranges() -> None:
    """WindowOffset bounds correlation to [-1, 1] and visibility to [0, 1]."""
    w = WindowOffset(
        slot=7,
        t_center=61.0,
        group_id="S2.g0",
        offset_s=0.12,
        offset_beats=0.25,
        corr=0.71,
        amp_ratio=0.95,
        visible_frac=1.0,
        valid=True,
    )
    assert WindowOffset.model_validate_json(w.model_dump_json()) == w
    with pytest.raises(ValidationError):
        w.corr = 1.5


def test_dancer_section_sync_fractions() -> None:
    """Per-dancer statistics keep fractions and confidence inside [0, 1]."""
    d = DancerSectionSync(
        slot=7,
        section_id="S2",
        group_id="S2.g0",
        median_offset_beats=0.4,
        offset_mad_beats=0.1,
        late_fraction=0.8,
        early_fraction=0.0,
        median_amp_ratio=1.02,
        mean_corr=0.7,
        n_valid_windows=20,
        confidence=0.85,
    )
    assert DancerSectionSync.model_validate_json(d.model_dump_json()) == d
    with pytest.raises(ValidationError):
        d.late_fraction = 1.2


def test_section_sync_and_sync_result_round_trip() -> None:
    """SectionSync and SyncResult nest correctly and round-trip through JSON."""
    section = SectionSync(
        section_id="S2",
        groups=[
            MotionGroup(group_id="S2.g0", slots=[0, 1], start_s=60, end_s=75, origin="declared")
        ],
        dancers=[],
        unmatched_slots=[9],
        timing_spread_beats=0.15,
        amp_spread_log=0.1,
        score_0_100=82.0,
        worst_slots=[7, 3],
    )
    result = SyncResult(sections=[section], overall_score_0_100=82.0, params={"window_s": 2.0})
    assert SyncResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        SyncResult(sections=[], overall_score_0_100=101.0)


def test_hit_event_offsets_and_participants() -> None:
    """HitEvent records per-dancer offsets (positive = late) and needs >= 1 participant."""
    hit = HitEvent(
        section_id="S2",
        group_id="S2.g0",
        kind="stop",
        t_group=61.2,
        beat_group=34.0,
        per_slot_offset_beats={3: 0.25, 7: 0.3, 0: -0.05},
        late_slots=[3, 7],
        early_slots=[],
        n_participants=3,
    )
    assert HitEvent.model_validate_json(hit.model_dump_json()) == hit
    assert all(hit.per_slot_offset_beats[s] > 0 for s in hit.late_slots)
    with pytest.raises(ValidationError):
        HitEvent(
            section_id="S2", group_id="g", kind="stop", t_group=1, beat_group=1, n_participants=0
        )
    with pytest.raises(ValidationError):
        HitEvent(
            section_id="S2", group_id="g", kind="hit", t_group=1, beat_group=1, n_participants=1
        )  # type: ignore[arg-type]
