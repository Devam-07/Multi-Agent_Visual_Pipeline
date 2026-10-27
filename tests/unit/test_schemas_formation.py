"""Tests for formsync.schemas.formation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from formsync.schemas.formation import FormationEvent, FormationResult, StageCalibration

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_stage_calibration_requires_four_corners_and_3x3() -> None:
    """Calibration needs exactly four image points and a 3x3 homography."""
    cal = StageCalibration(
        image_points=[(10, 900), (1900, 900), (1500, 300), (400, 300)],
        stage_width_m=12,
        stage_depth_m=8,
        homography=IDENTITY,
        frame_idx_used=0,
    )
    assert StageCalibration.model_validate_json(cal.model_dump_json()) == cal
    with pytest.raises(ValidationError):
        StageCalibration(
            image_points=[(0, 0), (1, 0), (1, 1)],
            stage_width_m=12,
            stage_depth_m=8,
            homography=IDENTITY,
            frame_idx_used=0,
        )
    with pytest.raises(ValidationError):
        StageCalibration(
            image_points=[(0, 0), (1, 0), (1, 1), (0, 1)],
            stage_width_m=12,
            stage_depth_m=8,
            homography=[[1, 0], [0, 1]],
            frame_idx_used=0,
        )


def test_formation_event_transition_fields_and_order() -> None:
    """A transition carries relative settle data; reversed time ranges are rejected."""
    ev = FormationEvent(
        kind="transition",
        start_s=40.0,
        end_s=43.0,
        start_beat=80.0,
        end_beat=86.0,
        group_settle_s=42.4,
        group_settle_beat=84.8,
        per_slot_settle_offset_beats={0: -0.1, 1: 0.0, 7: 0.6},
        arrival_spread_beats=0.7,
        late_slots=[7],
        early_slots=[],
        unmatched_slots=[12],
        nn_spacing_cv=None,
    )
    assert FormationEvent.model_validate_json(ev.model_dump_json()) == ev
    assert ev.per_slot_settle_offset_beats[7] > 0  # positive = late
    with pytest.raises(ValidationError):
        FormationEvent(kind="hold", start_s=5.0, end_s=4.0, start_beat=10, end_beat=8)


def test_formation_result_scores_and_flags() -> None:
    """FormationResult records calibration mode and bounds its scores."""
    res = FormationResult(calibrated=False, events=[], per_section={"S1": 88.0}, score_0_100=88.0)
    assert FormationResult.model_validate_json(res.model_dump_json()) == res
    with pytest.raises(ValidationError):
        FormationResult(calibrated=True, per_section={"S1": 120.0}, score_0_100=50.0)
