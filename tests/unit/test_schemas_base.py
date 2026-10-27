"""Tests for formsync.schemas.base."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from formsync.schemas.base import FormsyncModel, Score, SlotId, UnitInterval


class _Demo(FormsyncModel):
    """Minimal model exercising the shared field types."""

    score: Score
    frac: UnitInterval
    slot: SlotId


def test_formsync_model_rejects_unknown_fields() -> None:
    """Extra keys are an error, so misspelled JSON keys fail loudly."""
    with pytest.raises(ValidationError):
        _Demo.model_validate({"score": 1, "frac": 0.5, "slot": 0, "bogus": 1})


def test_formsync_model_validates_on_assignment() -> None:
    """Assigning an out-of-range value after construction also fails."""
    m = _Demo(score=50, frac=0.5, slot=0)
    with pytest.raises(ValidationError):
        m.score = 150


def test_score_range() -> None:
    """Score accepts 0..100 and rejects values outside."""
    assert _Demo(score=0, frac=0, slot=0).score == 0
    assert _Demo(score=100, frac=1, slot=0).score == 100
    with pytest.raises(ValidationError):
        _Demo(score=-1, frac=0, slot=0)


def test_unit_interval_range() -> None:
    """UnitInterval rejects values above 1."""
    with pytest.raises(ValidationError):
        _Demo(score=0, frac=1.01, slot=0)


def test_slot_id_non_negative() -> None:
    """SlotId rejects negative slots."""
    with pytest.raises(ValidationError):
        _Demo(score=0, frac=0, slot=-1)
