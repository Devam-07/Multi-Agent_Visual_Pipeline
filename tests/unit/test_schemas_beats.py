"""Tests for formsync.schemas.beats."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from formsync.schemas.beats import BeatGrid


def _grid() -> BeatGrid:
    """Hand-built grid: 120 bpm for four beats, then one slower 0.7 s beat."""
    # 120 bpm, four beats at 0, 0.5, 1.0, 1.5 s, then the tempo slows: next beat at 2.2 s.
    return BeatGrid(tempo_bpm=120, beat_times=[0.0, 0.5, 1.0, 1.5, 2.2], confidence=0.9)


def test_beat_grid_validation() -> None:
    """Beat times must be strictly increasing and at least two long."""
    with pytest.raises(ValidationError):
        BeatGrid(tempo_bpm=120, beat_times=[0.0, 0.5, 0.5], confidence=0.5)
    with pytest.raises(ValidationError):
        BeatGrid(tempo_bpm=120, beat_times=[0.0], confidence=0.5)
    assert BeatGrid.model_validate_json(_grid().model_dump_json()) == _grid()


def test_beat_at_interpolates_and_extrapolates() -> None:
    """beat_at is exact on beats, linear between them, and extrapolates past both ends."""
    g = _grid()
    assert g.beat_at(0.0) == 0.0
    assert g.beat_at(1.0) == 2.0
    assert g.beat_at(0.75) == pytest.approx(1.5)
    assert g.beat_at(1.85) == pytest.approx(3.5)  # halfway through the slower 0.7 s beat
    assert g.beat_at(-0.25) == pytest.approx(-0.5)  # before grid: uses first interval
    assert g.beat_at(2.9) == pytest.approx(5.0)  # after grid: uses last interval (0.7 s)


def test_local_period_uses_containing_interval() -> None:
    """local_period returns the interval around t, with edge intervals outside the grid."""
    g = _grid()
    assert g.local_period(0.25) == pytest.approx(0.5)
    assert g.local_period(1.6) == pytest.approx(0.7)
    assert g.local_period(-5.0) == pytest.approx(0.5)
    assert g.local_period(10.0) == pytest.approx(0.7)


def test_time_at_is_inverse_of_beat_at() -> None:
    """time_at(beat_at(t)) == t on hand-picked times inside and outside the grid."""
    g = _grid()
    for t in (-0.3, 0.0, 0.2, 0.5, 1.25, 1.5, 2.0, 2.2, 3.0):
        assert g.time_at(g.beat_at(t)) == pytest.approx(t)


@given(st.floats(min_value=-5, max_value=10, allow_nan=False, allow_infinity=False))
def test_beat_at_time_at_round_trip_property(t: float) -> None:
    """Property: the two conversions are inverses for any finite time."""
    g = _grid()
    assert g.time_at(g.beat_at(t)) == pytest.approx(t, abs=1e-9)
