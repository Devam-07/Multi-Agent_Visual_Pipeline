"""Tests for formsync.schemas.run and the package re-exports."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import formsync.schemas as schemas
from formsync.schemas import BeatGrid, RunResult, Section, SectionList, VideoMeta


def _minimal_run() -> RunResult:
    """A RunResult with only the mandatory fields filled in."""
    return RunResult(
        run_id="run-001",
        created_at=datetime(2026, 9, 13, 12, 0, tzinfo=UTC),
        video=VideoMeta(
            path="practice.mp4",
            fps=30,
            width=1920,
            height=1080,
            n_frames=900,
            duration_s=30,
            analysis_fps=15,
            frame_stride=2,
        ),
        roster_size=35,
        beats=BeatGrid(tempo_bpm=120, beat_times=[0.0, 0.5], confidence=0.8),
        sections=SectionList(
            source="user", sections=[Section(id="S1", name="All", start_s=0, end_s=30)]
        ),
    )


def test_run_result_partial_and_round_trip() -> None:
    """A run with only the mandatory stages filled in is valid and round-trips."""
    run = _minimal_run()
    assert run.sync is None and run.formation is None and run.feedback is None
    assert RunResult.model_validate_json(run.model_dump_json()) == run
    with pytest.raises(ValidationError):
        RunResult.model_validate({**run.model_dump(), "roster_size": 0})


def test_package_exports_every_public_model() -> None:
    """Everything listed in __all__ is importable from formsync.schemas."""
    for name in schemas.__all__:
        assert hasattr(schemas, name), name
