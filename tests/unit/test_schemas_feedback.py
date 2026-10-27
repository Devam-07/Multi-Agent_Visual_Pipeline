"""Tests for formsync.schemas.feedback."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from formsync.schemas.feedback import CoachingNote, FeedbackReport


def test_coaching_note_requires_evidence_and_bounds_priority() -> None:
    """A note must cite evidence, keep priority in 1..5, and stay under 400 characters."""
    note = CoachingNote(
        priority=1,
        category="timing",
        section_id="S2",
        slots=[3, 7],
        evidence_ids=["S2.slot3.offset", "S2.slot7.offset"],
        note="Dancers 3 and 7 are consistently a half-beat late on the chorus hit.",
        drill="Run the chorus hit at 80% tempo with a metronome, then full speed.",
    )
    assert CoachingNote.model_validate_json(note.model_dump_json()) == note
    with pytest.raises(ValidationError):
        CoachingNote(priority=1, category="timing", evidence_ids=[], note="x")
    with pytest.raises(ValidationError):
        CoachingNote(priority=6, category="timing", evidence_ids=["e"], note="x")
    with pytest.raises(ValidationError):
        CoachingNote(priority=1, category="timing", evidence_ids=["e"], note="x" * 401)
    with pytest.raises(ValidationError):
        CoachingNote(priority=1, category="vibes", evidence_ids=["e"], note="x")  # type: ignore[arg-type]


def test_feedback_report_round_trip() -> None:
    """FeedbackReport nests notes and records provenance."""
    report = FeedbackReport(
        run_id="run-001",
        summary="Strong first half; the chorus hit needs cleaning.",
        strengths=["Intro formation snapped together within a quarter beat."],
        notes=[
            CoachingNote(
                priority=2,
                category="formation_timing",
                section_id="S3",
                slots=[12],
                evidence_ids=["S3.transition1.slot12.settle"],
                note="Dancer 12 settles about 0.6 beats after the group in the diamond.",
            )
        ],
        provider="gemini",
        model="gemini-2.5-flash",
        prompt_version="feedback_v1",
        generated_at=datetime(2026, 9, 13, 12, 0, tzinfo=UTC),
    )
    assert FeedbackReport.model_validate_json(report.model_dump_json()) == report
