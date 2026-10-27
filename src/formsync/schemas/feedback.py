"""Feedback produced by the rehearsal feedback agent.

Every :class:`CoachingNote` must cite *evidence ids* that point at facts in the deterministic
analysis summary the agent was given. The agent layer validates those ids, so a note can never
rest on a number the pipeline did not compute.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from formsync.schemas.base import FormsyncModel, SlotId

NoteCategory = Literal[
    "timing",
    "amplitude",
    "formation_timing",
    "spacing",
    "grouping",
    "data_quality",
    "general",
]
"""What a coaching note is about. ``grouping`` covers unmatched dancers / unexpected layers;
``data_quality`` covers tracking or visibility problems that limit what can be concluded."""


class CoachingNote(FormsyncModel):
    """One actionable rehearsal note.

    Attributes:
        priority: 1 = fix first, 5 = minor.
        category: What the note is about.
        section_id: Section the note refers to, or ``None`` for whole-run notes.
        slots: Dancers the note refers to (may be empty for group-wide notes).
        evidence_ids: Ids of facts in the analysis summary that support the note; at least one.
        note: The coaching sentence(s), at most 400 characters.
        drill: Optional concrete exercise to address the issue.
    """

    priority: int = Field(ge=1, le=5)
    category: NoteCategory
    section_id: str | None = None
    slots: list[SlotId] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)
    note: str = Field(min_length=1, max_length=400)
    drill: str | None = Field(default=None, max_length=400)


class FeedbackReport(FormsyncModel):
    """The full feedback output for one run.

    Attributes:
        run_id: Run this report belongs to.
        summary: Two-to-four sentence overview of the practice.
        strengths: Things that went well (kept short so the notes stay the focus).
        notes: Coaching notes, ordered by priority.
        provider: LLM provider used (``gemini``, ``groq``, ``ollama``, ``anthropic``).
        model: Model id used.
        prompt_version: Version tag of the prompt template, for reproducibility.
        generated_at: Timestamp of generation.
    """

    run_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    strengths: list[str] = Field(default_factory=list)
    notes: list[CoachingNote] = Field(default_factory=list)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    generated_at: datetime
