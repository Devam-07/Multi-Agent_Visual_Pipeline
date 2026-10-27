"""Shared base class and reusable field types for every schema.

All result models inherit :class:`FormsyncModel`, which rejects unknown fields so a misspelled
key in a JSON artifact fails loudly instead of being silently dropped.

Conventions used across all schemas (documented once here, enforced everywhere):

* **Time** is in seconds (``_s`` suffix) or fractional beats (``_beats`` / ``_beat`` suffix).
* **Timing sign:** a positive offset always means *late* relative to the reference group.
* **Pixel coordinates:** origin top-left of the frame, x to the right, y downward.
* **Stage coordinates:** metres on the floor, origin at downstage-left, x across stage,
  y toward upstage. Only present when a :class:`~formsync.schemas.formation.StageCalibration`
  exists.
* **Slots** are stable dancer indices ``0..N-1`` assigned by the tracking stage; ``track_id``
  is the raw tracker id and may change when tracking is lost.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class FormsyncModel(BaseModel):
    """Base class for all formsync schemas: strict about unknown fields, validates on assignment.

    Subclasses are plain data containers. They may carry small unit-conversion helpers (see
    :class:`~formsync.schemas.beats.BeatGrid`) but never pipeline logic.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


Score = Annotated[float, Field(ge=0, le=100, description="Score on a 0-100 scale.")]
"""A 0-100 quality score. Higher is better."""

UnitInterval = Annotated[float, Field(ge=0, le=1, description="Fraction or probability in [0, 1].")]
"""A value constrained to [0, 1], used for confidences, fractions and coverage."""

SlotId = Annotated[int, Field(ge=0, description="Stable dancer slot index, 0-based.")]
"""A stable dancer slot index assigned by the tracking stage."""
