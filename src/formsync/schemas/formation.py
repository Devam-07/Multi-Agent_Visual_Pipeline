"""Formation results: relative settle timing per transition, plus optional stage calibration.

Formation feedback is *relative*: for each transition we find when the group as a whole settled
into the new formation and how far from that moment each dancer settled. There is no notion of
where anyone "should" stand. Spacing evenness is reported only when a stage calibration exists,
because it needs real distances.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from formsync.schemas.base import FormsyncModel, Score, SlotId


class StageCalibration(FormsyncModel):
    """Mapping from image pixels to metres on the stage floor (a planar homography).

    Produced once per camera setup by clicking the four stage corners on one frame. Optional:
    when absent, the formation stage falls back to torso-normalised pixel motion.

    Attributes:
        image_points: Pixel coordinates of the four stage corners in the order
            downstage-left, downstage-right, upstage-right, upstage-left.
        stage_width_m: Real stage width (downstage edge) in metres.
        stage_depth_m: Real stage depth (left edge) in metres.
        homography: 3x3 matrix mapping homogeneous pixel coords to stage coords in metres, with
            origin at downstage-left, x across stage, y toward upstage.
        frame_idx_used: Source frame index the corners were clicked on.
        video_hash: Optional content hash of the video the calibration was made for, used to
            look the calibration up again.
    """

    image_points: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    stage_width_m: float = Field(gt=0)
    stage_depth_m: float = Field(gt=0)
    homography: list[list[float]]
    frame_idx_used: int = Field(ge=0)
    video_hash: str | None = None

    @field_validator("homography")
    @classmethod
    def _is_3x3(cls, v: list[list[float]]) -> list[list[float]]:
        """The homography must be a 3x3 matrix."""
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("homography must be a 3x3 matrix")
        return v


class FormationEvent(FormsyncModel):
    """One hold or transition detected from the group's floor motion.

    For ``kind == "transition"`` the settle fields describe how the group arrived in the next
    formation. For ``kind == "hold"`` they are ``None``.

    Attributes:
        kind: ``"hold"`` (group mostly stationary) or ``"transition"`` (group moving).
        start_s: Start time in seconds.
        end_s: End time in seconds (>= ``start_s``).
        start_beat: Start expressed as a fractional beat index.
        end_beat: End expressed as a fractional beat index.
        group_settle_s: Time (s) at which the group settled (median of per-dancer settle times).
        group_settle_beat: The same moment in beats.
        per_slot_settle_offset_beats: For each slot that settled, ``settle - group_settle`` in
            beats. **Positive means late.**
        arrival_spread_beats: Spread of per-slot settle times in beats (last minus first).
        late_slots: Slots whose settle offset exceeded the lateness threshold.
        early_slots: Slots whose settle offset was below the negative threshold.
        unmatched_slots: Slots for which no settle moment could be detected (tracking loss or
            the dancer never stopped moving).
        nn_spacing_cv: Coefficient of variation of nearest-neighbour distances at the end of the
            event, a unit-free evenness measure (0 = perfectly even). ``None`` when uncalibrated.
    """

    kind: Literal["hold", "transition"]
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    start_beat: float
    end_beat: float
    group_settle_s: float | None = None
    group_settle_beat: float | None = None
    per_slot_settle_offset_beats: dict[SlotId, float] = Field(default_factory=dict)
    arrival_spread_beats: float | None = Field(default=None, ge=0)
    late_slots: list[SlotId] = Field(default_factory=list)
    early_slots: list[SlotId] = Field(default_factory=list)
    unmatched_slots: list[SlotId] = Field(default_factory=list)
    nn_spacing_cv: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_times(self) -> FormationEvent:
        """Ensure the time range is ordered."""
        if self.end_s < self.start_s:
            raise ValueError(f"end_s {self.end_s} precedes start_s {self.start_s}")
        if self.end_beat < self.start_beat:
            raise ValueError(f"end_beat {self.end_beat} precedes start_beat {self.start_beat}")
        return self


class FormationResult(FormsyncModel):
    """Formation analysis for a whole run.

    Attributes:
        calibrated: ``True`` if a :class:`StageCalibration` was used (metres); ``False`` if the
            torso-normalised pixel fallback was used. Downstream confidence is lower when False.
        events: Holds and transitions in time order.
        per_section: Formation score (0-100) per section id.
        score_0_100: Duration-weighted overall formation score.
        params: The constants used (thresholds, scales), recorded for reproducibility.
    """

    calibrated: bool
    events: list[FormationEvent] = Field(default_factory=list)
    per_section: dict[str, Score] = Field(default_factory=dict)
    score_0_100: Score
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)
