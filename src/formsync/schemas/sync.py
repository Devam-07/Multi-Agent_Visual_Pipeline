"""Synchronisation results: motion groups, per-window offsets and per-section scores.

Every offset is measured **relative to the dancer's own motion group** (the other dancers doing
the same choreography), never relative to the beat. Beats are only the unit. A positive offset
always means *late*.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from formsync.schemas.base import FormsyncModel, Score, SlotId, UnitInterval


class MotionGroup(FormsyncModel):
    """A choreographic layer: a set of dancers doing the same movement over a time range.

    Groups are either discovered by seeded correlation clustering or declared by the user in the
    section file.

    Attributes:
        group_id: Identifier unique within the section (e.g. ``"S2.g0"``).
        slots: Member slots.
        start_s: First time (s) the group was observed.
        end_s: Last time (s) the group was observed.
        origin: ``"discovered"`` (clustering) or ``"declared"`` (user section file).
        seed_slot: The most-central dancer the cluster was seeded from; ``None`` if declared.
        mean_internal_corr: Mean pairwise correlation among members; ``None`` if not computed.
    """

    group_id: str = Field(min_length=1)
    slots: list[SlotId] = Field(min_length=1)
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    origin: Literal["discovered", "declared"]
    seed_slot: SlotId | None = None
    mean_internal_corr: float | None = Field(default=None, ge=-1, le=1)

    @model_validator(mode="after")
    def _check(self) -> MotionGroup:
        """Time range must be ordered and the seed, if any, must be a member."""
        if self.end_s < self.start_s:
            raise ValueError("end_s precedes start_s")
        if self.seed_slot is not None and self.seed_slot not in self.slots:
            raise ValueError("seed_slot must be one of the group's slots")
        return self


class WindowOffset(FormsyncModel):
    """Timing and amplitude of one dancer in one analysis window, relative to their group.

    Attributes:
        slot: Dancer slot.
        t_center: Window centre time in seconds.
        group_id: Group the dancer was compared against; ``None`` if unmatched in this window.
        offset_s: Estimated lag in seconds. **Positive = late.**
        offset_beats: The same lag in beats (``offset_s / local beat period``).
        corr: Peak normalised cross-correlation with the group consensus, in [-1, 1].
        amp_ratio: Dancer's motion amplitude divided by the group median amplitude (1 = same).
        visible_frac: Fraction of the window in which the dancer's keypoints were present.
        valid: ``True`` if the window met the correlation and visibility thresholds.
    """

    slot: SlotId
    t_center: float = Field(ge=0)
    group_id: str | None = None
    offset_s: float
    offset_beats: float
    corr: float = Field(ge=-1, le=1)
    amp_ratio: float = Field(ge=0)
    visible_frac: UnitInterval
    valid: bool


class DancerSectionSync(FormsyncModel):
    """Aggregated timing/amplitude statistics for one dancer over one section.

    Attributes:
        slot: Dancer slot.
        section_id: Section id.
        group_id: The group the dancer belonged to for most of the section; ``None`` if unmatched.
        median_offset_beats: Median of valid window offsets in beats. **Positive = late.**
        offset_mad_beats: Median absolute deviation of the offsets (consistency of the timing).
        late_fraction: Fraction of valid windows with offset above the lateness threshold.
        early_fraction: Fraction of valid windows with offset below the negative threshold.
        median_amp_ratio: Median amplitude ratio (1 = matches the group).
        mean_corr: Mean peak correlation over valid windows.
        n_valid_windows: Number of windows that passed the validity thresholds.
        confidence: Overall confidence in this dancer's statistics, in [0, 1].
    """

    slot: SlotId
    section_id: str = Field(min_length=1)
    group_id: str | None = None
    median_offset_beats: float
    offset_mad_beats: float = Field(ge=0)
    late_fraction: UnitInterval
    early_fraction: UnitInterval
    median_amp_ratio: float = Field(ge=0)
    mean_corr: float = Field(ge=-1, le=1)
    n_valid_windows: int = Field(ge=0)
    confidence: UnitInterval


class HitEvent(FormsyncModel):
    """One group-level motion start or stop ("hit") and who was early or late on it.

    Detected from each dancer's motion-energy signal: a *start* is a sharp rise from rest, a
    *stop* is a sharp drop to rest. Events are matched across a motion group; the group time is
    the median of the members' event times and each member's offset is relative to that.

    Attributes:
        section_id: Section the hit falls in.
        group_id: Motion group the hit was matched within.
        kind: ``"start"`` (motion onset) or ``"stop"`` (the freeze at the accent).
        t_group: Group event time in seconds (median over participants).
        beat_group: The same moment in beats.
        per_slot_offset_beats: ``own_time - t_group`` in beats per participating slot.
            **Positive = late.**
        late_slots: Participants whose offset exceeded the lateness threshold.
        early_slots: Participants whose offset was below the negative threshold.
        n_participants: Number of group members with a detected event for this hit.
    """

    section_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    kind: Literal["start", "stop"]
    t_group: float = Field(ge=0)
    beat_group: float
    per_slot_offset_beats: dict[SlotId, float] = Field(default_factory=dict)
    late_slots: list[SlotId] = Field(default_factory=list)
    early_slots: list[SlotId] = Field(default_factory=list)
    n_participants: int = Field(ge=1)


class SectionSync(FormsyncModel):
    """Synchronisation summary for one section.

    Attributes:
        section_id: Section id.
        groups: Motion groups active in the section.
        dancers: Per-dancer statistics.
        unmatched_slots: Slots that had no group for most of the section (soloists, tracking
            loss, or genuinely different movement). Reported, not scored.
        hit_events: Group-level starts/stops with per-dancer offsets, in time order.
        timing_spread_beats: Robust spread of per-dancer median offsets within groups, in beats.
        amp_spread_log: Robust spread of per-dancer log amplitude ratios.
        score_0_100: Section synchronisation score.
        worst_slots: Slots ranked by ``|median_offset| * confidence``, worst first.
    """

    section_id: str = Field(min_length=1)
    groups: list[MotionGroup] = Field(default_factory=list)
    dancers: list[DancerSectionSync] = Field(default_factory=list)
    unmatched_slots: list[SlotId] = Field(default_factory=list)
    hit_events: list[HitEvent] = Field(default_factory=list)
    timing_spread_beats: float = Field(ge=0)
    amp_spread_log: float = Field(ge=0)
    score_0_100: Score
    worst_slots: list[SlotId] = Field(default_factory=list)


class SyncResult(FormsyncModel):
    """Synchronisation analysis for a whole run.

    Attributes:
        windows: All per-window offsets (may be large; also written to Parquet by the pipeline).
        sections: Per-section summaries in time order.
        overall_score_0_100: Duration-weighted mean of section scores.
        params: The constants used (window, hop, thresholds), recorded for reproducibility.
    """

    windows: list[WindowOffset] = Field(default_factory=list)
    sections: list[SectionSync] = Field(default_factory=list)
    overall_score_0_100: Score
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)
