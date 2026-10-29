"""Validated ByteTrack controls shared by replay and future inference adapters."""

from typing import Literal, Self

from pydantic import Field, model_validator

from formsync.schemas.base import FormsyncModel


class TrackerConfig(FormsyncModel):
    """ByteTrack thresholds and retention in analysis frames, independent of source fps.

    Lost tracks are retained for ``track_buffer`` updates. Low-confidence detections
    can continue active tracks but cannot start new ones. Slots remain unassigned;
    this configuration produces raw track IDs, not permanent dancer identities.
    """

    tracker_type: Literal["bytetrack"] = "bytetrack"
    track_high_thresh: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    track_low_thresh: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    new_track_thresh: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    track_buffer: int = Field(default=30, ge=0)
    match_thresh: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    fuse_score: bool = True

    @model_validator(mode="after")
    def check_thresholds(self) -> Self:
        """Ensure low-confidence rescue and new-track thresholds are ordered."""
        if not self.track_low_thresh < self.track_high_thresh <= self.new_track_thresh:
            raise ValueError("require track_low_thresh < track_high_thresh <= new_track_thresh")
        return self
