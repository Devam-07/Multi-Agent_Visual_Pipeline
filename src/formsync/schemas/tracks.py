"""Schemas describing the source video and the per-dancer tracks found in it.

Per-frame keypoints themselves are *not* modelled here; they live in Parquet files (see
``formsync.io.parquet``). These models hold the metadata and per-dancer summaries that the rest
of the pipeline and the dashboard need.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from formsync.schemas.base import FormsyncModel, SlotId, UnitInterval


class VideoMeta(FormsyncModel):
    """Properties of the analysed video and how it was sampled.

    Attributes:
        path: Location of the source video (local path or ``s3://`` URI).
        fps: Native frame rate of the file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        n_frames: Total frames in the file.
        duration_s: Duration in seconds.
        analysis_fps: Effective frame rate after striding (``fps / frame_stride``).
        frame_stride: Every ``frame_stride``-th source frame was analysed.
        roi: Optional crop ``(x1, y1, x2, y2)`` in pixels applied before inference. Keypoints are
            always stored in full-frame coordinates regardless of the crop.
    """

    path: str
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    n_frames: int = Field(ge=0)
    duration_s: float = Field(ge=0)
    analysis_fps: float = Field(gt=0)
    frame_stride: int = Field(ge=1)
    roi: tuple[int, int, int, int] | None = None

    @model_validator(mode="after")
    def _check_roi(self) -> VideoMeta:
        """Ensure the ROI, if present, is a non-empty box inside the frame."""
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            if not (0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height):
                raise ValueError(
                    f"roi {self.roi} is not a valid box inside {self.width}x{self.height}"
                )
        return self


class TrackSummary(FormsyncModel):
    """Summary of one dancer slot after tracking, stitching and slot assignment.

    Attributes:
        slot: Stable dancer index.
        track_ids: Raw tracker ids that were merged into this slot, in time order.
        first_t: First time (s) the slot is present.
        last_t: Last time (s) the slot is present.
        coverage: Fraction of analysed frames in which the slot has a detection.
        mean_kp_conf: Mean keypoint confidence over all present frames.
        n_id_switches: Number of times the raw tracker id changed for this slot; a data-quality
            signal (``len(track_ids) - 1``).
    """

    slot: SlotId
    track_ids: list[int] = Field(min_length=1)
    first_t: float = Field(ge=0)
    last_t: float = Field(ge=0)
    coverage: UnitInterval
    mean_kp_conf: UnitInterval
    n_id_switches: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_times(self) -> TrackSummary:
        """Ensure the time span is ordered."""
        if self.last_t < self.first_t:
            raise ValueError(f"last_t {self.last_t} precedes first_t {self.first_t}")
        return self
