"""Conservative, configurable evidence gates for joining raw track fragments."""

from pydantic import Field

from formsync.schemas.base import FormsyncModel


class IdentityConfig(FormsyncModel):
    """Fragment-matching limits; geometric distances are normalized by box height.

    A candidate must pass every gate and be the sole plausible outgoing/incoming
    link for both fragments. Thresholds are heuristic defaults, not probabilities
    or a calibrated claim of dancer recognition. Velocity uses endpoint observations.
    """

    max_gap_s: float = Field(default=2, gt=0, allow_inf_nan=False)
    max_distance_body: float = Field(default=0.75, gt=0, allow_inf_nan=False)
    max_pose_error: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    max_size_ratio: float = Field(default=1.5, ge=1, allow_inf_nan=False)
    confidence: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    min_common_joints: int = Field(default=6, ge=3, le=17)
    min_fragment_frames: int = Field(default=3, ge=2)
    velocity_window: int = Field(default=3, ge=2)
    max_fragments: int = Field(default=2000, ge=1, le=32767)
