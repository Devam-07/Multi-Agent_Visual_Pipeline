"""Configuration and timeline contracts for synthetic, untracked pose detections."""

from typing import Literal, Self

from pydantic import Field, model_validator

from formsync.schemas.base import FormsyncModel


class SyntheticDancer(FormsyncModel):
    """Motion controls for a dancer whose identity is its index in the scenario.

    Positive offsets mean late relative to the shared hit sequence. Amplitude scales
    joint displacement about the neutral pose, rather than changing body size.
    """

    offset_s: float = Field(default=0, allow_inf_nan=False)
    amp_scale: float = Field(default=1, ge=0, le=10, allow_inf_nan=False)


class ScenarioConfig(FormsyncModel):
    """Reproducible synthetic scenario with COCO-17 pixel coordinates.

    The default four-dancer example includes a 0.25-second delay, 70% movement
    amplitude, and independent 10% detection dropout. Dropout values are probabilities,
    not promises of an exact missing count. Low-confidence joints retain noisy finite
    estimates; entire missing detections are omitted. Frames cover [0, n_frames/fps).
    """

    schema_version: Literal[1] = 1
    seed: int = Field(default=0, ge=0)
    n_frames: int = Field(default=150, ge=1, le=90000)
    fps: float = Field(default=15, ge=1, le=240, allow_inf_nan=False)
    tempo_bpm: float = Field(default=120, ge=20, le=300, allow_inf_nan=False)
    noise_px: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    dropout_frac: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    joint_dropout_frac: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    dancers: list[SyntheticDancer] = Field(
        default=[
            SyntheticDancer(),
            SyntheticDancer(offset_s=0.25),
            SyntheticDancer(amp_scale=0.7),
            SyntheticDancer(),
        ],
        min_length=1,
        max_length=128,
    )

    @model_validator(mode="after")
    def check_size(self) -> Self:
        """Bound in-memory fixture size to half a million dancer/frame samples."""
        if self.n_frames * len(self.dancers) > 500000:
            raise ValueError("scenario exceeds 500000 dancer/frame samples")
        return self


class PoseStreamManifest(FormsyncModel):
    """Timeline needed to replay detections, including wholly empty frames.

    Contains no ground truth or motion parameters. The Parquet artifact is relative
    to this manifest. Version 1 uses constant frame rate, pixel coordinates and
    unassigned (-1) track/slot IDs; it is an input to tracking, not tracking output.
    """

    schema_version: Literal[1] = 1
    kind: Literal["untracked-coco17"] = "untracked-coco17"
    coordinates: Literal["pixels"] = "pixels"
    n_frames: int = Field(ge=1, le=90000)
    fps: float = Field(gt=0, le=240, allow_inf_nan=False)
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    detections: Literal["detections.parquet"] = "detections.parquet"
