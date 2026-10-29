"""Versioned ingestion manifests, independent of later analysis results."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field

from formsync.schemas.base import FormsyncModel
from formsync.schemas.tracks import VideoMeta


class IngestionManifest(FormsyncModel):
    """Describe one ingestion attempt and its usable artifacts.

    Source paths are absolute references, not copied inputs. Artifact paths are relative
    to the manifest directory. The fingerprint samples file size and ends; it is not a
    full content hash. Requested sampling rate may differ from the effective rate in
    ``video`` because frames are sampled at integer strides. A running manifest means
    ingestion has not committed a terminal outcome (for example after interruption).
    """

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    source_path: Path
    source_fingerprint: str | None = None
    fingerprint_method: Literal["sha256-size-and-ends-1MiB"] = "sha256-size-and-ends-1MiB"
    created_at: AwareDatetime
    completed_at: datetime | None = None
    status: Literal["running", "succeeded", "failed"] = "running"
    requested_analysis_fps: float = Field(gt=0, allow_inf_nan=False)
    audio_sample_rate: int = Field(gt=0)
    video: VideoMeta | None = None
    audio_status: Literal["not_checked", "extracted", "absent", "failed"] = "not_checked"
    audio_path: Literal["audio.wav"] | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
