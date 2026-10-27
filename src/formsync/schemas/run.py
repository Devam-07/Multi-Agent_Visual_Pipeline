"""The top-level result of one analysis run: everything the pipeline produced, in one object.

``RunResult`` is written to ``runs/<run_id>/manifest.json`` and is what the dashboard and the
feedback agent read. Large per-frame data is referenced through ``artifacts`` rather than
embedded.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from formsync.schemas.base import FormsyncModel
from formsync.schemas.beats import BeatGrid
from formsync.schemas.feedback import FeedbackReport
from formsync.schemas.formation import FormationResult, StageCalibration
from formsync.schemas.sections import SectionList
from formsync.schemas.sync import SyncResult
from formsync.schemas.tracks import TrackSummary, VideoMeta


class RunResult(FormsyncModel):
    """Manifest of one analysis run.

    Stage results are optional so a partially completed run (for example, pose finished but the
    LLM call failed) can still be saved and inspected.

    Attributes:
        run_id: Unique run identifier.
        created_at: When the run started.
        video: Source video metadata.
        roster_size: Number of dancer slots the tracker was asked to maintain.
        tracks: Per-slot tracking summaries.
        beats: Beat grid for the audio.
        sections: Sections analysed.
        calibration: Stage calibration used, if any.
        sync: Synchronisation results, if that stage completed.
        formation: Formation results, if that stage completed.
        feedback: Feedback report, if the agent ran.
        artifacts: Name -> URI of every file written for this run (Parquet, JSON, overlay video).
        timings_s: Wall-clock seconds per stage, for profiling.
        warnings: Human-readable data-quality warnings raised during the run.
    """

    run_id: str = Field(min_length=1)
    created_at: datetime
    video: VideoMeta
    roster_size: int = Field(ge=1)
    tracks: list[TrackSummary] = Field(default_factory=list)
    beats: BeatGrid
    sections: SectionList
    calibration: StageCalibration | None = None
    sync: SyncResult | None = None
    formation: FormationResult | None = None
    feedback: FeedbackReport | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    timings_s: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
