"""Pydantic data models shared by every pipeline stage. Contains no pipeline logic.

Conventions (see :mod:`formsync.schemas.base` for the full list): seconds and beats for time,
positive offset = late, pixel coordinates origin top-left, stage coordinates in metres with
origin downstage-left.
"""

from formsync.schemas.base import FormsyncModel, Score, SlotId, UnitInterval
from formsync.schemas.beats import BeatGrid
from formsync.schemas.feedback import CoachingNote, FeedbackReport, NoteCategory
from formsync.schemas.formation import FormationEvent, FormationResult, StageCalibration
from formsync.schemas.run import RunResult
from formsync.schemas.sections import DancerGroup, Section, SectionList
from formsync.schemas.sync import (
    DancerSectionSync,
    HitEvent,
    MotionGroup,
    SectionSync,
    SyncResult,
    WindowOffset,
)
from formsync.schemas.tracks import TrackSummary, VideoMeta

__all__ = [
    "BeatGrid",
    "CoachingNote",
    "DancerGroup",
    "DancerSectionSync",
    "FeedbackReport",
    "FormationEvent",
    "FormationResult",
    "FormsyncModel",
    "HitEvent",
    "MotionGroup",
    "NoteCategory",
    "RunResult",
    "Score",
    "Section",
    "SectionList",
    "SectionSync",
    "SlotId",
    "StageCalibration",
    "SyncResult",
    "TrackSummary",
    "UnitInterval",
    "VideoMeta",
    "WindowOffset",
]
