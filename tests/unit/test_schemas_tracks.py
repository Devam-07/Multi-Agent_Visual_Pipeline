"""Tests for formsync.schemas.tracks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from formsync.schemas.tracks import TrackSummary, VideoMeta


def _video(**overrides: object) -> VideoMeta:
    """Build a valid VideoMeta, overriding any fields given as keyword arguments."""
    base: dict[str, object] = {
        "path": "practice.mp4",
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "n_frames": 5400,
        "duration_s": 180.0,
        "analysis_fps": 15.0,
        "frame_stride": 2,
        "roi": None,
    }
    base.update(overrides)
    return VideoMeta.model_validate(base)


def test_video_meta_round_trip() -> None:
    """VideoMeta survives a JSON round trip unchanged."""
    v = _video(roi=(100, 200, 1800, 1000))
    assert VideoMeta.model_validate_json(v.model_dump_json()) == v


def test_video_meta_rejects_roi_outside_frame() -> None:
    """An ROI that extends past the frame or is inverted is rejected."""
    with pytest.raises(ValidationError):
        _video(roi=(0, 0, 2000, 500))
    with pytest.raises(ValidationError):
        _video(roi=(500, 0, 100, 500))


def test_track_summary_round_trip_and_order() -> None:
    """TrackSummary round-trips and rejects last_t before first_t."""
    t = TrackSummary(
        slot=3,
        track_ids=[7, 12],
        first_t=0.0,
        last_t=170.5,
        coverage=0.93,
        mean_kp_conf=0.81,
        n_id_switches=1,
    )
    assert TrackSummary.model_validate_json(t.model_dump_json()) == t
    with pytest.raises(ValidationError):
        TrackSummary(
            slot=0,
            track_ids=[1],
            first_t=10.0,
            last_t=5.0,
            coverage=1.0,
            mean_kp_conf=1.0,
            n_id_switches=0,
        )
