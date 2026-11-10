"""Opt-in local-weight smoke test running actual YOLO and ByteTrack on human imagery."""

import json
import os
from pathlib import Path

import cv2
import pytest

from formsync.config import PipelineConfig
from formsync.ingestion import ingest_video
from formsync.io.parquet import read_keypoints
from formsync.io.video import probe_video, write_video
from formsync.pose.inference import run_poses
from formsync.schemas.tracking import TrackerConfig

pytestmark = pytest.mark.slow


def test_real_yolo_pose_and_tracking(tmp_path: Path) -> None:
    """A repeated public person image yields poses, assigned tracks, and a readable overlay."""
    weights = os.environ.get("FORMSYNC_TEST_WEIGHTS")
    sample = os.environ.get("FORMSYNC_TEST_IMAGE")
    if not weights or not sample:
        pytest.skip("Set FORMSYNC_TEST_WEIGHTS and FORMSYNC_TEST_IMAGE for local smoke assets")
    image = cv2.imread(sample)
    assert image is not None
    image = cv2.resize(image, (320, 426))
    video = write_video(tmp_path / "people.mp4", [image] * 6, 6)
    cfg = PipelineConfig()
    cfg.pose.model_name = weights
    cfg.pose.imgsz = 320
    cfg.pose.target_fps = 3
    ingestion = ingest_video(video, tmp_path / "ingested", cfg)
    result = run_poses(ingestion, tmp_path / "poses", cfg, TrackerConfig())
    record = json.loads(result.read_text())
    assert record["status"] == "succeeded" and record["n_frames"] == 3
    detections = read_keypoints(result.parent / "detections.parquet")
    tracks = read_keypoints(result.parent / "tracks.parquet")
    assert len(detections) > 0 and detections.kp.shape[1:] == (17, 3)
    assert (tracks.track_id > 0).any()
    assert probe_video(result.parent / "overlay.mp4").duration_s == pytest.approx(1)
