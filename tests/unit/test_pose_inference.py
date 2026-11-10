"""Model conversion and full media workflow contracts without loading neural libraries."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from formsync.cli import app
from formsync.config import PipelineConfig
from formsync.ingestion import ingest_video
from formsync.io.video import Frame, probe_video
from formsync.pose.inference import pose_from_arrays, run_poses
from formsync.pose.replay import replay_frames
from formsync.schemas.tracking import TrackerConfig
from formsync.testing.video import make_synthetic_video
from formsync.viz.overlay import draw_poses


def test_conversion_preserves_pixels_confidence_and_source_time() -> None:
    """Predictions retain original pixel coordinates and timestamps after sampling."""
    frame = Frame(2, 4, 4 / 30, np.zeros((120, 160, 3), np.uint8))
    kp = np.full((1, 17, 3), 0.2, np.float32)
    kp[..., :2] = [30, 40]
    pose = pose_from_arrays(frame, np.array([[10, 20, 80, 90]]), np.array([0.9]), kp)
    assert pose.t == 4 / 30 and pose.frame_idx == 2
    np.testing.assert_array_equal(pose.detections.kp, kp)
    assert pose.detections.track_id.tolist() == [-1]
    assert pose.detections.slot.tolist() == [-1]
    empty = pose_from_arrays(frame, np.empty((0, 4)), np.empty(0), np.empty((0, 17, 3)))
    assert len(empty.detections) == 0
    with pytest.raises(ValueError):
        pose_from_arrays(frame, np.zeros((1, 4)), np.ones(1), np.zeros((1, 16, 3)))


def test_overlay_filters_uncertain_joints_and_preserves_image() -> None:
    """Low-confidence joints are hidden while confident joints appear on a copied image."""
    image = np.zeros((120, 160, 3), np.uint8)
    frame = Frame(0, 0, 0, image)
    kp = np.zeros((1, 17, 3), np.float32)
    kp[0, 9] = [70, 80, 0.9]
    kp[0, 10] = [100, 80, 0.1]
    pose = pose_from_arrays(frame, np.array([[5, 5, 150, 110]]), np.ones(1), kp)
    rendered = draw_poses(image, pose.detections, 0.3)
    assert rendered[78:83, 68:73].any()
    assert not rendered[78:83, 98:103].any()
    assert not image.any()


def empty_predictions(source: Path, config: object, stride: int) -> list:
    """Supply deterministic empty model outputs while retaining real decoded frame timing."""
    from formsync.io.video import iter_frames

    return [
        pose_from_arrays(f, np.empty((0, 4)), np.empty(0), np.empty((0, 17, 3)))
        for f in iter_frames(source, frame_stride=stride)
    ]


def test_run_outputs_replayable_timeline_and_overlay(tmp_path: Path) -> None:
    """Real codecs preserve sampled playback and a complete empty-detection timeline."""
    cfg = PipelineConfig()
    video = make_synthetic_video(tmp_path / "source.mp4", n_frames=61)
    ingested = ingest_video(video, tmp_path / "runs", cfg)
    with (
        patch("formsync.pose.inference.predict_frames", side_effect=empty_predictions),
        patch("formsync.pose.inference.track_frames", side_effect=lambda frames, config: frames),
    ):
        result = run_poses(ingested, tmp_path / "poses", cfg, TrackerConfig())
    record = json.loads(result.read_text())
    assert record["status"] == "succeeded"
    assert record["source_frame_stride"] == 2
    assert record["n_frames"] == 31
    assert len(list(replay_frames(result.parent / "stream.json"))) == 31
    meta = probe_video(result.parent / "overlay.mp4")
    assert meta.n_frames == 31 and meta.fps == 15
    assert abs(meta.duration_s - 61 / 30) < 1 / 15
    assert (result.parent / "tracks.parquet").exists()
    with pytest.raises(FileExistsError):
        run_poses(ingested, tmp_path / "poses", cfg, TrackerConfig())


@pytest.mark.parametrize("failure", ["weights unavailable", "inference failed", "disk full"])
def test_failure_has_no_success_artifacts(tmp_path: Path, failure: str) -> None:
    """Runtime failures record a failed outcome rather than a usable replay stream."""
    cfg = PipelineConfig()
    video = make_synthetic_video(tmp_path / "source.mp4")
    manifest = ingest_video(video, tmp_path / "runs", cfg)
    with (
        patch("formsync.pose.inference.predict_frames", side_effect=RuntimeError(failure)),
        pytest.raises(RuntimeError, match=failure),
    ):
        run_poses(manifest, tmp_path / "out", cfg, TrackerConfig())
    record = json.loads((tmp_path / "out" / "poses.json").read_text())
    assert record["status"] == "failed" and failure in record["error"]
    assert not (tmp_path / "out" / "stream.json").exists()


def test_changed_source_and_cli_error(tmp_path: Path) -> None:
    """Changed inputs are rejected before inference, with a readable CLI failure."""
    cfg = PipelineConfig()
    video = make_synthetic_video(tmp_path / "source.mp4")
    manifest = ingest_video(video, tmp_path / "runs", cfg)
    video.write_bytes(b"changed")
    result = CliRunner().invoke(app, ["poses", str(manifest), "--output", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "changed" in result.output.lower()


@pytest.mark.parametrize("stage", ["decode", "write", "overlay"])
def test_incomplete_outputs_never_publish_stream(tmp_path: Path, stage: str) -> None:
    """Early decoding and artifact failures leave a failed status without replay publication."""
    cfg = PipelineConfig()
    video = make_synthetic_video(tmp_path / "source.mp4")
    manifest = ingest_video(video, tmp_path / "runs", cfg)
    target = {"decode": "predict_frames", "write": "write_keypoints", "overlay": "write_video"}[
        stage
    ]
    with (
        patch("formsync.pose.inference.predict_frames", side_effect=empty_predictions),
        patch("formsync.pose.inference.track_frames", side_effect=lambda frames, config: frames),
        patch("formsync.pose.inference." + target) as failure,
    ):
        if stage == "decode":
            failure.return_value = []
        else:
            failure.side_effect = OSError("write failed")
        with pytest.raises((ValueError, OSError)):
            run_poses(manifest, tmp_path / "out", cfg, TrackerConfig())
    assert json.loads((tmp_path / "out" / "poses.json").read_text())["status"] == "failed"
    assert not (tmp_path / "out" / "stream.json").exists()


def test_cli_success_calls_workflow(tmp_path: Path) -> None:
    """The public poses command passes validated settings into the workflow."""
    with patch("formsync.pose.inference.run_poses", return_value=tmp_path / "poses.json") as run:
        result = CliRunner().invoke(
            app, ["poses", "manifest.json", "--output", str(tmp_path / "out")]
        )
    assert result.exit_code == 0, result.output
    assert "poses.json" in result.output
    assert run.call_args.args[0] == Path("manifest.json")
