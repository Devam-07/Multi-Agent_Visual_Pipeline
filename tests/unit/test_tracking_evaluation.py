"""Tracking metrics use arbitrary predicted IDs and separate row-aligned labels."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from formsync.cli import app
from formsync.io.parquet import KeypointTable
from formsync.pose.tracking import track_dataset, track_frames
from formsync.schemas.replay import ScenarioConfig
from formsync.schemas.tracking import TrackerConfig
from formsync.testing.pose_dataset import write_synthetic_dataset
from formsync.testing.tracking import evaluate_tracking


def table(ids: list[int]) -> KeypointTable:
    """Create a two-dancer row-aligned result table for metric arithmetic tests."""
    n = len(ids)
    return KeypointTable(
        np.arange(n) // 2,
        np.arange(n) // 2 / 15,
        np.array(ids),
        np.full(n, -1),
        np.zeros((n, 4)),
        np.ones(n),
        np.zeros((n, 17, 3)),
    )


def test_metrics_ignore_arbitrary_id_numbers_and_count_switches() -> None:
    """A global assignment distinguishes consistent IDs from identity swapping."""
    truth = [0, 1, 0, 1, 0, 1]
    perfect = evaluate_tracking(table([9, 7, 9, 7, 9, 7]), truth)
    assert perfect["identity_switches"] == 0
    assert perfect["assignment_accuracy"] == 1
    swapped = evaluate_tracking(table([9, 7, 7, 9, 7, 9]), truth)
    assert swapped["identity_switches"] == 2
    assert swapped["extra_tracks_per_dancer"] == 2
    assert swapped["mixed_tracks"] == 2
    assert swapped["assignment_accuracy"] == pytest.approx(4 / 6)


def test_metrics_count_unassigned_and_reject_wrong_labels() -> None:
    """Unassigned detections reduce coverage and accuracy rather than disappearing."""
    result = evaluate_tracking(table([9, 7, -1, 7]), [0, 1, 0, 1])
    assert result["assigned_fraction"] == 0.75
    assert result["assignment_accuracy"] == 0.75
    with pytest.raises(ValueError):
        evaluate_tracking(table([1, 2]), [0])
    with pytest.raises(ValueError):
        evaluate_tracking(table([1, 2]), [0, 0])
    assert evaluate_tracking(KeypointTable.empty(), [])["assignment_accuracy"] is None


def test_tracker_config_validates_threshold_order() -> None:
    """Confidence thresholds and buffer values cannot silently contradict each other."""
    assert TrackerConfig().track_buffer == 30
    for values in (
        {"track_low_thresh": 0.9},
        {"new_track_thresh": 0.1},
        {"track_buffer": -1},
        {"match_thresh": float("nan")},
        {"typo": 1},
    ):
        with pytest.raises(ValidationError):
            TrackerConfig.model_validate(values)


def test_missing_dependency_fails_without_auto_install() -> None:
    """Dependency preflight fails before importing Ultralytics or attempting installation."""
    with (
        patch("formsync.pose.tracking.find_spec", return_value=None),
        pytest.raises(RuntimeError, match="uv sync"),
    ):
        list(track_frames(iter([]), TrackerConfig()))


def test_tracking_write_failure_has_no_success_manifest(tmp_path: Path) -> None:
    """Partial artifact writes never publish the tracking completion marker."""
    source = write_synthetic_dataset(tmp_path / "source", ScenarioConfig(n_frames=1))
    output = tmp_path / "failed"
    with (
        patch("formsync.pose.tracking.track_frames", return_value=iter([])),
        patch("formsync.pose.tracking.write_keypoints", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        track_dataset(source, output, TrackerConfig())
    assert not (output / "tracking.json").exists()


def test_cli_bad_tracker_config_and_missing_evaluation(tmp_path: Path) -> None:
    """Invalid configuration and missing evaluation files produce readable CLI failures."""
    config = tmp_path / "bad.yaml"
    config.write_text("track_buffer: -1")
    runner = CliRunner()
    result = runner.invoke(
        app, ["track", "missing.json", "--output", str(tmp_path / "out"), "--tracker", str(config)]
    )
    assert result.exit_code == 1
    assert "Unable" in result.output
    assert not (tmp_path / "out").exists()
    result = runner.invoke(app, ["evaluate-tracks", "missing.json", "--truth", "missing.json"])
    assert result.exit_code == 1
