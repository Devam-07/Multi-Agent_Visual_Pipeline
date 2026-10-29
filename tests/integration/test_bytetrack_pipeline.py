"""Real ByteTrack acceptance checks; no neural weights or video are needed."""

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from formsync.cli import app
from formsync.io.parquet import KeypointTable, read_keypoints
from formsync.pose.replay import PoseFrame
from formsync.pose.tracking import track_dataset, track_frames
from formsync.schemas.replay import ScenarioConfig
from formsync.schemas.tracking import TrackerConfig
from formsync.testing.pose_dataset import write_synthetic_dataset
from formsync.testing.tracking import evaluate_tracking

pytestmark = pytest.mark.slow


def frame(index: int, xs: list[float], score: float = 0.9) -> PoseFrame:
    """Make detections with distinct motion and deliberately reversible row ordering."""
    n = len(xs)
    kp = np.zeros((n, 17, 3), np.float32)
    kp[..., 0] = np.asarray(xs)[:, None]
    kp[..., 2] = score
    table = KeypointTable(
        frame_idx=np.full(n, index),
        t=np.full(n, index / 15),
        track_id=np.full(n, -1),
        slot=np.full(n, -1),
        bbox=np.array([[x, 0, x + 20, 100] for x in xs]).reshape(n, 4),
        det_conf=np.full(n, score),
        kp=kp,
    )
    return PoseFrame(index, index / 15, table)


def test_identity_survives_row_shuffle_short_gap_and_low_confidence() -> None:
    """Position association, missing frames, and the low-score pass retain identities."""
    frames = [
        frame(0, [0, 200]),
        frame(1, [201, 1]),
        frame(2, []),
        frame(3, [3, 203]),
        frame(4, [204, 4], score=0.2),
    ]
    output = list(track_frames(iter(frames), TrackerConfig(track_buffer=5)))
    first = output[0].detections.track_id
    assert first[0] != first[1] and np.all(first > 0)
    np.testing.assert_array_equal(output[1].detections.track_id, first[::-1])
    assert len(output[2].detections) == 0
    np.testing.assert_array_equal(output[3].detections.track_id, first)
    np.testing.assert_array_equal(output[4].detections.track_id, first[::-1])
    for source, tracked in zip(frames, output, strict=True):
        np.testing.assert_array_equal(source.detections.kp, tracked.detections.kp)
        assert np.all(source.detections.track_id == -1)
        assert np.all(tracked.detections.slot == -1)


def test_long_gap_creates_new_track_and_unconfirmed_is_unassigned() -> None:
    """Expired tracks are not presented as recovered identities; birth needs confirmation."""
    frames = [frame(0, [0]), *[frame(i, []) for i in range(1, 8)], frame(8, [0]), frame(9, [0])]
    output = list(track_frames(iter(frames), TrackerConfig(track_buffer=2)))
    assert output[8].detections.track_id[0] == -1
    assert output[9].detections.track_id[0] > 0
    assert output[9].detections.track_id[0] != output[0].detections.track_id[0]


def test_track_dataset_and_cli_without_truth(tmp_path: Path) -> None:
    """Saved outputs preserve every detection and tracking does not read labels."""
    source = write_synthetic_dataset(tmp_path / "source", ScenarioConfig(dropout_frac=0))
    (source.parent / "ground_truth.json").unlink()
    target = track_dataset(source, tmp_path / "tracked", TrackerConfig())
    record = json.loads(target.read_text())
    assert record["status"] == "succeeded"
    table = read_keypoints(target.parent / "tracks.parquet")
    assert len(table) == 600 and np.all(table.track_id > 0)
    assert len(np.unique(table.track_id)) == 4
    runner = CliRunner()
    result = runner.invoke(app, ["track", str(source), "--output", str(tmp_path / "cli")])
    assert result.exit_code == 0, result.output
    assert "tracking.json" in result.output
    result = runner.invoke(app, ["track", str(source), "--output", str(tmp_path / "cli")])
    assert result.exit_code == 1


def test_crossing_has_unique_assignments_and_measurable_identity_errors() -> None:
    """Crossing boxes may confuse tracking, but never assign one ID twice in a frame."""
    output = list(
        track_frames(iter([frame(i, [i * 3, 100 - i * 3]) for i in range(35)]), TrackerConfig())
    )
    for item in output:
        assigned = item.detections.track_id[item.detections.track_id >= 0]
        assert len(assigned) == len(np.unique(assigned))
    metrics = evaluate_tracking(KeypointTable.concat([f.detections for f in output]), [0, 1] * 35)
    assert metrics["assigned_fraction"] == 1
    assert metrics["identity_switches"] == 0
    assert metrics["assignment_accuracy"] == 1


def test_empty_and_nonconsecutive_streams() -> None:
    """Empty frames advance tracking, while skipped timeline indices are rejected."""
    assert len(list(track_frames(iter([frame(0, []), frame(1, [])]), TrackerConfig()))) == 2
    with pytest.raises(ValueError, match="consecutive"):
        list(track_frames(iter([frame(0, []), frame(2, [])]), TrackerConfig()))


def test_evaluation_cli_uses_matching_dataset_only(tmp_path: Path) -> None:
    """Evaluation reports correct assignments and rejects mismatched source artifacts."""
    source = write_synthetic_dataset(tmp_path / "source", ScenarioConfig(dropout_frac=0))
    result = track_dataset(source, tmp_path / "tracked", TrackerConfig())
    runner = CliRunner()
    command = ["evaluate-tracks", str(result), "--truth", str(source.parent / "ground_truth.json")]
    evaluation = runner.invoke(app, command)
    assert evaluation.exit_code == 0, evaluation.output
    metrics = json.loads(evaluation.output)
    assert metrics["identity_switches"] == 0
    assert metrics["assignment_accuracy"] == 1
    (source.parent / "detections.parquet").write_bytes(b"changed")
    evaluation = runner.invoke(app, command)
    assert evaluation.exit_code == 1
    assert "match" in evaluation.output
