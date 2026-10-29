"""Numerical ground truth, schema validation, and failure contracts for pose fixtures."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from formsync.cli import app
from formsync.io.parquet import read_keypoints, write_keypoints
from formsync.pose.replay import replay_frames
from formsync.schemas.replay import PoseStreamManifest, ScenarioConfig
from formsync.testing.pose_dataset import write_synthetic_dataset


def test_saved_coordinates_have_known_lag_and_amplitude(tmp_path: Path) -> None:
    """Observed joint motion, rather than labels alone, has the intended lag and scale."""
    config = ScenarioConfig(fps=20, n_frames=120, dropout_frac=0)
    path = write_synthetic_dataset(tmp_path / "scene", config)
    table = read_keypoints(path.parent / "detections.parquet")
    truth = json.loads((path.parent / "ground_truth.json").read_text())
    labels = np.asarray(truth["row_dancer_ids"])
    tracks = [table.kp[labels == i, :, :2].astype(float) for i in range(4)]
    # Subtract each dancer's initial neutral pose to remove position on the stage.
    motion = [track - track[0] for track in tracks]
    np.testing.assert_allclose(motion[1][5:], motion[0][:-5], atol=1e-4)
    # Pixel values near 1920 have float32 resolution around 0.00012 pixels.
    np.testing.assert_allclose(motion[2], motion[0] * 0.7, atol=2e-4)
    assert np.max(np.abs(motion[0])) > 1
    np.testing.assert_allclose(
        np.asarray(truth["dancer_hit_stops_s"]["1"]),
        np.asarray(truth["group_hit_stops_s"]["main"]) + 0.25,
    )
    # Rows must not preserve the known dancer order on every frame.
    assert any(list(labels[table.frame_idx == i]) != [0, 1, 2, 3] for i in range(120))


@pytest.mark.parametrize(
    "settings",
    [
        {"fps": 0},
        {"fps": float("nan")},
        {"n_frames": 0},
        {"dropout_frac": 1.1},
        {"joint_dropout_frac": -1},
        {"noise_px": -1},
        {"seed": -1},
        {"dancers": []},
        {"dancers": [{"amp_scale": -0.1}]},
        {"dancers": [{"offset_s": float("inf")}]},
        {"n_frames": 90000, "dancers": [{}] * 6},
        {"unexpected": True},
    ],
)
def test_invalid_scenarios_are_rejected(settings: dict[str, object]) -> None:
    """Reject invalid numerical values, oversized fixtures, and unknown configuration keys."""
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(settings)


def test_manifest_does_not_allow_truth_or_path_traversal() -> None:
    """Replay only accepts the known relative artifact and untracked stream format."""
    for fields in (
        {"detections": "../other.parquet"},
        {"schema_version": 2},
        {"ground_truth": "secret"},
    ):
        with pytest.raises(ValidationError):
            PoseStreamManifest.model_validate({"n_frames": 10, "fps": 15} | fields)


def test_joint_dropout_and_seed_variation(tmp_path: Path) -> None:
    """Occluded joints remain finite low-confidence estimates and seeds change the sample."""
    a = write_synthetic_dataset(
        tmp_path / "a", ScenarioConfig(seed=1, joint_dropout_frac=1, dropout_frac=0)
    )
    b = write_synthetic_dataset(
        tmp_path / "b", ScenarioConfig(seed=2, joint_dropout_frac=1, dropout_frac=0)
    )
    first = read_keypoints(a.parent / "detections.parquet")
    second = read_keypoints(b.parent / "detections.parquet")
    assert np.isfinite(first.kp).all()
    assert np.all(first.kp[..., 2] <= 0.2)
    assert not np.array_equal(first.kp, second.kp)
    assert len(list(replay_frames(a))) == 150


def test_missing_frames_including_timeline_edges(tmp_path: Path) -> None:
    """Explicit timeline metadata retains gaps before, between, and after detections."""
    path = write_synthetic_dataset(tmp_path / "gaps", ScenarioConfig(n_frames=10, dropout_frac=0))
    table = read_keypoints(path.parent / "detections.parquet")
    table = table.select(np.isin(table.frame_idx, [2, 5]))
    write_keypoints(table, path.parent / "detections.parquet")
    frames = list(replay_frames(path))
    assert [f.frame_idx for f in frames if len(f.detections)] == [2, 5]
    assert len(frames) == 10
    assert frames[-1].t == pytest.approx(9 / 15)


def test_dataset_write_failure_has_no_completion_manifest(tmp_path: Path) -> None:
    """Failure to persist detections cannot leave a dataset that appears complete."""
    output = tmp_path / "partial"
    with (
        patch("formsync.testing.pose_dataset.write_keypoints", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        write_synthetic_dataset(output, ScenarioConfig())
    assert not (output / "stream.json").exists()


def test_existing_dataset_is_not_modified(tmp_path: Path) -> None:
    """Refusing an existing destination preserves every previously saved artifact."""
    output = tmp_path / "scene"
    write_synthetic_dataset(output, ScenarioConfig())
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        write_synthetic_dataset(output, ScenarioConfig(seed=9))
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_cli_invalid_scenario_and_corrupt_replay(tmp_path: Path) -> None:
    """Malformed configuration and broken Parquet produce readable nonzero CLI exits."""
    runner = CliRunner()
    config = tmp_path / "bad.json"
    config.write_text("{")
    result = runner.invoke(
        app,
        [
            "synthetic",
            "--output",
            str(tmp_path / "unused"),
            "--scenario",
            str(config),
        ],
    )
    assert result.exit_code == 1
    assert "Unable" in result.output
    assert not (tmp_path / "unused").exists()
    path = write_synthetic_dataset(tmp_path / "scene", ScenarioConfig())
    (path.parent / "detections.parquet").write_bytes(b"corrupt")
    result = runner.invoke(app, ["replay", str(path)])
    assert result.exit_code == 1
    assert "Unable" in result.output
