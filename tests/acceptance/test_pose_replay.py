"""Synthetic detections exercise the same replay boundary future inference will use."""

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from formsync.cli import app
from formsync.io.parquet import read_keypoints, write_keypoints
from formsync.pose.replay import replay_frames
from formsync.schemas.replay import PoseStreamManifest, ScenarioConfig
from formsync.testing.pose_dataset import write_synthetic_dataset


def test_dataset_replay_is_reproducible_and_truth_is_separate(tmp_path: Path) -> None:
    """Repeatable detections survive storage, without feeding true identities to replay."""
    config = ScenarioConfig(seed=17)
    a = write_synthetic_dataset(tmp_path / "a", config)
    b = write_synthetic_dataset(tmp_path / "b", config)
    manifest = PoseStreamManifest.model_validate_json(a.read_text())
    assert manifest.n_frames == config.n_frames
    observed = read_keypoints(a.parent / "detections.parquet")
    repeated = read_keypoints(b.parent / "detections.parquet")
    for field in ("frame_idx", "t", "kp", "bbox", "det_conf", "slot", "track_id"):
        np.testing.assert_array_equal(getattr(observed, field), getattr(repeated, field))
    assert np.all(observed.slot == -1)
    assert np.all(observed.track_id == -1)
    assert observed.kp.shape[1:] == (17, 3)
    assert np.all((observed.kp[..., 2] >= 0) & (observed.kp[..., 2] <= 1))
    truth = json.loads((a.parent / "ground_truth.json").read_text())
    assert len(truth["row_dancer_ids"]) == len(observed)
    assert truth["config"]["dancers"][1]["offset_s"] == 0.25
    assert truth["config"]["dancers"][2]["amp_scale"] == 0.7
    assert len(observed) < config.n_frames * len(config.dancers)
    (a.parent / "ground_truth.json").unlink()
    frames = list(replay_frames(a))
    assert len(frames) == config.n_frames
    assert [f.frame_idx for f in frames] == list(range(config.n_frames))
    assert [f.t for f in frames] == pytest.approx(np.arange(config.n_frames) / config.fps)
    np.testing.assert_array_equal(np.concatenate([f.detections.kp for f in frames]), observed.kp)
    assert sum(len(f.detections) for f in frames) == len(observed)


def test_all_missing_frames_are_replayed(tmp_path: Path) -> None:
    """The timeline survives even when no dancer is detected anywhere."""
    path = write_synthetic_dataset(tmp_path / "empty", ScenarioConfig(dropout_frac=1))
    frames = list(replay_frames(path))
    assert len(frames) == 150
    assert all(len(frame.detections) == 0 for frame in frames)


@pytest.mark.parametrize("damage", ["frame", "time", "confidence", "identity", "coordinate"])
def test_replay_rejects_invalid_detections(tmp_path: Path, damage: str) -> None:
    """Corrupt timing, values, and leaked labels cannot silently enter the pipeline."""
    path = write_synthetic_dataset(tmp_path / damage, ScenarioConfig(dropout_frac=0))
    table = read_keypoints(path.parent / "detections.parquet")
    table = table.select(np.arange(len(table)))  # writable copies of Arrow-backed arrays
    if damage == "frame":
        table.frame_idx[0] = -1
    elif damage == "time":
        table.t[0] = 30
    elif damage == "confidence":
        table.kp[0, 0, 2] = 2
    elif damage == "identity":
        table.slot[0] = 0
    else:
        table.kp[0, 0, 0] = np.nan
    write_keypoints(table, path.parent / "detections.parquet")
    with pytest.raises(ValueError):
        list(replay_frames(path))


def test_synthetic_and_replay_cli(tmp_path: Path) -> None:
    """Users generate a dataset, replay its timeline, and get readable errors."""
    runner = CliRunner()
    output = tmp_path / "demo"
    result = runner.invoke(app, ["synthetic", "--output", str(output), "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "stream.json" in result.output
    result = runner.invoke(app, ["replay", str(output / "stream.json")])
    assert result.exit_code == 0, result.output
    assert "150 frames" in result.output
    result = runner.invoke(app, ["synthetic", "--output", str(output)])
    assert result.exit_code == 1
    result = runner.invoke(app, ["replay", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
    assert "Unable" in result.output


def test_cli_custom_scenario(tmp_path: Path) -> None:
    """JSON scenario settings configure duration and full dropout without source edits."""
    scenario = tmp_path / "scenario.json"
    scenario.write_text('{"n_frames": 20, "fps": 10, "dropout_frac": 1}')
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "synthetic",
            "--output",
            str(tmp_path / "demo"),
            "--scenario",
            str(scenario),
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["replay", str(tmp_path / "demo" / "stream.json")])
    assert result.exit_code == 0
    assert "20 frames" in result.output
    assert "0 detections" in result.output
