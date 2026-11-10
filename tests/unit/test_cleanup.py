"""Numerical contracts for confidence filtering, gap filling and segment smoothing."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from formsync.cli import app
from formsync.io.parquet import KeypointTable, read_keypoints, write_keypoints
from formsync.pose.cleanup import CleanupConfig, clean_dataset, clean_joint, segments


def test_short_gaps_and_confidence_are_traceable() -> None:
    """Fill bounded gaps while retaining original confidence and observed/interpolated masks."""
    raw = np.column_stack((np.arange(10), np.arange(10) * 2, np.ones(10))).astype(float)
    raw[3:5, 2] = 0.1
    raw[0, 2] = 0
    raw[-1, 2] = 0
    cleaned, observed, filled, smoothed = clean_joint(raw, CleanupConfig())
    assert np.isnan(cleaned[[0, 9], :2]).all()
    np.testing.assert_allclose(cleaned[1:9, 0], np.arange(1, 9), atol=1e-6)
    np.testing.assert_array_equal(cleaned[:, 2], raw[:, 2])
    assert filled.tolist() == [False, False, False, True, True, False, False, False, False, False]
    assert not observed[3] and smoothed[3]
    assert not smoothed[0]


def test_long_gaps_and_short_segments_stay_separate() -> None:
    """A long gap is never filled or crossed by smoothing, and short segments are unchanged."""
    raw = np.column_stack((np.arange(12), np.arange(12), np.ones(12))).astype(float)
    raw[3:9] = np.nan
    result, _, filled, smoothed = clean_joint(raw, CleanupConfig(max_gap_frames=2))
    assert np.isnan(result[3:9, :2]).all()
    assert not filled.any() and not smoothed.any()
    np.testing.assert_array_equal(result[:3], raw[:3])
    np.testing.assert_array_equal(result[9:], raw[9:])


def test_smoothing_reduces_noise_with_limited_phase_and_amplitude_error() -> None:
    """Known sinusoidal motion must retain its phase and amplitude while reducing noise."""
    t = np.arange(300) / 30
    truth = 30 * np.sin(2 * np.pi * t)
    noisy = truth + np.random.default_rng(7).normal(0, 2, len(t))
    raw = np.column_stack((noisy, noisy, np.ones(len(t))))
    result, _, _, _ = clean_joint(raw, CleanupConfig(smooth_window=11, smooth_poly=3))
    middle = slice(11, -11)
    assert (
        np.mean((result[middle, 0] - truth[middle]) ** 2)
        < np.mean((noisy[middle] - truth[middle]) ** 2) * 0.5
    )
    design = np.column_stack((np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)))
    coefficients = np.linalg.lstsq(design[middle], result[middle, 0], rcond=None)[0]
    assert abs(np.linalg.norm(coefficients) / 30 - 1) < 0.03
    assert abs(np.arctan2(coefficients[1], coefficients[0]) / (2 * np.pi)) < 1 / 30


@pytest.mark.parametrize(
    "options",
    [
        {"smooth_window": 4},
        {"smooth_poly": 7},
        {"max_gap_frames": -1},
        {"confidence": float("nan")},
    ],
)
def test_invalid_settings(options: dict) -> None:
    """Reject invalid filter windows, interpolation limits, and non-finite confidence."""
    with pytest.raises(ValueError):
        CleanupConfig(**options)


def source_dataset(path: Path, kind: str = "tracking") -> Path:
    """Write a tiny result with a short internal gap, a distinct track, and an unassigned row."""
    path.mkdir()
    frames = np.array([0, 2, 3, 4, 5])
    ids = np.array([10, 10, 20, 20, -1])
    kp = np.ones((5, 17, 3), np.float32)
    kp[:, :, 0] = np.array([0, 2, 100, 101, 999])[:, None]
    kp[:, :, 1] = 5
    table = KeypointTable(
        frames, frames / 10, ids, np.full(5, -1), np.zeros((5, 4)), np.ones(5), kp
    )
    write_keypoints(table, path / "tracks.parquet")
    manifest = path / ("poses.json" if kind == "poses" else "tracking.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "succeeded",
                "n_frames": 6,
                "analysis_fps" if kind == "poses" else "fps": 10,
            }
        )
    )
    return manifest


@pytest.mark.parametrize("kind", ["tracking", "poses"])
def test_cleanup_artifacts_and_cli(tmp_path: Path, kind: str) -> None:
    """Both producer formats yield isolated tracks, complete samples and quality metrics."""
    source = source_dataset(tmp_path / "source", kind)
    original = (source.parent / "tracks.parquet").read_bytes()
    result = CliRunner().invoke(app, ["clean", str(source), "--output", str(tmp_path / "clean")])
    assert result.exit_code == 0, result.output
    rows = pq.read_table(tmp_path / "clean" / "trajectories.parquet").to_pandas()
    assert len(rows) == (3 + 2) * 17
    assert set(rows.track_id) == {10, 20}
    gap = rows[(rows.track_id == 10) & (rows.frame_idx == 1)]
    assert np.allclose(gap.x, 1) and gap.interpolated.all()
    assert gap.confidence.isna().all()
    assert rows[rows.track_id == 20].x.min() >= 100
    quality = json.loads((tmp_path / "clean" / "quality.json").read_text())
    assert quality["unassigned_detections"] == 1
    assert quality["tracks"][0]["interpolated_samples"] == 17
    assert (source.parent / "tracks.parquet").read_bytes() == original
    assert json.loads((tmp_path / "clean" / "cleanup.json").read_text())["status"] == "succeeded"
    with pytest.raises(FileExistsError):
        clean_dataset(source, tmp_path / "clean", CleanupConfig())


def test_failed_write_does_not_publish_success(tmp_path: Path) -> None:
    """Write failures cannot leave a success manifest."""
    source = source_dataset(tmp_path / "source")
    with (
        patch("formsync.pose.cleanup.pq.write_table", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        clean_dataset(source, tmp_path / "out", CleanupConfig())
    assert not (tmp_path / "out" / "cleanup.json").exists()


def test_empty_and_all_missing_series() -> None:
    """Empty or wholly missing series have no inferred observations or smoothing."""
    for raw in (np.empty((0, 3)), np.full((10, 3), np.nan)):
        result, observed, filled, smoothed = clean_joint(raw, CleanupConfig())
        assert result.shape == raw.shape
        assert np.isnan(result).all()
        assert not observed.any() and not filled.any() and not smoothed.any()
    assert segments(np.array([True, True, False, True])) == [(0, 2), (3, 4)]


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("inf")])
def test_invalid_joint_confidence_is_rejected(confidence: float) -> None:
    """Invalid confidences must not produce trusted observed coordinates."""
    raw = np.array([[1, 2, confidence]])
    with pytest.raises(ValueError, match="confidence"):
        clean_joint(raw, CleanupConfig())


@pytest.mark.parametrize("empty", [True, False])
def test_empty_and_unassigned_datasets_have_typed_outputs(tmp_path: Path, empty: bool) -> None:
    """No usable tracks is a valid empty result, with explicit excluded-detection counts."""
    source = source_dataset(tmp_path / "source")
    table = KeypointTable.empty()
    if not empty:
        table = read_keypoints(source.parent / "tracks.parquet")
        table.track_id = np.full(len(table), -1)
    write_keypoints(table, source.parent / "tracks.parquet")
    result = clean_dataset(source, tmp_path / "out", CleanupConfig())
    output = pq.read_table(result.parent / "trajectories.parquet")
    assert output.num_rows == 0
    assert "interpolated" in output.column_names
    quality = json.loads((result.parent / "quality.json").read_text())
    assert quality["raw_track_count"] == 0
    assert quality["unassigned_detections"] == (0 if empty else 5)


@pytest.mark.parametrize("damage", ["duplicate", "timestamp", "frame", "status", "fps"])
def test_bad_source_contract_is_rejected(tmp_path: Path, damage: str) -> None:
    """Invalid timelines and duplicate observations cannot silently overwrite trajectory data."""
    source = source_dataset(tmp_path / "source")
    table = read_keypoints(source.parent / "tracks.parquet")
    if damage == "duplicate":
        table = KeypointTable.concat([table, table.select(np.array([0]))])
    elif damage == "timestamp":
        table.t = np.zeros(len(table))
    elif damage == "frame":
        table.frame_idx = np.full(len(table), -1)
    else:
        record = json.loads(source.read_text())
        record[damage] = "failed" if damage == "status" else 0
        source.write_text(json.dumps(record))
    write_keypoints(table, source.parent / "tracks.parquet")
    with pytest.raises(ValueError):
        clean_dataset(source, tmp_path / "out", CleanupConfig())
    assert not (tmp_path / "out").exists()
