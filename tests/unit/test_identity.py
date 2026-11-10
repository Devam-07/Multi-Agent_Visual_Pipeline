"""Test-first contracts for conservative track-fragment identity reconciliation."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from formsync.cli import app
from formsync.io.parquet import KeypointTable, read_keypoints, write_keypoints
from formsync.pose.cleanup import CleanupConfig, clean_dataset
from formsync.pose.identity import reconcile, reconcile_dataset
from formsync.schemas.identity import IdentityConfig
from formsync.testing.synthetic import BASE_SKELETON


def fragments(specs: list[tuple[int, int, int, float, float]]) -> KeypointTable:
    """Generate known fragments (ID, first frame, last frame, initial x, velocity/frame)."""
    rows = []
    for identity, start, end, x, velocity in specs:
        for frame in range(start, end + 1):
            center = x + (frame - start) * velocity
            kp = np.column_stack((BASE_SKELETON * 40 + [center, 70], np.ones(17)))
            rows.append(
                KeypointTable(
                    np.array([frame]),
                    np.array([frame / 10]),
                    np.array([identity]),
                    np.array([-1]),
                    np.array([[center - 40, 0, center + 40, 160]]),
                    np.array([0.9]),
                    kp[None],
                )
            )
    return KeypointTable.concat(rows).sorted_by_frame()


def test_motion_continuity_joins_fragments_without_changing_measurements() -> None:
    """A moving dancer reappearing under a new ID receives the same stable slot."""
    table = fragments([(10, 0, 2, 100, 5), (30, 6, 8, 130, 5)])
    result, report = reconcile(table, 10, IdentityConfig())
    assert set(result.slot) == {0}
    assert report["accepted_links"] == [{"from_track": 10, "to_track": 30}]
    for name in ("frame_idx", "t", "track_id", "kp", "bbox", "det_conf"):
        np.testing.assert_array_equal(getattr(result, name), getattr(table, name))
    assert np.all(table.slot == -1)


def test_competing_matches_remain_separate() -> None:
    """Two plausible returns must not be resolved by arbitrary row or ID order."""
    table = fragments([(10, 0, 2, 100, 0), (20, 6, 8, 99, 0), (30, 6, 8, 101, 0)])
    result, report = reconcile(table, 10, IdentityConfig())
    assert len(np.unique(result.slot)) == 3
    assert report["accepted_links"] == []
    assert len(report["ambiguous_links"]) == 2
    shuffled, _ = reconcile(table.select(np.arange(len(table))[::-1]), 10, IdentityConfig())
    for identity in [10, 20, 30]:
        assert set(shuffled.slot[shuffled.track_id == identity]) == set(
            result.slot[result.track_id == identity]
        )


def test_two_predecessors_cannot_claim_one_return() -> None:
    """Reciprocal uniqueness rejects a merge when the successor has competing predecessors."""
    table = fragments([(10, 0, 2, 99, 0), (20, 0, 2, 101, 0), (30, 6, 8, 100, 0)])
    result, report = reconcile(table, 10, IdentityConfig())
    assert len(np.unique(result.slot)) == 3
    assert not report["accepted_links"]


@pytest.mark.parametrize("case", ["overlap", "far", "long_gap", "pose", "confidence", "short"])
def test_insufficient_or_inconsistent_evidence_is_not_joined(case: str) -> None:
    """Timing, location, pose, confidence and minimum duration all gate a link."""
    start, end, x = 6, 8, 100
    if case == "overlap":
        start, end = 1, 3
    if case == "far":
        x = 1000
    if case == "long_gap":
        start, end = 50, 52
    if case == "short":
        end = start
    table = fragments([(10, 0, 2, 100, 0), (20, start, end, x, 0)])
    if case == "pose":
        table.kp[table.track_id == 20, :8, 0] += 100
    if case == "confidence":
        table.kp[table.track_id == 20, :, 2] = 0.1
    result, report = reconcile(table, 10, IdentityConfig())
    assert len(np.unique(result.slot)) == 2
    assert not report["accepted_links"]


def test_chain_and_crossing_assignments() -> None:
    """Unambiguous chains are transitive; opposing trajectories retain separate slots."""
    chain = fragments([(1, 0, 2, 100, 0), (2, 10, 12, 100, 0), (3, 20, 22, 100, 0)])
    result, _ = reconcile(chain, 10, IdentityConfig(max_gap_s=1))
    assert len(np.unique(result.slot)) == 1
    crossing = fragments(
        [(10, 0, 2, 100, 10), (20, 0, 2, 200, -10), (30, 10, 12, 200, 10), (40, 10, 12, 100, -10)]
    )
    result, report = reconcile(crossing, 10, IdentityConfig(max_distance_body=0.3))
    mapping = {int(i): int(result.slot[result.track_id == i][0]) for i in [10, 20, 30, 40]}
    assert mapping[10] == mapping[30] and mapping[20] == mapping[40]
    assert mapping[10] != mapping[20]
    assert len(report["accepted_links"]) == 2


def test_empty_unassigned_and_duplicate_input() -> None:
    """Empty input is valid, unassigned rows stay unassigned, and duplicate track frames fail."""
    result, report = reconcile(KeypointTable.empty(), 10, IdentityConfig())
    assert len(result) == 0 and report["slot_count"] == 0
    table = fragments([(1, 0, 2, 100, 0)])
    table.track_id[:] = -1
    result, report = reconcile(table, 10, IdentityConfig())
    assert np.all(result.slot == -1) and report["unassigned_detections"] == 3
    table = fragments([(1, 0, 2, 100, 0), (1, 0, 2, 100, 0)])
    with pytest.raises(ValueError, match="Duplicate"):
        reconcile(table, 10, IdentityConfig())


def test_dataset_cli_and_failed_writes(tmp_path: Path) -> None:
    """The CLI publishes slots with a provenance report and refuses existing destinations."""
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "tracking.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "status": "succeeded", "fps": 10, "n_frames": 9})
    )
    table = fragments([(10, 0, 2, 100, 0), (30, 6, 8, 100, 0)])
    write_keypoints(table, source / "tracks.parquet")
    original = (source / "tracks.parquet").read_bytes()
    runner = CliRunner()
    result = runner.invoke(app, ["reconcile", str(manifest), "--output", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    output = read_keypoints(tmp_path / "out" / "slots.parquet")
    assert set(output.slot) == {0}
    record = json.loads((tmp_path / "out" / "identities.json").read_text())
    assert record["status"] == "succeeded" and record["source_sha256"]
    assert (source / "tracks.parquet").read_bytes() == original
    result = runner.invoke(app, ["reconcile", str(manifest), "--output", str(tmp_path / "out")])
    assert result.exit_code == 1
    with (
        patch("formsync.pose.identity.write_keypoints", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        reconcile_dataset(manifest, tmp_path / "failed", IdentityConfig())
    assert not (tmp_path / "failed" / "identities.json").exists()


def test_config_rejects_invalid_thresholds() -> None:
    """All matching thresholds are finite and enough joints/frames must support a link."""
    for options in (
        {"max_gap_s": 0},
        {"max_distance_body": float("nan")},
        {"min_common_joints": 18},
        {"min_fragment_frames": 1},
    ):
        with pytest.raises(ValueError):
            IdentityConfig(**options)


def test_exact_time_limit_is_not_changed_by_float32_rounding() -> None:
    """Measure the configured gap from frame indices, without float32 timestamp rounding."""
    table = fragments([(10, 0, 2, 100, 0), (20, 5, 7, 100, 0)])
    result, _ = reconcile(table, 10, IdentityConfig(max_gap_s=0.3))
    assert len(np.unique(result.slot)) == 1


def test_cleanup_preserves_slots_without_smoothing_across_fragments(tmp_path: Path) -> None:
    """Reconciled slots flow into cleanup while original raw fragment boundaries remain intact."""
    import pyarrow.parquet as pq

    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "poses.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "status": "succeeded", "analysis_fps": 10, "n_frames": 9})
    )
    write_keypoints(fragments([(10, 0, 2, 100, 0), (30, 6, 8, 100, 0)]), source / "tracks.parquet")
    identity = reconcile_dataset(manifest, tmp_path / "identities", IdentityConfig())
    cleanup = clean_dataset(identity, tmp_path / "clean", CleanupConfig())
    rows = pq.read_table(cleanup.parent / "trajectories.parquet").to_pandas()
    assert set(rows.slot) == {0}
    assert set(rows.track_id) == {10, 30}
    assert set(rows.frame_idx) == {0, 1, 2, 6, 7, 8}
    assert not rows.interpolated.any()


def test_cleanup_rejects_simultaneous_people_in_one_slot(tmp_path: Path) -> None:
    """A corrupted slot assignment must not flow into scoring as a consistent dancer."""
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "tracking.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "status": "succeeded", "fps": 10, "n_frames": 3})
    )
    write_keypoints(fragments([(10, 0, 2, 100, 0), (20, 0, 2, 200, 0)]), source / "tracks.parquet")
    identity = reconcile_dataset(manifest, tmp_path / "ids", IdentityConfig())
    table = read_keypoints(identity.parent / "slots.parquet")
    table.slot = np.zeros(len(table), dtype=np.int16)
    write_keypoints(table, identity.parent / "slots.parquet")
    with pytest.raises(ValueError, match="slot"):
        clean_dataset(identity, tmp_path / "clean", CleanupConfig())


@pytest.mark.parametrize("damage", ["time", "box", "confidence", "assigned", "negative_id"])
def test_invalid_reconciliation_input(damage: str) -> None:
    """Reject malformed tracking evidence before creating any identity assignments."""
    table = fragments([(10, 0, 2, 100, 0)])
    if damage == "time":
        table.t[0] = np.nan
    elif damage == "box":
        table.bbox[0, 3] = 0
    elif damage == "confidence":
        table.kp[0, 0, 2] = 2
    elif damage == "assigned":
        table.slot[0] = 0
    else:
        table.track_id[0] = -2
    with pytest.raises(ValueError):
        reconcile(table, 10, IdentityConfig())


def test_yaml_configuration_and_invalid_manifest_cli(tmp_path: Path) -> None:
    """YAML overrides control acceptance; failed upstream runs are refused."""
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "tracking.json"
    record = {"schema_version": 1, "status": "succeeded", "fps": 10, "n_frames": 9}
    manifest.write_text(json.dumps(record))
    write_keypoints(fragments([(10, 0, 2, 100, 0), (30, 6, 8, 100, 0)]), source / "tracks.parquet")
    settings = tmp_path / "identity.yaml"
    settings.write_text("max_gap_s: 0.1\n")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["reconcile", str(manifest), "--config", str(settings), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "out" / "identities.json").read_text())["slot_count"] == 2
    record["status"] = "failed"
    manifest.write_text(json.dumps(record))
    result = runner.invoke(app, ["reconcile", str(manifest), "--output", str(tmp_path / "bad")])
    assert result.exit_code == 1 and "Unable" in result.output
    assert not (tmp_path / "bad").exists()
