"""Offline tracking evaluation; labels are never passed into the tracker."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from formsync.io.parquet import KeypointTable, read_keypoints


def evaluate_tracking(
    table: KeypointTable, row_dancer_ids: list[int]
) -> dict[str, int | float | None]:
    """Measure row-aligned assignments against synthetic identities, ignoring ID numbering.

    Identity switches count changes between successive assigned observations of a
    dancer, including across gaps. Extra tracks sum distinct IDs beyond the first for
    each dancer (a fragmentation proxy). Mixed tracks serve more than one dancer.
    Accuracy is the best global one-to-one dancer/track assignment divided by all
    detections, so unassigned rows count against it. Empty-data rates are undefined.
    These are fixture diagnostics, not benchmark MOTA/HOTA/IDF1 metrics.
    """
    labels = np.asarray(row_dancer_ids)
    if len(labels) != len(table) or (
        len(labels) and (not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0))
    ):
        raise ValueError("Ground-truth IDs must be nonnegative integers aligned to every row")
    if len(labels) and len(np.unique(np.column_stack((table.frame_idx, labels)), axis=0)) != len(
        labels
    ):
        raise ValueError("A dancer cannot have two detections in the same frame")
    dancers = np.unique(labels)
    ids = np.unique(table.track_id[table.track_id >= 0])
    counts = np.zeros((len(dancers), len(ids)), dtype=int)
    switches = extra = 0
    for i, dancer in enumerate(dancers):
        rows = np.flatnonzero(labels == dancer)
        rows = rows[np.argsort(table.frame_idx[rows], kind="stable")]
        assigned = table.track_id[rows]
        assigned = assigned[assigned >= 0]
        switches += int(np.count_nonzero(np.diff(assigned)))
        extra += max(0, len(np.unique(assigned)) - 1)
        for j, track in enumerate(ids):
            counts[i, j] = np.count_nonzero(assigned == track)
    matches = linear_sum_assignment(-counts)
    correct = int(counts[matches].sum())
    return {
        "detections": len(table),
        "assigned_detections": int(counts.sum()),
        "identity_switches": switches,
        "extra_tracks_per_dancer": extra,
        "mixed_tracks": int(np.count_nonzero((counts > 0).sum(axis=0) > 1)),
        "assigned_fraction": float(counts.sum() / len(table)) if len(table) else None,
        "assignment_accuracy": correct / len(table) if len(table) else None,
    }


def evaluate_dataset(manifest: Path, truth: Path) -> dict[str, int | float | None]:
    """Evaluate saved tracks against labels paired with the original detections artifact.

    Verifies the sibling detections SHA-256 against tracking provenance and requires
    unchanged row order and measurements in tracks.parquet. This catches accidental
    dataset mismatches; evaluation labels themselves must be trusted and unmodified.
    Returns metrics without altering either dataset or loading the tracker.
    """
    record = json.loads(manifest.read_text(encoding="utf-8"))
    labels = json.loads(truth.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(labels, dict):
        raise ValueError("Expected tracking and ground-truth JSON objects")
    if record.get("schema_version") != 1 or record.get("status") != "succeeded":
        raise ValueError("Expected a successful version 1 tracking manifest")
    if "row_dancer_ids" not in labels or not isinstance(labels["row_dancer_ids"], list):
        raise ValueError("Ground truth must contain row_dancer_ids")
    with (truth.parent / "detections.parquet").open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != record.get("source_sha256"):
        raise ValueError("Ground-truth dataset does not match the tracked source")
    source = read_keypoints(truth.parent / "detections.parquet")
    tracked = read_keypoints(manifest.parent / "tracks.parquet")
    for field in ("frame_idx", "t", "kp", "bbox", "det_conf", "slot"):
        if not np.array_equal(getattr(source, field), getattr(tracked, field)):
            raise ValueError("Tracked rows do not match the source detections")
    return evaluate_tracking(tracked, labels["row_dancer_ids"])
