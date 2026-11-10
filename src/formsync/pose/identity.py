"""Offline identity reconciliation with explicit uncertainty and unchanged observations."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from formsync.io.parquet import KeypointTable, read_keypoints, write_keypoints
from formsync.schemas.identity import IdentityConfig


@dataclass
class Fragment:
    """Time-ordered rows of one raw track, used solely to assess endpoint continuity."""

    track_id: int
    rows: np.ndarray


def assess_link(
    table: KeypointTable,
    before: Fragment,
    after: Fragment,
    config: IdentityConfig,
    fps: float,
) -> dict[str, Any]:
    """Report temporal, bidirectional motion, scale, and normalized pose evidence.

    Endpoint velocity is estimated from up to velocity_window observations on each
    side. Both forward and backward predictions must agree. Pose error is RMS
    Euclidean joint distance after centering/scaling by each bounding box. Only
    common finite confident joints count. Rejected links report the first failed gate.
    """
    a, b = before.rows, after.rows
    end, start = int(a[-1]), int(b[0])
    gap = float((int(table.frame_idx[start]) - int(table.frame_idx[end])) / fps)
    evidence: dict[str, Any] = {
        "from_track": before.track_id,
        "to_track": after.track_id,
        "gap_s": gap,
        "status": "rejected",
    }
    if gap <= 0 or gap > config.max_gap_s:
        evidence["reason"] = "time_gap"
        return evidence
    if min(len(a), len(b)) < config.min_fragment_frames:
        evidence["reason"] = "short_fragment"
        return evidence
    earlier = int(a[max(0, len(a) - config.velocity_window)])
    later = int(b[min(len(b) - 1, config.velocity_window - 1)])
    boxes = table.bbox[[end, start, earlier, later]].astype(float)
    center_a, center_b, center_earlier, center_later = (boxes[:, :2] + boxes[:, 2:]) / 2
    ha = float(table.bbox[end, 3] - table.bbox[end, 1])
    hb = float(table.bbox[start, 3] - table.bbox[start, 1])
    ratio = max(ha, hb) / min(ha, hb)
    evidence["size_ratio"] = ratio
    if ratio > config.max_size_ratio:
        evidence["reason"] = "body_scale"
        return evidence
    velocity_a = (center_a - center_earlier) / (
        (int(table.frame_idx[end]) - int(table.frame_idx[earlier])) / fps
    )
    velocity_b = (center_later - center_b) / (
        (int(table.frame_idx[later]) - int(table.frame_idx[start])) / fps
    )
    distance = float(
        max(
            np.linalg.norm(center_a + velocity_a * gap - center_b),
            np.linalg.norm(center_b - velocity_b * gap - center_a),
        )
        / ((ha + hb) / 2)
    )
    evidence["distance_body"] = distance
    if distance > config.max_distance_body:
        evidence["reason"] = "motion"
        return evidence
    ka, kb = table.kp[end], table.kp[start]
    common = (
        np.isfinite(ka).all(axis=1)
        & np.isfinite(kb).all(axis=1)
        & (ka[:, 2] >= config.confidence)
        & (kb[:, 2] >= config.confidence)
    )
    evidence["common_joints"] = int(common.sum())
    if common.sum() < config.min_common_joints:
        evidence["reason"] = "insufficient_pose"
        return evidence
    normalized_a = (ka[common, :2] - center_a) / ha
    normalized_b = (kb[common, :2] - center_b) / hb
    error = float(np.sqrt(np.mean(np.sum((normalized_a - normalized_b) ** 2, axis=1))))
    evidence["pose_error"] = error
    if error > config.max_pose_error:
        evidence["reason"] = "pose_difference"
        return evidence
    evidence.update(status="plausible", reason="all_gates_passed")
    return evidence


def reconcile(
    table: KeypointTable,
    fps: float,
    config: IdentityConfig,
) -> tuple[KeypointTable, dict[str, Any]]:
    """Add deterministic zero-based slots, joining only reciprocal unique plausible links.

    All row measurements and raw IDs are preserved. Temporal overlap prevents joins;
    chains are transitive, but ambiguity prevents a merge rather than forcing a best
    guess. Separate slots for unresolved fragments are provisional, not evidence of
    distinct people. Cannot split an identity switch already inside one raw track.
    """
    config = IdentityConfig.model_validate(config.model_dump())
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid fps")
    if (
        np.any(table.track_id < -1)
        or np.any(table.frame_idx < 0)
        or np.any(table.slot != -1)
        or not np.allclose(table.t, table.frame_idx / fps, rtol=1e-6, atol=1e-6)
    ):
        raise ValueError("Invalid raw IDs, preassigned slots, or timeline")
    assigned = table.track_id >= 0
    pairs = np.column_stack((table.track_id[assigned], table.frame_idx[assigned]))
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate track/frame observation")
    boxes = table.bbox[assigned]
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Assigned tracks require finite positive-area bounding boxes")
    confidence = table.kp[..., 2]
    if np.any(np.isinf(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Joint confidence must be NaN or in [0,1]")
    fragments = []
    for track_id in np.unique(table.track_id[assigned]):
        rows = np.flatnonzero(table.track_id == track_id)
        rows = rows[np.argsort(table.frame_idx[rows], kind="stable")]
        fragments.append(Fragment(int(track_id), rows))
    if len(fragments) > config.max_fragments:
        raise ValueError("Too many track fragments for in-memory reconciliation")
    fragments.sort(key=lambda f: (int(table.frame_idx[f.rows[0]]), f.track_id))
    evidence = []
    for before in fragments:
        for after in fragments:
            gap = (
                int(table.frame_idx[after.rows[0]]) - int(table.frame_idx[before.rows[-1]])
            ) / fps
            if 0 < gap <= config.max_gap_s:
                evidence.append(assess_link(table, before, after, config, fps))
    plausible = [e for e in evidence if e["status"] == "plausible"]
    outgoing = Counter(e["from_track"] for e in plausible)
    incoming = Counter(e["to_track"] for e in plausible)
    predecessors: dict[int, int] = {}
    accepted = []
    ambiguous = []
    for item in plausible:
        pair = {"from_track": item["from_track"], "to_track": item["to_track"]}
        if outgoing[item["from_track"]] == incoming[item["to_track"]] == 1:
            item["status"] = "accepted"
            predecessors[item["to_track"]] = item["from_track"]
            accepted.append(pair)
        else:
            item.update(status="ambiguous", reason="competing_plausible_links")
            ambiguous.append(pair)
    mapping: dict[int, int] = {}
    next_slot = 0
    for fragment in fragments:
        if fragment.track_id in predecessors:
            mapping[fragment.track_id] = mapping[predecessors[fragment.track_id]]
        else:
            mapping[fragment.track_id] = next_slot
            next_slot += 1
    result = table.select(np.arange(len(table)))
    for track_id, slot in mapping.items():
        result.slot[result.track_id == track_id] = slot
    slot_pairs = np.column_stack((result.slot[assigned], result.frame_idx[assigned]))
    if len(np.unique(slot_pairs, axis=0)) != len(slot_pairs):
        raise ValueError("Reconciliation would assign simultaneous detections to one slot")
    report = {
        "slot_count": next_slot,
        "raw_track_count": len(fragments),
        "unassigned_detections": int((~assigned).sum()),
        "accepted_links": accepted,
        "ambiguous_links": ambiguous,
        "candidate_links": evidence,
        "track_to_slot": {str(k): v for k, v in mapping.items()},
    }
    return result, report


def reconcile_dataset(source: Path, output: Path, config: IdentityConfig) -> Path:
    """Reconcile successful pose/tracking results, publishing identities.json last.

    Saves slots.parquet with unchanged row order and full source-file SHA-256 plus
    configuration and link evidence. Does not read ground truth or modify source data.
    Existing destinations are refused. A failed write can leave a partial directory
    without a completion manifest. Assigned slots are provisional estimates.
    """
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if source.name not in {"poses.json", "tracking.json"}:
        raise ValueError("Expected poses.json or tracking.json")
    record = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 1
        or record.get("status") != "succeeded"
    ):
        raise ValueError("Expected a successful version-1 source manifest")
    fps = record.get("analysis_fps" if source.name == "poses.json" else "fps")
    frames = record.get("n_frames")
    if (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not isinstance(frames, int)
        or isinstance(frames, bool)
        or not 1 <= frames <= 90000
    ):
        raise ValueError("Invalid source timeline")
    input_path = source.parent / "tracks.parquet"
    with input_path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    table = read_keypoints(input_path)
    if np.any(table.frame_idx >= frames):
        raise ValueError("Tracked frame outside source timeline")
    result, report = reconcile(table, fps, config)
    report.update(
        schema_version=1,
        status="succeeded",
        source_manifest=str(source.resolve()),
        source_sha256=digest,
        n_frames=frames,
        fps=fps,
        config=config.model_dump(),
        artifacts=["slots.parquet"],
        algorithm="reciprocal-unique-gates-v1",
    )
    output.mkdir(parents=True, exist_ok=False)
    write_keypoints(result, output / "slots.parquet")
    temporary = output / "identities.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    path = output / "identities.json"
    temporary.replace(path)
    return path
