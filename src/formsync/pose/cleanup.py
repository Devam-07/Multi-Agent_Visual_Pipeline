"""Traceable offline cleanup of joint trajectories within individual raw tracks."""

import hashlib
import json
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_validator
from scipy.signal import savgol_filter

from formsync.io.parquet import read_keypoints
from formsync.schemas.base import FormsyncModel


class CleanupConfig(FormsyncModel):
    """Cleanup controls in analysis-frame units, applied independently to each joint.

    Confidence below the threshold masks both coordinates. Short internal gaps can
    be filled, then an odd Savitzky-Golay window smooths each continuous valid segment.
    Segments shorter than the window remain unchanged. This is an offline filter.
    """

    confidence: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    max_gap_frames: int = Field(default=5, ge=0)
    smooth_window: int = Field(default=7, ge=3)
    smooth_poly: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def check_filter(self) -> Self:
        """Require an odd window and a polynomial order smaller than that window."""
        if self.smooth_window % 2 == 0 or self.smooth_poly >= self.smooth_window:
            raise ValueError("smooth_window must be odd and smooth_poly must be smaller")
        return self


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open ranges of consecutive true samples, including boundary runs."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


def clean_joint(
    raw: np.ndarray,
    config: CleanupConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clean an (N,3) x/y/confidence series and return observed/interpolated/smoothed masks.

    Confidence is copied without modification, including NaN for absent detections.
    Both coordinates must be finite and confident to count as observed. Interpolation
    needs observations on both sides within the same series; boundaries are never
    extrapolated. Smoothing only touches finite runs after filling. Flags record
    operations applied, even when a filter leaves an exactly linear signal unchanged.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("Expected joint samples shaped (N,3)")
    if np.any(np.isinf(raw[:, 2])) or np.any((raw[:, 2] < 0) | (raw[:, 2] > 1)):
        raise ValueError("Joint confidence must be NaN or in [0,1]")
    result = raw.copy()
    observed = np.isfinite(raw).all(axis=1) & (raw[:, 2] >= config.confidence)
    filled = np.zeros(len(raw), dtype=bool)
    smoothed = np.zeros(len(raw), dtype=bool)
    result[~observed, :2] = np.nan
    for start, end in segments(~observed):
        if start > 0 and end < len(raw) and end - start <= config.max_gap_frames:
            fraction = np.arange(1, end - start + 1)[:, None] / (end - start + 1)
            result[start:end, :2] = raw[start - 1, :2] + fraction * (
                raw[end, :2] - raw[start - 1, :2]
            )
            filled[start:end] = True
    for start, end in segments(observed | filled):
        if end - start >= config.smooth_window:
            result[start:end, :2] = savgol_filter(
                result[start:end, :2],
                config.smooth_window,
                config.smooth_poly,
                axis=0,
                mode="interp",
            )
            smoothed[start:end] = True
    return result, observed, filled, smoothed


TRAJECTORY_SCHEMA = pa.schema(
    [
        ("track_id", pa.int32()),
        ("slot", pa.int16()),
        ("frame_idx", pa.int32()),
        ("t", pa.float64()),
        ("joint", pa.int8()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("confidence", pa.float32()),
        ("observed", pa.bool_()),
        ("interpolated", pa.bool_()),
        ("smoothed", pa.bool_()),
        ("missing", pa.bool_()),
    ]
)


def clean_dataset(source: Path, output: Path, config: CleanupConfig) -> Path:
    """Save separate long-format joint trajectories and quality metrics from tracked artifacts.

    Accepts successful poses.json, tracking.json or identities.json. Every track spans its
    first through last detection; absent intermediate frames are represented as NaN
    samples, but different IDs are never joined. Output has one row per track/frame/
    COCO joint. Raw confidence and processing masks support downstream quality gates.
    Unassigned detections are excluded and counted. Existing destinations are refused;
    cleanup.json is atomically published last, so partial writes are not successes.
    """
    config = CleanupConfig.model_validate(config.model_dump())
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if source.name not in {"poses.json", "tracking.json", "identities.json"}:
        raise ValueError("Expected poses.json, tracking.json or identities.json")
    record = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 1
        or record.get("status") != "succeeded"
    ):
        raise ValueError("Cleanup requires a successful version-1 source manifest")
    fps = record.get("analysis_fps" if source.name == "poses.json" else "fps")
    n_frames = record.get("n_frames")
    if (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not np.isfinite(fps)
        or fps <= 0
        or not isinstance(n_frames, int)
        or isinstance(n_frames, bool)
        or not 1 <= n_frames <= 90000
    ):
        raise ValueError("Invalid source timeline")
    tracks_path = source.parent / (
        "slots.parquet" if source.name == "identities.json" else "tracks.parquet"
    )
    with tracks_path.open("rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    table = read_keypoints(tracks_path)
    if (
        np.any(table.frame_idx < 0)
        or np.any(table.frame_idx >= n_frames)
        or np.any(table.track_id < -1)
        or not np.allclose(table.t, table.frame_idx / fps, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("Invalid tracked frame indices, IDs or timestamps")
    conf = table.kp[..., 2]
    if np.any(np.isinf(conf)) or np.any((conf < 0) | (conf > 1)):
        raise ValueError("Joint confidence must be NaN or in [0,1]")
    assigned = table.track_id >= 0
    pairs = np.column_stack((table.track_id[assigned], table.frame_idx[assigned]))
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate track/frame observations")
    if np.any(table.slot < -1) or np.any((table.track_id < 0) & (table.slot >= 0)):
        raise ValueError("Invalid slot assignment")
    if source.name == "identities.json" and np.any(table.slot[assigned] < 0):
        raise ValueError("Reconciled tracks require assigned slots")
    slotted = table.slot >= 0
    slot_pairs = np.column_stack((table.slot[slotted], table.frame_idx[slotted]))
    if len(np.unique(slot_pairs, axis=0)) != len(slot_pairs):
        raise ValueError("Duplicate slot/frame observations")
    parts = []
    reports: list[dict[str, object]] = []
    total_samples = 0
    for track_id in np.unique(table.track_id[assigned]):
        selected = table.track_id == track_id
        slots = np.unique(table.slot[selected])
        if len(slots) != 1 or slots[0] < -1:
            raise ValueError("Each raw track must have a consistent slot")
        first, last = int(table.frame_idx[selected].min()), int(table.frame_idx[selected].max())
        length = last - first + 1
        total_samples += length * 17
        if total_samples > 5000000:
            raise ValueError("Cleanup exceeds the 5000000 joint-sample in-memory limit")
        raw = np.full((length, 17, 3), np.nan)
        raw[table.frame_idx[selected] - first] = table.kp[selected]
        cleaned = np.empty_like(raw)
        observed = np.zeros((length, 17), dtype=bool)
        filled, smoothed = observed.copy(), observed.copy()
        gaps: list[int] = []
        for joint in range(17):
            cleaned[:, joint], observed[:, joint], filled[:, joint], smoothed[:, joint] = (
                clean_joint(raw[:, joint], config)
            )
            gaps.extend(end - start for start, end in segments(~observed[:, joint]))
        missing = ~(observed | filled)
        indices = np.repeat(np.arange(first, last + 1), 17)
        data = {
            "track_id": np.full(length * 17, track_id),
            "slot": np.full(length * 17, slots[0]),
            "frame_idx": indices,
            "t": indices / fps,
            "joint": np.tile(np.arange(17), length),
            "x": cleaned[..., 0].ravel(),
            "y": cleaned[..., 1].ravel(),
            "confidence": raw[..., 2].ravel(),
            "observed": observed.ravel(),
            "interpolated": filled.ravel(),
            "smoothed": smoothed.ravel(),
            "missing": missing.ravel(),
        }
        parts.append(pa.Table.from_pydict(data, schema=TRAJECTORY_SCHEMA))
        reports.append(
            {
                "track_id": int(track_id),
                "slot": int(slots[0]),
                "first_frame": first,
                "last_frame": last,
                "span_frames": length,
                "detected_frames": int(selected.sum()),
                "detection_coverage": float(selected.sum() / length),
                "joint_samples": length * 17,
                "observed_samples": int(observed.sum()),
                "observed_coverage": float(observed.mean()),
                "interpolated_samples": int(filled.sum()),
                "smoothed_samples": int(smoothed.sum()),
                "missing_samples": int(missing.sum()),
                "usable_coverage": float((~missing).mean()),
                "gap_lengths_frames": gaps,
                "longest_gap_frames": max(gaps, default=0),
            }
        )
    quality = {
        "schema_version": 1,
        "raw_track_count": len(reports),
        "input_detections": len(table),
        "unassigned_detections": int((~assigned).sum()),
        "tracks": reports,
    }
    result = (
        pa.concat_tables(parts) if parts else pa.Table.from_pylist([], schema=TRAJECTORY_SCHEMA)
    )
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(result, output / "trajectories.parquet", compression="zstd")
    (output / "quality.json").write_text(
        json.dumps(quality, indent=2, allow_nan=False), encoding="utf-8"
    )
    completed = {
        "schema_version": 1,
        "status": "succeeded",
        "source_manifest": str(source.resolve()),
        "tracks_sha256": fingerprint,
        "n_frames": n_frames,
        "fps": fps,
        "config": config.model_dump(),
        "coordinate_units": "pixels",
        "artifacts": ["trajectories.parquet", "quality.json"],
    }
    temporary = output / "cleanup.json.tmp"
    temporary.write_text(json.dumps(completed, indent=2, allow_nan=False), encoding="utf-8")
    path = output / "cleanup.json"
    temporary.replace(path)
    return path
