"""Frame-by-frame pose input without model loading or access to evaluation labels."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from formsync.io.parquet import KeypointTable, read_keypoints
from formsync.schemas.replay import PoseStreamManifest


@dataclass(frozen=True)
class PoseFrame:
    """One analysis frame passed to downstream tracking.

    ``frame_idx`` is zero-based and ``t`` is seconds. ``detections`` may be empty;
    each present row contains COCO-17 pixel coordinates and confidence values, with
    track and slot IDs unassigned. Future inference can yield this same frame type.
    """

    frame_idx: int
    t: float
    detections: KeypointTable


def replay_frames(manifest_path: Path) -> Iterator[PoseFrame]:
    """Yield every timeline frame from a validated, untracked pose dataset.

    Loads the Parquet table once and slices it in frame order, preserving detection
    order within each frame. Includes leading, middle, and trailing empty frames.
    Reads only the stream manifest and detections, never ground truth. Raises
    ``OSError`` or ``ValueError`` for missing artifacts or invalid values. This initial
    implementation is for bounded fixtures, not out-of-core production replay.
    """
    manifest = PoseStreamManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    table = read_keypoints(manifest_path.parent / manifest.detections)
    frames = table.frame_idx
    if np.any(frames < 0) or np.any(frames >= manifest.n_frames) or np.any(np.diff(frames) < 0):
        raise ValueError("Detection frames must be ordered and within the manifest timeline")
    if not np.allclose(table.t, frames / manifest.fps, rtol=1e-6, atol=1e-6):
        raise ValueError("Detection timestamps disagree with the manifest timeline")
    if np.any(table.slot != -1) or np.any(table.track_id != -1):
        raise ValueError("Untracked replay must not contain assigned dancer or track IDs")
    if not all(np.isfinite(a).all() for a in (table.kp, table.bbox, table.det_conf)):
        raise ValueError("Detection coordinates and confidences must be finite")
    if any(np.any((a < 0) | (a > 1)) for a in (table.kp[..., 2], table.det_conf)):
        raise ValueError("Confidences must be in [0, 1]")
    if np.any(table.bbox[:, :2] > table.bbox[:, 2:]):
        raise ValueError("Bounding boxes must have ordered corners")
    boundaries = np.searchsorted(frames, np.arange(manifest.n_frames + 1))
    for index in range(manifest.n_frames):
        rows = np.arange(boundaries[index], boundaries[index + 1])
        yield PoseFrame(index, index / manifest.fps, table.select(rows))
