"""Lazy ByteTrack integration preserving detection rows and original joint estimates."""

import hashlib
import json
from collections.abc import Iterator
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from formsync.io.parquet import KeypointTable, write_keypoints
from formsync.pose.replay import PoseFrame, replay_frames
from formsync.schemas.replay import PoseStreamManifest
from formsync.schemas.tracking import TrackerConfig


def track_frames(frames: Iterator[PoseFrame], config: TrackerConfig) -> Iterator[PoseFrame]:
    """Assign raw IDs using real ByteTrack, yielding all input frames and detection rows.

    Heavy dependencies are imported only when iteration begins. Inputs must start at
    frame zero and include every analysis frame, even if empty. Missing/unconfirmed
    associations stay -1; joints are never invented or replaced by Kalman predictions.
    This adapter is tested against the Ultralytics API locked in uv.lock.
    """
    if find_spec("lap") is None:
        raise RuntimeError("ByteTrack requires lap; install project dependencies with uv sync")
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker

    tracker = BYTETracker(SimpleNamespace(**config.model_dump()))  # type: ignore[no-untyped-call]
    for expected, frame in enumerate(frames):
        if frame.frame_idx != expected or not np.all(frame.detections.frame_idx == expected):
            raise ValueError("Tracking requires consecutive frames starting at zero")
        source = frame.detections
        if np.any(source.track_id != -1) or np.any(source.slot != -1):
            raise ValueError("Tracking requires unassigned input detections")
        if not np.allclose(source.t, frame.t, atol=1e-6, rtol=1e-6):
            raise ValueError("Detection timestamps must match their frame")
        rows = source.select(np.arange(len(source)))
        data = np.column_stack((source.bbox, source.det_conf, np.zeros(len(source))))
        results = np.asarray(tracker.update(Boxes(data.astype(np.float32), orig_shape=(1, 1))))
        if results.size:
            indices = results[:, -1].astype(int)
            ids = results[:, 4].astype(int)
            if (
                np.any(indices < 0)
                or np.any(indices >= len(source))
                or len(np.unique(indices)) != len(indices)
                or len(np.unique(ids)) != len(ids)
                or np.any(ids <= 0)
            ):
                raise RuntimeError("ByteTrack returned invalid detection associations")
            rows.track_id[indices] = ids
        yield PoseFrame(frame.frame_idx, frame.t, rows)


def track_dataset(source: Path, output: Path, config: TrackerConfig) -> Path:
    """Track a replay dataset into a new directory, publishing its manifest last.

    Preserves every input row so evaluation labels remain aligned. Stores the exact
    input Parquet SHA-256, settings, backend version, and counts for provenance.
    Never reads ground truth. Failed writes may leave a partial directory without
    ``tracking.json``. Refuses existing destinations and propagates failures.
    """
    config = TrackerConfig.model_validate(config.model_dump())
    source = source.resolve()
    timeline = PoseStreamManifest.model_validate_json(source.read_text(encoding="utf-8"))
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    with (source.parent / timeline.detections).open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    table = KeypointTable.concat(
        [frame.detections for frame in track_frames(replay_frames(source), config)]
    )
    output.mkdir(parents=True, exist_ok=False)
    write_keypoints(table, output / "tracks.parquet")
    assigned = table.track_id[table.track_id >= 0]
    record = {
        "schema_version": 1,
        "status": "succeeded",
        "backend": "ultralytics-bytetrack",
        "backend_version": version("ultralytics"),
        "source_manifest": str(source),
        "source_sha256": digest,
        "n_frames": timeline.n_frames,
        "fps": timeline.fps,
        "config": config.model_dump(),
        "tracks": "tracks.parquet",
        "detections": len(table),
        "assigned_detections": len(assigned),
        "track_count": len(np.unique(assigned)),
    }
    temporary = output / "tracking.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    path = output / "tracking.json"
    temporary.replace(path)
    return path
