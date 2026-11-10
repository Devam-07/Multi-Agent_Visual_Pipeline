"""CPU pose inference, replayable artifacts, tracking, and silent inspection video."""

import json
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast

import numpy as np

from formsync.config import PipelineConfig, PoseConfig
from formsync.io.parquet import KeypointTable, write_keypoints
from formsync.io.video import Frame, hash_video_file, iter_frames, probe_video, write_video
from formsync.pose.replay import PoseFrame
from formsync.pose.tracking import track_frames
from formsync.schemas.ingestion import IngestionManifest
from formsync.schemas.replay import PoseStreamManifest
from formsync.schemas.tracking import TrackerConfig
from formsync.viz.overlay import draw_poses

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


def pose_from_arrays(
    frame: Frame, boxes: np.ndarray, scores: np.ndarray, kp: np.ndarray
) -> PoseFrame:
    """Convert full-image pixel predictions to untracked COCO-17 rows without normalization.

    The caller supplies coordinates already restored by YOLO to the source image.
    Confidence is retained even for uncertain joints; rendering filters it later.
    Empty predictions remain an explicit timeline frame. Invalid model shapes or
    non-finite values fail rather than becoming misleading stored detections.
    """
    n = len(scores)
    if boxes.shape != (n, 4) or scores.shape != (n,) or kp.shape != (n, 17, 3):
        raise ValueError("Expected boxes (N,4), scores (N,), and COCO keypoints (N,17,3)")
    if not all(np.isfinite(a).all() for a in (boxes, scores, kp)):
        raise ValueError("Model predictions must be finite")
    if any(np.any((a < 0) | (a > 1)) for a in (scores, kp[..., 2])):
        raise ValueError("Model confidence must be in [0,1]")
    if np.any(boxes[:, :2] > boxes[:, 2:]):
        raise ValueError("Model boxes have inverted corners")
    table = KeypointTable(
        np.full(n, frame.frame_idx),
        np.full(n, frame.t),
        np.full(n, -1),
        np.full(n, -1),
        boxes,
        scores,
        kp,
    )
    return PoseFrame(frame.frame_idx, frame.t, table)


def predict_frames(source: Path, config: PoseConfig, stride: int) -> Iterator[PoseFrame]:
    """Load one pretrained pose model lazily and infer sampled BGR frames on CPU.

    Official model names resolve through Ultralytics and may download weights on first
    use. An explicit local .pt path can be used offline. No resizing or normalization
    is applied to returned original-image coordinates. No weights are trained here.
    """
    from ultralytics.models.yolo.model import YOLO

    name = config.model_name
    if not Path(name).suffix:
        name += ".pt"
    model = YOLO(name)
    if model.task != "pose":
        raise ValueError("Selected weights are not a pose model")
    for frame in iter_frames(source, frame_stride=stride):
        predictions = cast(
            "list[Results]",
            model.predict(
                frame.image,
                device="cpu",
                imgsz=config.imgsz,
                conf=config.conf,
                iou=config.iou,
                max_det=config.max_det,
                classes=[0],
                verbose=False,
            ),
        )
        result = cast("Results", predictions[0].cpu().numpy())  # type: ignore[no-untyped-call]
        if result.boxes is None or result.keypoints is None:
            raise ValueError("Pose model did not return boxes and keypoints")
        yield pose_from_arrays(
            frame,
            np.asarray(result.boxes.xyxy),
            np.asarray(result.boxes.conf),
            np.asarray(result.keypoints.data),
        )


def run_poses(manifest: Path, output: Path, config: PipelineConfig, tracker: TrackerConfig) -> Path:
    """Process a successful ingestion into detections, tracks, and a silent overlay.

    Uses constant-frame-rate metadata and integer-stride sampling. The overlay runs
    at the effective analysis fps, so duration differs by at most one sampled interval.
    Frames are decoded again for rendering; images are not accumulated in memory.
    Detection arrays are held in memory as in replay. Existing directories are refused.
    ``poses.json`` records running/succeeded/failed; ``stream.json`` is published only
    after all outputs are verified. A failed run may retain incomplete artifacts.
    """
    config = PipelineConfig.model_validate(config.model_dump())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "poses.json"
    started = perf_counter()
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "source_manifest": str(manifest.resolve()),
        "model": config.pose.model_name,
        "backend_version": version("ultralytics"),
        "device": "cpu",
        "config": config.model_dump(mode="json"),
        "tracker": tracker.model_dump(),
        "overlay_audio": False,
    }
    save_pose_record(path, record)
    try:
        ingestion = IngestionManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
        if ingestion.status != "succeeded" or ingestion.video is None:
            raise ValueError("A successful video ingestion is required")
        source = ingestion.source_path
        if hash_video_file(source) != ingestion.source_fingerprint:
            raise ValueError("Source video changed since ingestion; ingest it again")
        meta = probe_video(source, target_fps=config.pose.target_fps)
        expected = (meta.n_frames + meta.frame_stride - 1) // meta.frame_stride
        timeline = PoseStreamManifest(
            n_frames=expected, fps=meta.analysis_fps, width=meta.width, height=meta.height
        )
        frames = list(predict_frames(source, config.pose, meta.frame_stride))
        if len(frames) != expected:
            raise ValueError(
                f"Video decoding ended early: expected {expected} frames, got {len(frames)}"
            )
        detections = KeypointTable.concat([frame.detections for frame in frames])
        tracked = list(track_frames(iter(frames), tracker))
        tracks = KeypointTable.concat([frame.detections for frame in tracked])
        write_keypoints(detections, output / "detections.parquet")
        write_keypoints(tracks, output / "tracks.parquet")

        def images() -> Iterator[np.ndarray]:
            """Pair each decoded source sample with its tracked pose frame for rendering."""
            for raw, poses in zip(
                iter_frames(source, frame_stride=meta.frame_stride), tracked, strict=True
            ):
                if raw.frame_idx != poses.frame_idx or abs(raw.t - poses.t) > 1e-6:
                    raise ValueError("Overlay and pose timestamps disagree")
                yield draw_poses(raw.image, poses.detections, config.pose.keypoint_conf_min)

        write_video(output / "overlay.mp4", images(), meta.analysis_fps)
        rendered = probe_video(output / "overlay.mp4")
        if rendered.n_frames != expected or abs(rendered.fps - meta.analysis_fps) > 0.01:
            raise ValueError("Overlay output failed frame count or playback-rate verification")
        if hash_video_file(source) != ingestion.source_fingerprint:
            raise ValueError("Source video changed during pose analysis")
        record.update(
            status="succeeded",
            source_video=str(source),
            source_fingerprint=ingestion.source_fingerprint,
            source_fps=meta.fps,
            source_frame_stride=meta.frame_stride,
            n_frames=expected,
            analysis_fps=meta.analysis_fps,
            detections=len(detections),
            assigned_detections=int((tracks.track_id >= 0).sum()),
            artifacts=["detections.parquet", "tracks.parquet", "overlay.mp4", "stream.json"],
            runtime_s=perf_counter() - started,
        )
        temporary = output / "stream.json.tmp"
        temporary.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(output / "stream.json")
        save_pose_record(path, record)
    except Exception as exc:
        (output / "stream.json").unlink(missing_ok=True)
        record.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            runtime_s=perf_counter() - started,
            artifacts=[],
        )
        save_pose_record(path, record)
        raise
    return path


def save_pose_record(path: Path, record: dict[str, object]) -> None:
    """Atomically publish JSON status; filesystem errors propagate to the caller."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
