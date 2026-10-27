"""Video decoding: probe metadata, stream frames at a stride, random-access single frames.

Frames are yielded one at a time (never all loaded at once), so memory stays at one frame
regardless of clip length. Pixel coordinates follow OpenCV: origin top-left, ``x`` right,
``y`` down, BGR channel order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from formsync.schemas.tracks import VideoMeta


class Frame(NamedTuple):
    """One decoded frame.

    Attributes:
        frame_idx: Index among analysed frames (0-based). This is the index stored in keypoint
            tables and used as the time axis of dense arrays.
        source_idx: Index of the frame in the source file (``frame_idx * frame_stride``).
        t: Timestamp in seconds (``source_idx / fps``).
        image: ``(H, W, 3)`` uint8 BGR array. If an ROI was requested this is the crop.
    """

    frame_idx: int
    source_idx: int
    t: float
    image: np.ndarray


def _open(path: Path | str) -> cv2.VideoCapture:
    """Open a video with OpenCV, raising a clear error if it cannot be read."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {p}")
    return cap


def probe_video(
    path: Path | str,
    target_fps: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> VideoMeta:
    """Read a video's metadata and decide the frame stride for analysis.

    Args:
        path: Video file.
        target_fps: Desired analysis frame rate. The stride is ``round(fps / target_fps)``,
            at least 1. ``None`` analyses every frame.
        roi: Optional crop ``(x1, y1, x2, y2)`` recorded in the metadata.

    Returns:
        A :class:`~formsync.schemas.tracks.VideoMeta`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV cannot open the file or reports a zero frame rate.
    """
    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0:
        raise ValueError(f"video reports non-positive fps ({fps}): {path}")
    stride = max(1, round(fps / target_fps)) if target_fps else 1
    return VideoMeta(
        path=str(path),
        fps=fps,
        width=width,
        height=height,
        n_frames=n_frames,
        duration_s=n_frames / fps,
        analysis_fps=fps / stride,
        frame_stride=stride,
        roi=roi,
    )


def iter_frames(
    path: Path | str,
    frame_stride: int = 1,
    roi: tuple[int, int, int, int] | None = None,
    max_frames: int | None = None,
) -> Iterator[Frame]:
    """Yield every ``frame_stride``-th frame of a video, optionally cropped.

    Skipped frames are still decoded (video codecs store most frames as differences from the
    previous one) but not converted to arrays, which is the cheap part. The expensive work, the
    pose model, only sees the yielded frames.

    Args:
        path: Video file.
        frame_stride: Keep one frame in every ``frame_stride`` (1 = every frame).
        roi: Optional crop ``(x1, y1, x2, y2)`` in pixels applied to each yielded image.
        max_frames: Stop after yielding this many frames (useful for tests and benchmarks).

    Yields:
        :class:`Frame` objects in time order.

    Raises:
        ValueError: If ``frame_stride`` < 1.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
        source_idx = 0
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                return
            if source_idx % frame_stride:
                if not cap.grab():
                    return
                source_idx += 1
                continue
            ok, image = cap.read()
            if not ok:
                return
            if roi is not None:
                x1, y1, x2, y2 = roi
                image = image[y1:y2, x1:x2]
            yield Frame(frame_idx=frame_idx, source_idx=source_idx, t=source_idx / fps, image=image)
            frame_idx += 1
            source_idx += 1
    finally:
        cap.release()


def read_frame(path: Path | str, source_idx: int) -> np.ndarray:
    """Decode a single frame by source index (random access, e.g. for calibration UIs).

    Args:
        path: Video file.
        source_idx: Frame index in the source file.

    Returns:
        ``(H, W, 3)`` uint8 BGR array.

    Raises:
        IndexError: If the frame cannot be read (index past the end).
    """
    cap = _open(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_idx)
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok:
        raise IndexError(f"frame {source_idx} not readable from {path}")
    return image


def hash_video_file(path: Path | str, chunk_bytes: int = 1 << 20) -> str:
    """Return a fast content fingerprint of a video file.

    Hashes the file size plus the first and last ``chunk_bytes`` bytes, which is enough to tell
    videos apart without reading gigabytes. Used to key cached calibrations.

    Args:
        path: Video file.
        chunk_bytes: Bytes to read from each end.

    Returns:
        Hex SHA-256 digest string.
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with p.open("rb") as fh:
        h.update(fh.read(chunk_bytes))
        if size > chunk_bytes:
            fh.seek(max(chunk_bytes, size - chunk_bytes))
            h.update(fh.read(chunk_bytes))
    return h.hexdigest()


def write_video(
    path: Path | str, frames: Iterator[np.ndarray] | list[np.ndarray], fps: float
) -> Path:
    """Write BGR frames to an MP4 file (used for overlays and synthetic test clips).

    Args:
        path: Destination ``.mp4`` path; parent directories are created.
        frames: Iterable of ``(H, W, 3)`` uint8 arrays, all the same size.
        fps: Frame rate to record.

    Returns:
        The path written.

    Raises:
        ValueError: If no frames are given or OpenCV cannot open the writer.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    n = 0
    try:
        for image in frames:
            if writer is None:
                h, w = image.shape[:2]
                # OpenCV's stubs omit VideoWriter_fourcc; the runtime attribute exists.
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise ValueError(f"OpenCV could not open a video writer for {out}")
            writer.write(image)
            n += 1
    finally:
        if writer is not None:
            writer.release()
    if n == 0:
        raise ValueError("no frames to write")
    return out
