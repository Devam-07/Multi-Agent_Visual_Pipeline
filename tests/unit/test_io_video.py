"""Tests for formsync.io.video, using synthetic clips from formsync.testing.video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from formsync.io.video import hash_video_file, iter_frames, probe_video, read_frame, write_video
from formsync.testing.video import make_synthetic_video, square_frames


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 30-frame, 30 fps, 160x120 synthetic clip shared by the tests in this module."""
    return make_synthetic_video(tmp_path_factory.mktemp("vid") / "clip.mp4", n_frames=30, fps=30)


def test_probe_video_reports_metadata_and_stride(clip: Path) -> None:
    """probe_video reads fps/size/count and derives the stride from target_fps."""
    meta = probe_video(clip, target_fps=15)
    assert (meta.fps, meta.width, meta.height, meta.n_frames) == (30.0, 160, 120, 30)
    assert meta.frame_stride == 2 and meta.analysis_fps == 15.0
    assert meta.duration_s == pytest.approx(1.0)
    assert probe_video(clip).frame_stride == 1
    with pytest.raises(FileNotFoundError):
        probe_video(clip.with_name("nope.mp4"))


def test_iter_frames_stride_and_indices(clip: Path) -> None:
    """Striding yields every k-th source frame with consistent indices and timestamps."""
    frames = list(iter_frames(clip, frame_stride=3))
    assert len(frames) == 10
    assert [f.frame_idx for f in frames] == list(range(10))
    assert [f.source_idx for f in frames] == list(range(0, 30, 3))
    assert frames[4].t == pytest.approx(12 / 30)
    assert frames[0].image.shape == (120, 160, 3)
    with pytest.raises(ValueError):
        next(iter_frames(clip, frame_stride=0))


def test_iter_frames_roi_and_max_frames(clip: Path) -> None:
    """ROI crops the yielded image; max_frames stops early."""
    frames = list(iter_frames(clip, roi=(10, 20, 60, 50), max_frames=4))
    assert len(frames) == 4
    assert frames[0].image.shape == (30, 50, 3)


def test_read_frame_matches_iter_frames(clip: Path) -> None:
    """Random access returns the same pixels as streaming, and errors past the end."""
    streamed = {f.source_idx: f.image for f in iter_frames(clip)}
    for idx in (0, 7, 29):
        # Lossy codec: compare the red square's centre position rather than exact bytes.
        a, b = streamed[idx], read_frame(clip, idx)
        assert np.abs(a.astype(int) - b.astype(int)).mean() < 2.0
    with pytest.raises(IndexError):
        read_frame(clip, 500)


def test_write_video_round_trip(tmp_path: Path) -> None:
    """write_video produces a readable clip with the given frame count and size."""
    p = write_video(tmp_path / "out.mp4", square_frames(12, 64, 48), fps=24)
    meta = probe_video(p)
    assert (meta.n_frames, meta.width, meta.height, meta.fps) == (12, 64, 48, 24.0)
    with pytest.raises(ValueError):
        write_video(tmp_path / "empty.mp4", [], fps=24)


def test_hash_video_file_is_stable_and_content_sensitive(clip: Path, tmp_path: Path) -> None:
    """Same file hashes identically; a different clip hashes differently."""
    assert hash_video_file(clip) == hash_video_file(clip)
    other = make_synthetic_video(tmp_path / "other.mp4", n_frames=12, fps=30)
    assert hash_video_file(clip) != hash_video_file(other)
