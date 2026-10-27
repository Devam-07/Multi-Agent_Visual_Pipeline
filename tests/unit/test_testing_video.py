"""Tests for formsync.testing.video (the synthetic clip generator itself)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from formsync.io.video import probe_video
from formsync.testing.video import make_synthetic_video, square_frames


def test_square_frames_positions_are_exact() -> None:
    """The red square sits at x = speed*i and the blue square at y = speed*i."""
    frames = list(square_frames(5, width=100, height=80, square=10, speed_px=3))
    assert len(frames) == 5
    f = frames[4]
    assert f.shape == (80, 100, 3)
    # Red covers x=12..21, y=10..19; blue covers x=10..19, y=12..21. Sample outside the overlap.
    assert tuple(f[10, 21]) == (0, 0, 255)  # red only
    assert tuple(f[21, 10]) == (255, 0, 0)  # blue only
    assert not np.any(f[60:, 60:])  # empty corner


def test_make_synthetic_video_with_and_without_audio(tmp_path: Path) -> None:
    """Both variants produce a readable clip with the requested frame count and size."""
    silent = make_synthetic_video(tmp_path / "s.mp4", n_frames=20, fps=25, width=96, height=64)
    m = probe_video(silent)
    assert (m.n_frames, m.width, m.height, m.fps) == (20, 96, 64, 25.0)
    with_audio = make_synthetic_video(tmp_path / "a.mp4", n_frames=20, fps=25, bpm=100)
    assert probe_video(with_audio).n_frames == 20
    assert with_audio.stat().st_size > silent.stat().st_size  # audio stream adds bytes
    assert not (tmp_path / "a_silent.mp4").exists()  # temp files cleaned up
