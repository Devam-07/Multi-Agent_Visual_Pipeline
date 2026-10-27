"""Acceptance fixture for the media foundation, without model inference or network calls."""

from pathlib import Path

import numpy as np
import pytest

from formsync.io.audio import extract_audio, load_audio
from formsync.io.video import iter_frames, probe_video
from formsync.testing.video import make_synthetic_video


def test_synthetic_clip_preserves_video_and_audio_timing(tmp_path: Path) -> None:
    """A two-second clip retains frame timestamps and half-second beats through real codecs.

    Moving squares exercise decoding only; they are not a pose-inference fixture. Check
    audio energy around known beats rather than requiring lossless AAC sample equality.
    """
    clip = make_synthetic_video(tmp_path / "practice.mp4", n_frames=60, fps=30, bpm=120)
    meta = probe_video(clip, target_fps=15)
    frames = list(iter_frames(clip, frame_stride=meta.frame_stride))
    assert meta.duration_s == pytest.approx(2.0)
    assert len(frames) == 30
    assert [frame.source_idx for frame in frames] == list(range(0, 60, 2))
    assert [frame.t for frame in frames] == pytest.approx(np.arange(30) / 15)

    wav = extract_audio(clip, tmp_path / "artifacts" / "audio.wav", sample_rate=22050)
    samples, rate = load_audio(wav)
    assert rate == 22050
    assert samples.ndim == 1
    assert abs(len(samples) / rate - meta.duration_s) < 0.1
    for beat in (0.5, 1.0, 1.5):
        on_beat = np.abs(samples[int(beat * rate) : int((beat + 0.03) * rate)]).mean()
        off_beat = np.abs(samples[int((beat - 0.25) * rate) : int((beat - 0.22) * rate)])
        assert on_beat > 0.01
        assert on_beat > 5 * off_beat.mean()
