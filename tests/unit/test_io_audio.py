"""Tests for formsync.io.audio."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import numpy as np
import pytest

from formsync.io.audio import (
    AudioExtractionError,
    NoAudioStreamError,
    extract_audio,
    ffmpeg_path,
    load_audio,
    write_audio,
)
from formsync.testing.video import click_track, make_synthetic_video


def test_ffmpeg_path_exists() -> None:
    """The bundled ffmpeg binary is present."""
    assert Path(ffmpeg_path()).exists()


def test_write_and_load_audio_round_trip(tmp_path: Path) -> None:
    """A mono float signal survives a 16-bit WAV round trip within quantisation error."""
    sr = 8000
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    p = write_audio(tmp_path / "a" / "tone.wav", x, sr)
    y, sr_back = load_audio(p)
    assert sr_back == sr and y.shape == x.shape and y.dtype == np.float32
    np.testing.assert_allclose(y, x, atol=1e-3)


def test_extract_audio_from_synthetic_video(tmp_path: Path) -> None:
    """Audio muxed into a clip comes back out at the requested rate with clicks intact."""
    clip = make_synthetic_video(tmp_path / "clip.mp4", n_frames=60, fps=30, bpm=120)
    wav = extract_audio(clip, tmp_path / "out" / "clip.wav", sample_rate=22050)
    y, sr = load_audio(wav)
    assert sr == 22050
    assert abs(len(y) - 2 * sr) < sr * 0.1  # ~2 s long

    # Clicks are 0.5 s apart: energy near t=0.5 s should dwarf energy at t=0.25 s.
    def energy(t0: float) -> float:
        """Mean absolute amplitude over 30 ms starting at t0 seconds."""
        i = int(t0 * sr)
        return float(np.abs(y[i : i + int(0.03 * sr)]).mean())

    assert energy(0.5) > 5 * energy(0.25)


def test_extract_audio_without_audio_stream_raises(tmp_path: Path) -> None:
    """A silent clip (no audio stream) raises AudioExtractionError; missing file raises."""
    clip = make_synthetic_video(tmp_path / "silent.mp4", n_frames=10, fps=30)
    with pytest.raises(NoAudioStreamError):
        extract_audio(clip, tmp_path / "silent.wav")
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "missing.mp4", tmp_path / "x.wav")


def test_click_track_has_clicks_on_beats() -> None:
    """click_track places bursts every 60/bpm seconds and silence between."""
    sr = 8000
    y = click_track(2.0, bpm=120, sample_rate=sr)
    assert y.shape == (2 * sr,)
    assert np.abs(y[: int(0.03 * sr)]).max() > 0.5
    assert np.abs(y[int(0.1 * sr) : int(0.4 * sr)]).max() == 0.0


@pytest.mark.parametrize("detail", ["decoder failed", "Permission denied", ""])
def test_audio_errors_do_not_imply_absent_stream(tmp_path: Path, detail: str) -> None:
    """Only an explicit unmatched stream map identifies a silent input."""
    clip = tmp_path / "source.mp4"
    clip.write_bytes(b"placeholder")
    result = CompletedProcess([], 1, stdout="", stderr=detail)
    with (
        patch("formsync.io.audio.subprocess.run", return_value=result),
        pytest.raises(AudioExtractionError) as raised,
    ):
        extract_audio(clip, tmp_path / "audio.wav")
    assert not isinstance(raised.value, NoAudioStreamError)
