"""Synthetic video clips for tests: moving coloured squares, optionally with a click-track.

These clips exercise the io layer end to end (decode, stride, crop, audio extraction) without
shipping real footage. The click track has a known tempo so beat-tracking tests can check the
recovered bpm exactly.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from formsync.io.audio import ffmpeg_path, write_audio
from formsync.io.video import write_video


def square_frames(
    n_frames: int,
    width: int = 160,
    height: int = 120,
    square: int = 20,
    speed_px: int = 2,
) -> Iterator[np.ndarray]:
    """Yield frames with a red square moving right and a blue square moving down.

    Frame ``i`` has the red square's top-left corner at ``(speed_px * i, 10)`` and the blue
    square's at ``(10, speed_px * i)``, wrapping around the frame. Pixel values are exact, so
    tests can locate the squares.

    Args:
        n_frames: Number of frames to yield.
        width: Frame width in pixels.
        height: Frame height in pixels.
        square: Square side length in pixels.
        speed_px: Pixels moved per frame.

    Yields:
        ``(height, width, 3)`` uint8 BGR arrays.
    """
    for i in range(n_frames):
        img = np.zeros((height, width, 3), np.uint8)
        rx = (speed_px * i) % (width - square)
        by = (speed_px * i) % (height - square)
        img[10 : 10 + square, rx : rx + square] = (0, 0, 255)  # red (BGR)
        img[by : by + square, 10 : 10 + square] = (255, 0, 0)  # blue (BGR)
        yield img


def click_track(duration_s: float, bpm: float, sample_rate: int = 22050) -> np.ndarray:
    """Generate a metronome click track: short 1 kHz bursts on every beat.

    Args:
        duration_s: Length in seconds.
        bpm: Beats per minute.
        sample_rate: Sample rate in Hz.

    Returns:
        ``(n,)`` float32 array in [-1, 1].
    """
    n = int(duration_s * sample_rate)
    samples = np.zeros(n, np.float32)
    period = 60.0 / bpm
    click_len = int(0.03 * sample_rate)
    tone = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(click_len) / sample_rate)
    tone *= np.linspace(1, 0, click_len)  # quick decay
    t = 0.0
    while t < duration_s:
        start = int(t * sample_rate)
        end = min(start + click_len, n)
        samples[start:end] += tone[: end - start].astype(np.float32)
        t += period
    return np.clip(samples, -1, 1)


def make_synthetic_video(
    path: Path | str,
    n_frames: int = 60,
    fps: float = 30.0,
    width: int = 160,
    height: int = 120,
    bpm: float | None = None,
    sample_rate: int = 22050,
) -> Path:
    """Write a small MP4 with moving squares and, optionally, a click-track audio stream.

    Args:
        path: Destination ``.mp4`` path.
        n_frames: Number of video frames.
        fps: Frame rate.
        width: Frame width.
        height: Frame height.
        bpm: If given, mux a click track at this tempo into the file; ``None`` = no audio.
        sample_rate: Audio sample rate.

    Returns:
        The path written.

    Raises:
        RuntimeError: If ffmpeg fails to mux the audio.
    """
    out = Path(path)
    if bpm is None:
        return write_video(out, square_frames(n_frames, width, height), fps)

    silent = out.with_name(out.stem + "_silent.mp4")
    wav = out.with_name(out.stem + "_click.wav")
    write_video(silent, square_frames(n_frames, width, height), fps)
    write_audio(wav, click_track(n_frames / fps, bpm, sample_rate), sample_rate)
    cmd = [
        ffmpeg_path(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent),
        "-i",
        str(wav),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    silent.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr.strip()}")
    return out
