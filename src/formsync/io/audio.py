"""Audio extraction and loading.

The audio track is pulled out of the video with the ffmpeg binary bundled in the
``imageio-ffmpeg`` wheel (no system install), written as a mono WAV at the requested sample
rate, and loaded into a float32 numpy array with ``soundfile``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf


class AudioExtractionError(RuntimeError):
    """Raised when ffmpeg cannot extract an audio track (e.g. the video has none)."""


class NoAudioStreamError(AudioExtractionError):
    """Raised only when ffmpeg explicitly reports no matching audio stream."""


def ffmpeg_path() -> str:
    """Return the path of the bundled ffmpeg executable.

    Returns:
        Absolute path string usable with :mod:`subprocess`.
    """
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def extract_audio(video_path: Path | str, wav_path: Path | str, sample_rate: int = 22050) -> Path:
    """Extract the audio track of a video to a mono WAV file.

    Args:
        video_path: Source video.
        wav_path: Destination ``.wav`` path; parent directories are created.
        sample_rate: Output sample rate in Hz.

    Returns:
        The WAV path written.

    Raises:
        FileNotFoundError: If the video does not exist.
        NoAudioStreamError: If the input has no audio stream.
        AudioExtractionError: If ffmpeg fails to decode or write audio.
    """
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(src)
    out = Path(wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-map",
        "0:a:0",  # explicitly distinguish absent audio from decoding/output failures
        "-vn",  # drop video
        "-ac",
        "1",  # mono
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        detail = result.stderr.strip() or "no audio stream"
        if "matches no streams" in detail:
            raise NoAudioStreamError(f"No audio stream in {src}")
        raise AudioExtractionError(f"ffmpeg could not extract audio from {src}: {detail}")
    return out


def load_audio(wav_path: Path | str) -> tuple[np.ndarray, int]:
    """Load a WAV file as a mono float32 array.

    Multi-channel files are averaged to mono.

    Args:
        wav_path: WAV file.

    Returns:
        ``(samples, sample_rate)`` where ``samples`` has shape ``(n,)`` and values in [-1, 1].
    """
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)


def write_audio(wav_path: Path | str, samples: np.ndarray, sample_rate: int) -> Path:
    """Write a mono float array to a 16-bit WAV file.

    Args:
        wav_path: Destination path; parent directories are created.
        samples: ``(n,)`` array with values in [-1, 1].
        sample_rate: Sample rate in Hz.

    Returns:
        The path written.
    """
    out = Path(wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), np.asarray(samples, np.float32), sample_rate, subtype="PCM_16")
    return out
