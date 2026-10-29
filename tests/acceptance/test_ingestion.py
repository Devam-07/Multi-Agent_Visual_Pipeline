"""Ingestion contracts exercised with actual generated media and controlled failures."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from formsync.cli import app
from formsync.config import PipelineConfig
from formsync.ingestion import ingest_video
from formsync.io.audio import AudioExtractionError, load_audio
from formsync.schemas.ingestion import IngestionManifest
from formsync.testing.video import make_synthetic_video


@pytest.mark.parametrize("bpm", [None, 120])
def test_ingest_media(tmp_path: Path, bpm: int | None) -> None:
    """Successful manifests round-trip and describe real audio and frame timing."""
    clip = make_synthetic_video(tmp_path / "practice.mp4", bpm=bpm)
    cfg = PipelineConfig()
    cfg.pose.target_fps = 12
    path = ingest_video(clip, tmp_path / "runs", cfg)
    manifest = IngestionManifest.model_validate_json(path.read_text(encoding="utf-8"))
    assert manifest.status == "succeeded"
    assert manifest.source_path == clip.resolve()
    assert manifest.run_id == path.parent.name
    assert manifest.completed_at >= manifest.created_at
    assert manifest.created_at.utcoffset().total_seconds() == 0
    assert manifest.requested_analysis_fps == 12
    assert manifest.video is not None
    assert (manifest.video.width, manifest.video.height, manifest.video.n_frames) == (160, 120, 60)
    assert manifest.video.duration_s == pytest.approx(2)
    assert manifest.video.analysis_fps == 15
    assert manifest.video.frame_stride == 2
    assert manifest.error is None
    assert manifest.fingerprint_method == "sha256-size-and-ends-1MiB"
    assert manifest.source_fingerprint
    if bpm is None:
        assert manifest.audio_status == "absent"
        assert manifest.audio_path is None
        assert manifest.warnings
        assert not (path.parent / "audio.wav").exists()
    else:
        assert manifest.audio_status == "extracted"
        assert manifest.audio_path == "audio.wav"
        samples, rate = load_audio(path.parent / manifest.audio_path)
        assert rate == manifest.audio_sample_rate == 22050
        assert abs(len(samples) / rate - 2) < 0.1


def test_repeated_inputs_have_separate_runs(tmp_path: Path) -> None:
    """Run identity differs even when the sampled input fingerprint is identical."""
    clip = make_synthetic_video(tmp_path / "practice.mp4")
    paths = [ingest_video(clip, tmp_path / "runs", PipelineConfig()) for _ in range(2)]
    manifests = [IngestionManifest.model_validate_json(p.read_text()) for p in paths]
    assert paths[0] != paths[1]
    assert manifests[0].source_fingerprint == manifests[1].source_fingerprint


@pytest.mark.parametrize("kind", ["missing", "corrupt", "directory", "unreadable"])
def test_invalid_video_records_failure(tmp_path: Path, kind: str) -> None:
    """Bad inputs leave a failed manifest, including metadata-only undecodable media."""
    clip = tmp_path / "bad.mp4"
    if kind == "corrupt":
        clip.write_bytes(b"not a video")
    elif kind == "directory":
        clip.mkdir()
    elif kind == "unreadable":
        make_synthetic_video(clip)
    with patch("formsync.ingestion.read_frame", side_effect=IndexError("cannot decode")):
        path = ingest_video(clip, tmp_path / "runs", PipelineConfig())
    manifest = IngestionManifest.model_validate_json(path.read_text())
    assert manifest.status == "failed"
    assert manifest.error
    assert manifest.audio_path is None


def test_extraction_failure_is_not_silent_video(tmp_path: Path) -> None:
    """A decoder error must fail ingestion rather than masquerade as absent audio."""
    clip = make_synthetic_video(tmp_path / "practice.mp4", bpm=120)
    with patch("formsync.ingestion.extract_audio", side_effect=AudioExtractionError("decoder")):
        path = ingest_video(clip, tmp_path / "runs", PipelineConfig())
    manifest = IngestionManifest.model_validate_json(path.read_text())
    assert manifest.status == "failed"
    assert "decoder" in manifest.error
    assert manifest.audio_status == "failed"


def test_output_failure_is_reported(tmp_path: Path) -> None:
    """An unusable output location raises an actionable filesystem error."""
    output = tmp_path / "not-a-directory"
    output.write_text("occupied")
    with pytest.raises(OSError):
        ingest_video(tmp_path / "missing.mp4", output, PipelineConfig())


@pytest.mark.usefixtures("clean_env")
def test_ingest_cli(tmp_path: Path) -> None:
    """The public command reports manifest locations and uses meaningful exit codes."""
    clip = make_synthetic_video(tmp_path / "practice.mp4")
    runner = CliRunner()
    output = tmp_path / "runs"
    result = runner.invoke(app, ["ingest", str(clip), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert "manifest.json" in result.output
    assert "beat" in result.output.lower()
    result = runner.invoke(app, ["ingest", str(tmp_path / "missing"), "--output", str(output)])
    assert result.exit_code == 1
    assert "manifest.json" in result.output
    assert "failed" in result.output.lower()
    occupied = tmp_path / "occupied"
    occupied.write_text("file")
    result = runner.invoke(app, ["ingest", str(clip), "--output", str(occupied)])
    assert result.exit_code == 1
    assert "Unable" in result.output


@pytest.mark.usefixtures("clean_env")
def test_ingest_cli_invalid_yaml(tmp_path: Path) -> None:
    """Malformed configuration produces a readable CLI error before creating a run."""
    config = tmp_path / "broken.yaml"
    config.write_text("pose: [")
    result = CliRunner().invoke(app, ["ingest", "clip.mp4", "--pipeline", str(config)])
    assert result.exit_code == 1
    assert "Unable to ingest" in result.output
