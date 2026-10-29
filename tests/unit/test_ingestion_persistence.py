"""Manifest persistence tests for output errors and interrupted runs."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from formsync.config import PipelineConfig
from formsync.ingestion import ingest_video, write_manifest
from formsync.schemas.ingestion import IngestionManifest
from formsync.testing.video import make_synthetic_video


def test_manifest_replacement_failure_keeps_original(tmp_path: Path) -> None:
    """A failed atomic replacement cannot turn the previous manifest into partial JSON."""
    manifest = IngestionManifest(
        run_id="test",
        source_path=tmp_path / "video.mp4",
        created_at=datetime.now(UTC),
        requested_analysis_fps=15,
        audio_sample_rate=22050,
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    original = path.read_bytes()
    with (
        patch.object(Path, "replace", side_effect=PermissionError("output locked")),
        pytest.raises(PermissionError, match="output locked"),
    ):
        write_manifest(path, manifest)
    assert path.read_bytes() == original
    assert IngestionManifest.model_validate_json(original).status == "running"


def test_manifest_rejects_invalid_settings(tmp_path: Path) -> None:
    """Manifest configuration and schema version reject invalid persisted values."""
    fields = dict(
        run_id="test",
        source_path=tmp_path / "video.mp4",
        created_at=datetime.now(UTC),
        requested_analysis_fps=15,
        audio_sample_rate=22050,
    )
    for updates in ({"requested_analysis_fps": 0}, {"schema_version": 2}, {"typo": True}):
        with pytest.raises(ValidationError):
            IngestionManifest.model_validate(fields | updates)


def test_partial_audio_is_removed_on_failure(tmp_path: Path) -> None:
    """A failed extraction records the cause and does not expose a partial WAV artifact."""
    clip = make_synthetic_video(tmp_path / "clip.mp4")

    def fail_extraction(source: Path, target: Path, rate: int) -> Path:
        """Simulate a disk error after ffmpeg has started writing its output."""
        target.write_bytes(b"partial WAV")
        raise OSError("disk full")

    with patch("formsync.ingestion.extract_audio", side_effect=fail_extraction):
        path = ingest_video(clip, tmp_path / "runs", PipelineConfig())
    manifest = IngestionManifest.model_validate_json(path.read_text())
    assert manifest.status == "failed"
    assert "disk full" in manifest.error
    assert manifest.audio_path is None
    assert not (path.parent / "audio.wav").exists()


def test_final_manifest_write_failure_is_propagated(tmp_path: Path) -> None:
    """A failed final write never reports success and leaves the initial running record."""
    clip = make_synthetic_video(tmp_path / "clip.mp4")

    def fail_final(path: Path, manifest: IngestionManifest) -> None:
        """Allow the initial write and reject the terminal write."""
        if manifest.status != "running":
            raise PermissionError("manifest locked")
        write_manifest(path, manifest)

    with (
        patch("formsync.ingestion.write_manifest", side_effect=fail_final),
        pytest.raises(PermissionError, match="manifest locked"),
    ):
        ingest_video(clip, tmp_path / "runs", PipelineConfig())
    paths = list((tmp_path / "runs").glob("*/manifest.json"))
    assert len(paths) == 1
    assert IngestionManifest.model_validate_json(paths[0].read_text()).status == "running"
