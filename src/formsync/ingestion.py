"""Ingest local media into isolated run directories without loading a pose model."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from formsync.config import PipelineConfig
from formsync.io.audio import NoAudioStreamError, extract_audio
from formsync.io.video import hash_video_file, probe_video, read_frame
from formsync.schemas.ingestion import IngestionManifest


def write_manifest(path: Path, manifest: IngestionManifest) -> None:
    """Atomically replace a UTF-8 JSON manifest using a sibling temporary file.

    The parent directory must exist. Filesystem errors propagate to the caller; an
    existing manifest stays intact if writing the replacement fails.
    """
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ingest_video(source: Path, output: Path, config: PipelineConfig) -> Path:
    """Validate a video, extract optional audio, and persist an ingestion manifest.

    Every invocation creates a distinct run, even for repeated inputs. Validation reads
    the first frame and container metadata; it does not certify every frame or variable
    frame-rate timestamps. Media failures produce a failed manifest returned normally.
    Output failures raise ``OSError`` because a manifest cannot be guaranteed on an
    unwritable filesystem. The source is referenced by absolute path and must remain
    available. No inference, scoring, database registration, or source copying occurs.

    Args:
        source: Local video to reference.
        output: Root directory for isolated runs.
        config: Validated pipeline settings for sampling and audio extraction.

    Returns:
        Absolute path to the manifest; inspect its status before using artifacts.
    """
    run_id = str(uuid4())
    directory = output.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "manifest.json"
    manifest = IngestionManifest(
        run_id=run_id,
        source_path=source.resolve(),
        created_at=datetime.now(UTC),
        requested_analysis_fps=config.pose.target_fps,
        audio_sample_rate=config.audio.sample_rate,
    )
    write_manifest(path, manifest)
    try:
        if not manifest.source_path.is_file():
            raise ValueError(f"Input is missing or is not a file: {manifest.source_path}")
        manifest.video = probe_video(manifest.source_path, target_fps=config.pose.target_fps)
        if manifest.video.n_frames < 1:
            raise ValueError("Input reports no video frames")
        read_frame(manifest.source_path, 0)
        manifest.source_fingerprint = hash_video_file(manifest.source_path)
        manifest.audio_status = "failed"
        try:
            extract_audio(manifest.source_path, directory / "audio.wav", config.audio.sample_rate)
        except NoAudioStreamError:
            manifest.audio_status = "absent"
            manifest.warnings.append("No audio stream; beat-based analysis is unavailable.")
        else:
            manifest.audio_status = "extracted"
            manifest.audio_path = "audio.wav"
        manifest.status = "succeeded"
    except (OSError, ValueError, IndexError, RuntimeError) as exc:
        manifest.status = "failed"
        manifest.error = f"{type(exc).__name__}: {exc}"
    if manifest.audio_path is None:
        try:
            (directory / "audio.wav").unlink(missing_ok=True)
        except OSError as exc:
            manifest.warnings.append(f"Unable to remove incomplete audio artifact: {exc}")
    manifest.completed_at = datetime.now(UTC)
    write_manifest(path, manifest)
    return path
