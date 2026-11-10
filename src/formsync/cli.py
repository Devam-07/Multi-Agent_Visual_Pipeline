"""Command-line entry point (``formsync ...``), built with Typer.

Provides configuration, ingestion, synthetic datasets and replay; later steps add analysis,
``beats``, ``feedback`` and ``bench``. Each command is a thin wrapper that parses arguments and
delegates to library code, so all logic stays testable without the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from formsync import __version__
from formsync.config import get_settings, load_pipeline_config
from formsync.ingestion import ingest_video
from formsync.schemas.ingestion import IngestionManifest

app = typer.Typer(
    name="formsync",
    help="Dance formation & synchronization analyzer.",
    no_args_is_help=True,
    add_completion=False,
)
# highlight=False keeps output plain (no colour codes around numbers), which is friendlier for
# piping and for tests that assert on the text.
console = Console(highlight=False)


@app.command()
def version() -> None:
    """Print the installed formsync version."""
    console.print(f"formsync {__version__}")


@app.command()
def config(
    pipeline: Annotated[
        Path | None,
        typer.Option("--pipeline", "-p", help="Pipeline YAML to load (default: bundled config)."),
    ] = None,
) -> None:
    """Show the effective configuration: environment settings and pipeline constants.

    Secrets are masked. Useful to confirm which provider, model and thresholds a run will use.
    """
    settings = get_settings()
    cfg = load_pipeline_config(pipeline or settings.pipeline_config)
    console.print("[bold]Settings[/bold]")
    console.print_json(settings.model_dump_json(indent=2))
    console.print(f"[bold]Resolved LLM model:[/bold] {settings.resolved_llm_model()}")
    console.print("[bold]Pipeline config[/bold]")
    console.print_json(cfg.model_dump_json(indent=2))


@app.command()
def ingest(
    video: Annotated[Path, typer.Argument(help="Local source video (referenced, not copied).")],
    output: Annotated[Path | None, typer.Option(help="Run directory root.")] = None,
    pipeline: Annotated[
        Path | None, typer.Option("--pipeline", "-p", help="Pipeline YAML settings.")
    ] = None,
) -> None:
    """Validate a video and save metadata and optional audio in a new run.

    Prints the manifest path. Exits with code 1 on media, configuration, or output
    failures; absent audio is a successful ingestion with a warning.
    """
    try:
        settings = get_settings()
        cfg = load_pipeline_config(pipeline or settings.pipeline_config)
        path = ingest_video(video, output or settings.artifact_root, cfg)
        manifest = IngestionManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        console.print(f"Unable to ingest video: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Ingestion {manifest.status}: {path}", markup=False)
    for warning in manifest.warnings:
        console.print(warning, markup=False)
    if manifest.status != "succeeded":
        console.print(manifest.error, markup=False)
        raise typer.Exit(code=1)


@app.command()
def synthetic(
    output: Annotated[Path, typer.Option(help="New directory for a synthetic pose dataset.")],
    scenario: Annotated[Path | None, typer.Option(help="Optional JSON scenario settings.")] = None,
    seed: Annotated[int | None, typer.Option(help="Override the scenario random seed.")] = None,
) -> None:
    """Save synthetic joint detections and separate evaluation labels without video.

    The default scenario has four dancers, a timing delay, reduced movement amplitude,
    and missing detections. An existing output directory is never overwritten.
    """
    from formsync.schemas.replay import ScenarioConfig
    from formsync.testing.pose_dataset import write_synthetic_dataset

    try:
        cfg = (
            ScenarioConfig.model_validate_json(scenario.read_text(encoding="utf-8"))
            if scenario
            else ScenarioConfig()
        )
        if seed is not None:
            cfg.seed = seed
        path = write_synthetic_dataset(output, cfg)
    except (OSError, ValueError) as exc:
        console.print(f"Unable to generate synthetic dataset: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Synthetic pose dataset: {path}", markup=False)


@app.command()
def replay(manifest: Annotated[Path, typer.Argument(help="Path to a pose stream.json.")]) -> None:
    """Replay stored detections and report frame, detection, and empty-frame counts.

    Exercises the frame input boundary only; use ``track`` to assign raw track IDs.
    Does not read ground truth, load a model, or produce a video.
    """
    from formsync.pose.replay import replay_frames

    frames = detections = empty = 0
    try:
        for frame in replay_frames(manifest):
            frames += 1
            detections += len(frame.detections)
            empty += int(len(frame.detections) == 0)
    except (OSError, ValueError) as exc:
        console.print(f"Unable to replay dataset: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Replayed {frames} frames, {detections} detections, {empty} empty frames.")


@app.command()
def track(
    manifest: Annotated[Path, typer.Argument(help="Untracked pose stream.json.")],
    output: Annotated[Path, typer.Option(help="New directory for tracking results.")],
    tracker: Annotated[Path | None, typer.Option(help="ByteTrack YAML configuration.")] = None,
) -> None:
    """Run ByteTrack on saved pose detections and preserve every row in tracks.parquet.

    Unconfirmed detections keep ID -1. Permanent dancer slots remain unassigned.
    Ground truth is never read. Errors exit with code 1 without a success manifest.
    """
    from formsync.config import REPO_ROOT
    from formsync.pose.tracking import track_dataset
    from formsync.schemas.tracking import TrackerConfig

    try:
        if tracker is None:
            settings = get_settings()
            tracker = load_pipeline_config(settings.pipeline_config).pose.tracker_config
            if not tracker.is_absolute():
                tracker = REPO_ROOT / tracker
        cfg = TrackerConfig.model_validate(yaml.safe_load(tracker.read_text(encoding="utf-8")))
        path = track_dataset(manifest, output, cfg)
    except (OSError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        console.print(f"Unable to track dataset: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Tracking complete: {path}", markup=False)


@app.command("evaluate-tracks")
def evaluate_tracks(
    manifest: Annotated[Path, typer.Argument(help="Saved tracking.json.")],
    truth: Annotated[Path, typer.Option(help="Original synthetic ground_truth.json.")],
) -> None:
    """Print offline identity metrics using separate synthetic evaluation labels.

    Requires the original detections next to the truth file for provenance checks.
    Does not run inference or modify artifacts. Rates for an empty dataset are null.
    """
    import json

    from formsync.testing.tracking import evaluate_dataset

    try:
        metrics = evaluate_dataset(manifest, truth)
    except (OSError, ValueError) as exc:
        console.print(f"Unable to evaluate tracks: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(metrics))


@app.command()
def poses(
    manifest: Annotated[Path, typer.Argument(help="Successful ingestion manifest.json.")],
    output: Annotated[Path, typer.Option(help="New directory for poses, tracks and overlay.")],
    pipeline: Annotated[Path | None, typer.Option(help="Pipeline YAML settings.")] = None,
) -> None:
    """Run pretrained YOLO Pose and ByteTrack on CPU and save an inspection video.

    Official weights may download on first use. Use pose.model_name in a pipeline
    YAML to select cached/local weights. Existing output directories are refused.
    """
    from formsync.config import REPO_ROOT
    from formsync.pose.inference import run_poses
    from formsync.schemas.tracking import TrackerConfig

    try:
        cfg = load_pipeline_config(pipeline or get_settings().pipeline_config)
        tracker_path = cfg.pose.tracker_config
        if not tracker_path.is_absolute():
            tracker_path = REPO_ROOT / tracker_path
        tracker = TrackerConfig.model_validate(
            yaml.safe_load(tracker_path.read_text(encoding="utf-8"))
        )
        result = run_poses(manifest, output, cfg, tracker)
    except (OSError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        console.print(f"Unable to process poses: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Pose analysis complete: {result}", markup=False)


@app.command()
def clean(
    manifest: Annotated[
        Path, typer.Argument(help="Successful poses.json, tracking.json or identities.json.")
    ],
    output: Annotated[Path, typer.Option(help="New directory for cleaned trajectories.")],
    pipeline: Annotated[
        Path | None, typer.Option(help="Pipeline YAML with cleanup settings.")
    ] = None,
) -> None:
    """Filter, interpolate and smooth each raw track, saving quality and processing flags.

    Uses pose.keypoint_conf_min, max_gap_frames, smooth_window and smooth_poly from
    pipeline settings. Does not merge track identities or modify source measurements.
    """
    from formsync.pose.cleanup import CleanupConfig, clean_dataset

    try:
        pose = load_pipeline_config(pipeline or get_settings().pipeline_config).pose
        cfg = CleanupConfig(
            confidence=pose.keypoint_conf_min,
            max_gap_frames=pose.max_gap_frames,
            smooth_window=pose.smooth_window,
            smooth_poly=pose.smooth_poly,
        )
        result = clean_dataset(manifest, output, cfg)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        console.print(f"Unable to clean trajectories: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Cleanup complete: {result}", markup=False)


@app.command("reconcile")
def reconcile_identities(
    manifest: Annotated[Path, typer.Argument(help="Successful poses.json or tracking.json.")],
    output: Annotated[Path, typer.Option(help="New directory for slot assignments and evidence.")],
    config: Annotated[Path | None, typer.Option(help="Optional identity matching YAML.")] = None,
) -> None:
    """Join unambiguous track fragments into provisional dancer slots.

    Competing candidates remain separate. Saved evidence explains accepted, rejected,
    and ambiguous links. Does not change raw measurements or repair within-track swaps.
    """
    from formsync.pose.identity import reconcile_dataset
    from formsync.schemas.identity import IdentityConfig

    try:
        settings = (
            IdentityConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
            if config
            else IdentityConfig()
        )
        result = reconcile_dataset(manifest, output, settings)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        console.print(f"Unable to reconcile identities: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Identity reconciliation complete: {result}", markup=False)


if __name__ == "__main__":  # pragma: no cover
    app()
