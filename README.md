# formsync: Dance Formation & Synchronization Analyzer

Analyzes fixed-camera practice footage of a full dance set (~35 dancers), scores each dancer's
timing and amplitude against the group, tracks formation transitions on the stage floor, and
uses an LLM agent to turn the numbers into prioritized rehearsal notes such as
"dancers 3 and 7 are consistently a half-beat late on the chorus hit".

## How it works

```
video ──► pose estimation ──► tracking + smoothing ──► per-dancer motion signals
  │        (YOLO11-pose)        (ByteTrack, slots)              │
  └──► audio ──► beat grid                                       ▼
                    │                       sync scoring (offset in beats, amplitude ratio)
                    ▼                       formation scoring (holds, transitions, spacing)
             sections / segments                                 │
                                                                 ▼
                                             feedback agent (LangGraph, provider-agnostic LLM)
                                                                 │
                                                                 ▼
                                    JSON artifacts ──► CLI report / dashboard (later phase)
```

The source tree under `src/formsync/` is ordered the same way: `io → pose → audio → sync →
formation → agents`, with shared Pydantic models in `schemas/`.

## Project status

Phase 1 (local CLI: CV core, sync/formation scoring, single feedback agent) is in progress.
Foundation, ingestion, synthetic pose datasets, replay, ByteTrack tracking, and CPU YOLO pose
inference with inspection overlays are implemented. Smoothing, beat extraction,
scoring, and feedback generation remain upcoming slices.
Later phases: multi-agent rubric scoring, human-labelled eval harness, FastAPI + React dashboard,
Kubernetes batch jobs with MinIO/S3 storage, staging visualization.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Docker (for later phases). uv installs
Python 3.14 automatically.

```
uv sync --group dev          # core deps + dev tools
uv sync --all-extras --group dev   # also groq/ollama/anthropic/openvino/s3 extras
cp .env.example .env         # then fill in the keys you use
uv run formsync --help
uv run formsync config       # show effective settings and pipeline constants
```

LLM providers: Gemini (default, free tier), Groq (free tier), Ollama (local), Anthropic. Select
with `LLM_PROVIDER` in `.env`.

## Ingest a video

```powershell
uv run formsync ingest practice.mp4 --output runs/
```

Each invocation creates `runs/<run-id>/manifest.json` and, when the input has audio,
`audio.wav` (mono, at the configured sample rate). The source video is referenced by
absolute path, not copied; keep it available for subsequent analysis. Use `--pipeline`
to select a YAML configuration. Without `--output`, the `ARTIFACT_ROOT` setting applies.

The manifest records input metadata, requested and effective analysis frame rates,
UTC timestamps, a sampled input fingerprint, artifact paths, and ingestion status.
Silent videos succeed with a warning that beat-based analysis is unavailable.
Media failures save a failed manifest and exit with code 1. Filesystem failures may
prevent saving a manifest; the command reports the error. A `running` manifest means
the run did not commit a final outcome. Repeated inputs always create separate runs.

This validates container metadata and the first readable frame, not every frame in
the video. It performs no pose estimation or scoring. See
[the ingestion contract](docs/slice-1-ingestion.md) for details and limitations.

## Generate and replay synthetic poses

```powershell
uv run formsync synthetic --output runs/pose-demo --seed 7
uv run formsync replay runs/pose-demo/stream.json
```

This creates numerical joint detections, not an MP4. The default scenario contains
four dancers over ten seconds at 15 fps: dancer index 1 is 0.25 seconds late, dancer
index 2 moves at 70% amplitude, and each dancer has a 10% chance of missing a detection
per frame. Replay reports frame and detection counts; it does not score performance.

The dataset contains `detections.parquet`, `stream.json` (timeline), and a separate
`ground_truth.json` (answers for evaluation). Known dancer identities are removed
from detections, and replay never reads ground truth. The output folder must be new.
Use `--scenario path/to/scenario.json` for custom settings; see
[the synthetic pose and replay contract](docs/slice-2-pose-replay.md).

## Track and evaluate saved poses

```powershell
uv run formsync track runs/pose-demo/stream.json --output runs/tracked-demo
uv run formsync evaluate-tracks runs/tracked-demo/tracking.json --truth runs/pose-demo/ground_truth.json
```

Tracking writes `tracks.parquet` and `tracking.json` into a new directory. It uses
ByteTrack directly, without YOLO weights or ground truth. Every detection is retained;
unconfirmed detections keep track ID `-1`. IDs are raw track identities; permanent
dancer slots remain unassigned. Settings live in `configs/bytetrack_dance.yaml`, or
pass `--tracker custom.yaml`.

Evaluation is a separate operation using the original synthetic dataset. It reports
identity switches, extra tracks per dancer, mixed identities, assigned coverage, and
global assignment accuracy. These are fixture diagnostics, not real-video accuracy.
See [the tracking contract and test-first evidence](docs/slice-3-tracking.md).

## Run YOLO Pose on a video

First ingest your video, then use the manifest path printed by ingestion:

```powershell
uv run formsync ingest practice.mp4 --output runs/
uv run formsync poses runs/<run-id>/manifest.json --output runs/pose-results/
uv run formsync replay runs/pose-results/stream.json
```

The new output directory contains `detections.parquet`, `tracks.parquet`, `stream.json`,
`overlay.mp4`, and `poses.json`. The overlay shows boxes, confident skeleton joints, and
raw track IDs. It is silent and runs at the sampled analysis rate. `poses.json` records
the outcome, model/settings, runtime, and mapping to source frame indices.

The default model is YOLO11s Pose on CPU; official weights may download on first use.
Use `--pipeline custom.yaml` to choose a smaller model or local weights and adjust
inference resolution. For example:

```yaml
pose:
  model_name: runs/yolo-assets/yolo11n-pose.pt
  imgsz: 640
  target_fps: 15
```

This performs pose detection and raw tracking, not dance scoring. See
[the YOLO integration contract](docs/slice-4-yolo.md) for timing, test evidence, and limits.

## Development

```
uv run pytest                # fast unit tests (never loads torch)
uv run pytest -m slow        # real backend integration tests (currently ByteTrack)
uv run ruff check . && uv run ruff format .
uv run mypy
uv run interrogate           # docstring coverage must be 100%
```

Every function and class has a docstring and a test. Tunable constants live in
`configs/pipeline.default.yaml`; environment-specific values (keys, URLs) in `.env`.

## Hardware notes

Developed on a 4-core laptop CPU with no GPU. Pose inference is CPU-only (optionally accelerated
with OpenVINO), so a three-minute clip is a batch job of several minutes, which is exactly the
workload the later Kubernetes Job phase targets.

## License note

This project depends on [Ultralytics](https://github.com/ultralytics/ultralytics), which is
licensed under AGPL-3.0. Keep that in mind if you redistribute or host it as a service.
