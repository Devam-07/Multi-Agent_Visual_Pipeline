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

## Development

```
uv run pytest                # fast unit tests (never loads torch)
uv run pytest -m slow        # integration test with the real pose model
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
