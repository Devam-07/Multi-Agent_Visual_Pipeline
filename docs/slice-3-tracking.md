# Slice 3: ByteTrack on replayed pose detections

## Scope and interface

This slice integrates the real Ultralytics ByteTrack implementation, rather than a
second homegrown tracker. It uses the existing `PoseFrame` input boundary and lazily
loads the backend when tracking starts. No YOLO weights or inference are needed.
The adapter was checked against the installed source and the official
[ByteTrack reference](https://docs.ultralytics.com/reference/trackers/byte_tracker/).
`lap` is now an explicit locked dependency; a missing dependency fails with setup
instructions before Ultralytics can attempt runtime installation.

```powershell
uv run formsync track runs/pose-demo/stream.json --output runs/tracked-demo
uv run formsync evaluate-tracks runs/tracked-demo/tracking.json --truth runs/pose-demo/ground_truth.json
```

Tracking reads only the timeline and detections. `--tracker` accepts a validated YAML
file; otherwise the pipeline's tracker_config path is used, relative to the repository.
The supplied `configs/bytetrack_dance.yaml` retains lost tracks for 30 analysis-frame
updates (two seconds at 15 fps). Defaults are a baseline, not calibrated for dance.

## Output contract

- A new output directory contains `tracks.parquet` and completion marker `tracking.json`.
- Every input detection remains in its original row order, with its original joints,
  timestamp, box, and confidence. Only raw `track_id` is assigned.
- Unconfirmed or rejected associations remain -1; missing observations are not
  synthesized. Empty frames still advance the tracker and its retention timer.
- Tracks born after the initial frame require subsequent confirmation under ByteTrack's
  rules. IDs can change after long gaps; this slice does not stitch identities.
- `slot` stays -1. Stable dancer slots across a full routine require later identity
  reconciliation; raw tracking is not a claim of permanent identity.
- The manifest records input path and full Parquet SHA-256, backend version, settings,
  timeline, detection count, assigned count, and number of distinct tracks.
- Existing output directories are refused. The manifest is atomically published last;
  failed writes can leave a partial directory without a success manifest. CLI errors
  return code 1. This is a bounded in-memory implementation, like current replay.

`track_frames` can later consume YOLO-produced `PoseFrame` objects. Frames must start
at zero and be consecutive, with unassigned input IDs and timestamps matching their
detection rows. Tracking uses bounding boxes/confidences, not joint similarity.

## Evaluation semantics

Evaluation reads labels separately and checks the original detections' hash and all
unchanged measurement columns before trusting row alignment. Original ground-truth
files must remain unmodified; these checks detect accidental dataset mismatch, not
malicious label alteration.

- Identity switches: changes between successive assigned observations of a true dancer,
  including across gaps.
- Extra tracks per dancer: distinct predicted IDs beyond the first, summed over dancers.
  This is a fragmentation proxy, not a standard benchmark fragmentation metric.
- Mixed tracks: predicted IDs associated with more than one true dancer.
- Assigned fraction: assigned detections divided by all input detections.
- Assignment accuracy: correct detections under the best global one-to-one matching
  between arbitrary track IDs and true dancers, divided by all detections. Unassigned
  rows count against accuracy. Empty-data rates are null.

These diagnostics are not MOTA, IDF1, or HOTA. They do not measure detector misses
against invisible ground truth, since their denominator is observed detections.

## Test-first evidence

Before implementation, real-backend acceptance tests failed during collection because
`formsync.pose.tracking` did not exist. Metric/configuration tests similarly failed
before their modules existed. The evaluation CLI acceptance test subsequently failed
with `No such command 'evaluate-tracks'` before that command was implemented.

Acceptance tests exercise real ByteTrack, not a fake tracker: shuffled row order,
short gaps, low-confidence second-stage matching, long-gap expiry and confirmation,
empty frames, invalid timeline order, source identity isolation, persistence, CLI,
and evaluation provenance. Unit tests verify metric arithmetic using known swaps and
missing assignments, configuration validation, dependency preflight, and partial writes.
The write-failure tests characterize the implemented completion-marker behavior.

The 600-detection, four-dancer no-dropout fixture achieved zero switches and 100%
assignment accuracy. A simple 35-frame constant-velocity crossing (70 detections) also
achieved zero switches and 100%. These controlled examples do not establish accuracy
for crowded dance footage, changing boxes, or prolonged occlusion.

Run fast tests and real-backend tests separately so the fast suite retains its guard
against importing torch/Ultralytics. CI now runs both:

Final validation: 157 fast tests passed and six real ByteTrack integration tests passed.
Ruff lint/format checks, strict mypy (35 source files), and the 100% docstring coverage
gate passed.

```powershell
uv run pytest
uv run pytest -m slow tests/integration/test_bytetrack_pipeline.py
```

## Next slice

Integrate pretrained YOLO pose inference on actual video, convert detections to
`PoseFrame`, feed them through this tracker, and inspect skeleton/ID overlays.
Smoothing, dancer-slot reconciliation, synchronization, and formation scoring remain
subsequent work. No training data is needed for the initial pretrained integration.
