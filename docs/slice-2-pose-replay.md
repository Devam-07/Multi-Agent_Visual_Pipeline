# Slice 2: synthetic pose datasets and replay

## Usage

```powershell
uv run formsync synthetic --output runs/pose-demo --seed 7
uv run formsync replay runs/pose-demo/stream.json
```

Use a new output directory each time. This is separate from video ingestion: it
generates numerical pose detections, not video or audio, and does not need a model.
Replay validates the artifacts and reports frame, detection, and empty-frame counts.
It does not run tracking, scoring, or real-time playback.

## Scenario configuration

Save a JSON file and pass `--scenario scenario.json` to `synthetic`. Omitted fields
use defaults; `--seed` overrides the JSON seed. Dancer indices are zero-based.

```json
{
  "seed": 7,
  "n_frames": 150,
  "fps": 15,
  "tempo_bpm": 120,
  "noise_px": 0,
  "dropout_frac": 0.1,
  "joint_dropout_frac": 0,
  "dancers": [
    {"offset_s": 0, "amp_scale": 1},
    {"offset_s": 0.25, "amp_scale": 1},
    {"offset_s": 0, "amp_scale": 0.7},
    {"offset_s": 0, "amp_scale": 1}
  ]
}
```

The existing `make_dancers` generator provides shared keyframe hits: dancers move
between target poses, then hold them. It uses a non-repeating hit sequence to avoid
ambiguous timing matches. Positive offsets mean late; amplitude scales motion about
the neutral pose, not body size. The default 0.25-second delay is half a beat at 120
BPM. Bodies stand in a fixed line on a 1920 by 1080 pixel coordinate plane.

Dropout is an independent probability per dancer per frame, not an exact count.
Dropped detections have no row. Joint dropout retains a noisy coordinate with low
confidence, allowing future code to decide when to ignore a joint. Coordinates are
not clipped to image bounds. Same configuration and generator version reproduce the
same arrays; byte equality across different Parquet library versions is not promised.

Scenario validation bounds frame rate to 1–240 fps, dancer count to 1–128, frames to
1–90000, and total dancer/frame samples to 500000. This is an in-memory fixture tool.

## Saved artifacts

- `stream.json`: versioned timeline, frame dimensions, coordinate convention, and
  relative detections filename. Frame timestamps are `frame_idx / fps`; duration is
  `n_frames / fps`, with the final sample one frame interval before that duration.
- `detections.parquet`: existing `KeypointTable` format, one row per person detection
  containing COCO-17 `(x, y, confidence)` joints, bounding box, and detection confidence.
  Rows are ordered by frame, shuffled within frames, and both `slot` and `track_id`
  are -1. Stable identities must be recovered by future tracking.
- `ground_truth.json`: scenario configuration, seed, generator version, hit/beat
  times, and `row_dancer_ids` mapping each saved Parquet row to its true dancer.
  This mapping is valid for the original row order only. Missing detections can be
  derived from absent `(frame_idx, dancer)` pairs. This file is for evaluation only.

Existing destinations are refused. The stream manifest is atomically published last;
failed writes may leave a partial directory without a stream manifest. Choose a new
directory after a failed generation. No database records or ingestion runs are created.

## Downstream interface

`formsync.pose.replay.replay_frames(path)` yields `PoseFrame` objects with `frame_idx`,
`t`, and `detections`. Empty frames are yielded too, including leading and trailing
gaps and streams with no detections at all. Future inference can yield the same type.

Replay reads only `stream.json` and `detections.parquet`. It validates frame order,
frame bounds, timestamps, finite coordinates, confidence ranges, bounding-box corner
order, and unassigned identities. It loads the table once; it is not out-of-core.
It rejects malformed artifacts rather than silently repairing them.

## Validation and remaining work

Tests cover reproducibility, storage/replay equality, delayed and scaled numerical
motion, identity removal, ground-truth independence, dropout and empty frames,
confidence values, invalid configuration/artifacts, failed writes, overwrite refusal,
and CLI behavior. The initial acceptance suite failed before replay existed.

Final checks: 151 tests passed; Ruff lint and formatting passed; strict mypy passed
for 32 source files; the 100% docstring coverage gate passed.

This slice does not establish pose-estimation accuracy, physical realism, stable
tracking, or meaningful scoring. The numerical generator predates this slice; this
work exposes it as reproducible saved detections with an isolated evaluation contract.
Next: implement tracking against this frame interface and evaluate identity continuity
using the separate ground truth. More difficult crossing/occlusion scenarios can then
extend this baseline before real-video evaluation.
