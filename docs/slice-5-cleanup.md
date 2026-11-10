# Slice 5: traceable joint trajectory cleanup

## Usage

```powershell
uv run formsync clean runs/pose-results/poses.json --output runs/clean-results/
uv run formsync clean runs/tracked-demo/tracking.json --output runs/clean-synthetic/
```

The command accepts successful version-1 `poses.json` and `tracking.json` results.
It reads the adjacent `tracks.parquet`, validates frame indices/times and uniqueness
of each assigned track/frame pair, and preserves the source files. Existing output
directories are refused. CLI errors return code 1.

## Algorithm and configuration

All settings are taken from the pipeline YAML (`--pipeline` overrides the default):

```yaml
pose:
  keypoint_conf_min: 0.3
  max_gap_frames: 5
  smooth_window: 7
  smooth_poly: 2
```

Each raw track is processed independently from its first through last detection.
Absent intermediate detections become missing samples. There is no assumption that
two different raw track IDs belong to the same person, and no samples are invented
outside a track's lifetime. Track ID -1 observations are excluded and counted.

For each COCO joint:

1. Both pixel coordinates and confidence must be finite, with confidence at or above
   the threshold, to count as observed. Otherwise both coordinates become NaN.
2. A missing run is interpolated linearly only when bounded on both sides by valid
   observations and no longer than max_gap_frames. Setting it to zero disables filling.
   Boundary gaps and long gaps remain missing.
3. Each remaining continuous valid segment is smoothed independently using SciPy's
   [Savitzky-Golay filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html),
   with `mode="interp"`. Segments shorter than the configured window remain unchanged.
   The window must be odd and the polynomial order smaller than the window.

This is offline processing: interpolation and centered smoothing use future samples.
Smoothing can alter sharp hits and amplitudes. Passing the controlled numerical tests
below does not guarantee acceptable distortion for every choreography or frame rate.
Different frame rates give the same window a different duration in seconds.

## Artifacts

`trajectories.parquet` is a new long-format table, distinct from detection Parquet.
It contains one row per track/frame/joint over each track's observed lifetime:

| Field | Meaning |
|---|---|
| track_id | Raw tracker identity; permanent dancer slots are not assigned |
| frame_idx, t | Global analysis-frame index and timestamp in seconds |
| joint | Zero-based COCO-17 joint index, using the existing keypoint name order |
| x, y | Cleaned pixel coordinates; NaN where still missing |
| confidence | Original confidence; NaN if that detection never existed |
| observed | Original joint was finite and passed the confidence threshold |
| interpolated | Coordinates were filled across a bounded short gap |
| smoothed | A smoothing filter was applied, even if the value did not change |
| missing | Coordinates remain unavailable after filling |

Confidence is never increased to make interpolated points look like detections.
Flags are complementary provenance: an observed or interpolated sample can also be
smoothed. Downstream consumers must consult these flags and quality metrics instead
of blindly applying the original confidence threshold to interpolated points.

`quality.json` reports input and unassigned detection counts, raw track count, and
per-track lifetimes, observed detection coverage, observed/usable joint coverage,
interpolation/smoothing/missing counts, and original missing-run lengths. Gap lengths
are joint-level runs pooled across the track's 17 joints, so a wholly missing frame
can contribute to 17 gap runs. Coverage denominators are the track's first-to-last
detection span, not the full video. No true fragmentation count can be inferred
without ground-truth or reconciled dancer identities; raw counts/spans expose the
available evidence without making that claim.

`cleanup.json` records source manifest, a full SHA-256 of the input tracks, timeline,
settings, coordinate units, and artifact names. It is atomically published last.
Failed writes may leave partial files, but no successful cleanup manifest. Empty or
wholly unassigned inputs produce a typed empty Parquet and explicit zero-track report.
This first implementation holds output tables in memory and limits expansion to
5,000,000 joint samples; it does not support arbitrarily long recordings.

## Test-first evidence

The initial numerical and CLI tests failed before `formsync.pose.cleanup` existed.
After implementation, a further regression test showed invalid confidence values
(-0.1, 1.1, infinity) were accepted by the joint helper. All three cases failed with
`DID NOT RAISE` before the validation fix.

Tests cover thresholding, original confidence preservation, bounded gaps, unfilled
long/boundary gaps, track separation, short segments, empty/missing data, both source
formats, duplicate observations, invalid timelines/settings, overwrite refusal, and
partial-write behavior. A seeded 1 Hz sinusoid at 30 fps with 30-pixel amplitude and
2-pixel noise is checked with an 11-frame cubic filter: interior mean squared error
must fall by at least 50%, fitted amplitude error must stay below 3%, and fitted phase
error must stay below one frame. These tolerances apply to this fixture only.

Final validation: 189 fast tests passed (seven model/tracker integration tests were
deselected); Ruff lint/format checks, strict mypy for 38 source files, and the 100%
docstring coverage gate passed. No neural inference changes were made in this slice.

No smoothing crosses a long gap or different track ID. Cleanup cannot detect or fix
an identity switch that already occurred within one raw track. Identity reconciliation,
beat extraction, synchronization and formation scoring, and feedback remain future work.
