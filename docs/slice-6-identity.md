# Slice 6: conservative track identity reconciliation

## Usage

```powershell
uv run formsync reconcile runs/pose-results/poses.json --output runs/identities/
uv run formsync reconcile runs/tracked-demo/tracking.json --output runs/synthetic-identities/ --config configs/identity.default.yaml
uv run formsync clean runs/identities/identities.json --output runs/clean-with-slots/
```

The command accepts successful version-1 `poses.json` and `tracking.json` plus their
adjacent `tracks.parquet`. It does not load a neural model or read ground truth. The
optional YAML overrides matching gates; omitted keys use `IdentityConfig` defaults.
The example YAML contains all current defaults. Existing output directories are refused.

## Matching rule

Each raw track ID is a fragment. Rows are sorted by frame to assess its endpoints.
A candidate predecessor must finish strictly before the successor starts, and the
gap between their observed endpoints must be no greater than `max_gap_s` (default
2 seconds). Calculations use integer frame indices and FPS, avoiding float32 timestamp
rounding at the threshold. Temporally overlapping fragments can never be joined.

Candidates must satisfy every gate:

- Both fragments have at least `min_fragment_frames` observations (default 3).
- Endpoint box-height ratio is within `max_size_ratio` (default 1.5).
- Velocity is estimated from up to `velocity_window` observations (default 3) at each
  endpoint. Forward and backward position predictions must both agree within
  `max_distance_body` (default 0.75 times mean endpoint box height).
- At least `min_common_joints` (default 6) have finite coordinates and confidence at
  least `confidence` (default 0.3) at both endpoints.
- Joint shapes, centered on each endpoint box and divided by its height, have RMS
  Euclidean error at most `max_pose_error` (default 0.25).

Among candidates passing all gates, a link is accepted only if it is the sole
plausible outgoing link of its predecessor AND the sole plausible incoming link of
its successor. No nearest-neighbor tie breaking or forced assignment occurs. This
intentionally favors leaving fragments unresolved. Even a candidate farther into
the future can block a nearer candidate if both pass the gates.

Accepted links form time-ordered chains. Each chain receives a zero-based slot;
unlinked fragments get separate provisional slots. Slot numbering is deterministic
for fixed input evidence: order by first frame, then raw ID. Unassigned raw detections
remain slot -1. Neither raw IDs nor measurement values are changed.

## Artifacts and downstream use

- `slots.parquet`: existing `KeypointTable` schema with slot filled for assigned raw
  tracks. Preserves input row order, boxes, joint coordinates/confidences, timestamps,
  detection confidence, and raw track ID.
- `identities.json`: completion marker, source path/full tracks SHA-256, timeline,
  settings, algorithm version, raw-to-slot mapping, slot/raw-track counts, unassigned
  count, accepted and ambiguous links, and detailed candidate evidence. Candidates
  within the temporal window report the first failed evidence gate or acceptance/
  ambiguity; pairs outside that window are not expanded into the report.

The manifest is atomically published last. A failed write can leave partial files
without a success manifest. Empty inputs are supported. Invalid source timelines,
duplicate raw track/frame observations, preassigned slots, malformed boxes, and
invalid confidence are rejected. The in-memory matcher is capped at 2000 fragments
by default and considers pairs quadratically; it is not a large-scale global solver.

Cleanup now accepts `identities.json` and adjacent `slots.parquet`. Its trajectory
schema adds a `slot` column (also -1 for old unslotted inputs), and per-track quality
includes slot. This additive change retains raw `track_id`; cleanup still processes
each raw fragment separately and never fills the disappearance between fragments.
It rejects inconsistent slots within a raw track or simultaneous observations in
the same slot. Slot assignments can be used to group cleaned rows in later scoring.

## Test-first evidence

The initial matching, ambiguity, chain, crossing, and CLI tests failed because the
identity modules did not exist. Implementation followed. A cleanup integration test
then failed because `identities.json` was unsupported; slot-preserving cleanup was
added afterward. A gap exactly at 0.3 seconds failed because float32 subtraction
made it slightly larger; frame-index calculations fixed it. A corrupted duplicate
slot/frame test failed with `DID NOT RAISE` before cleanup validation was added.

Controlled tests cover known reappearances, opposing trajectories at a crossing,
one-to-many and many-to-one ambiguity, transitive chains, temporal overlap, position
and pose mismatch, low confidence, short fragments, overly long gaps, deterministic
results under row reordering, source immutability, empty/unassigned input, invalid
settings/media metadata, CLI configuration, and failed writes. These tests exercise
known identities but do not establish accuracy on real dancers.

Final validation: 211 fast tests and seven real YOLO/ByteTrack integration tests
passed (218 total). Ruff lint/format checks, strict mypy for 40 source files,
100% docstring coverage, and the Git whitespace check passed.

## Limits

The gates are geometric heuristics, not calibrated identity probabilities. Similar
dancers can still be incorrectly joined; changed pose or unstable boxes can prevent
a correct join. A new provisional slot does not prove a different person, so slot
count is not a reliable dancer count. No appearance embeddings, clothing comparison,
camera motion compensation, or human confirmation workflow are included.

This cannot detect or split an identity swap already present inside one raw track.
Threshold tuning and real dance footage evaluation are still needed. Subsequent work
is beat/section extraction and synchronization scoring, with uncertainty preserved
through the scoring workflow.
