# Slice 1: ingestion

## Approved scope

Provide `formsync ingest VIDEO --output ROOT`: validate local video, allocate a unique
run, record metadata, extract optional audio, and persist a manifest. Repeated inputs
get separate runs. Source media is referenced, not copied. No model or training data
is required, and no pose inference, scoring, or feedback is performed.

## Run contract

- Each run owns `ROOT/<UUID>/manifest.json` and optional `audio.wav`.
- Schema version 1 is defined by `IngestionManifest`. Source paths are absolute;
  audio paths are relative to the manifest directory.
- UTC creation/completion times distinguish wall-clock run timing from video timing.
- `requested_analysis_fps` records the configuration; `video.analysis_fps` and
  `frame_stride` record the effective integer-stride sampling plan. For example,
  requesting 12 fps from a 30 fps video yields stride 2 and effective 15 fps.
- `source_fingerprint` uses the existing size-and-file-ends SHA-256 algorithm with
  1 MiB chunks. It is explicitly labeled and is not a complete content hash or run ID.
- `audio_sample_rate` records the requested mono WAV sample rate.
- Audio states: `not_checked`, `extracted`, `absent`, or `failed`.
- Status starts as `running` and finishes as `succeeded` or `failed`. Writes use a
  sibling temporary file and atomic replacement. Interrupted or failed final writes
  leave the previous manifest; consumers must not treat `running` as success.
- Silent video succeeds with a warning. An explicit unmatched ffmpeg audio stream
  map identifies absence; other extraction failures fail the run. Partial WAV files
  are removed when possible and never listed as usable artifacts.
- Media failures record their exception type/message and any metadata already read.
  Output errors that prevent writing a manifest propagate to the CLI. Exit code 0
  means successful ingestion, including silent video; code 1 reports failure.

## Validation and limits

Tests use generated clips with actual video/audio codecs. They cover metadata,
sampling, UTC timestamps, silent/audio inputs, repeated inputs, missing/corrupt media,
an unreadable first frame, extraction errors, incomplete audio cleanup, atomic-write
failures, schema validation, CLI errors, and malformed YAML. The initial acceptance
suite failed because the ingestion module did not exist; the malformed-YAML regression
also failed before its error handling was added.

Final validation: 123 tests passed; Ruff lint and formatting checks passed; strict
mypy passed for 29 source files; the 100% docstring-coverage gate passed.

Input validation checks container metadata and decodes the first frame. It does not
certify all frames, detect corruption later in a clip, or establish accurate timestamps
for variable-frame-rate footage. Duration and sampling follow the existing OpenCV
metadata contract. Referenced source files must remain available and unchanged.
The manifest is local to the run; database registration and resumable jobs are outside
this slice. No real-dancer accuracy claims follow from these tests.

## Next slice

Define synthetic pose scenarios and a replay interface using the existing keypoint
schemas/storage, then connect actual downstream tracking/scoring in subsequent slices.
