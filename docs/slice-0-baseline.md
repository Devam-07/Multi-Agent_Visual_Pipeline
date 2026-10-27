# Slice 0: foundation baseline

## Scope and working agreement

Follow `CLAUDE.md` as the agent instructions. Develop behavior test-first: write the
acceptance/regression test, observe the intended failure, implement, then refactor and
run the relevant quality checks. Existing behavior can receive characterization tests
that pass immediately; do not manufacture failures for already implemented features.

This slice audits the existing foundation and fixes confirmed defects. It does not
implement pose inference, scoring, or a feedback agent.

## Baseline observed on 2026-10-06

- Original suite: 97 tests passed.
- Ruff lint and format checks passed.
- Strict mypy passed for 27 source files.
- Interrogate reported 100% docstring coverage. This is documentation coverage, not
  execution coverage or proof that every function is adequately tested.
- The local virtual environment works, but its Python launcher needs access to the
  interpreter outside the workspace sandbox. `uv` was not found on this shell's PATH.
- Most project files were already untracked at the start; no commit was created.

## Confirmed defect and regression evidence

`load_pipeline_config` used `yaml.safe_load(...) or {}`. Consequently, `false`, `0`,
`[]`, and an empty-string scalar silently selected defaults instead of failing model
validation. Four new parametrized regression cases failed with "DID NOT RAISE"
before the fix. Only `None` now selects the empty default mapping; empty documents
and empty mappings retain their existing behavior. The focused checks then passed.

Final validation: 105 tests passed; Ruff lint, formatting, strict mypy, and the
100% docstring-coverage gate all passed.

## Small acceptance fixture

`tests/acceptance/test_media_foundation.py` generates a two-second, 30 fps clip with
moving squares and 120 bpm audio. It uses real video/audio codecs to verify:

- 15 fps analysis produces 30 frames with the expected source indices and timestamps.
- Extracted audio is mono at 22050 Hz and agrees with video duration within 100 ms.
- Energy remains concentrated around the known half-second beats.

This is an end-to-end test of the current media foundation, not the future full
analysis pipeline. Squares cannot validate human pose estimation. Real dancer clips
and labeled measurements are still needed for tracking and scoring evaluation.

## Audit observations for subsequent slices

- Existing tests exercise file round trips, numerical synthetic data, schema validation,
  CLI behavior, and lazy imports. Passing tests do not establish real-video accuracy.
- `pose`, `audio`, `sync`, `formation`, `agents`, and `viz` are package placeholders.
- `configs/bytetrack_dance.yaml` is referenced but absent; supply it with the tracking slice.
- The default YAML lookup assumes a repository checkout. Verify installed-wheel behavior
  before distributing the CLI and decide how to package its configuration resource.
- `hash_video_file` intentionally samples the file ends and size. It is a fast calibration
  cache fingerprint, not a complete content hash; the ingestion contract must distinguish them.
- Smoothing parameter relationships and tempo-bound ordering are documented but not
  fully validated by configuration models. Define these invariants with their algorithms.

## Next slice contract to develop

Ingestion should validate the input, record metadata and a run identity, and persist
an artifact manifest. Its tests should cover corrupt/missing media, absent audio,
timestamps, repeated inputs, and failure reporting. Specify the run/output contract
before adding the `analyze` command; downstream analysis stages remain separate slices.
