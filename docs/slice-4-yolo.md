# Slice 4: YOLO pose inference and inspection overlays

## Scope

`formsync poses INGESTION_MANIFEST --output NEW_DIRECTORY [--pipeline SETTINGS]`
connects pretrained YOLO11 Pose to our existing `PoseFrame`/ByteTrack pipeline. It
runs locally on CPU. Model loading is lazy, so fast tests and ordinary CLI imports
do not import torch. Official names resolve through Ultralytics, which can download
weights on first use; a local .pt path supports offline inference. Non-pose models
and predictions that do not match COCO-17 are rejected.

Model API reference: [Ultralytics pose documentation](https://docs.ultralytics.com/tasks/pose).
The default configuration uses `yolo11s-pose`, 1600-pixel inference size, and a target
15 fps. For a smaller CPU smoke test use `yolo11n-pose.pt` and `imgsz: 320` or `640`.
These settings affect quality and runtime. Detection confidence is governed by
`pose.conf`; detections below it never reach ByteTrack's low-confidence matching pass.

## Artifacts and timing

- `detections.parquet`: original-image bounding boxes, detection confidence, COCO-17
  pixel coordinates, and keypoint confidence. IDs/slots are -1 before tracking.
- `tracks.parquet`: same measurements with raw ByteTrack IDs. Unconfirmed detections
  remain -1; permanent dancer slots are not assigned.
- `stream.json`: existing replay format; includes empty frames. Detection inference
  can therefore be reused by `replay` and `track` without running the model again.
- `overlay.mp4`: sampled source images with boxes, labels, and confidence-filtered
  skeletons. Both endpoints must be confident to draw a skeleton edge. Stored
  keypoints retain all confidence values regardless of display filtering.
- `poses.json`: schema version, status, source manifest/video/fingerprint, model and
  backend version, full pipeline/tracker settings, CPU device, counts, runtime, and
  artifact names. A question-mark ID on the overlay means unassigned.

Frames use the existing constant-frame-rate contract: source index equals analysis
index times `source_frame_stride`, and time equals source index divided by
`source_fps`. The pose command's pipeline settings determine sampling; it records
the effective rate even if it differs from ingestion's sampling plan. No ROI is
applied, so all coordinates remain in the full source-image frame.

The overlay is intentionally silent and uses the effective analysis fps. For videos
whose frame count is not divisible by the stride, the final sample extends playback
by less than one analysis-frame interval. Variable-frame-rate presentation timestamps
are not supported by the current metadata-based timing contract. The command checks
the number of decoded samples and the encoded overlay's frame count and frame rate.

## Failure handling and resource use

Only successful ingestion manifests are accepted. The existing sampled fingerprint
is checked before and after processing to catch source changes; it is not a full
content integrity hash. Existing output directories are refused. JSON status writes
are atomic. On failure, `poses.json` records the error and no `stream.json` is
published; partial Parquet/video files may remain and are not listed as usable.
If the filesystem cannot save status, the CLI still returns an error, but persistence
cannot be guaranteed. An interrupted run may remain `running`.

The source is decoded for inference and again for rendering. Images are not collected
in memory, but detection/track arrays are; this first implementation is not out-of-core.
Current replay bounds (90000 analysis frames, at most 240 fps) also apply here.

## Test-first evidence and validation

The initial conversion/media/overlay acceptance tests failed because
`formsync.pose.inference` did not exist. Implementation followed those failing tests.
Further tests cover partial decoding, output failures, changed source media, and CLI
errors; existing behavior also received characterization tests.

Fast tests use controlled predictions with actual video codecs to verify pixel/time
preservation, empty detections, low-confidence rendering, frame counts, playback rate,
and failure publication. A separate opt-in test uses real pretrained YOLO11n Pose and
ByteTrack on six repeated frames of Ultralytics' bundled `bus.jpg` person image. It
checks nonempty COCO-17 predictions, assigned tracks, and a one-second readable overlay.
This is an inference/integration smoke test, not an accuracy evaluation on dance footage.

Validation completed: 168 fast tests passed; the real YOLO11n/ByteTrack smoke test
passed; Ruff lint and formatting, strict mypy (37 source files), and the 100%
docstring coverage gate passed. A frame extracted from the generated overlay was
visually inspected: skeletons and raw ID labels align with the detected people.
The local preview is `runs/yolo-overlay-preview.png` (ignored generated output).

To run with already downloaded local assets:

```powershell
$env:FORMSYNC_TEST_WEIGHTS = (Resolve-Path runs/yolo-assets/yolo11n-pose.pt).Path
$env:FORMSYNC_TEST_IMAGE = (Resolve-Path .venv/Lib/site-packages/ultralytics/assets/bus.jpg).Path
uv run pytest -m slow tests/integration/test_yolo_pipeline.py
```

The real-model test skips if these environment variables are absent; it does not
silently download assets in CI. The existing ByteTrack integration tests remain
separate and require no model weights. No user dance footage was available during
this slice. Real choreography, dense crowds, occlusion, and long-term identity
continuity remain to be evaluated. Smoothing, identity stitching, beats, scoring,
and feedback are subsequent slices.
