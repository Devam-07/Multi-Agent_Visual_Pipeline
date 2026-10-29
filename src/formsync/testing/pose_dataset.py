"""Save synthetic model-shaped detections separately from evaluation ground truth."""

import json
from pathlib import Path

import numpy as np

from formsync.io.parquet import write_keypoints
from formsync.schemas.replay import PoseStreamManifest, ScenarioConfig
from formsync.testing.synthetic import DancerSpec, make_dancers


def write_synthetic_dataset(output: Path, config: ScenarioConfig) -> Path:
    """Generate a deterministic dataset in a new directory and return its manifest.

    Reuses the keyframe-hit generator: dancers share a sequence of target poses with
    configurable time shifts, amplitudes, noise, and dropout. Detection rows are
    shuffled within each frame, then true identities are removed. The separate
    ``ground_truth.json`` maps each Parquet row to its dancer and records the exact
    configuration and hit times. Replay never reads this file.

    Refuses existing destinations to avoid mixing datasets. Writes ``stream.json``
    last as the completion marker; on failure, a partial directory can remain with
    no valid stream manifest. Repeatability means equal arrays for the same config
    and generator version, not byte-identical Parquet across library versions.
    """
    # Validate again in case a caller mutated a nested collection after construction.
    config = ScenarioConfig.model_validate(config.model_dump())
    scene = make_dancers(
        [
            DancerSpec(slot=i, offset_s=d.offset_s, amp_scale=d.amp_scale)
            for i, d in enumerate(config.dancers)
        ],
        duration_s=config.n_frames / config.fps,
        fps=config.fps,
        tempo_bpm=config.tempo_bpm,
        noise_px=config.noise_px,
        dropout_frac=config.dropout_frac,
        joint_dropout_frac=config.joint_dropout_frac,
        seed=config.seed,
    )
    rng = np.random.default_rng(config.seed)
    order = np.lexsort((rng.random(len(scene.table)), scene.table.frame_idx))
    table = scene.table.select(order)
    labels = table.slot.tolist()
    table.slot[:] = -1
    table.track_id[:] = -1
    manifest = PoseStreamManifest(n_frames=config.n_frames, fps=config.fps)
    truth = {
        "schema_version": 1,
        "generator": "keyframe-hits-v1",
        "config": config.model_dump(mode="json"),
        "row_dancer_ids": labels,
        "beat_times_s": scene.truth.beat_times.tolist(),
        "group_hit_stops_s": {k: v.tolist() for k, v in scene.truth.group_hit_stops.items()},
        "dancer_hit_stops_s": {str(k): v.tolist() for k, v in scene.truth.dancer_hit_stops.items()},
        "hit_duration_s": scene.truth.hit_duration_s,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_keypoints(table, output / manifest.detections)
    (output / "ground_truth.json").write_text(
        json.dumps(truth, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    path = output / "stream.json"
    temporary = output / "stream.json.tmp"
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
