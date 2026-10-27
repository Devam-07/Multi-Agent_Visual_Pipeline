"""Tests for formsync.testing.synthetic (the keyframe-hit dancer generator)."""

from __future__ import annotations

import numpy as np
import pytest

from formsync.io.parquet import KEYPOINT_NAMES, N_KEYPOINTS
from formsync.testing.synthetic import (
    ANKLE_DROP,
    BASE_SKELETON,
    DEFAULT_JOINT_AMPLITUDES,
    DancerSpec,
    Formation,
    SyntheticScene,
    add_noise_and_dropout,
    assemble_keypoints,
    floor_trajectory,
    generate_hit_times,
    generate_poses,
    hit_curve,
    limb_trajectory,
    make_dancers,
    smoothstep,
)

L_WRIST, R_WRIST = KEYPOINT_NAMES.index("left_wrist"), KEYPOINT_NAMES.index("right_wrist")
L_ANKLE, R_ANKLE = KEYPOINT_NAMES.index("left_ankle"), KEYPOINT_NAMES.index("right_ankle")


def test_base_skeleton_is_symmetric_and_upright() -> None:
    """The neutral pose has 17 joints, mirrored left/right, head above hips above ankles."""
    assert BASE_SKELETON.shape == (N_KEYPOINTS, 2)
    np.testing.assert_allclose(BASE_SKELETON[L_WRIST, 0], -BASE_SKELETON[R_WRIST, 0])
    assert BASE_SKELETON[0, 1] < 0 < BASE_SKELETON[L_ANKLE, 1]
    assert BASE_SKELETON[L_ANKLE, 1] == BASE_SKELETON[R_ANKLE, 1] == ANKLE_DROP
    assert L_ANKLE not in DEFAULT_JOINT_AMPLITUDES  # ankles stay put by default


def test_smoothstep_endpoints_monotone_and_clipped() -> None:
    """smoothstep maps 0->0, 0.5->0.5, 1->1, is increasing, and clips outside [0, 1]."""
    p = np.linspace(-0.5, 1.5, 41)
    s = smoothstep(p)
    assert s[0] == 0.0 and s[-1] == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)
    assert np.all(np.diff(s) >= 0)


def test_hit_curve_easings() -> None:
    """Linear easing is the identity on [0, 1]; unknown names raise."""
    p = np.linspace(0, 1, 5)
    np.testing.assert_allclose(hit_curve(p, "linear"), p)
    np.testing.assert_allclose(hit_curve(p, "smoothstep"), smoothstep(p))
    with pytest.raises(ValueError):
        hit_curve(p, "bounce")  # type: ignore[arg-type]


def test_generate_hit_times_layout_and_determinism() -> None:
    """Hits start on beat one, use only the given spacings, stay in range, repeat by seed."""
    a = generate_hit_times(np.random.default_rng(3), 10.0, 120.0, (1.0, 0.75, 1.5))
    b = generate_hit_times(np.random.default_rng(3), 10.0, 120.0, (1.0, 0.75, 1.5))
    np.testing.assert_array_equal(a, b)
    assert a[0] == pytest.approx(0.5)  # beat 1 at 120 bpm
    assert a[-1] <= 9.5
    gaps_beats = np.diff(a) / 0.5
    assert set(np.round(gaps_beats, 6)) <= {1.0, 0.75, 1.5}
    assert len(set(np.round(gaps_beats, 6))) > 1  # not a single repeated step
    assert len(generate_hit_times(np.random.default_rng(0), 0.6, 120.0)) == 0


def test_generate_poses_respects_amplitudes() -> None:
    """Listed joints move within their amplitude (scaled by torso), others stay at zero."""
    poses = generate_poses(
        np.random.default_rng(0), 50, torso_px=100.0, joint_amplitudes={L_WRIST: 0.5}
    )
    assert poses.shape == (50, N_KEYPOINTS, 2)
    assert np.abs(poses[:, L_WRIST]).max() <= 50.0
    assert np.abs(poses[:, L_WRIST]).max() > 25.0
    assert not np.any(poses[:, R_WRIST])


def test_limb_trajectory_holds_and_hits() -> None:
    """Neutral before the first hit, exact pose at each stop, still in holds, moving in hits."""
    fps = 100.0
    t = np.arange(0, 3.0, 1 / fps)
    stops = np.array([1.0, 2.0])
    poses = np.zeros((2, N_KEYPOINTS, 2))
    poses[0, L_WRIST] = (10.0, 0.0)
    poses[1, L_WRIST] = (-4.0, 6.0)
    limb = limb_trajectory(t, stops, hit_duration_s=0.2, poses=poses)
    assert not np.any(limb[t <= 0.8])  # neutral before first start
    np.testing.assert_allclose(limb[np.argmin(np.abs(t - 1.0)), L_WRIST], (10.0, 0.0), atol=1e-9)
    hold = limb[(t > 1.0) & (t < 1.8)]
    np.testing.assert_allclose(hold, np.broadcast_to(hold[0], hold.shape))  # frozen in hold
    mid = limb[np.argmin(np.abs(t - 0.9)), L_WRIST]
    assert 0 < mid[0] < 10.0  # halfway through the first hit
    np.testing.assert_allclose(limb[-1, L_WRIST], (-4.0, 6.0))
    with pytest.raises(ValueError):
        limb_trajectory(t, stops, hit_duration_s=0.0, poses=poses)
    with pytest.raises(ValueError):
        limb_trajectory(t, stops, hit_duration_s=0.2, poses=poses[:1])


def test_floor_trajectory_transitions_and_settle_delay() -> None:
    """Positions hold, ease between formations, and a settle delay stretches the arrival."""
    t = np.arange(0, 6.0, 0.01)
    positions = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0]])
    hold_untils = np.array([1.0, 3.0, 6.0])
    on_time = floor_trajectory(t, positions, hold_untils, transition_s=1.0)
    assert not np.any(on_time[t <= 1.0])
    np.testing.assert_allclose(on_time[np.argmin(np.abs(t - 1.5))], (50.0, 0.0), atol=1e-6)
    held = on_time[(t >= 2.0) & (t <= 3.0)]
    np.testing.assert_allclose(held, np.broadcast_to((100.0, 0.0), held.shape), atol=1e-9)
    np.testing.assert_allclose(on_time[-1], (100.0, 50.0), atol=1e-9)
    late = floor_trajectory(t, positions, hold_untils, transition_s=1.0, settle_delay_s=0.5)
    assert late[np.argmin(np.abs(t - 2.0)), 0] < 100.0  # not there yet at the nominal arrival
    np.testing.assert_allclose(late[np.argmin(np.abs(t - 2.5))], (100.0, 0.0), atol=1e-6)
    single = floor_trajectory(t, positions[:1], hold_untils[:1], transition_s=1.0)
    assert not np.any(single)
    with pytest.raises(ValueError):
        floor_trajectory(t, positions, hold_untils, transition_s=1.0, settle_delay_s=-1.0)
    with pytest.raises(ValueError):
        floor_trajectory(t, positions[:0], hold_untils[:0], transition_s=1.0)


def test_assemble_keypoints_floor_point_and_mirror() -> None:
    """The ankle midpoint equals the floor point; mirroring swaps and negates limb x."""
    floor = np.array([[500.0, 900.0], [510.0, 900.0]])
    limb = np.zeros((2, N_KEYPOINTS, 2))
    limb[:, L_WRIST] = (30.0, -10.0)
    kp = assemble_keypoints(floor, torso_px=80.0, limb=limb)
    ankle_mid = (kp[:, L_ANKLE] + kp[:, R_ANKLE]) / 2
    np.testing.assert_allclose(ankle_mid, floor)
    np.testing.assert_allclose(
        kp[0, L_WRIST] - kp[0, R_WRIST],
        BASE_SKELETON[L_WRIST] * 80 - BASE_SKELETON[R_WRIST] * 80 + (30.0, -10.0),
    )
    mirrored = assemble_keypoints(floor, torso_px=80.0, limb=limb, mirror=True)
    # The displacement moved to the right wrist with x negated; left wrist is now neutral.
    np.testing.assert_allclose(mirrored[0, R_WRIST] - kp[0, R_WRIST], (-30.0, -10.0))
    np.testing.assert_allclose(mirrored[0, L_WRIST], kp[0, L_WRIST] - (30.0, -10.0))


def test_add_noise_and_dropout_statistics() -> None:
    """Noise has the requested spread, dropout removes the requested fraction, conf in [0, 1]."""
    rng = np.random.default_rng(0)
    clean = np.zeros((2000, N_KEYPOINTS, 2))
    kp = add_noise_and_dropout(clean, rng, noise_px=3.0, dropout_frac=0.1, joint_dropout_frac=0.05)
    assert kp.shape == (2000, N_KEYPOINTS, 3) and kp.dtype == np.float32
    dropped = np.isnan(kp[:, 0, 0])
    assert 0.07 < dropped.mean() < 0.13
    good = kp[~dropped]
    normal_joints = good[..., 2] > 0.5
    assert 2.5 < good[..., 0][normal_joints].std() < 3.5
    assert 0.03 < (~normal_joints).mean() < 0.07
    assert good[..., 2].min() >= 0 and good[..., 2].max() <= 1
    exact = add_noise_and_dropout(clean, rng)
    np.testing.assert_array_equal(exact[..., :2], 0)


def test_make_dancers_defaults_and_truth_consistency() -> None:
    """Default scene: n dancers in one group, full coverage, truth agrees with the table."""
    scene = make_dancers(4, duration_s=6.0, fps=15.0, seed=1)
    assert isinstance(scene, SyntheticScene)
    tr = scene.truth
    assert tr.n_frames == 90 and len(tr.t) == 90 and len(tr.dancers) == 4
    assert len(scene.table) == 4 * 90  # no dropout by default
    assert set(scene.table.slot.tolist()) == {0, 1, 2, 3}
    dense = scene.dense()
    assert dense.shape == (4, 90, 17, 3) and not np.isnan(dense).any()
    np.testing.assert_allclose(dense[..., :2], tr.clean_kp, atol=1e-3)  # float32 storage
    assert list(tr.group_hit_stops) == ["main"]
    for d in tr.dancers:
        np.testing.assert_allclose(tr.dancer_hit_stops[d.slot], tr.group_hit_stops["main"])
        np.testing.assert_allclose(tr.dancer_hit_starts[d.slot], tr.dancer_hit_stops[d.slot] - 0.2)
    assert tr.beat_times[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        make_dancers([DancerSpec(slot=0), DancerSpec(slot=2)])


def test_make_dancers_offset_shifts_motion_exactly() -> None:
    """A dancer late by one frame reproduces the reference dancer's limb motion one frame later."""
    fps = 15.0
    specs = [DancerSpec(slot=0), DancerSpec(slot=1, offset_s=1 / fps)]
    tr = make_dancers(specs, duration_s=6.0, fps=fps, seed=2).truth
    rel = tr.clean_kp - tr.floor_positions[:, :, None, :]  # limb motion relative to floor point
    np.testing.assert_allclose(rel[1, 1:], rel[0, :-1], atol=1e-6)
    np.testing.assert_allclose(tr.dancer_hit_stops[1], tr.dancer_hit_stops[0] + 1 / fps)


def test_make_dancers_groups_amplitude_and_mirror() -> None:
    """Groups get different hits; amp_scale scales displacement; mirror flips it."""
    specs = [
        DancerSpec(slot=0, group="main"),
        DancerSpec(slot=1, group="main", amp_scale=0.5),
        DancerSpec(slot=2, group="main", mirror=True),
        DancerSpec(slot=3, group="accent"),
    ]
    tr = make_dancers(specs, duration_s=8.0, fps=15.0, seed=3).truth
    assert set(tr.group_hit_stops) == {"accent", "main"}
    assert not np.array_equal(tr.group_hit_stops["main"], tr.group_hit_stops["accent"])
    rel = tr.clean_kp - tr.floor_positions[:, :, None, :]
    neutral = rel[0, 0]  # frame 0 is before any hit: neutral pose
    disp = rel - neutral  # displacement from neutral
    np.testing.assert_allclose(disp[1], 0.5 * disp[0], atol=1e-6)
    np.testing.assert_allclose(disp[2][:, R_WRIST, 0], -disp[0][:, L_WRIST, 0], atol=1e-6)
    np.testing.assert_allclose(disp[2][:, R_WRIST, 1], disp[0][:, L_WRIST, 1], atol=1e-6)
    assert np.abs(disp[3] - disp[0]).max() > 1.0  # accent layer moves differently


def test_make_dancers_formations_and_settle_delay() -> None:
    """Floor points follow the formations; a delayed dancer arrives after the nominal time."""
    formations = [
        Formation(positions=((100.0, 900.0), (300.0, 900.0)), hold_until_s=2.0),
        Formation(positions=((100.0, 700.0), (300.0, 500.0)), hold_until_s=8.0),
    ]
    specs = [DancerSpec(slot=0), DancerSpec(slot=1, settle_delay_s=0.6)]
    tr = make_dancers(
        specs, duration_s=8.0, fps=15.0, formations=formations, transition_s=1.0, seed=4
    ).truth
    assert tr.nominal_arrivals_s == [3.0]
    at = lambda s: int(round(s * tr.fps))  # noqa: E731
    np.testing.assert_allclose(tr.floor_positions[:, at(1.0)], [[100, 900], [300, 900]])
    np.testing.assert_allclose(tr.floor_positions[0, at(3.0)], (100, 700), atol=1e-6)
    assert tr.floor_positions[1, at(3.0), 1] > 500.0  # slot 1 still on its way
    np.testing.assert_allclose(tr.floor_positions[1, at(3.6)], (300, 500), atol=1e-6)
    # Keypoints ride on the floor path: ankle midpoint tracks floor_positions.
    ankles = (tr.clean_kp[:, :, L_ANKLE] + tr.clean_kp[:, :, R_ANKLE]) / 2
    np.testing.assert_allclose(ankles, tr.floor_positions, atol=1e-6)
    with pytest.raises(ValueError):
        make_dancers(3, formations=formations)


def test_make_dancers_noise_and_dropout_produce_gaps() -> None:
    """With dropout the table has fewer rows and the dense array has NaN frames."""
    scene = make_dancers(3, duration_s=10.0, fps=15.0, noise_px=2.0, dropout_frac=0.2, seed=5)
    assert len(scene.table) < 3 * 150
    dense = scene.dense()
    assert np.isnan(dense[..., 0]).any()
    present = ~np.isnan(dense[..., 0, 0])
    assert 0.7 < present.mean() < 0.9
