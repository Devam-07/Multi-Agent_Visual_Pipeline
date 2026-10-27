"""Synthetic dancers with known ground truth, for testing the sync and formation stages.

The motion model is **keyframe hits**, matching a hard-hitting choreography style: each group
of dancers shares a non-repeating sequence of hit times; at each hit the limbs move quickly
(over ``hit_duration_s``) to a new random pose and then freeze until the next hit. Floor
positions move between formations with smooth (smoothstep) transitions. On top of that,
per-dancer knobs create the "answers" the pipeline must recover:

* ``offset_s`` shifts every hit of that dancer (positive = late),
* ``amp_scale`` scales how far their limbs travel,
* ``group`` puts them in a different choreographic layer (different hits and poses),
* ``mirror`` flips their limb motion left-right,
* ``settle_delay_s`` makes them arrive late (or early) in each formation transition,
* ``torso_px`` varies body size, like real height differences.

The output is a real :class:`~formsync.io.parquet.KeypointTable` (the same type the pose stage
produces) plus a :class:`SyntheticGroundTruth` holding every chosen value.

Example:
    >>> scene = make_dancers(4, duration_s=6, fps=15, seed=1)
    >>> scene.truth.n_frames
    90
    >>> scene.dense().shape
    (4, 90, 17, 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from formsync.io.parquet import (
    KP_DIM,
    LR_SWAP_INDICES,
    N_KEYPOINTS,
    KeypointTable,
    from_dense,
    to_dense,
)

Easing = Literal["smoothstep", "linear"]

# Neutral standing pose in torso units, relative to the mid-hip point; y grows downward.
# Index order is COCO-17 (see formsync.io.parquet.KEYPOINT_NAMES).
BASE_SKELETON: np.ndarray = np.array(
    [
        [0.00, -1.45],  # nose
        [-0.06, -1.52],  # left eye
        [0.06, -1.52],  # right eye
        [-0.14, -1.48],  # left ear
        [0.14, -1.48],  # right ear
        [-0.35, -1.00],  # left shoulder
        [0.35, -1.00],  # right shoulder
        [-0.55, -0.55],  # left elbow
        [0.55, -0.55],  # right elbow
        [-0.60, -0.10],  # left wrist
        [0.60, -0.10],  # right wrist
        [-0.20, 0.00],  # left hip
        [0.20, 0.00],  # right hip
        [-0.22, 0.90],  # left knee
        [0.22, 0.90],  # right knee
        [-0.22, 1.80],  # left ankle
        [0.22, 1.80],  # right ankle
    ],
    dtype=np.float32,
)
"""Neutral pose, ``(17, 2)`` in torso units relative to the mid-hip."""

ANKLE_DROP = 1.80
"""Vertical distance (torso units) from mid-hip down to the ankle midpoint (the floor point)."""

DEFAULT_JOINT_AMPLITUDES: dict[int, float] = {
    0: 0.05,
    1: 0.05,
    2: 0.05,
    3: 0.05,
    4: 0.05,  # head
    5: 0.05,
    6: 0.05,  # shoulders
    7: 0.35,
    8: 0.35,  # elbows
    9: 0.60,
    10: 0.60,  # wrists
    13: 0.15,
    14: 0.15,  # knees
}
"""How far each joint may move per hit, in torso units. Hips and ankles stay put by default."""


@dataclass(frozen=True)
class DancerSpec:
    """Ground-truth knobs for one synthetic dancer.

    Attributes:
        slot: Dancer slot (0-based, unique).
        group: Choreographic layer; dancers in the same group share hits and poses.
        offset_s: Time shift applied to every hit. **Positive = late.**
        amp_scale: Multiplier on limb travel distance (1.0 = same as the group).
        mirror: If True, limb motion is mirrored left-right.
        settle_delay_s: Extra time to arrive in each formation (negative = arrives early).
        torso_px: Torso length in pixels (body size).
    """

    slot: int
    group: str = "main"
    offset_s: float = 0.0
    amp_scale: float = 1.0
    mirror: bool = False
    settle_delay_s: float = 0.0
    torso_px: float = 80.0


@dataclass(frozen=True)
class Formation:
    """One formation: a floor point per dancer and how long it is held.

    Attributes:
        positions: ``(x, y)`` floor point in pixels for each dancer, indexed by dancer order.
        hold_until_s: Time at which the transition to the *next* formation starts. Ignored for
            the last formation.
    """

    positions: tuple[tuple[float, float], ...]
    hold_until_s: float


@dataclass
class SyntheticGroundTruth:
    """Everything that was chosen when generating a scene.

    Attributes:
        fps: Analysis frame rate.
        n_frames: Number of frames.
        t: ``(n_frames,)`` frame times in seconds.
        tempo_bpm: Tempo used to lay out hits.
        beat_times: ``(n_beats,)`` beat times in seconds.
        dancers: The specs, in slot order.
        hit_duration_s: How long each hit's motion lasts (start to stop).
        group_hit_stops: Nominal (offset-free) stop time of every hit, per group.
        dancer_hit_stops: Stop times per slot, including that dancer's offset.
        dancer_hit_starts: Start times per slot (``stop - hit_duration_s``).
        formations: Formations used.
        transition_s: Nominal transition duration.
        nominal_arrivals_s: Nominal arrival time of each transition (group, no delays).
        floor_positions: ``(n_dancers, n_frames, 2)`` noise-free floor points in pixels.
        clean_kp: ``(n_dancers, n_frames, 17, 2)`` noise-free keypoint pixels.
    """

    fps: float
    n_frames: int
    t: np.ndarray
    tempo_bpm: float
    beat_times: np.ndarray
    dancers: list[DancerSpec]
    hit_duration_s: float
    group_hit_stops: dict[str, np.ndarray]
    dancer_hit_stops: dict[int, np.ndarray]
    dancer_hit_starts: dict[int, np.ndarray]
    formations: list[Formation]
    transition_s: float
    nominal_arrivals_s: list[float]
    floor_positions: np.ndarray
    clean_kp: np.ndarray


@dataclass
class SyntheticScene:
    """A generated scene: the keypoint table plus its ground truth.

    Attributes:
        table: Long-format keypoints, slots assigned, noise and dropout applied.
        truth: The ground truth used to generate it.
    """

    table: KeypointTable
    truth: SyntheticGroundTruth = field(repr=False)

    def dense(self) -> np.ndarray:
        """Return the table as a dense ``(n_dancers, n_frames, 17, 3)`` array (NaN = missing)."""
        return to_dense(self.table, n_slots=len(self.truth.dancers), n_frames=self.truth.n_frames)


def smoothstep(p: np.ndarray | float) -> np.ndarray:
    """Smooth ease-in-out curve ``3p^2 - 2p^3``, clipped to [0, 1] outside the unit interval.

    Speed is zero at both ends and peaks in the middle, like a person walking to a mark.

    Args:
        p: Progress value(s); anything below 0 maps to 0 and above 1 maps to 1.

    Returns:
        Eased progress, same shape as ``p``.
    """
    q = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return q * q * (3.0 - 2.0 * q)


def hit_curve(p: np.ndarray | float, easing: Easing = "smoothstep") -> np.ndarray:
    """Progress curve for a hit's motion from the previous pose to the new one.

    Args:
        p: Progress through the hit, 0 at the start and 1 at the stop.
        easing: ``"smoothstep"`` (default) or ``"linear"`` (constant speed, hard edges).

    Returns:
        Eased progress in [0, 1].

    Raises:
        ValueError: For an unknown easing name.
    """
    if easing == "smoothstep":
        return smoothstep(p)
    if easing == "linear":
        return np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    raise ValueError(f"unknown easing {easing!r}")


def generate_hit_times(
    rng: np.random.Generator,
    duration_s: float,
    tempo_bpm: float,
    spacings_beats: tuple[float, ...] = (1.0, 1.5, 2.0, 0.75),
    first_beat: float = 1.0,
    end_margin_s: float = 0.5,
) -> np.ndarray:
    """Lay out a non-repeating sequence of hit (stop) times along the beat grid.

    Successive hits are separated by a randomly chosen spacing from ``spacings_beats``, so hits
    land on beats, half-beats, three-quarter beats and so on, never as a fixed repeating step.

    Args:
        rng: Random generator (pass a seeded one for reproducibility).
        duration_s: Clip length; no hit is placed after ``duration_s - end_margin_s``.
        tempo_bpm: Tempo defining the beat length.
        spacings_beats: Candidate gaps between consecutive hits, in beats.
        first_beat: Beat index of the first hit.
        end_margin_s: Keep this much silence at the end.

    Returns:
        ``(n_hits,)`` increasing stop times in seconds (possibly empty for very short clips).
    """
    beat_s = 60.0 / tempo_bpm
    times: list[float] = []
    beat = first_beat
    limit = duration_s - end_margin_s
    while beat * beat_s <= limit:
        times.append(beat * beat_s)
        beat += float(rng.choice(spacings_beats))
    return np.asarray(times, dtype=np.float64)


def generate_poses(
    rng: np.random.Generator,
    n_hits: int,
    torso_px: float = 1.0,
    joint_amplitudes: dict[int, float] | None = None,
) -> np.ndarray:
    """Draw a random target pose (joint displacement from neutral) for each hit.

    Args:
        rng: Random generator.
        n_hits: Number of poses to draw.
        torso_px: Scale to convert torso units to pixels (use 1.0 to keep torso units).
        joint_amplitudes: Max displacement per joint index in torso units; joints not listed do
            not move. Defaults to :data:`DEFAULT_JOINT_AMPLITUDES`.

    Returns:
        ``(n_hits, 17, 2)`` displacements in pixels (or torso units if ``torso_px == 1``).
    """
    amps = DEFAULT_JOINT_AMPLITUDES if joint_amplitudes is None else joint_amplitudes
    poses = np.zeros((n_hits, N_KEYPOINTS, 2), dtype=np.float64)
    for j, amp in amps.items():
        poses[:, j, :] = rng.uniform(-amp, amp, size=(n_hits, 2)) * torso_px
    return poses


def limb_trajectory(
    t: np.ndarray,
    hit_stops: np.ndarray,
    hit_duration_s: float,
    poses: np.ndarray,
    easing: Easing = "smoothstep",
) -> np.ndarray:
    """Limb displacement over time for a keyframe-hit sequence.

    Before the first hit the pose is neutral (zero displacement). During the ``hit_duration_s``
    window ending at each stop time, the pose eases from the previous pose to that hit's pose;
    between hits it holds still.

    Args:
        t: ``(n_frames,)`` frame times in seconds.
        hit_stops: ``(n_hits,)`` increasing stop times.
        hit_duration_s: Motion duration of each hit; must be positive.
        poses: ``(n_hits, 17, 2)`` target displacements per hit.
        easing: Progress curve, see :func:`hit_curve`.

    Returns:
        ``(n_frames, 17, 2)`` displacements.

    Raises:
        ValueError: If ``hit_duration_s`` is not positive or the shapes disagree.
    """
    if hit_duration_s <= 0:
        raise ValueError("hit_duration_s must be positive")
    if len(hit_stops) != len(poses):
        raise ValueError("hit_stops and poses must have the same length")
    out = np.zeros((len(t), N_KEYPOINTS, 2), dtype=np.float64)
    prev = np.zeros((N_KEYPOINTS, 2), dtype=np.float64)
    for stop, pose in zip(hit_stops, poses, strict=True):
        start = stop - hit_duration_s
        progress = hit_curve((t - start) / hit_duration_s, easing)  # 0 before start, 1 after stop
        active = t > start
        out[active] = prev + progress[active, None, None] * (pose - prev)
        prev = pose
    return out


def floor_trajectory(
    t: np.ndarray,
    positions: np.ndarray,
    hold_untils: np.ndarray,
    transition_s: float,
    settle_delay_s: float = 0.0,
) -> np.ndarray:
    """Floor position over time through a list of formations.

    Transition ``k`` starts at ``hold_untils[k]`` and lasts ``transition_s + settle_delay_s``,
    easing (smoothstep) from ``positions[k]`` to ``positions[k + 1]``.

    Args:
        t: ``(n_frames,)`` frame times.
        positions: ``(n_formations, 2)`` floor point per formation.
        hold_untils: ``(n_formations,)`` transition start times (last entry unused).
        transition_s: Nominal transition duration.
        settle_delay_s: Per-dancer stretch (positive) or shrink (negative) of the transition.

    Returns:
        ``(n_frames, 2)`` positions.

    Raises:
        ValueError: If the effective transition duration is not positive, or if fewer than one
            formation is given.
    """
    if len(positions) == 0:
        raise ValueError("at least one formation is required")
    duration = transition_s + settle_delay_s
    if len(positions) > 1 and duration <= 0:
        raise ValueError("transition_s + settle_delay_s must be positive")
    out = np.tile(np.asarray(positions[0], dtype=np.float64), (len(t), 1))
    for k in range(len(positions) - 1):
        start = float(hold_untils[k])
        progress = smoothstep((t - start) / duration)
        active = t > start
        out[active] = positions[k] + progress[active, None] * (positions[k + 1] - positions[k])
    return out


def assemble_keypoints(
    floor: np.ndarray,
    torso_px: float,
    limb: np.ndarray,
    mirror: bool = False,
) -> np.ndarray:
    """Combine a floor path, body size and limb displacements into pixel keypoints.

    The mid-hip sits ``ANKLE_DROP * torso_px`` above the floor point; the neutral skeleton is
    scaled by ``torso_px`` and the limb displacement (already in pixels) is added.

    Args:
        floor: ``(n_frames, 2)`` floor (ankle midpoint) positions in pixels.
        torso_px: Torso length in pixels.
        limb: ``(n_frames, 17, 2)`` limb displacements in pixels.
        mirror: If True, mirror the displacements left-right (swap L/R joints, negate x).

    Returns:
        ``(n_frames, 17, 2)`` keypoint pixel coordinates.
    """
    if mirror:
        limb = limb[:, list(LR_SWAP_INDICES), :].copy()
        limb[..., 0] *= -1
    mid_hip = np.asarray(floor, dtype=np.float64) - np.array([0.0, ANKLE_DROP * torso_px])
    kp: np.ndarray = mid_hip[:, None, :] + BASE_SKELETON[None] * torso_px + limb
    return kp


def add_noise_and_dropout(
    kp_clean: np.ndarray,
    rng: np.random.Generator,
    noise_px: float = 0.0,
    dropout_frac: float = 0.0,
    joint_dropout_frac: float = 0.0,
    base_conf: float = 0.9,
) -> np.ndarray:
    """Turn clean keypoint coordinates into pose-model-like output with confidence values.

    Args:
        kp_clean: ``(n_frames, 17, 2)`` exact pixel coordinates.
        rng: Random generator.
        noise_px: Standard deviation of Gaussian jitter added to every coordinate.
        dropout_frac: Fraction of frames in which the dancer is not detected at all (all NaN).
        joint_dropout_frac: Fraction of individual joints that are "occluded": confidence drops
            to ~0.1 and the coordinate gets large noise, like a real bad estimate.
        base_conf: Confidence assigned to good joints (small jitter added).

    Returns:
        ``(n_frames, 17, 3)`` float32 array of ``x, y, conf`` with NaN rows for dropped frames.
    """
    n_frames = kp_clean.shape[0]
    kp = np.empty((n_frames, N_KEYPOINTS, KP_DIM), dtype=np.float32)
    xy = kp_clean + rng.normal(0.0, noise_px, size=kp_clean.shape) if noise_px > 0 else kp_clean
    kp[..., :2] = xy
    kp[..., 2] = np.clip(base_conf + rng.normal(0.0, 0.03, size=(n_frames, N_KEYPOINTS)), 0, 1)
    if joint_dropout_frac > 0:
        bad = rng.random((n_frames, N_KEYPOINTS)) < joint_dropout_frac
        kp[bad, 2] = np.clip(rng.uniform(0.02, 0.2, size=int(bad.sum())), 0, 1)
        kp[bad, :2] += rng.normal(0.0, 40.0, size=(int(bad.sum()), 2)).astype(np.float32)
    if dropout_frac > 0:
        dropped = rng.random(n_frames) < dropout_frac
        kp[dropped] = np.nan
    return kp


def make_dancers(
    dancers: int | list[DancerSpec],
    duration_s: float = 10.0,
    fps: float = 15.0,
    tempo_bpm: float = 120.0,
    hit_duration_s: float = 0.2,
    hit_spacings_beats: tuple[float, ...] = (1.0, 1.5, 2.0, 0.75),
    formations: list[Formation] | None = None,
    transition_s: float = 1.5,
    frame_size: tuple[int, int] = (1920, 1080),
    noise_px: float = 0.0,
    dropout_frac: float = 0.0,
    joint_dropout_frac: float = 0.0,
    easing: Easing = "smoothstep",
    seed: int = 0,
) -> SyntheticScene:
    """Generate a synthetic scene of dancers doing keyframe hits with known ground truth.

    Args:
        dancers: Either a number of default dancers (one group, no offsets, spread in a line)
            or explicit :class:`DancerSpec` objects with slots ``0..n-1``.
        duration_s: Clip length in seconds.
        fps: Analysis frame rate.
        tempo_bpm: Tempo used to lay out hits and the returned beat grid.
        hit_duration_s: Motion duration of each hit.
        hit_spacings_beats: Candidate gaps between hits, in beats (see :func:`generate_hit_times`).
        formations: Formations to move through; ``None`` = one static line formation.
        transition_s: Nominal formation transition duration.
        frame_size: ``(width, height)`` in pixels, used for the default formation and bounds.
        noise_px: Keypoint jitter standard deviation.
        dropout_frac: Fraction of frames each dancer goes undetected.
        joint_dropout_frac: Fraction of joints that get occluded-style bad estimates.
        easing: Hit motion curve.
        seed: Random seed; the same seed always produces the same scene.

    Returns:
        A :class:`SyntheticScene`.

    Raises:
        ValueError: If dancer slots are not ``0..n-1`` or a formation has the wrong size.
    """
    rng = np.random.default_rng(seed)
    specs = (
        [DancerSpec(slot=i) for i in range(dancers)] if isinstance(dancers, int) else list(dancers)
    )
    n = len(specs)
    if sorted(d.slot for d in specs) != list(range(n)):
        raise ValueError("dancer slots must be exactly 0..n-1")
    specs.sort(key=lambda d: d.slot)

    n_frames = int(round(duration_s * fps))
    t = np.arange(n_frames, dtype=np.float64) / fps
    beat_times = np.arange(0.0, duration_s, 60.0 / tempo_bpm)

    width, height = frame_size
    if formations is None:
        xs = np.linspace(0.15 * width, 0.85 * width, n) if n > 1 else np.array([0.5 * width])
        formations = [
            Formation(
                positions=tuple((float(x), 0.85 * height) for x in xs), hold_until_s=duration_s
            )
        ]
    for f in formations:
        if len(f.positions) != n:
            raise ValueError(f"formation has {len(f.positions)} positions for {n} dancers")
    hold_untils = np.array([f.hold_until_s for f in formations], dtype=np.float64)
    nominal_arrivals = [float(h + transition_s) for h in hold_untils[:-1]]

    # One hit sequence and one pose sequence per group (in torso units; scaled per dancer).
    group_hits: dict[str, np.ndarray] = {}
    group_poses: dict[str, np.ndarray] = {}
    for g in sorted({d.group for d in specs}):
        stops = generate_hit_times(rng, duration_s, tempo_bpm, hit_spacings_beats)
        group_hits[g] = stops
        group_poses[g] = generate_poses(rng, len(stops))

    floor_all = np.zeros((n, n_frames, 2), dtype=np.float64)
    clean_all = np.zeros((n, n_frames, N_KEYPOINTS, 2), dtype=np.float64)
    dense = np.empty((n, n_frames, N_KEYPOINTS, KP_DIM), dtype=np.float32)
    dancer_stops: dict[int, np.ndarray] = {}
    dancer_starts: dict[int, np.ndarray] = {}
    for i, d in enumerate(specs):
        stops = group_hits[d.group] + d.offset_s
        dancer_stops[d.slot] = stops
        dancer_starts[d.slot] = stops - hit_duration_s
        poses_px = group_poses[d.group] * d.amp_scale * d.torso_px
        limb = limb_trajectory(t, stops, hit_duration_s, poses_px, easing)
        positions = np.array([f.positions[i] for f in formations], dtype=np.float64)
        floor = floor_trajectory(t, positions, hold_untils, transition_s, d.settle_delay_s)
        clean = assemble_keypoints(floor, d.torso_px, limb, mirror=d.mirror)
        floor_all[i], clean_all[i] = floor, clean
        dense[i] = add_noise_and_dropout(clean, rng, noise_px, dropout_frac, joint_dropout_frac)

    table = from_dense(dense, t)
    truth = SyntheticGroundTruth(
        fps=fps,
        n_frames=n_frames,
        t=t,
        tempo_bpm=tempo_bpm,
        beat_times=beat_times,
        dancers=specs,
        hit_duration_s=hit_duration_s,
        group_hit_stops=group_hits,
        dancer_hit_stops=dancer_stops,
        dancer_hit_starts=dancer_starts,
        formations=list(formations),
        transition_s=transition_s,
        nominal_arrivals_s=nominal_arrivals,
        floor_positions=floor_all,
        clean_kp=clean_all,
    )
    return SyntheticScene(table=table, truth=truth)
