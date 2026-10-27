"""Tests for formsync.io.parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from formsync.io.parquet import (
    KEYPOINT_NAMES,
    KP_DIM,
    LR_SWAP_INDICES,
    N_KEYPOINTS,
    UNASSIGNED_SLOT,
    KeypointTable,
    from_dense,
    read_keypoints,
    to_dense,
    write_keypoints,
)


def _table(n: int = 6, seed: int = 0) -> KeypointTable:
    """Random valid table with ``n`` rows: two frames, three slots, no duplicates."""
    rng = np.random.default_rng(seed)
    frames = np.repeat(np.arange(n // 3), 3)[:n]
    slots = np.tile(np.arange(3), n // 3 + 1)[:n]
    return KeypointTable(
        frame_idx=frames,
        t=frames / 15.0,
        track_id=slots + 100,
        slot=slots,
        bbox=rng.uniform(0, 500, (n, 4)),
        det_conf=rng.uniform(0.5, 1, n),
        kp=rng.uniform(0, 1000, (n, N_KEYPOINTS, KP_DIM)),
    )


def test_keypoint_table_constants() -> None:
    """COCO-17 layout: 17 names, 3 values each."""
    assert len(KEYPOINT_NAMES) == N_KEYPOINTS == 17
    assert KP_DIM == 3
    assert KEYPOINT_NAMES[9] == "left_wrist" and KEYPOINT_NAMES[16] == "right_ankle"


def test_keypoint_table_post_init_casts_and_validates() -> None:
    """Construction casts dtypes, reshapes kp, and rejects mismatched row counts."""
    t = _table()
    assert t.frame_idx.dtype == np.int32 and t.slot.dtype == np.int16
    assert t.kp.shape == (6, 17, 3) and t.kp.dtype == np.float32
    assert len(t) == 6
    with pytest.raises(ValueError):
        KeypointTable(
            frame_idx=[0, 1],
            t=[0.0],
            track_id=[0, 1],
            slot=[0, 1],
            bbox=np.zeros((2, 4)),
            det_conf=[1, 1],
            kp=np.zeros((2, 17, 3)),
        )


def test_keypoint_table_empty_and_concat() -> None:
    """empty() has zero rows; concat joins rows in order and handles the empty list."""
    e = KeypointTable.empty()
    assert len(e) == 0 and e.kp.shape == (0, 17, 3)
    a, b = _table(3, seed=1), _table(6, seed=2)
    c = KeypointTable.concat([a, b])
    assert len(c) == 9
    np.testing.assert_array_equal(c.kp[:3], a.kp)
    np.testing.assert_array_equal(c.kp[3:], b.kp)
    assert len(KeypointTable.concat([])) == 0


def test_keypoint_table_select_and_sorted_by_frame() -> None:
    """select keeps masked rows; sorted_by_frame orders by frame then slot."""
    t = _table()
    sub = t.select(t.slot == 1)
    assert len(sub) == 2 and set(sub.slot.tolist()) == {1}
    shuffled = t.select(np.array([5, 2, 0, 4, 1, 3]))
    s = shuffled.sorted_by_frame()
    assert s.frame_idx.tolist() == [0, 0, 0, 1, 1, 1]
    assert s.slot.tolist() == [0, 1, 2, 0, 1, 2]


def test_keypoint_table_to_pandas() -> None:
    """to_pandas flattens keypoints into named columns."""
    df = _table().to_pandas()
    assert len(df) == 6
    assert {"frame_idx", "slot", "x1", "left_wrist_x", "right_ankle_c"} <= set(df.columns)


def test_write_and_read_keypoints_round_trip(tmp_path: Path) -> None:
    """A table survives Parquet round-trip exactly; missing file raises."""
    t = _table(9, seed=3)
    p = write_keypoints(t, tmp_path / "sub" / "kp.parquet")
    back = read_keypoints(p)
    for name in ("frame_idx", "t", "track_id", "slot", "bbox", "det_conf", "kp"):
        np.testing.assert_array_equal(getattr(back, name), getattr(t, name))
    with pytest.raises(FileNotFoundError):
        read_keypoints(tmp_path / "missing.parquet")


def test_write_and_read_empty_table(tmp_path: Path) -> None:
    """An empty table round-trips to an empty table with correct shapes."""
    p = write_keypoints(KeypointTable.empty(), tmp_path / "empty.parquet")
    back = read_keypoints(p)
    assert len(back) == 0 and back.kp.shape == (0, 17, 3)


def test_to_dense_fills_nan_and_skips_unassigned() -> None:
    """Missing (slot, frame) cells are NaN; unassigned rows are ignored."""
    t = _table()  # frames 0,1 x slots 0,1,2
    t.slot[5] = UNASSIGNED_SLOT  # drop slot 2 in frame 1
    dense = to_dense(t, n_slots=4, n_frames=3)
    assert dense.shape == (4, 3, 17, 3)
    np.testing.assert_array_equal(dense[0, 0], t.kp[0])
    assert np.isnan(dense[2, 1]).all()  # unassigned
    assert np.isnan(dense[3]).all()  # slot never seen
    assert np.isnan(dense[:, 2]).all()  # frame never seen


def test_to_dense_rejects_out_of_range_and_duplicates() -> None:
    """Out-of-range indices and duplicate (slot, frame) pairs are errors."""
    t = _table()
    with pytest.raises(ValueError):
        to_dense(t, n_slots=2, n_frames=2)
    with pytest.raises(ValueError):
        to_dense(t, n_slots=3, n_frames=1)
    dup = _table()
    dup.slot[1] = 0  # frame 0 now has slot 0 twice
    with pytest.raises(ValueError):
        to_dense(dup, n_slots=3, n_frames=2)


def test_from_dense_inverts_to_dense() -> None:
    """from_dense(to_dense(t)) reproduces frame, slot and keypoints of every present row."""
    t = _table(9, seed=4).sorted_by_frame()
    dense = to_dense(t, n_slots=3, n_frames=3)
    back = from_dense(dense, t=np.arange(3) / 15.0)
    assert len(back) == 9
    np.testing.assert_array_equal(back.frame_idx, t.frame_idx)
    np.testing.assert_array_equal(back.slot, t.slot)
    np.testing.assert_array_equal(back.kp, t.kp)
    np.testing.assert_allclose(back.bbox[:, 0], t.kp[:, :, 0].min(axis=1))
    with pytest.raises(ValueError):
        from_dense(dense, t=np.zeros(2))


def test_lr_swap_indices_is_an_involution_pairing_left_and_right() -> None:
    """Applying the swap twice is the identity, and every left_* maps to its right_* twin."""
    idx = np.array(LR_SWAP_INDICES)
    np.testing.assert_array_equal(idx[idx], np.arange(N_KEYPOINTS))
    for j, name in enumerate(KEYPOINT_NAMES):
        twin = KEYPOINT_NAMES[LR_SWAP_INDICES[j]]
        expected = (
            name.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")
        )
        assert twin == expected
