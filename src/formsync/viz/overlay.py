"""Confidence-filtered COCO-17 skeletons for visual inspection of raw tracks."""

import cv2
import numpy as np

from formsync.io.parquet import KeypointTable

SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def draw_poses(image: np.ndarray, table: KeypointTable, confidence: float) -> np.ndarray:
    """Draw copied BGR pixels with boxes, IDs, and only sufficiently confident joints.

    Both endpoints must pass the confidence threshold to draw a skeleton edge.
    Unassigned IDs are labeled with a question mark. Coordinates outside the image
    are excluded from skeleton drawing; stored measurements remain unchanged.
    """
    out = image.copy()
    height, width = image.shape[:2]
    for box, kp, track_id in zip(table.bbox, table.kp, table.track_id, strict=True):
        color = (0, 200, 255) if track_id < 0 else (60, 220, 60)
        x1, y1, x2, y2 = np.clip(
            box, [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1]
        ).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        label = f"ID {track_id}" if track_id >= 0 else "ID ?"
        cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        valid = (
            np.isfinite(kp).all(axis=1)
            & (kp[:, 2] >= confidence)
            & (kp[:, 0] >= 0)
            & (kp[:, 0] < width)
            & (kp[:, 1] >= 0)
            & (kp[:, 1] < height)
        )
        for a, b in SKELETON:
            if valid[a] and valid[b]:
                cv2.line(out, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), color, 2)
        for joint in kp[valid]:
            cv2.circle(out, tuple(joint[:2].astype(int)), 3, color, -1)
    return out
