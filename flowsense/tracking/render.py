"""Render tracked boxes and IDs on video frames."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .motion_prediction import MotionSnapshot
from .schemas import TrackedDetection


_COLORS = (
    (255, 120, 40),
    (40, 210, 255),
    (80, 220, 80),
    (230, 80, 190),
    (170, 120, 255),
)


def render_tracking_ids(
    frame: np.ndarray, tracks: Sequence[TrackedDetection]
) -> np.ndarray:
    """Return a copy of ``frame`` with bounding boxes and tracking labels."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Rendering dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    annotated = frame.copy()
    for track in tracks:
        color = _COLORS[track.track_id % len(_COLORS)]
        start = int(round(track.x1)), int(round(track.y1))
        end = int(round(track.x2)), int(round(track.y2))
        label = f"ID {track.track_id} | {track.class_name} {track.confidence:.2f}"
        cv2.rectangle(annotated, start, end, color, 2)
        cv2.putText(
            annotated,
            label,
            (start[0], max(20, start[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def render_motion_paths(
    frame: np.ndarray, snapshots: Sequence[MotionSnapshot]
) -> np.ndarray:
    """Draw solid observed paths and dashed short-term predictions."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Rendering dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    annotated = frame.copy()
    for snapshot in snapshots:
        color = _COLORS[snapshot.track_id % len(_COLORS)]
        observed_points = [
            (int(round(point.x)), int(round(point.y)))
            for point in snapshot.observed_points
        ]
        if len(observed_points) >= 2:
            cv2.polylines(
                annotated,
                [np.asarray(observed_points, dtype=np.int32)],
                False,
                color,
                3,
                cv2.LINE_AA,
            )

        predicted_points = [
            (int(round(x)), int(round(y)))
            for x, y in snapshot.predicted_points
        ]
        for index in range(0, len(predicted_points) - 1, 2):
            cv2.line(
                annotated,
                predicted_points[index],
                predicted_points[index + 1],
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.circle(annotated, predicted_points[-1], 5, color, 2, cv2.LINE_AA)
    return annotated
