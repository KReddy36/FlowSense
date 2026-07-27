"""TEST-ONLY: track mock vehicles, render IDs, and verify ID stability.

This development demo is not required by the final FlowSense application.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flowsense.tracking import ByteTrackTracker, Detection
from flowsense.tracking.render import render_tracking_ids
from flowsense.tracking.verification import BBox, verify_id_stability


OUTPUT_PATH = Path("outputs/day1_mock_tracking.mp4")
FRAME_SIZE = (800, 450)
FPS = 30
FRAME_COUNT = 90


def make_mock_frame(frame_id: int) -> tuple[list[Detection], dict[str, BBox]]:
    """Generate two independently moving cars; names are only verification truth."""
    timestamp = frame_id / FPS
    car_a_x = 55.0 + frame_id * 4.2
    car_b_x = 650.0 - frame_id * 3.5
    truth = {
        "eastbound_car": (car_a_x, 105.0, car_a_x + 92.0, 165.0),
        "westbound_car": (car_b_x, 285.0, car_b_x + 92.0, 345.0),
    }
    detections = [
        Detection(
            frame_id=frame_id,
            timestamp=timestamp,
            class_id=2,
            class_name="car",
            confidence=0.94,
            x1=box[0],
            y1=box[1],
            x2=box[2],
            y2=box[3],
        )
        for box in truth.values()
    ]
    # Reverse detector output order every frame. Stable IDs must come from motion
    # association, not from an object's position in the input list.
    if frame_id % 2:
        detections.reverse()
    return detections, truth


def make_background(frame_id: int) -> np.ndarray:
    """Create a road-like synthetic frame without needing input footage."""
    import cv2

    frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), (45, 48, 50), np.uint8)
    cv2.line(frame, (0, 225), (FRAME_SIZE[0], 225), (210, 210, 210), 2)
    for x in range(-80 + (frame_id * 3) % 80, FRAME_SIZE[0], 80):
        cv2.line(frame, (x, 225), (x + 42, 225), (35, 190, 240), 4)
    cv2.putText(
        frame,
        f"Mock frame {frame_id:03d}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return frame


def run_demo(output_path: Path = OUTPUT_PATH) -> dict[str, int]:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        FRAME_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    tracker = ByteTrackTracker(frame_rate=FPS)
    tracks_by_frame = {}
    truth_by_frame = {}
    try:
        for frame_id in range(FRAME_COUNT):
            detections, truth = make_mock_frame(frame_id)
            tracks = tracker.update(detections)
            tracks_by_frame[frame_id] = tracks
            truth_by_frame[frame_id] = truth
            writer.write(render_tracking_ids(make_background(frame_id), tracks))
    finally:
        writer.release()

    stable_ids = verify_id_stability(tracks_by_frame, truth_by_frame)
    print(f"PASS: no ID exchanges across {FRAME_COUNT} mock frames")
    print(f"Stable IDs: {stable_ids}")
    print(f"Annotated video: {output_path.resolve()}")
    return stable_ids


if __name__ == "__main__":
    run_demo()
