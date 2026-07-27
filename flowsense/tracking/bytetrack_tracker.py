"""Small adapter from FlowSense detections to Supervision ByteTrack."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .schemas import Detection, TrackedDetection


class ByteTrackTracker:
    """Assign persistent IDs to frame-by-frame object detections."""

    def __init__(
        self,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "ByteTrack dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        self._sv = sv
        self._tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        self._last_frame_id: int | None = None

    def update(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        """Process exactly one frame and return detections that have track IDs."""
        if not detections:
            # ByteTrack must still advance when a frame has no detections.
            empty = self._sv.Detections.empty()
            self._tracker.update_with_detections(empty)
            return []

        frame_ids = {item.frame_id for item in detections}
        timestamps = {item.timestamp for item in detections}
        if len(frame_ids) != 1 or len(timestamps) != 1:
            raise ValueError("update() accepts detections from exactly one frame")

        frame_id = detections[0].frame_id
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("frames must be supplied in strictly increasing order")
        self._last_frame_id = frame_id

        xyxy = np.asarray([item.bbox for item in detections], dtype=np.float32)
        confidence = np.asarray(
            [item.confidence for item in detections], dtype=np.float32
        )
        class_ids = np.asarray([item.class_id for item in detections], dtype=int)
        class_names = {item.class_id: item.class_name for item in detections}

        batch = self._sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
        )
        tracked = self._tracker.update_with_detections(batch)

        results: list[TrackedDetection] = []
        if tracked.tracker_id is None:
            return results

        for box, score, class_id, track_id in zip(
            tracked.xyxy,
            tracked.confidence,
            tracked.class_id,
            tracked.tracker_id,
            strict=True,
        ):
            if track_id is None:
                continue
            results.append(
                TrackedDetection(
                    frame_id=frame_id,
                    timestamp=detections[0].timestamp,
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=class_names.get(int(class_id), f"class_{class_id}"),
                    confidence=float(score),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                )
            )
        return results

    def reset(self) -> None:
        """Clear all identities before processing a different video."""
        self._tracker.reset()
        self._last_frame_id = None
