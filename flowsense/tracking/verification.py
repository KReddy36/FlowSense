"""TEST-ONLY checks for stable identities in generated mock detections.

The production video pipeline does not import this module.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .schemas import TrackedDetection


BBox = tuple[float, float, float, float]


def _iou(first: BBox, second: BBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def verify_id_stability(
    tracks_by_frame: Mapping[int, Sequence[TrackedDetection]],
    truth_by_frame: Mapping[int, Mapping[str, BBox]],
    *,
    warmup_frames: int = 2,
    minimum_iou: float = 0.5,
) -> dict[str, int]:
    """Return the stable ID for each truth object or raise on an ID exchange."""
    observed_ids: dict[str, set[int]] = defaultdict(set)

    for frame_id in sorted(truth_by_frame):
        if frame_id < warmup_frames:
            continue
        available = list(tracks_by_frame.get(frame_id, ()))
        used_track_ids: set[int] = set()

        for object_name, truth_box in truth_by_frame[frame_id].items():
            candidates = [
                (_iou(truth_box, track.bbox), track)
                for track in available
                if track.track_id not in used_track_ids
            ]
            if not candidates:
                raise AssertionError(
                    f"{object_name} has no tracked detection in frame {frame_id}"
                )
            overlap, best_track = max(candidates, key=lambda item: item[0])
            if overlap < minimum_iou:
                raise AssertionError(
                    f"{object_name} has no sufficiently matching track in frame "
                    f"{frame_id} (best IoU={overlap:.3f})"
                )
            observed_ids[object_name].add(best_track.track_id)
            used_track_ids.add(best_track.track_id)

    unstable = {
        object_name: sorted(track_ids)
        for object_name, track_ids in observed_ids.items()
        if len(track_ids) != 1
    }
    if unstable:
        raise AssertionError(f"tracking IDs changed or exchanged: {unstable}")

    stable = {
        object_name: next(iter(track_ids))
        for object_name, track_ids in observed_ids.items()
    }
    if len(set(stable.values())) != len(stable):
        raise AssertionError(f"multiple objects share the same tracking ID: {stable}")
    return stable
