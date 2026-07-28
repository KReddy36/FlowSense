"""Consolidate duplicate and briefly interrupted ByteTrack identities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from math import hypot

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


def _center(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _translate(box: BBox, dx: float, dy: float) -> BBox:
    return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy


@dataclass(slots=True)
class ConsolidationResult:
    """Canonicalized tracks and the subset that should be rendered."""

    all_tracks: list[TrackedDetection]
    visible_tracks: list[TrackedDetection]
    suppressed_tracks: list[TrackedDetection]


@dataclass(slots=True)
class _IdentityState:
    canonical_id: int
    last_frame: int
    bbox: BBox
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    class_scores: dict[tuple[int, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )

    @property
    def stable_class(self) -> tuple[int, str]:
        return max(self.class_scores, key=self.class_scores.get)


class IdentityConsolidator:
    """Map duplicate or reappearing raw tracks onto stable canonical identities."""

    def __init__(
        self,
        *,
        duplicate_iou_threshold: float = 0.85,
        duplicate_center_distance_ratio: float = 0.20,
        reidentification_iou_threshold: float = 0.20,
        reidentification_distance_ratio: float = 0.75,
        maximum_reidentification_gap: int = 15,
    ) -> None:
        if not 0.0 <= duplicate_iou_threshold <= 1.0:
            raise ValueError("duplicate_iou_threshold must be between 0 and 1")
        if duplicate_center_distance_ratio < 0:
            raise ValueError(
                "duplicate_center_distance_ratio cannot be negative"
            )
        if not 0.0 <= reidentification_iou_threshold <= 1.0:
            raise ValueError(
                "reidentification_iou_threshold must be between 0 and 1"
            )
        if reidentification_distance_ratio < 0:
            raise ValueError("reidentification_distance_ratio cannot be negative")
        if maximum_reidentification_gap < 1:
            raise ValueError("maximum_reidentification_gap must be positive")

        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.duplicate_center_distance_ratio = duplicate_center_distance_ratio
        self.reidentification_iou_threshold = reidentification_iou_threshold
        self.reidentification_distance_ratio = reidentification_distance_ratio
        self.maximum_reidentification_gap = maximum_reidentification_gap
        self._raw_to_canonical: dict[int, int] = {}
        self._aliases: dict[int, int] = {}
        self._states: dict[int, _IdentityState] = {}
        self._next_canonical_id = 1

    def update(
        self,
        tracks: list[TrackedDetection],
        *,
        frame_id: int,
    ) -> ConsolidationResult:
        """Canonicalize one frame and suppress duplicate visible detections."""
        if any(track.frame_id != frame_id for track in tracks):
            raise ValueError("all tracks must belong to frame_id")
        if not tracks:
            return ConsolidationResult([], [], [])

        clusters = self._overlap_clusters(tracks)
        cluster_records: list[
            tuple[int, list[TrackedDetection], TrackedDetection]
        ] = []
        assigned_this_frame: set[int] = set()

        clusters.sort(
            key=lambda cluster: max(track.confidence for track in cluster),
            reverse=True,
        )
        for cluster in clusters:
            representative = max(cluster, key=lambda track: track.confidence)
            mapped = {
                self._resolve(self._raw_to_canonical[track.track_id])
                for track in cluster
                if track.track_id in self._raw_to_canonical
            }
            if mapped:
                canonical_id = min(mapped)
                # If a prior reidentification mapped two different raw tracks
                # onto one identity, they can later appear together as
                # spatially separate clusters. Split the lower-confidence
                # cluster instead of alternating which vehicle is displayed.
                if canonical_id in assigned_this_frame:
                    canonical_id = self._create_identity(representative)
            else:
                match = self._find_recent_identity(
                    representative,
                    frame_id,
                    excluded=assigned_this_frame,
                )
                canonical_id = (
                    match if match is not None else self._create_identity(representative)
                )

            canonical_id = self._resolve(canonical_id)
            for track in cluster:
                self._raw_to_canonical[track.track_id] = canonical_id
            assigned_this_frame.add(canonical_id)
            cluster_records.append((canonical_id, cluster, representative))

        # Multiple raw tracks can already map to one canonical identity. Keep
        # only the strongest representative visible in the current frame.
        records_by_identity: dict[
            int, list[tuple[list[TrackedDetection], TrackedDetection]]
        ] = defaultdict(list)
        for canonical_id, cluster, representative in cluster_records:
            records_by_identity[self._resolve(canonical_id)].append(
                (cluster, representative)
            )

        all_tracks: list[TrackedDetection] = []
        visible_tracks: list[TrackedDetection] = []
        suppressed_tracks: list[TrackedDetection] = []

        for canonical_id, records in records_by_identity.items():
            _, winner = max(records, key=lambda record: record[1].confidence)
            state = self._states[canonical_id]
            state.class_scores[(winner.class_id, winner.class_name)] += winner.confidence
            stable_class_id, stable_class_name = state.stable_class

            canonical_records: list[
                tuple[list[TrackedDetection], TrackedDetection]
            ] = []
            for cluster, representative in records:
                canonical_cluster = [
                    replace(
                        track,
                        track_id=canonical_id,
                        class_id=stable_class_id,
                        class_name=stable_class_name,
                    )
                    for track in cluster
                ]
                canonical_representative = replace(
                    representative,
                    track_id=canonical_id,
                    class_id=stable_class_id,
                    class_name=stable_class_name,
                )
                canonical_records.append(
                    (canonical_cluster, canonical_representative)
                )
                all_tracks.extend(canonical_cluster)

            visible_winner = replace(
                winner,
                track_id=canonical_id,
                class_id=stable_class_id,
                class_name=stable_class_name,
            )
            visible_tracks.append(visible_winner)
            winner_hidden = False
            for canonical_cluster, canonical_representative in canonical_records:
                for track in canonical_cluster:
                    if (
                        not winner_hidden
                        and canonical_representative == visible_winner
                        and track == visible_winner
                    ):
                        winner_hidden = True
                    else:
                        suppressed_tracks.append(track)

            self._update_state(state, winner, frame_id)

        return ConsolidationResult(
            all_tracks=all_tracks,
            visible_tracks=visible_tracks,
            suppressed_tracks=suppressed_tracks,
        )

    def _overlap_clusters(
        self, tracks: list[TrackedDetection]
    ) -> list[list[TrackedDetection]]:
        parents = list(range(len(tracks)))
        canonical_ids: list[set[int]] = []
        for track in tracks:
            mapped_id = self._raw_to_canonical.get(track.track_id)
            canonical_ids.append(
                {self._resolve(mapped_id)} if mapped_id is not None else set()
            )

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root == second_root:
                return

            # Two established tracks may overlap while passing, queuing, or
            # crossing in perspective. Never permanently merge their
            # canonical identities solely because their boxes overlap.
            combined_ids = canonical_ids[first_root] | canonical_ids[second_root]
            if len(combined_ids) > 1:
                return
            parents[second_root] = first_root
            canonical_ids[first_root] = combined_ids

        for first in range(len(tracks)):
            for second in range(first + 1, len(tracks)):
                overlap = _iou(tracks[first].bbox, tracks[second].bbox)
                if overlap < self.duplicate_iou_threshold:
                    continue
                first_center = tracks[first].center
                second_center = tracks[second].center
                distance = hypot(
                    first_center[0] - second_center[0],
                    first_center[1] - second_center[1],
                )
                width = max(
                    tracks[first].x2 - tracks[first].x1,
                    tracks[second].x2 - tracks[second].x1,
                )
                height = max(
                    tracks[first].y2 - tracks[first].y1,
                    tracks[second].y2 - tracks[second].y1,
                )
                distance_ratio = distance / max(1.0, hypot(width, height))
                if distance_ratio <= self.duplicate_center_distance_ratio:
                    union(first, second)

        grouped: dict[int, list[TrackedDetection]] = defaultdict(list)
        for index, track in enumerate(tracks):
            grouped[find(index)].append(track)
        return list(grouped.values())

    def _find_recent_identity(
        self,
        track: TrackedDetection,
        frame_id: int,
        *,
        excluded: set[int],
    ) -> int | None:
        best: tuple[float, int] | None = None
        for canonical_id, state in self._states.items():
            if canonical_id in excluded:
                continue
            gap = frame_id - state.last_frame
            if gap <= 0 or gap > self.maximum_reidentification_gap:
                continue

            predicted = _translate(
                state.bbox,
                state.velocity_x * gap,
                state.velocity_y * gap,
            )
            overlap = _iou(predicted, track.bbox)
            predicted_center = _center(predicted)
            current_center = track.center
            distance = hypot(
                predicted_center[0] - current_center[0],
                predicted_center[1] - current_center[1],
            )
            width = max(predicted[2] - predicted[0], track.x2 - track.x1)
            height = max(predicted[3] - predicted[1], track.y2 - track.y1)
            distance_ratio = distance / max(1.0, hypot(width, height))

            if (
                overlap < self.reidentification_iou_threshold
                and distance_ratio > self.reidentification_distance_ratio
            ):
                continue
            score = overlap - 0.15 * distance_ratio - 0.005 * gap
            if best is None or score > best[0]:
                best = score, canonical_id
        return None if best is None else best[1]

    def _create_identity(self, track: TrackedDetection) -> int:
        canonical_id = self._next_canonical_id
        self._next_canonical_id += 1
        state = _IdentityState(
            canonical_id=canonical_id,
            last_frame=track.frame_id,
            bbox=track.bbox,
        )
        state.class_scores[(track.class_id, track.class_name)] += track.confidence
        self._states[canonical_id] = state
        return canonical_id

    def _update_state(
        self,
        state: _IdentityState,
        track: TrackedDetection,
        frame_id: int,
    ) -> None:
        gap = frame_id - state.last_frame
        if gap > 0:
            old_center = _center(state.bbox)
            new_center = track.center
            measured_x = (new_center[0] - old_center[0]) / gap
            measured_y = (new_center[1] - old_center[1]) / gap
            state.velocity_x = 0.6 * measured_x + 0.4 * state.velocity_x
            state.velocity_y = 0.6 * measured_y + 0.4 * state.velocity_y
        state.bbox = track.bbox
        state.last_frame = frame_id

    def _resolve(self, canonical_id: int) -> int:
        while canonical_id in self._aliases:
            canonical_id = self._aliases[canonical_id]
        return canonical_id
