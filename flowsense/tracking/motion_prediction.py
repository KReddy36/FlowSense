"""Bounded per-track motion histories and short-term linear predictions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import atan2, degrees, hypot
from statistics import median

from .learned_motion import LearnedMotionCorrector
from .schemas import TrackedDetection


@dataclass(frozen=True, slots=True)
class MotionPoint:
    """One observed center point for a canonical track."""

    frame_id: int
    timestamp: float
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class MotionSnapshot:
    """Motion and prediction values for one active track in one frame."""

    frame_id: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    observed: bool
    frames_since_seen: int
    center_x: float
    center_y: float
    velocity_x: float
    velocity_y: float
    speed: float
    direction_degrees: float
    prediction_horizon_frames: int
    predicted_frame: int
    predicted_time: float
    predicted_center_x: float
    predicted_center_y: float
    prediction_scale: float
    observed_points: tuple[MotionPoint, ...]
    predicted_points: tuple[tuple[float, float], ...]


@dataclass(slots=True)
class _TrackState:
    track_id: int
    class_id: int
    class_name: str
    last_seen_frame: int
    last_seen_timestamp: float
    confidence: float
    box_width: float
    box_height: float
    points: deque[MotionPoint] = field(default_factory=deque)


class MotionPredictor:
    """Maintain active histories and predict motion with averaged velocities."""

    def __init__(
        self,
        *,
        fps: float,
        history_points: int = 30,
        velocity_window: int = 5,
        prediction_horizon_frames: int = 15,
        inactive_timeout_frames: int = 30,
        prediction_segments: int = 6,
        frame_width: float | None = None,
        frame_height: float | None = None,
        learned_corrector: LearnedMotionCorrector | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if history_points < 2:
            raise ValueError("history_points must be at least 2")
        if velocity_window < 1:
            raise ValueError("velocity_window must be positive")
        if prediction_horizon_frames < 1:
            raise ValueError("prediction_horizon_frames must be positive")
        if inactive_timeout_frames < 0:
            raise ValueError("inactive_timeout_frames cannot be negative")
        if prediction_segments < 1:
            raise ValueError("prediction_segments must be positive")

        self.fps = fps
        self.history_points = history_points
        self.velocity_window = velocity_window
        self.prediction_horizon_frames = prediction_horizon_frames
        self.inactive_timeout_frames = inactive_timeout_frames
        self.prediction_segments = prediction_segments
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.learned_corrector = (
            learned_corrector
            if learned_corrector is not None
            and learned_corrector.compatible_with(
                horizon_frames=prediction_horizon_frames,
                velocity_window=velocity_window,
            )
            else None
        )
        self._states: dict[int, _TrackState] = {}
        self._last_frame_id: int | None = None

    @property
    def active_track_ids(self) -> frozenset[int]:
        """Return the IDs whose histories have not timed out."""
        return frozenset(self._states)

    def update(
        self,
        tracks: list[TrackedDetection],
        *,
        frame_id: int,
        timestamp: float,
    ) -> list[MotionSnapshot]:
        """Update one frame and return observed plus temporarily missing tracks."""
        if frame_id < 0 or timestamp < 0:
            raise ValueError("frame_id and timestamp must be non-negative")
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("frames must be supplied in strictly increasing order")
        if any(track.frame_id != frame_id for track in tracks):
            raise ValueError("all tracks must belong to frame_id")
        if len({track.track_id for track in tracks}) != len(tracks):
            raise ValueError("tracks must contain at most one row per track_id")
        self._last_frame_id = frame_id

        observed_ids: set[int] = set()
        for track in tracks:
            observed_ids.add(track.track_id)
            state = self._states.get(track.track_id)
            if state is None:
                state = _TrackState(
                    track_id=track.track_id,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    last_seen_frame=frame_id,
                    last_seen_timestamp=track.timestamp,
                    confidence=track.confidence,
                    box_width=track.x2 - track.x1,
                    box_height=track.y2 - track.y1,
                    points=deque(maxlen=self.history_points),
                )
                self._states[track.track_id] = state

            center_x, center_y = track.center
            state.points.append(
                MotionPoint(
                    frame_id=frame_id,
                    timestamp=track.timestamp,
                    x=center_x,
                    y=center_y,
                )
            )
            state.class_id = track.class_id
            state.class_name = track.class_name
            state.last_seen_frame = frame_id
            state.last_seen_timestamp = track.timestamp
            state.confidence = track.confidence
            state.box_width = track.x2 - track.x1
            state.box_height = track.y2 - track.y1

        expired = [
            track_id
            for track_id, state in self._states.items()
            if frame_id - state.last_seen_frame > self.inactive_timeout_frames
        ]
        for track_id in expired:
            del self._states[track_id]

        return [
            self._snapshot(
                state,
                frame_id=frame_id,
                timestamp=timestamp,
                observed=track_id in observed_ids,
            )
            for track_id, state in sorted(self._states.items())
        ]

    def _snapshot(
        self,
        state: _TrackState,
        *,
        frame_id: int,
        timestamp: float,
        observed: bool,
    ) -> MotionSnapshot:
        velocity_x, velocity_y = self._average_velocity(state.points)
        last_point = state.points[-1]
        elapsed = max(0.0, timestamp - last_point.timestamp)
        current_x = last_point.x + velocity_x * elapsed
        current_y = last_point.y + velocity_y * elapsed
        horizon_seconds = self.prediction_horizon_frames / self.fps
        baseline_dx = velocity_x * horizon_seconds
        baseline_dy = velocity_y * horizon_seconds
        applied_scale = 1.0
        velocities = self._recent_velocities(state.points)
        if (
            self.learned_corrector is not None
            and self.frame_width is not None
            and self.frame_height is not None
            and len(state.points) >= self.velocity_window + 1
        ):
            correction = self.learned_corrector.correct(
                class_name=state.class_name,
                confidence=state.confidence,
                center_x=current_x,
                center_y=current_y,
                box_width=state.box_width,
                box_height=state.box_height,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                baseline_dx=baseline_dx,
                baseline_dy=baseline_dy,
                mean_velocity_x=velocity_x,
                mean_velocity_y=velocity_y,
                velocities=velocities,
                velocity_sample_seconds=self._median_elapsed(state.points),
                history_points=len(state.points),
            )
            applied_scale = correction.applied_scale
        predicted_x = current_x + baseline_dx * applied_scale
        predicted_y = current_y + baseline_dy * applied_scale
        predicted_points = tuple(
            (
                current_x
                + (predicted_x - current_x) * step / self.prediction_segments,
                current_y
                + (predicted_y - current_y) * step / self.prediction_segments,
            )
            for step in range(self.prediction_segments + 1)
        )
        speed = hypot(velocity_x, velocity_y)
        direction = (
            (degrees(atan2(velocity_y, velocity_x)) + 360.0) % 360.0
            if speed > 0
            else 0.0
        )
        return MotionSnapshot(
            frame_id=frame_id,
            timestamp=timestamp,
            track_id=state.track_id,
            class_id=state.class_id,
            class_name=state.class_name,
            observed=observed,
            frames_since_seen=frame_id - state.last_seen_frame,
            center_x=current_x,
            center_y=current_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            speed=speed,
            direction_degrees=direction,
            prediction_horizon_frames=self.prediction_horizon_frames,
            predicted_frame=frame_id + self.prediction_horizon_frames,
            predicted_time=timestamp + horizon_seconds,
            predicted_center_x=predicted_x,
            predicted_center_y=predicted_y,
            prediction_scale=applied_scale,
            observed_points=tuple(state.points),
            predicted_points=predicted_points,
        )

    def _average_velocity(
        self, points: deque[MotionPoint]
    ) -> tuple[float, float]:
        recent = self._recent_velocities(points)
        if not recent:
            return 0.0, 0.0
        return (
            sum(velocity[0] for velocity in recent) / len(recent),
            sum(velocity[1] for velocity in recent) / len(recent),
        )

    def _recent_velocities(
        self, points: deque[MotionPoint]
    ) -> list[tuple[float, float]]:
        velocities: list[tuple[float, float]] = []
        point_list = list(points)
        for previous, current in zip(point_list, point_list[1:]):
            elapsed = current.timestamp - previous.timestamp
            if elapsed <= 0:
                continue
            velocities.append(
                (
                    (current.x - previous.x) / elapsed,
                    (current.y - previous.y) / elapsed,
                )
            )
        recent = velocities[-self.velocity_window :]
        return recent

    def _median_elapsed(self, points: deque[MotionPoint]) -> float:
        point_list = list(points)[-(self.velocity_window + 1) :]
        elapsed = [
            current.timestamp - previous.timestamp
            for previous, current in zip(point_list, point_list[1:])
            if current.timestamp > previous.timestamp
        ]
        return median(elapsed) if elapsed else 1.0 / self.fps
