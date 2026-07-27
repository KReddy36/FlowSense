"""Data contracts shared by the detector and tracker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """One object detection in image-pixel coordinates."""

    frame_id: int
    timestamp: float
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("x2/y2 must be greater than x1/y1")
        if not self.class_name.strip():
            raise ValueError("class_name cannot be blank")

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detection after ByteTrack assigns a persistent identity."""

    frame_id: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0
