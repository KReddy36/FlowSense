"""Ultralytics YOLO adapter for FlowSense's in-memory detection contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .tracking import Detection


DEFAULT_ROAD_USER_CLASS_IDS = (0, 1, 2, 3, 5, 7)


class YoloDetector:
    """Detect supported road users in individual video frames."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        *,
        confidence: float = 0.35,
        iou: float = 0.50,
        device: str | None = None,
        class_ids: Sequence[int] = DEFAULT_ROAD_USER_CLASS_IDS,
        model: Any | None = None,
    ) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be greater than 0 and at most 1")
        if not 0 < iou <= 1:
            raise ValueError("iou must be greater than 0 and at most 1")
        if not class_ids:
            raise ValueError("class_ids cannot be empty")

        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is missing. Run: "
                    "python -m pip install -r requirements.txt"
                ) from exc
            model = YOLO(model_name)

        self.model = model
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.class_ids = tuple(int(class_id) for class_id in class_ids)

    def detect(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
        timestamp: float,
    ) -> list[Detection]:
        """Return FlowSense detections for one BGR video frame."""
        prediction_options: dict[str, object] = {
            "source": frame,
            "classes": list(self.class_ids),
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device:
            prediction_options["device"] = self.device

        results = self.model.predict(**prediction_options)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_numbers = boxes.cls.int().cpu().tolist()
        detections: list[Detection] = []
        for coordinates_row, confidence, class_number in zip(
            coordinates,
            confidences,
            class_numbers,
        ):
            class_id = int(class_number)
            if class_id not in self.class_ids:
                continue
            x1, y1, x2, y2 = (float(value) for value in coordinates_row)
            detections.append(
                Detection(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    class_id=class_id,
                    class_name=self._class_name(class_id),
                    confidence=float(confidence),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def _class_name(self, class_id: int) -> str:
        names = self.model.names
        if isinstance(names, Mapping):
            name = names.get(class_id)
        else:
            name = names[class_id] if class_id < len(names) else None
        if name is None:
            raise ValueError(f"YOLO model has no name for class ID {class_id}")
        return str(name).strip().casefold()
