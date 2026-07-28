"""TEST-ONLY: verifies conversion from Ultralytics results to FlowSense detections."""

from __future__ import annotations

import unittest

import numpy as np

from flowsense.yolo_detector import YoloDetector


class _Tensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def cpu(self) -> _Tensor:
        return self

    def int(self) -> _Tensor:
        return _Tensor(
            [
                int(value)
                for value in self.values
            ]
        )

    def tolist(self) -> list[object]:
        return self.values


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = _Tensor([[10.0, 20.0, 40.0, 60.0]])
        self.conf = _Tensor([0.91])
        self.cls = _Tensor([2.0])

    def __len__(self) -> int:
        return 1


class _Result:
    boxes = _Boxes()


class _Model:
    names = {2: "car"}

    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def predict(self, **options: object) -> list[_Result]:
        self.options = options
        return [_Result()]


class YoloDetectorTests(unittest.TestCase):
    def test_converts_yolo_boxes_to_flow_sense_detections(self) -> None:
        model = _Model()
        detector = YoloDetector(model=model)

        detections = detector.detect(
            np.zeros((80, 100, 3), dtype=np.uint8),
            frame_id=7,
            timestamp=0.28,
        )

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.frame_id, 7)
        self.assertEqual(detection.class_id, 2)
        self.assertEqual(detection.class_name, "car")
        self.assertEqual(detection.bbox, (10.0, 20.0, 40.0, 60.0))
        self.assertEqual(model.options["classes"], [0, 1, 2, 3, 5, 7])
        self.assertFalse(model.options["verbose"])


if __name__ == "__main__":
    unittest.main()
