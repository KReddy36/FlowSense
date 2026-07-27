"""TEST-ONLY: verifies Member 1 CSV parsing and class mapping."""

from __future__ import annotations

import unittest
from pathlib import Path

from flowsense.csv_detections import load_detection_csv


FIXTURE = Path(__file__).parent / "fixtures" / "member1_sample.csv"


class DetectionCsvTests(unittest.TestCase):
    def test_groups_rows_and_maps_coco_class_ids(self) -> None:
        detections = load_detection_csv(FIXTURE)

        self.assertEqual(sorted(detections), [0, 1])
        self.assertEqual(len(detections[0]), 2)
        self.assertEqual(detections[0][0].class_name, "car")
        self.assertEqual(detections[0][0].class_id, 2)
        self.assertEqual(detections[0][1].class_id, 0)

    def test_member_one_track_id_is_not_reused(self) -> None:
        detections = load_detection_csv(FIXTURE)

        # Detection intentionally has no track_id. ByteTrack assigns it later.
        self.assertFalse(hasattr(detections[0][0], "track_id"))


if __name__ == "__main__":
    unittest.main()
