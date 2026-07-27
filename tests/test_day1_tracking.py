"""TEST-ONLY: verifies mock ByteTrack identity stability."""

from __future__ import annotations

import unittest

from demo_day1 import FRAME_COUNT, FPS, make_mock_frame
from flowsense.tracking import ByteTrackTracker, Detection
from flowsense.tracking.verification import verify_id_stability


class DetectionTests(unittest.TestCase):
    def test_rejects_inverted_box(self) -> None:
        with self.assertRaises(ValueError):
            Detection(0, 0.0, 2, "car", 0.9, 20, 10, 10, 30)


class ByteTrackIntegrationTests(unittest.TestCase):
    def test_mock_vehicles_do_not_exchange_ids(self) -> None:
        tracker = ByteTrackTracker(frame_rate=FPS)
        tracks_by_frame = {}
        truth_by_frame = {}

        for frame_id in range(FRAME_COUNT):
            detections, truth = make_mock_frame(frame_id)
            tracks_by_frame[frame_id] = tracker.update(detections)
            truth_by_frame[frame_id] = truth

        stable_ids = verify_id_stability(tracks_by_frame, truth_by_frame)
        self.assertEqual(set(stable_ids), {"eastbound_car", "westbound_car"})
        self.assertEqual(len(set(stable_ids.values())), 2)

    def test_rejects_mixed_frames(self) -> None:
        tracker = ByteTrackTracker(frame_rate=FPS)
        first, _ = make_mock_frame(0)
        second, _ = make_mock_frame(1)
        with self.assertRaises(ValueError):
            tracker.update([first[0], second[0]])


if __name__ == "__main__":
    unittest.main()
