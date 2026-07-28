"""TEST-ONLY: verifies bounded motion history and short-term prediction."""

from __future__ import annotations

import unittest

from flowsense.tracking import MotionPredictor, TrackedDetection


def make_track(frame_id: int, center_x: float, center_y: float = 50.0) -> TrackedDetection:
    return TrackedDetection(
        frame_id=frame_id,
        timestamp=float(frame_id),
        track_id=7,
        class_id=2,
        class_name="car",
        confidence=0.9,
        x1=center_x - 5,
        y1=center_y - 5,
        x2=center_x + 5,
        y2=center_y + 5,
    )


class MotionPredictorTests(unittest.TestCase):
    def test_history_is_bounded_and_recent_velocities_are_averaged(self) -> None:
        predictor = MotionPredictor(
            fps=1,
            history_points=3,
            velocity_window=2,
            prediction_horizon_frames=2,
        )
        centers = (0.0, 10.0, 30.0, 60.0)
        snapshot = None
        for frame_id, center_x in enumerate(centers):
            snapshot = predictor.update(
                [make_track(frame_id, center_x)],
                frame_id=frame_id,
                timestamp=float(frame_id),
            )[0]

        assert snapshot is not None
        self.assertEqual(len(snapshot.observed_points), 3)
        self.assertEqual(
            [point.x for point in snapshot.observed_points],
            [10.0, 30.0, 60.0],
        )
        self.assertAlmostEqual(snapshot.velocity_x, 25.0)
        self.assertAlmostEqual(snapshot.velocity_y, 0.0)
        self.assertAlmostEqual(snapshot.predicted_center_x, 110.0)

    def test_missing_track_is_predicted_until_timeout_then_deleted(self) -> None:
        predictor = MotionPredictor(
            fps=1,
            inactive_timeout_frames=2,
            prediction_horizon_frames=1,
        )
        predictor.update([make_track(0, 0)], frame_id=0, timestamp=0.0)
        predictor.update([make_track(1, 10)], frame_id=1, timestamp=1.0)

        missing = predictor.update([], frame_id=2, timestamp=2.0)
        self.assertEqual(len(missing), 1)
        self.assertFalse(missing[0].observed)
        self.assertEqual(missing[0].frames_since_seen, 1)
        self.assertAlmostEqual(missing[0].center_x, 20.0)
        self.assertAlmostEqual(missing[0].predicted_center_x, 30.0)

        still_active = predictor.update([], frame_id=3, timestamp=3.0)
        self.assertEqual(len(still_active), 1)
        expired = predictor.update([], frame_id=4, timestamp=4.0)
        self.assertEqual(expired, [])
        self.assertEqual(predictor.active_track_ids, frozenset())


if __name__ == "__main__":
    unittest.main()
