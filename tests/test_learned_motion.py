"""TEST-ONLY: verifies the portable learned motion corrector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from flowsense.tracking import (
    LearnedMotionCorrector,
    MotionPredictor,
    TrackedDetection,
    default_model_path,
)


def make_track(frame_id: int, center_x: float) -> TrackedDetection:
    return TrackedDetection(
        frame_id=frame_id,
        timestamp=frame_id / 10.0,
        track_id=3,
        class_id=2,
        class_name="car",
        confidence=0.8,
        x1=center_x - 10,
        y1=30,
        x2=center_x + 10,
        y2=50,
    )


class LearnedMotionCorrectorTests(unittest.TestCase):
    def test_bundled_model_is_valid_and_dependency_free(self) -> None:
        corrector = LearnedMotionCorrector.from_json(default_model_path())
        self.assertEqual(corrector.horizon_frames, 15)
        self.assertEqual(corrector.velocity_window, 5)
        self.assertEqual(len(corrector.coefficients), 24)
        self.assertTrue(
            corrector.compatible_with(horizon_frames=15, velocity_window=5)
        )

    def test_predictor_applies_bounded_learned_scale_after_enough_history(self) -> None:
        corrector = LearnedMotionCorrector.from_json(default_model_path())
        baseline = MotionPredictor(
            fps=10,
            velocity_window=5,
            prediction_horizon_frames=15,
        )
        learned = MotionPredictor(
            fps=10,
            velocity_window=5,
            prediction_horizon_frames=15,
            frame_width=200,
            frame_height=100,
            learned_corrector=corrector,
        )
        baseline_snapshot = None
        learned_snapshot = None
        for frame_id in range(6):
            track = make_track(frame_id, 20 + frame_id * 4)
            baseline_snapshot = baseline.update(
                [track],
                frame_id=frame_id,
                timestamp=frame_id / 10.0,
            )[0]
            learned_snapshot = learned.update(
                [track],
                frame_id=frame_id,
                timestamp=frame_id / 10.0,
            )[0]

        assert baseline_snapshot is not None and learned_snapshot is not None
        baseline_dx = (
            baseline_snapshot.predicted_center_x - baseline_snapshot.center_x
        )
        learned_dx = (
            learned_snapshot.predicted_center_x - learned_snapshot.center_x
        )
        applied_scale = learned_dx / baseline_dx
        self.assertGreaterEqual(applied_scale, 0.7375)
        self.assertLessEqual(applied_scale, 1.2625)
        self.assertNotAlmostEqual(applied_scale, 1.0)

    def test_incompatible_runtime_settings_use_original_predictor(self) -> None:
        corrector = LearnedMotionCorrector.from_json(default_model_path())
        predictor = MotionPredictor(
            fps=10,
            velocity_window=2,
            prediction_horizon_frames=3,
            frame_width=200,
            frame_height=100,
            learned_corrector=corrector,
        )
        snapshot = None
        for frame_id in range(4):
            snapshot = predictor.update(
                [make_track(frame_id, 20 + frame_id * 4)],
                frame_id=frame_id,
                timestamp=frame_id / 10.0,
            )[0]
        assert snapshot is not None
        self.assertAlmostEqual(snapshot.predicted_center_x, 44.0)

    def test_malformed_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps({"format": "wrong"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                LearnedMotionCorrector.from_json(path)


if __name__ == "__main__":
    unittest.main()
