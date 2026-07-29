"""TEST-ONLY: verifies learned prediction wiring for official video generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from track_member1_video import (
    build_motion_predictor,
    load_prediction_corrector,
    validate_clean_source_path,
)


class OfficialHybridGeneratorTests(unittest.TestCase):
    def test_default_loads_bundled_corrector_and_frame_dimensions(self) -> None:
        predictor, mode = build_motion_predictor(
            fps=30.0,
            frame_width=1920,
            frame_height=1080,
        )
        self.assertIsNotNone(predictor.learned_corrector)
        self.assertEqual(predictor.frame_width, 1920)
        self.assertEqual(predictor.frame_height, 1080)
        self.assertEqual(predictor.velocity_window, 5)
        self.assertEqual(predictor.prediction_horizon_frames, 15)
        self.assertIn("learned hybrid", mode)

    def test_disable_option_uses_original_predictor(self) -> None:
        predictor, mode = build_motion_predictor(
            fps=25.0,
            frame_width=1280,
            frame_height=720,
            disable_learned_prediction=True,
        )
        self.assertIsNone(predictor.learned_corrector)
        self.assertIn("constant-velocity baseline", mode)

    def test_missing_and_invalid_models_fall_back_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.json"
            corrector, mode = load_prediction_corrector(
                disabled=False,
                model_path=missing,
                prediction_horizon_frames=15,
                velocity_window=5,
            )
            self.assertIsNone(corrector)
            self.assertIn("fallback", mode)

            invalid = root / "invalid.json"
            invalid.write_text("{not json", encoding="utf-8")
            corrector, mode = load_prediction_corrector(
                disabled=False,
                model_path=invalid,
                prediction_horizon_frames=15,
                velocity_window=5,
            )
            self.assertIsNone(corrector)
            self.assertIn("fallback", mode)

    def test_known_annotated_outputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "clean source"):
            validate_clean_source_path("videos/flowsense_tracking2.mp4")


if __name__ == "__main__":
    unittest.main()
