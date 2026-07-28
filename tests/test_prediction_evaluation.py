"""TEST-ONLY: verifies measured prediction error and baseline comparison."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from flowsense.prediction_evaluation import (
    evaluate_tracking_csv,
    summarize_errors,
)


class PredictionEvaluationTests(unittest.TestCase):
    def test_constant_velocity_beats_stationary_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "member2_canonical_tracks.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "frame",
                        "time_seconds",
                        "track_id",
                        "class_id",
                        "class_name",
                        "confidence",
                        "center_x",
                        "center_y",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    ],
                )
                writer.writeheader()
                for frame in range(40):
                    center_x = 20.0 + 2.0 * frame
                    writer.writerow(
                        {
                            "frame": frame,
                            "time_seconds": frame / 10.0,
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.9,
                            "center_x": center_x,
                            "center_y": 30.0,
                            "x1": center_x - 5.0,
                            "y1": 25.0,
                            "x2": center_x + 5.0,
                            "y2": 35.0,
                        }
                    )

            errors = evaluate_tracking_csv(
                path,
                velocity_window=3,
                prediction_horizon_frames=5,
            )
            summary = summarize_errors(
                errors,
                dataset="Video 1",
                prediction_horizon_frames=5,
            )

        self.assertGreater(len(errors), 0)
        self.assertAlmostEqual(
            float(summary["Median prediction error (px)"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(summary["Median stationary baseline error (px)"]),
            10.0,
        )
        self.assertEqual(
            float(summary["Prediction win rate (%)"]),
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
