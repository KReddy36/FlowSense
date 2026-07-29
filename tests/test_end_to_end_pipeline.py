"""TEST-ONLY: exercises the one-command pipeline without downloading YOLO."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from flowsense.pipeline import PipelineConfig, run_pipeline
from flowsense.tracking import Detection
from flowsense.video_compat import VideoConversionError


class _MovingCarDetector:
    def detect(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
        timestamp: float,
    ) -> list[Detection]:
        x1 = 10.0 + frame_id * 5.0
        return [
            Detection(
                frame_id=frame_id,
                timestamp=timestamp,
                class_id=2,
                class_name="car",
                confidence=0.95,
                x1=x1,
                y1=20.0,
                x2=x1 + 25.0,
                y2=50.0,
            )
        ]


def _write_test_video(path: Path, frame_count: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (96, 72),
    )
    if not writer.isOpened():
        raise RuntimeError("Test video writer could not open")
    try:
        for _ in range(frame_count):
            writer.write(np.zeros((72, 96, 3), dtype=np.uint8))
    finally:
        writer.release()


class EndToEndPipelineTests(unittest.TestCase):
    def test_one_call_creates_only_final_video_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_video = root / "intersection.mp4"
            output_dir = root / "results"
            _write_test_video(input_video)
            progress: list[tuple[int, int]] = []

            result = run_pipeline(
                PipelineConfig(
                    input_video=input_video,
                    output_dir=output_dir,
                    movement_threshold_pixels=1.0,
                ),
                detector=_MovingCarDetector(),
                progress_callback=lambda processed, total: progress.append(
                    (processed, total)
                ),
            )

            self.assertTrue(result.output_video.is_file())
            self.assertGreater(result.output_video.stat().st_size, 0)
            self.assertTrue(result.output_report.is_file())
            report = result.output_report.read_text(encoding="utf-8")
            self.assertIn("intersection.mp4", report)
            self.assertIn("FlowSense Traffic Results", report)
            self.assertEqual(result.processed_frames, 8)
            self.assertGreater(result.unique_track_ids, 0)
            self.assertIsNone(result.prediction_accuracy_percent)
            self.assertEqual(result.prediction_accuracy_samples, 0)
            self.assertIn("Prediction accuracy", report)
            self.assertIn("N/A", report)
            self.assertTrue(result.browser_compatible_video)
            self.assertIsNone(result.video_preview_warning)
            self.assertEqual(progress[0], (0, 8))
            self.assertEqual(progress[-1], (8, 8))
            self.assertIsNone(result.intermediate_dir)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["intersection_flowsense.mp4", "intersection_report.html"],
            )

    def test_report_includes_prediction_accuracy_percentage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_video = root / "intersection.mp4"
            output_dir = root / "results"
            _write_test_video(input_video, frame_count=30)

            result = run_pipeline(
                PipelineConfig(
                    input_video=input_video,
                    output_dir=output_dir,
                    movement_threshold_pixels=1.0,
                    velocity_window=2,
                    prediction_horizon_frames=3,
                ),
                detector=_MovingCarDetector(),
            )

            self.assertIsNotNone(result.prediction_accuracy_percent)
            assert result.prediction_accuracy_percent is not None
            self.assertGreater(result.prediction_accuracy_percent, 50.0)
            self.assertGreater(result.prediction_accuracy_samples, 0)
            report = result.output_report.read_text(encoding="utf-8")
            self.assertIn(
                f"{result.prediction_accuracy_percent:.1f}%",
                report,
            )
            self.assertIn("stationary baseline", report)

    def test_debug_option_retains_internal_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_video = root / "intersection.mp4"
            output_dir = root / "results"
            _write_test_video(input_video)

            result = run_pipeline(
                PipelineConfig(
                    input_video=input_video,
                    output_dir=output_dir,
                    movement_threshold_pixels=1.0,
                    keep_intermediates=True,
                ),
                detector=_MovingCarDetector(),
            )

            self.assertTrue(
                result.intermediate_dir.samefile(output_dir / "intermediates")
            )
            self.assertTrue(
                (result.intermediate_dir / "canonical_tracks.csv").is_file()
            )
            self.assertTrue(
                (result.intermediate_dir / "motion_predictions.csv").is_file()
            )
            self.assertTrue(
                (
                    result.intermediate_dir
                    / "counter_outputs"
                    / "automatic_counts.csv"
                ).is_file()
            )

    def test_h264_failure_retains_original_annotated_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_video = root / "intersection.mp4"
            output_dir = root / "results"
            _write_test_video(input_video)

            with patch(
                "flowsense.pipeline.convert_to_browser_mp4",
                side_effect=VideoConversionError("converter unavailable"),
            ):
                result = run_pipeline(
                    PipelineConfig(
                        input_video=input_video,
                        output_dir=output_dir,
                        movement_threshold_pixels=1.0,
                    ),
                    detector=_MovingCarDetector(),
                )

            self.assertTrue(result.output_video.is_file())
            self.assertFalse(result.browser_compatible_video)
            self.assertIn(
                "still available to download",
                result.video_preview_warning or "",
            )

    def test_progress_callback_failure_does_not_stop_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_video = root / "intersection.mp4"
            output_dir = root / "results"
            _write_test_video(input_video)

            def broken_progress(_processed: int, _total: int) -> None:
                raise RuntimeError("test callback failure")

            result = run_pipeline(
                PipelineConfig(
                    input_video=input_video,
                    output_dir=output_dir,
                    movement_threshold_pixels=1.0,
                ),
                detector=_MovingCarDetector(),
                progress_callback=broken_progress,
            )

            self.assertTrue(result.output_video.is_file())
            self.assertTrue(result.output_report.is_file())


if __name__ == "__main__":
    unittest.main()
