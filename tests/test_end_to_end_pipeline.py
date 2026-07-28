"""TEST-ONLY: exercises the one-command pipeline without downloading YOLO."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from flowsense.pipeline import PipelineConfig, run_pipeline
from flowsense.tracking import Detection


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

            result = run_pipeline(
                PipelineConfig(
                    input_video=input_video,
                    output_dir=output_dir,
                    movement_threshold_pixels=1.0,
                ),
                detector=_MovingCarDetector(),
            )

            self.assertTrue(result.output_video.is_file())
            self.assertGreater(result.output_video.stat().st_size, 0)
            self.assertTrue(result.output_report.is_file())
            report = result.output_report.read_text(encoding="utf-8")
            self.assertIn("intersection.mp4", report)
            self.assertIn("FlowSense Traffic Results", report)
            self.assertEqual(result.processed_frames, 8)
            self.assertGreater(result.unique_track_ids, 0)
            self.assertIsNone(result.intermediate_dir)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["intersection_flowsense.mp4", "intersection_report.html"],
            )

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


if __name__ == "__main__":
    unittest.main()
