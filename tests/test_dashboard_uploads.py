"""TEST-ONLY: verifies isolated Streamlit upload analysis without YOLO."""

from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

from flowsense.dashboard_uploads import (
    MissingOutputError,
    UploadValidationError,
    analyze_saved_upload,
    build_intermediates_zip,
    cleanup_abandoned_runs,
    friendly_analysis_error,
    remove_run_directory,
    save_uploaded_mp4,
    validate_pipeline_outputs,
)
from flowsense.tracking import Detection


class _FakeDetector:
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


def _video_bytes(path: Path, frame_count: int = 8) -> bytes:
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
    return path.read_bytes()


class DashboardUploadTests(unittest.TestCase):
    def test_uploaded_video_uses_fixed_name_in_unique_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _video_bytes(root / "fixture.mp4")
            runs_root = root / "runs"

            first = save_uploaded_mp4(
                r"..\unsafe folder\first video.mp4",
                data,
                runs_root=runs_root,
            )
            second = save_uploaded_mp4(
                "first video.mp4",
                data,
                runs_root=runs_root,
            )

            self.assertNotEqual(first.run_dir, second.run_dir)
            self.assertEqual(first.input_path.name, "uploaded_video.mp4")
            self.assertEqual(first.input_path.parent, first.run_dir)
            self.assertEqual(first.run_dir.parent, runs_root.resolve())
            self.assertNotIn("unsafe", str(first.input_path))
            self.assertEqual(first.download_stem, "first-video")

    def test_invalid_uploads_are_rejected_before_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "runs"
            cases = [
                ("video.mov", b"\x00\x00\x00\x18ftypisom", 100),
                ("video.mp4", b"", 100),
                ("video.mp4", b"not an mp4 file", 100),
                ("video.mp4", b"\x00\x00\x00\x18ftypisom", 4),
            ]
            for filename, data, maximum in cases:
                with self.subTest(filename=filename, maximum=maximum):
                    with self.assertRaises(UploadValidationError):
                        save_uploaded_mp4(
                            filename,
                            data,
                            runs_root=runs_root,
                            maximum_bytes=maximum,
                        )

            self.assertFalse(runs_root.exists())

    def test_pipeline_failures_have_readable_dashboard_messages(self) -> None:
        cases = [
            (
                RuntimeError("Could not open input video: uploaded_video.mp4"),
                "processing",
                "could not be read",
            ),
            (
                RuntimeError("The input video contains no readable frames"),
                "processing",
                "empty",
            ),
            (
                RuntimeError(
                    "YOLO did not produce any trackable road-user detections."
                ),
                "processing",
                "No supported road users",
            ),
            (
                RuntimeError("download failed"),
                "model",
                "download or load the YOLO model",
            ),
        ]
        for error, stage, expected in cases:
            with self.subTest(stage=stage, error=str(error)):
                self.assertIn(
                    expected,
                    friendly_analysis_error(error, stage=stage),
                )

    def test_missing_pipeline_outputs_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            fake_result = type(
                "Result",
                (),
                {
                    "output_video": missing / "video.mp4",
                    "output_report": missing / "report.html",
                },
            )()

            with self.assertRaises(MissingOutputError):
                validate_pipeline_outputs(fake_result)

    def test_fake_detector_analysis_creates_downloadable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _video_bytes(root / "fixture.mp4")
            saved = save_uploaded_mp4(
                "traffic.mp4",
                data,
                runs_root=root / "runs",
            )
            progress: list[tuple[int, int]] = []

            result = analyze_saved_upload(
                saved,
                detector=_FakeDetector(),
                progress_callback=lambda processed, total: progress.append(
                    (processed, total)
                ),
            )

            self.assertTrue(result.output_video.is_file())
            self.assertTrue(result.output_report.is_file())
            self.assertTrue(result.browser_compatible_video)
            self.assertEqual(progress[0], (0, 8))
            self.assertEqual(progress[-1], (8, 8))
            self.assertIsNotNone(result.intermediate_dir)
            assert result.intermediate_dir is not None
            archive_bytes = build_intermediates_zip(result.intermediate_dir)
            with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
                names = set(archive.namelist())
            self.assertIn("canonical_tracks.csv", names)
            self.assertIn("motion_predictions.csv", names)
            self.assertIn("counting/automatic_counts.csv", names)

    def test_clear_and_abandoned_cleanup_stay_inside_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _video_bytes(root / "fixture.mp4")
            runs_root = root / "runs"
            old_run = save_uploaded_mp4(
                "old.mp4",
                data,
                runs_root=runs_root,
            )
            active_run = save_uploaded_mp4(
                "active.mp4",
                data,
                runs_root=runs_root,
            )
            os.utime(old_run.run_dir, (10.0, 10.0))
            os.utime(active_run.run_dir, (10.0, 10.0))

            removed = cleanup_abandoned_runs(
                runs_root=runs_root,
                maximum_age_seconds=1.0,
                exclude=[active_run.run_dir],
                now=100.0,
            )

            self.assertEqual(removed, 1)
            self.assertFalse(old_run.run_dir.exists())
            self.assertTrue(active_run.run_dir.exists())
            self.assertTrue(
                remove_run_directory(
                    active_run.run_dir,
                    runs_root=runs_root,
                )
            )
            with self.assertRaises(ValueError):
                remove_run_directory(root, runs_root=runs_root)


if __name__ == "__main__":
    unittest.main()
