"""TEST-ONLY: confirms both Streamlit dashboard tabs render."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from flowsense.pipeline import PipelineResult


class DashboardSmokeTests(unittest.TestCase):
    def test_project_results_and_upload_tabs_render(self) -> None:
        dashboard = Path(__file__).resolve().parents[1] / "Dashboard.py"

        app = AppTest.from_file(str(dashboard)).run(timeout=60)

        self.assertEqual(
            [tab.label for tab in app.get("tab")],
            ["Project Results", "Analyze Your Video"],
        )
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertEqual(
            list(app.selectbox[0].options),
            ["Video 1", "Video 2", "Video 3", "Video 4"],
        )
        expected_totals = {
            "Video 1": "6",
            "Video 2": "77",
            "Video 3": "47",
            "Video 4": "264",
        }
        for video, expected in expected_totals.items():
            app.selectbox[0].select(video)
            app.run(timeout=60)
            self.assertEqual(
                app.metric[0].value,
                expected,
                msg=f"Unexpected saved vehicle total for {video}",
            )
        self.assertEqual(list(app.exception), [])

    def test_uploaded_result_displays_prediction_accuracy(self) -> None:
        dashboard = Path(__file__).resolve().parents[1] / "Dashboard.py"

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            video = run_dir / "annotated.mp4"
            report = run_dir / "report.html"
            video.write_bytes(b"test video")
            report.write_text("<html>test report</html>", encoding="utf-8")

            app = AppTest.from_file(str(dashboard)).run(timeout=60)
            app.session_state["uploaded_analysis"] = {
                "original_filename": "sample.mp4",
                "download_stem": "sample",
                "run_dir": str(run_dir),
                "result": PipelineResult(
                    output_video=video,
                    output_report=report,
                    browser_compatible_video=True,
                    video_preview_warning=None,
                    processed_frames=120,
                    detected_instances=80,
                    rendered_track_instances=70,
                    unique_track_ids=12,
                    suppressed_duplicate_instances=2,
                    counts_by_class={"car": 10},
                    counts_by_direction={"toward camera": 6},
                    prediction_accuracy_percent=83.36,
                    prediction_accuracy_samples=61_388,
                    intermediate_dir=None,
                ),
            }

            app.run(timeout=60)

            prediction_metrics = [
                metric
                for metric in app.metric
                if metric.label == "Prediction accuracy"
            ]
            self.assertEqual(len(prediction_metrics), 1)
            self.assertEqual(prediction_metrics[0].value, "83.4%")
            self.assertTrue(
                any(
                    "61,388 forecasts" in caption.value
                    for caption in app.caption
                )
            )
            self.assertEqual(list(app.exception), [])


if __name__ == "__main__":
    unittest.main()
