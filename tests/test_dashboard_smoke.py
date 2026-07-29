"""TEST-ONLY: confirms both Streamlit dashboard tabs render."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


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


if __name__ == "__main__":
    unittest.main()
