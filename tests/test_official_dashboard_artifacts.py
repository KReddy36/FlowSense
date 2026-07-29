"""TEST-ONLY: validates official dashboard videos and preserved result data."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

from flowsense.video_compat import is_h264_mp4


ROOT = Path(__file__).resolve().parents[1]


class OfficialDashboardArtifactTests(unittest.TestCase):
    def test_streamlit_upload_limit_is_100_mb(self) -> None:
        config = (ROOT / ".streamlit" / "config.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("maxUploadSize = 100", config)

    def test_dashboard_requires_learned_results_and_maps_all_hybrid_videos(self) -> None:
        source = (ROOT / "Dashboard.py").read_text(encoding="utf-8")
        self.assertIn('"learned_prediction_results.csv"', source)
        for video_id in range(1, 5):
            self.assertIn(
                f'"Video {video_id}": VIDEOS_DIR / '
                f'"flowsense_hybrid_video_{video_id}.mp4"',
                source,
            )

    def test_official_videos_are_materialized_h264_files(self) -> None:
        for video_id in range(1, 5):
            path = ROOT / "videos" / f"flowsense_hybrid_video_{video_id}.mp4"
            self.assertTrue(path.is_file(), msg=f"Missing official Video {video_id}")
            with path.open("rb") as handle:
                header = handle.read(128)
            self.assertFalse(
                header.startswith(b"version https://git-lfs"),
                msg=f"Video {video_id} is an unmaterialized Git LFS pointer",
            )
            self.assertTrue(
                is_h264_mp4(path),
                msg=f"Video {video_id} is not a browser-compatible H.264 MP4",
            )

    def test_official_vehicle_counts_are_unchanged(self) -> None:
        path = ROOT / "easy_results" / "automatic_counts.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        actual = {
            row["Video"]: int(float(row["Vehicles counted"]))
            for row in rows
        }
        self.assertEqual(
            actual,
            {"Video 1": 6, "Video 2": 77, "Video 3": 47, "Video 4": 264},
        )


if __name__ == "__main__":
    unittest.main()
