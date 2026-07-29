"""TEST-ONLY: validates official dashboard videos and preserved result data."""

from __future__ import annotations

import csv
import subprocess
import unittest
from pathlib import Path

import cv2
import imageio_ffmpeg

from flowsense.video_compat import is_h264_mp4


ROOT = Path(__file__).resolve().parents[1]


class OfficialDashboardArtifactTests(unittest.TestCase):
    def test_streamlit_upload_limit_is_100_mb(self) -> None:
        config = (ROOT / ".streamlit" / "config.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("maxUploadSize = 100", config)

    def test_dashboard_requires_learned_results_and_uses_local_web_videos(
        self,
    ) -> None:
        source = (ROOT / "Dashboard.py").read_text(encoding="utf-8")
        self.assertIn('"learned_prediction_results.csv"', source)
        self.assertNotIn("media.githubusercontent.com", source)
        self.assertNotIn("GITHUB_VIDEO_BASE_URL", source)
        self.assertIn("VIDEO_FILES", source)
        self.assertIn('ROOT / "videos"', source)
        self.assertIn('f"Video {number}"', source)
        self.assertIn("flowsense_web_hybrid_video_{number}.mp4", source)
        self.assertIn("for number in range(1, 5)", source)

    def test_web_videos_are_real_h264_yuv420p_mp4_files(self) -> None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        for video_id in range(1, 5):
            path = (
                ROOT
                / "videos"
                / f"flowsense_web_hybrid_video_{video_id}.mp4"
            )
            self.assertTrue(path.is_file(), msg=f"Missing web Video {video_id}")
            with path.open("rb") as handle:
                header = handle.read(2 * 1024 * 1024)
            self.assertFalse(
                header.startswith(b"version https://git-lfs"),
                msg=f"Web Video {video_id} is a Git LFS pointer",
            )
            self.assertIn(b"ftyp", header[:64])
            self.assertTrue(
                is_h264_mp4(path),
                msg=f"Web Video {video_id} is not H.264",
            )
            probe = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(path)],
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
            self.assertIn("Video: h264", probe.stderr)
            self.assertIn("yuv420p", probe.stderr)
            self.assertLess(path.stat().st_size, 100_000_000)
            self.assertGreaterEqual(header.find(b"moov"), 0)
            self.assertGreater(header.find(b"mdat"), header.find(b"moov"))

    def test_web_videos_preserve_original_frame_counts_and_rates(self) -> None:
        for video_id in range(1, 5):
            original = (
                ROOT / "videos" / f"flowsense_hybrid_video_{video_id}.mp4"
            )
            web = (
                ROOT
                / "videos"
                / f"flowsense_web_hybrid_video_{video_id}.mp4"
            )
            original_capture = cv2.VideoCapture(str(original))
            web_capture = cv2.VideoCapture(str(web))
            try:
                self.assertTrue(original_capture.isOpened())
                self.assertTrue(web_capture.isOpened())
                self.assertEqual(
                    int(original_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                    int(web_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                )
                self.assertAlmostEqual(
                    original_capture.get(cv2.CAP_PROP_FPS),
                    web_capture.get(cv2.CAP_PROP_FPS),
                    places=3,
                )
                self.assertEqual(
                    int(web_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    1280,
                )
                self.assertEqual(
                    int(web_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    720,
                )
            finally:
                original_capture.release()
                web_capture.release()

    def test_web_videos_are_excluded_from_lfs_pattern(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn(
            "videos/flowsense_web_hybrid_video_*.mp4 -text",
            attributes,
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
