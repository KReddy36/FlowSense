"""TEST-ONLY: verifies the three built-in CSV/video dataset pairings."""

from __future__ import annotations

import unittest
from pathlib import Path

from track_member1_video import DATASETS, track_csv_on_video


class DatasetConfigTests(unittest.TestCase):
    def test_all_member_one_datasets_are_configured(self) -> None:
        self.assertEqual(set(DATASETS), {"1", "2", "3", "4"})

    def test_dataset_suffixes_match(self) -> None:
        self.assertEqual(DATASETS["1"].csv_path.name, "tracking_data.csv")
        self.assertEqual(DATASETS["1"].video_path.name, "source_traffic.mp4")
        for dataset_id in ("2", "3", "4"):
            config = DATASETS[dataset_id]
            self.assertEqual(
                config.csv_path.name,
                f"tracking_data{dataset_id}.csv",
            )
            self.assertEqual(
                config.video_path.name,
                f"flowsense_tracking{dataset_id}.mp4",
            )
            self.assertIn(dataset_id, config.output_path.stem)
            self.assertIn(dataset_id, config.tracks_output_path.stem)

    def test_no_video_writes_only_canonical_csv(self) -> None:
        config = DATASETS["1"]
        test_output_root = Path("outputs")
        test_output_root.mkdir(parents=True, exist_ok=True)
        video_output = test_output_root / "_test_no_video_must_not_exist.mp4"
        tracks_output = test_output_root / "_test_no_video_tracks.csv"
        video_output.unlink(missing_ok=True)
        tracks_output.unlink(missing_ok=True)
        try:
            summary = track_csv_on_video(
                config.csv_path,
                config.video_path,
                video_output,
                tracks_output,
                maximum_frames=2,
                generate_video=False,
            )

            self.assertEqual(summary["output_video"], "disabled")
            self.assertFalse(video_output.exists())
            self.assertTrue(tracks_output.is_file())
        finally:
            video_output.unlink(missing_ok=True)
            tracks_output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
