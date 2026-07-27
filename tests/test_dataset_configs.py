"""TEST-ONLY: verifies the three built-in CSV/video dataset pairings."""

from __future__ import annotations

import unittest

from track_member1_video import DATASETS


class DatasetConfigTests(unittest.TestCase):
    def test_all_member_one_datasets_are_configured(self) -> None:
        self.assertEqual(set(DATASETS), {"1", "2", "3"})

    def test_dataset_suffixes_match(self) -> None:
        self.assertEqual(DATASETS["1"].csv_path.name, "tracking_data.csv")
        self.assertEqual(DATASETS["1"].video_path.name, "source_traffic.mp4")
        for dataset_id in ("2", "3"):
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


if __name__ == "__main__":
    unittest.main()
