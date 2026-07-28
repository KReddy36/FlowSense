import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from automatic_counter import (
    Detection,
    analyze_tracks,
    class_count_phrase,
    choose_counting_method,
    default_video_label,
    discover_csvs,
    discover_kelvin_file,
    find_horizontal_passage,
    load_kelvin_class_totals,
    normalize_class_name,
    refine_passage_classes,
    vehicle_id_rate_per_minute,
)


def detection(
    frame: int,
    time_seconds: float,
    canonical_id: str,
    class_name: str,
    center_x: float,
    center_y: float,
) -> Detection:
    return Detection(
        frame=frame,
        time_seconds=time_seconds,
        canonical_id=canonical_id,
        class_id=None,
        class_name=class_name,
        confidence=0.9,
        center_x=center_x,
        center_y=center_y,
        frame_bottom=720.0,
    )


class AutomaticCounterTests(unittest.TestCase):
    def test_repository_discovery_prefers_tracking_data_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_data = root / "tracking_data"
            outputs = root / "outputs"
            tracking_data.mkdir()
            outputs.mkdir()
            canonical = tracking_data / "member2_canonical_tracks2.csv"
            duplicate = outputs / "member2_canonical_tracks2.csv"
            canonical.touch()
            duplicate.touch()

            with patch.object(Path, "cwd", return_value=root):
                discovered = discover_csvs([])

        self.assertEqual(discovered, [canonical])

    def test_clear_video_labels(self) -> None:
        self.assertEqual(
            default_video_label(Path("member2_canonical_tracks.csv"), 1),
            "Video 1",
        )
        self.assertEqual(
            default_video_label(Path("member2_canonical_tracks2.csv"), 9),
            "Video 2",
        )
        self.assertEqual(
            default_video_label(Path("member2_canonical_tracks_video4.csv"), 9),
            "Video 4",
        )

    def test_passage_crossing_counts_once(self) -> None:
        rows = [
            detection(0, 0.0, "A", "car", 100, 100),
            detection(1, 1.0, "A", "car", 100, 280),
            detection(2, 2.0, "A", "car", 100, 500),
            detection(3, 3.0, "A", "car", 100, 100),
        ]
        crossing = find_horizontal_passage(rows, 288.0, 15.0)
        self.assertIsNotNone(crossing)
        assert crossing is not None
        self.assertGreater(crossing[0], 0.0)
        self.assertLess(crossing[0], 2.0)

    def test_stationary_track_does_not_cross(self) -> None:
        rows = [
            detection(0, 0.0, "A", "car", 100, 200),
            detection(1, 1.0, "A", "car", 101, 202),
            detection(2, 2.0, "A", "car", 99, 199),
        ]
        self.assertIsNone(find_horizontal_passage(rows, 288.0, 15.0))

    def test_auto_mode_uses_fragmentation_rate(self) -> None:
        self.assertEqual(
            choose_counting_method("auto", 149.9, 150.0), "movement"
        )
        self.assertEqual(
            choose_counting_method("auto", 150.0, 150.0), "passage"
        )
        self.assertEqual(
            choose_counting_method("movement", 999.0, 150.0), "movement"
        )

    def test_vehicle_rate_excludes_pedestrians(self) -> None:
        rows = [
            detection(0, 0.0, "car-1", "car", 0, 0),
            detection(1, 60.0, "car-1", "car", 10, 10),
            detection(0, 0.0, "person-1", "person", 0, 0),
            detection(1, 60.0, "person-1", "person", 10, 10),
        ]
        self.assertEqual(vehicle_id_rate_per_minute(rows), 1.0)

    def test_kelvin_wide_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kelvin.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["Video", "Cars", "Trucks", "Buses", "Motorcycles"]
                )
                writer.writerow(["Video 4", 212, 20, 4, 3])
            totals = load_kelvin_class_totals(
                path, {"video 4": "Video 4"}
            )
        self.assertEqual(totals[("Video 4", "Car")], 212)
        self.assertEqual(totals[("Video 4", "Truck")], 20)
        self.assertEqual(totals[("Video 4", "Bus")], 4)
        self.assertEqual(totals[("Video 4", "Motorcycle")], 3)

    def test_kelvin_real_headers_and_annotated_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kelvin.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "Video number",
                        "Direction of the road",
                        "Cars crossing",
                        "Trucks",
                        "Buses",
                        "Motorcycles",
                        "Bicycles",
                        "Pedestrians",
                    ]
                )
                writer.writerow(
                    ["Video 2", "down", 4, 2, "", 3, 0, 0]
                )
                writer.writerow(
                    [
                        "Video 2",
                        "up",
                        62,
                        1,
                        0,
                        0,
                        0,
                        "3 (only one identified)",
                    ]
                )
            totals = load_kelvin_class_totals(
                path, {"video 2": "Video 2"}
            )
        self.assertEqual(totals[("Video 2", "Car")], 66)
        self.assertEqual(totals[("Video 2", "Truck")], 3)
        self.assertEqual(totals[("Video 2", "Motorcycle")], 3)
        self.assertEqual(totals[("Video 2", "Pedestrian")], 3)

    def test_kelvin_file_is_discovered_beside_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks = root / "member2_canonical_tracks.csv"
            tracks.write_text("placeholder", encoding="utf-8")
            kelvin = root / "kelvin_vehicle_counts (1).csv"
            kelvin.write_text("placeholder", encoding="utf-8")
            discovered = discover_kelvin_file([tracks])
            self.assertIsNotNone(discovered)
            assert discovered is not None
            self.assertTrue(discovered.samefile(kelvin))

    def test_passage_class_refinement_preserves_total(self) -> None:
        rows = [
            detection(0, 0.0, "car-crossing", "car", 100, 200),
            detection(1, 1.0, "car-crossing", "car", 100, 280),
            detection(2, 2.0, "car-crossing", "car", 100, 400),
            detection(3, 2.5, "truck-fragment", "truck", 105, 500),
            detection(4, 2.6, "truck-fragment", "truck", 106, 510),
        ]
        original = analyze_tracks(
            detections=rows,
            video="Video 2",
            source_csv=Path("video2.csv"),
            movement_threshold=50.0,
            toward="down",
            cross_traffic_ratio=0.35,
            min_track_frames=2,
            counting_method="passage",
            passage_line_y=288.0,
            passage_hysteresis_pixels=5.0,
        )
        refined = refine_passage_classes(original, 6.0)
        counted = [item for item in refined if item.counted]
        self.assertEqual(len(counted), 1)
        self.assertEqual(counted[0].class_name, "Truck")

    def test_class_aliases(self) -> None:
        self.assertEqual(normalize_class_name("person"), "Pedestrian")
        self.assertEqual(normalize_class_name("motorbike"), "Motorcycle")

    def test_class_pluralization(self) -> None:
        self.assertEqual(class_count_phrase("Bus", 1), "1 bus")
        self.assertEqual(class_count_phrase("Bus", 2), "2 buses")
        self.assertEqual(class_count_phrase("Car", 2), "2 cars")


if __name__ == "__main__":
    unittest.main()
