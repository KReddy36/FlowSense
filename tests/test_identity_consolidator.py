"""TEST-ONLY: verifies duplicate suppression and identity reconnection."""

from __future__ import annotations

import unittest

from flowsense.tracking import IdentityConsolidator, TrackedDetection


def make_track(
    frame_id: int,
    raw_id: int,
    bbox: tuple[float, float, float, float],
    *,
    class_id: int = 2,
    class_name: str = "car",
    confidence: float = 0.9,
) -> TrackedDetection:
    return TrackedDetection(
        frame_id=frame_id,
        timestamp=frame_id / 25,
        track_id=raw_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
    )


class IdentityConsolidatorTests(unittest.TestCase):
    def test_overlapping_misclassifications_share_id_and_one_is_hidden(self) -> None:
        consolidator = IdentityConsolidator()
        car = make_track(0, 10, (100, 100, 200, 180), confidence=0.93)
        truck = make_track(
            0,
            11,
            (102, 101, 201, 181),
            class_id=7,
            class_name="truck",
            confidence=0.41,
        )

        result = consolidator.update([car, truck], frame_id=0)

        self.assertEqual(len(result.all_tracks), 2)
        self.assertEqual(len({track.track_id for track in result.all_tracks}), 1)
        self.assertEqual(len(result.visible_tracks), 1)
        self.assertEqual(len(result.suppressed_tracks), 1)
        self.assertEqual(result.visible_tracks[0].class_name, "car")

    def test_new_raw_id_after_short_gap_reuses_canonical_id(self) -> None:
        consolidator = IdentityConsolidator()
        first = consolidator.update(
            [make_track(0, 10, (100, 100, 200, 180))],
            frame_id=0,
        )
        consolidator.update([], frame_id=1)
        returned = consolidator.update(
            [make_track(2, 99, (110, 100, 210, 180))],
            frame_id=2,
        )

        self.assertEqual(
            first.visible_tracks[0].track_id,
            returned.visible_tracks[0].track_id,
        )

    def test_separate_objects_remain_separate(self) -> None:
        consolidator = IdentityConsolidator()
        result = consolidator.update(
            [
                make_track(0, 10, (100, 100, 200, 180)),
                make_track(0, 11, (500, 300, 600, 380)),
            ],
            frame_id=0,
        )

        self.assertEqual(len(result.visible_tracks), 2)
        self.assertEqual(len({track.track_id for track in result.all_tracks}), 2)
        self.assertEqual(result.suppressed_tracks, [])


if __name__ == "__main__":
    unittest.main()
