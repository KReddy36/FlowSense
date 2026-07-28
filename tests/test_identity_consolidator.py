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

    def test_established_vehicles_do_not_merge_when_boxes_overlap(self) -> None:
        consolidator = IdentityConsolidator()
        initial = consolidator.update(
            [
                make_track(0, 10, (100, 100, 200, 200)),
                make_track(0, 20, (300, 100, 400, 200)),
            ],
            frame_id=0,
        )
        initial_ids = {track.track_id for track in initial.visible_tracks}

        overlapping = consolidator.update(
            [
                make_track(1, 10, (150, 100, 250, 200)),
                make_track(1, 20, (153, 103, 253, 203)),
            ],
            frame_id=1,
        )
        separated = consolidator.update(
            [
                make_track(2, 10, (200, 100, 300, 200)),
                make_track(2, 20, (320, 100, 420, 200)),
            ],
            frame_id=2,
        )

        self.assertEqual(len(overlapping.visible_tracks), 2)
        self.assertEqual(
            {track.track_id for track in overlapping.visible_tracks},
            initial_ids,
        )
        self.assertEqual(len(separated.visible_tracks), 2)
        self.assertEqual(
            {track.track_id for track in separated.visible_tracks},
            initial_ids,
        )

    def test_reidentification_collision_splits_two_visible_vehicles(self) -> None:
        consolidator = IdentityConsolidator()
        first = consolidator.update(
            [make_track(0, 10, (100, 100, 200, 200))],
            frame_id=0,
        )
        original_id = first.visible_tracks[0].track_id

        # A new raw ID appears near the old location while ID 10 is missing,
        # so short-gap reidentification intentionally reconnects it.
        reidentified = consolidator.update(
            [make_track(1, 20, (105, 100, 205, 200))],
            frame_id=1,
        )
        self.assertEqual(reidentified.visible_tracks[0].track_id, original_id)

        # Both raw tracks then appear far apart. They must be split instead of
        # alternating as one canonical trajectory.
        collision = consolidator.update(
            [
                make_track(2, 10, (110, 100, 210, 200), confidence=0.95),
                make_track(2, 20, (500, 400, 600, 500), confidence=0.90),
            ],
            frame_id=2,
        )
        self.assertEqual(len(collision.visible_tracks), 2)
        self.assertEqual(
            len({track.track_id for track in collision.visible_tracks}),
            2,
        )
        self.assertEqual(collision.suppressed_tracks, [])


if __name__ == "__main__":
    unittest.main()
