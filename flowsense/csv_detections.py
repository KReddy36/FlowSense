"""Load Member 1's frame-by-frame YOLO detections from CSV."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from flowsense.tracking import Detection


# Standard COCO category IDs used by pretrained Ultralytics YOLO models.
COCO_CLASS_IDS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}

REQUIRED_COLUMNS = {
    "frame",
    "time_seconds",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
}


def load_detection_csv(
    csv_path: str | Path,
    *,
    class_ids: Mapping[str, int] = COCO_CLASS_IDS,
) -> dict[int, list[Detection]]:
    """Read detections grouped by frame.

    A CSV ``track_id`` column is deliberately ignored: Member 2's ByteTrack
    instance assigns the tracking identities used in the output video.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Detection CSV not found: {path}")

    detections_by_frame: dict[int, list[Detection]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"Detection CSV is missing required columns: {sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                class_name = row["class_name"].strip().casefold()
                if class_name not in class_ids:
                    raise ValueError(
                        f"unknown class_name {row['class_name']!r}; "
                        f"known classes are {sorted(class_ids)}"
                    )
                frame_id = int(row["frame"])
                detection = Detection(
                    frame_id=frame_id,
                    timestamp=float(row["time_seconds"]),
                    class_id=class_ids[class_name],
                    class_name=class_name,
                    confidence=float(row["confidence"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid detection CSV row at line {line_number}: {exc}"
                ) from exc
            detections_by_frame[frame_id].append(detection)

    if not detections_by_frame:
        raise ValueError(f"Detection CSV contains no detections: {path}")
    return dict(detections_by_frame)
