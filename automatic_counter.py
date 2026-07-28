#!/usr/bin/env python3
"""Convert Kellan's canonical tracks into FlowSense traffic totals.

The program:
* discovers member2_canonical_tracks*.csv files or accepts explicit inputs;
* groups rows by canonical_id (falling back to track_id);
* decides whether each object moved or stayed parked;
* counts each moving canonical ID exactly once;
* classifies object type and camera-relative direction; and
* exports simple counts, an ID-level audit, interval volume, and JSON summary.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_REQUIRED_COLUMNS = {
    "frame",
    "time_seconds",
    "class_name",
    "center_x",
    "center_y",
}
ID_COLUMNS = ("canonical_id", "track_id")
PROGRAM_VERSION = "hybrid-v5"
CLASS_NAMES = {
    "person": "Pedestrian",
    "pedestrian": "Pedestrian",
    "car": "Car",
    "truck": "Truck",
    "motorcycle": "Motorcycle",
    "motorbike": "Motorcycle",
    "bicycle": "Bicycle",
    "bike": "Bicycle",
    "bus": "Bus",
    "van": "Van",
}
DIRECTIONS = ("Toward camera", "Away from camera", "Cross-traffic", "Mixed/unclear")
SUMMARY_CLASSES = (
    "Car",
    "Truck",
    "Motorcycle",
    "Bicycle",
    "Pedestrian",
    "Bus",
    "Van",
)
VEHICLE_CLASSES = {
    "Car",
    "Truck",
    "Motorcycle",
    "Bicycle",
    "Bus",
    "Van",
}


@dataclass(frozen=True)
class Detection:
    frame: int
    time_seconds: float
    canonical_id: str
    class_id: int | None
    class_name: str
    confidence: float | None
    center_x: float
    center_y: float
    frame_bottom: float | None


@dataclass(frozen=True)
class TrackResult:
    video: str
    source_csv: str
    canonical_id: str
    class_name: str
    row_count: int
    first_time_seconds: float
    last_time_seconds: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    delta_x: float
    delta_y: float
    net_displacement_pixels: float
    trajectory_span_pixels: float
    movement_threshold_pixels: float
    counting_method: str
    passage_line_y: float | None
    count_time_seconds: float
    status: str
    counted: bool
    direction: str
    mean_confidence: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Turn Kellan's canonical tracking CSVs into final automatic "
            "traffic counts."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "CSV files, folders, or wildcard patterns. If omitted, searches "
            "recursively for member2_canonical_tracks*.csv."
        ),
    )
    parser.add_argument(
        "--movement-threshold-pixels",
        type=float,
        default=50.0,
        help="Minimum trajectory span counted as movement (default: 50)",
    )
    parser.add_argument(
        "--toward-camera",
        choices=("down", "up", "left", "right"),
        default="down",
        help=(
            "Image direction that means toward the camera. In most road "
            "videos this is down (default: down)."
        ),
    )
    parser.add_argument(
        "--cross-traffic-ratio",
        type=float,
        default=0.15,
        help=(
            "Camera-axis motion below this fraction of net motion is labeled "
            "Cross-traffic (default: 0.15)"
        ),
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        choices=(5, 10),
        default=5,
        help="Traffic-volume interval size (default: 5)",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=1,
        help="Ignore IDs with fewer canonical rows (default: 1)",
    )
    parser.add_argument(
        "--counting-mode",
        choices=("auto", "movement", "passage"),
        default="auto",
        help=(
            "auto uses passage counting only for heavily fragmented videos "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--passage-line-fraction",
        type=float,
        default=0.40,
        help="Horizontal passage line as a fraction of frame height (default: 0.40)",
    )
    parser.add_argument(
        "--passage-hysteresis-pixels",
        type=float,
        default=15.0,
        help="Distance required on both sides of a passage line (default: 15)",
    )
    parser.add_argument(
        "--fragmentation-ids-per-minute",
        type=float,
        default=150.0,
        help=(
            "Auto mode switches to passage counting above this vehicle-ID "
            "rate (default: 150)"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional JSON with per-video label, toward-camera direction, "
            "and movement threshold"
        ),
    )
    parser.add_argument(
        "--manual-counts",
        type=Path,
        help="Optional Kelvin count CSV used to create comparison_with_kelvin.csv",
    )
    parser.add_argument(
        "--evaluation-video",
        help="Video label to reserve as evaluation data, usually Video 4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("easy_results"),
        help="Output directory (default: easy_results)",
    )
    return parser.parse_args()


def discover_csvs(inputs: list[str]) -> list[Path]:
    candidates: list[Path] = []
    if not inputs:
        current_directory = Path.cwd()
        tracking_data_directory = current_directory / "tracking_data"
        repository_inputs = (
            list(
                tracking_data_directory.glob(
                    "member2_canonical_tracks*.csv"
                )
            )
            if tracking_data_directory.is_dir()
            else []
        )
        standalone_inputs = list(
            current_directory.glob("member2_canonical_tracks*.csv")
        )
        if repository_inputs:
            candidates.extend(repository_inputs)
        elif standalone_inputs:
            candidates.extend(standalone_inputs)
        else:
            candidates.extend(
                current_directory.rglob("member2_canonical_tracks*.csv")
            )
    else:
        for raw in inputs:
            if any(character in raw for character in "*?[]"):
                candidates.extend(Path(match) for match in glob.glob(raw, recursive=True))
                continue
            path = Path(raw)
            if path.is_dir():
                candidates.extend(path.rglob("member2_canonical_tracks*.csv"))
            else:
                candidates.append(path)

    unique: dict[str, Path] = {}
    for path in candidates:
        if path.is_file():
            unique[str(path.resolve())] = path
    def natural_path_key(item: Path) -> list[int | str]:
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(item))
        ]

    paths = sorted(unique.values(), key=natural_path_key)
    if not paths:
        raise FileNotFoundError(
            "No canonical CSVs found. Put member2_canonical_tracks*.csv "
            "beside the program or pass the file paths."
        )
    return paths


def required_text(row: dict[str, str], key: str, row_number: int) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"row {row_number}: missing {key!r}")
    return value.strip()


def optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {key!r}")
    return parsed


def load_detections(path: Path) -> tuple[list[Detection], str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(BASE_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {', '.join(missing)}"
            )
        id_column = next((column for column in ID_COLUMNS if column in columns), None)
        if id_column is None:
            raise ValueError(
                f"{path.name} needs either canonical_id or track_id"
            )

        detections: list[Detection] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                class_id_text = (row.get("class_id") or "").strip()
                detection = Detection(
                    frame=int(float(required_text(row, "frame", row_number))),
                    time_seconds=float(
                        required_text(row, "time_seconds", row_number)
                    ),
                    canonical_id=required_text(row, id_column, row_number),
                    class_id=(
                        int(float(class_id_text)) if class_id_text else None
                    ),
                    class_name=required_text(row, "class_name", row_number),
                    confidence=optional_float(row, "confidence"),
                    center_x=float(required_text(row, "center_x", row_number)),
                    center_y=float(required_text(row, "center_y", row_number)),
                    frame_bottom=optional_float(row, "y2"),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse {path.name} row {row_number}: {exc}"
                ) from exc
            numeric = (
                detection.time_seconds,
                detection.center_x,
                detection.center_y,
            )
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(
                    f"{path.name} row {row_number} contains a non-finite number"
                )
            detections.append(detection)
    if not detections:
        raise ValueError(f"{path.name} contains no tracking rows")
    return detections, id_column


def normalize_class_name(name: str) -> str:
    cleaned = " ".join(name.strip().lower().replace("_", " ").split())
    return CLASS_NAMES.get(cleaned, cleaned.title())


def inferred_video_number(path: Path) -> int | None:
    stem = path.stem
    match = re.search(r"(?:video|vid|v)[ _-]?0*(\d+)", stem, re.IGNORECASE)
    if match:
        return int(match.group(1))
    trailing_number = re.search(r"0*(\d+)\)?$", stem)
    if trailing_number:
        return int(trailing_number.group(1))
    return None


def default_video_label(path: Path, fallback_number: int) -> str:
    number = inferred_video_number(path)
    return f"Video {number if number is not None else fallback_number}"


def load_config(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    videos = data.get("videos")
    if not isinstance(videos, dict):
        raise ValueError("Configuration must contain a 'videos' object")
    result: dict[str, dict[str, object]] = {}
    for key, settings in videos.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Configuration entry {key!r} must be an object")
        result[str(key)] = settings
    return result


def settings_for_path(
    path: Path,
    config: dict[str, dict[str, object]],
    default_threshold: float,
    default_toward: str,
    default_counting_mode: str,
    default_passage_fraction: float,
    fallback_video_number: int,
) -> tuple[str, float, str, str, float]:
    settings: dict[str, object] = {}
    default_label = default_video_label(path, fallback_video_number)
    for candidate in (
        str(path),
        str(path.resolve()),
        path.name,
        path.stem,
        default_label,
    ):
        if candidate in config:
            settings = config[candidate]
            break
    label = str(
        settings.get(
            "video_label",
            default_label,
        )
    )
    threshold = float(
        settings.get("movement_threshold_pixels", default_threshold)
    )
    toward = str(settings.get("toward_camera", default_toward)).lower()
    counting_mode = str(
        settings.get("counting_mode", default_counting_mode)
    ).lower()
    passage_fraction = float(
        settings.get("passage_line_fraction", default_passage_fraction)
    )
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError(f"{path.name}: movement threshold must be positive")
    if toward not in {"down", "up", "left", "right"}:
        raise ValueError(
            f"{path.name}: toward_camera must be down, up, left, or right"
        )
    if counting_mode not in {"auto", "movement", "passage"}:
        raise ValueError(
            f"{path.name}: counting_mode must be auto, movement, or passage"
        )
    if not 0 < passage_fraction < 1:
        raise ValueError(
            f"{path.name}: passage_line_fraction must be between 0 and 1"
        )
    return label, threshold, toward, counting_mode, passage_fraction


def robust_endpoint(rows: list[Detection], beginning: bool) -> tuple[float, float]:
    window_size = max(1, min(10, len(rows) // 4 or 1))
    sample = rows[:window_size] if beginning else rows[-window_size:]
    return (
        statistics.median(row.center_x for row in sample),
        statistics.median(row.center_y for row in sample),
    )


def camera_projection(delta_x: float, delta_y: float, toward: str) -> float:
    return {
        "down": delta_y,
        "up": -delta_y,
        "right": delta_x,
        "left": -delta_x,
    }[toward]


def direction_from_motion(
    delta_x: float,
    delta_y: float,
    toward: str,
    cross_traffic_ratio: float,
) -> str:
    net = math.hypot(delta_x, delta_y)
    if net < 1e-9:
        return "Mixed/unclear"
    projection = camera_projection(delta_x, delta_y, toward)
    if abs(projection) / net < cross_traffic_ratio:
        return "Cross-traffic"
    return "Toward camera" if projection > 0 else "Away from camera"


def find_horizontal_passage(
    rows: list[Detection],
    line_y: float,
    hysteresis_pixels: float,
) -> tuple[float, Detection, Detection] | None:
    """Return the first stable crossing of a horizontal passage line."""
    previous: Detection | None = None
    previous_side = 0
    for current in rows:
        distance = current.center_y - line_y
        side = (
            -1
            if distance < -hysteresis_pixels
            else 1
            if distance > hysteresis_pixels
            else 0
        )
        if side == 0:
            continue
        if previous is None:
            previous = current
            previous_side = side
            continue
        if side != previous_side:
            before_distance = previous.center_y - line_y
            after_distance = current.center_y - line_y
            denominator = before_distance - after_distance
            fraction = (
                0.5
                if denominator == 0
                else before_distance / denominator
            )
            fraction = min(1.0, max(0.0, fraction))
            crossing_time = previous.time_seconds + fraction * (
                current.time_seconds - previous.time_seconds
            )
            return crossing_time, previous, current
        previous = current
        previous_side = side
    return None


def vehicle_id_rate_per_minute(
    detections: Iterable[Detection],
) -> float:
    rows = list(detections)
    duration = max(row.time_seconds for row in rows) - min(
        row.time_seconds for row in rows
    )
    vehicle_ids = {
        row.canonical_id
        for row in rows
        if normalize_class_name(row.class_name) in VEHICLE_CLASSES
    }
    return len(vehicle_ids) / max(duration / 60.0, 1e-9)


def choose_counting_method(
    requested_mode: str,
    vehicle_ids_per_minute: float,
    fragmentation_threshold: float,
) -> str:
    if requested_mode == "auto":
        return (
            "passage"
            if vehicle_ids_per_minute >= fragmentation_threshold
            else "movement"
        )
    return requested_mode


def analyze_tracks(
    detections: Iterable[Detection],
    video: str,
    source_csv: Path,
    movement_threshold: float,
    toward: str,
    cross_traffic_ratio: float,
    min_track_frames: int,
    counting_method: str,
    passage_line_y: float | None,
    passage_hysteresis_pixels: float,
) -> list[TrackResult]:
    grouped: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.canonical_id].append(detection)

    results: list[TrackResult] = []
    for canonical_id, rows in grouped.items():
        rows.sort(key=lambda item: (item.time_seconds, item.frame))
        class_votes = Counter(normalize_class_name(row.class_name) for row in rows)
        class_name = class_votes.most_common(1)[0][0]
        start_x, start_y = robust_endpoint(rows, True)
        end_x, end_y = robust_endpoint(rows, False)
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        net_displacement = math.hypot(delta_x, delta_y)
        center_xs = [row.center_x for row in rows]
        center_ys = [row.center_y for row in rows]
        trajectory_span = math.hypot(
            max(center_xs) - min(center_xs),
            max(center_ys) - min(center_ys),
        )
        enough_rows = len(rows) >= min_track_frames
        moved = enough_rows and trajectory_span >= movement_threshold
        passage = (
            find_horizontal_passage(
                rows,
                passage_line_y,
                passage_hysteresis_pixels,
            )
            if counting_method == "passage" and passage_line_y is not None
            else None
        )
        if counting_method == "passage":
            counted = enough_rows and passage is not None
            status = (
                "Passed counting line"
                if counted
                else "Too few rows"
                if not enough_rows
                else "Did not pass counting line"
            )
            if passage is not None:
                count_time, direction_before, direction_after = passage
                direction = direction_from_motion(
                    direction_after.center_x - direction_before.center_x,
                    direction_after.center_y - direction_before.center_y,
                    toward,
                    cross_traffic_ratio,
                )
            else:
                count_time = rows[0].time_seconds
                direction = ""
        else:
            counted = moved
            status = (
                "Moving"
                if moved
                else "Too few rows"
                if not enough_rows
                else "Parked/stationary"
            )
            count_time = rows[0].time_seconds
            direction = (
                direction_from_motion(
                    delta_x, delta_y, toward, cross_traffic_ratio
                )
                if counted
                else ""
            )
        direction = (
            direction
            if counted
            else ""
        )
        confidences = [
            row.confidence for row in rows if row.confidence is not None
        ]
        results.append(
            TrackResult(
                video=video,
                source_csv=str(source_csv.resolve()),
                canonical_id=canonical_id,
                class_name=class_name,
                row_count=len(rows),
                first_time_seconds=rows[0].time_seconds,
                last_time_seconds=rows[-1].time_seconds,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                delta_x=delta_x,
                delta_y=delta_y,
                net_displacement_pixels=net_displacement,
                trajectory_span_pixels=trajectory_span,
                movement_threshold_pixels=movement_threshold,
                counting_method=counting_method,
                passage_line_y=passage_line_y,
                count_time_seconds=count_time,
                status=status,
                counted=counted,
                direction=direction,
                mean_confidence=(
                    sum(confidences) / len(confidences)
                    if confidences
                    else None
                ),
            )
        )
    return sorted(results, key=lambda item: id_sort_key(item.canonical_id))


def id_sort_key(identifier: str) -> tuple[int, float | str]:
    try:
        return 0, float(identifier)
    except ValueError:
        return 1, identifier


def write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_rows(results: list[TrackResult]) -> list[dict[str, object]]:
    return [
        {
            "Video": item.video,
            "Source CSV": item.source_csv,
            "Canonical ID": item.canonical_id,
            "Class": item.class_name,
            "Rows": item.row_count,
            "First time (s)": round(item.first_time_seconds, 3),
            "Last time (s)": round(item.last_time_seconds, 3),
            "Start X": round(item.start_x, 3),
            "Start Y": round(item.start_y, 3),
            "End X": round(item.end_x, 3),
            "End Y": round(item.end_y, 3),
            "Delta X": round(item.delta_x, 3),
            "Delta Y": round(item.delta_y, 3),
            "Net displacement (px)": round(item.net_displacement_pixels, 3),
            "Trajectory span (px)": round(item.trajectory_span_pixels, 3),
            "Movement threshold (px)": round(
                item.movement_threshold_pixels, 3
            ),
            "Counting method": item.counting_method,
            "Passage line Y": (
                round(item.passage_line_y, 3)
                if item.passage_line_y is not None
                else ""
            ),
            "Count time (s)": round(item.count_time_seconds, 3),
            "Status": item.status,
            "Counted": "Yes" if item.counted else "No",
            "Direction": item.direction,
            "Mean confidence": (
                round(item.mean_confidence, 6)
                if item.mean_confidence is not None
                else ""
            ),
        }
        for item in results
    ]


def automatic_count_rows(
    videos: list[str], results: list[TrackResult]
) -> list[dict[str, object]]:
    moving = [item for item in results if item.counted]
    counts = Counter(
        (item.video, item.direction, item.class_name) for item in moving
    )
    rows: list[dict[str, object]] = []
    for video in videos:
        combinations = sorted(
            {
                (item.direction, item.class_name)
                for item in moving
                if item.video == video
            }
        )
        for direction, class_name in combinations:
            rows.append(
                {
                    "Video": video,
                    "Direction": direction,
                    "Class": class_name,
                    "Automatic count": counts[(video, direction, class_name)],
                }
            )
    return rows


def easy_summary_rows(
    videos: list[str], results: list[TrackResult]
) -> list[dict[str, object]]:
    """Create one immediately readable row per video."""
    rows: list[dict[str, object]] = []
    for video in videos:
        video_results = [item for item in results if item.video == video]
        moving = [item for item in video_results if item.counted]
        class_counts = Counter(item.class_name for item in moving)
        direction_counts = Counter(item.direction for item in moving)
        standard_total = sum(class_counts[name] for name in SUMMARY_CLASSES)
        vehicle_total = sum(
            class_counts[name] for name in VEHICLE_CLASSES
        )
        row: dict[str, object] = {
            "Video": video,
            "Moving objects counted": len(moving),
            "Vehicles counted": vehicle_total,
            "Pedestrians counted": class_counts["Pedestrian"],
            "Parked/excluded": len(video_results) - len(moving),
            "Toward camera": direction_counts["Toward camera"],
            "Away from camera": direction_counts["Away from camera"],
            "Cross-traffic": direction_counts["Cross-traffic"],
            "Mixed/unclear": direction_counts["Mixed/unclear"],
            "Cars": class_counts["Car"],
            "Trucks": class_counts["Truck"],
            "Motorcycles": class_counts["Motorcycle"],
            "Bicycles": class_counts["Bicycle"],
            "Pedestrians": class_counts["Pedestrian"],
            "Buses": class_counts["Bus"],
            "Vans": class_counts["Van"],
            "Other classes": len(moving) - standard_total,
        }
        rows.append(row)
    return rows


def _html_table(headers: list[str], rows: list[list[object]]) -> str:
    header_html = "".join(
        f"<th>{html.escape(str(header))}</th>" for header in headers
    )
    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + header_html
        + "</tr></thead><tbody>"
        + body_html
        + "</tbody></table></div>"
    )


def write_html_report(
    path: Path,
    videos: list[str],
    results: list[TrackResult],
    video_settings: dict[str, dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    moving_all = [item for item in results if item.counted]
    total_moving = len(moving_all)
    total_excluded = len(results) - total_moving
    summary_headers = list(summary_rows[0]) if summary_rows else ["Video"]
    summary_table = _html_table(
        summary_headers,
        [[row[header] for header in summary_headers] for row in summary_rows],
    )

    sections: list[str] = []
    for video in videos:
        video_results = [item for item in results if item.video == video]
        moving = [item for item in video_results if item.counted]
        class_counts = Counter(item.class_name for item in moving)
        direction_counts = Counter(item.direction for item in moving)
        settings = video_settings[video]
        method_label = (
            f"Passage line at y={float(settings['passage_line_y']):g}"
            if settings["selected_counting_method"] == "passage"
            else (
                "Movement threshold: "
                f"{float(settings['movement_threshold_pixels']):g} px"
            )
        )

        direction_phrases = {
            "Toward camera": "toward the camera",
            "Away from camera": "away from the camera",
            "Cross-traffic": "across the camera view",
            "Mixed/unclear": "in a mixed or unclear direction",
        }
        sentences = [
            f"{count} {class_name.lower()}{'' if count == 1 else 's'} "
            f"moving {direction_phrases[direction]}"
            for (direction, class_name), count in sorted(
                Counter(
                    (item.direction, item.class_name) for item in moving
                ).items()
            )
        ]
        plain_english = (
            "; ".join(sentences) + "."
            if sentences
            else "No moving objects passed the movement threshold."
        )
        class_table = _html_table(
            ["Class", "Count"],
            [
                [class_name, count]
                for class_name, count in sorted(class_counts.items())
            ]
            or [["No moving classes", 0]],
        )
        direction_table = _html_table(
            ["Direction", "Count"],
            [
                [direction, count]
                for direction, count in sorted(direction_counts.items())
            ]
            or [["No movement directions", 0]],
        )
        sections.append(
            f"""
            <section>
              <div class="section-heading">
                <div>
                  <p class="eyebrow">VIDEO RESULT</p>
                  <h2>{html.escape(video)}</h2>
                  <p class="source-file">Source file:
                    {html.escape(Path(str(settings['source_csv'])).name)}
                  </p>
                </div>
                <span class="threshold">{html.escape(method_label)}</span>
              </div>
              <div class="metrics">
                <div class="metric primary"><strong>{len(moving)}</strong><span>Moving objects counted</span></div>
                <div class="metric primary"><strong>{sum(class_counts[name] for name in VEHICLE_CLASSES)}</strong><span>Vehicles counted</span></div>
                <div class="metric"><strong>{class_counts['Pedestrian']}</strong><span>Pedestrians counted</span></div>
                <div class="metric"><strong>{len(video_results) - len(moving)}</strong><span>Parked or excluded</span></div>
              </div>
              <p class="interpretation"><strong>Plain-English result:</strong>
                {html.escape(plain_english)}
              </p>
              <div class="two-column">
                <div><h3>Counts by class</h3>{class_table}</div>
                <div><h3>Counts by direction</h3>{direction_table}</div>
              </div>
            </section>
            """
        )

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FlowSense Traffic Results</title>
  <style>
    :root {{
      --navy: #132238; --blue: #2563eb; --pale: #eff6ff;
      --line: #dbe4f0; --muted: #5f6f82; --white: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background: #f4f7fb; color: var(--navy);
      font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 40px 22px 64px; }}
    header {{
      color: white; padding: 34px; border-radius: 20px;
      background: linear-gradient(135deg, #0f172a, #1d4ed8);
      box-shadow: 0 18px 48px rgba(29, 78, 216, .18);
    }}
    header h1 {{ margin: 4px 0 8px; font-size: clamp(30px, 5vw, 48px); }}
    header p {{ margin: 0; max-width: 760px; color: #dbeafe; }}
    .eyebrow {{ margin: 0; letter-spacing: .14em; font-size: 12px; font-weight: 800; color: #60a5fa; }}
    .overview, section {{
      margin-top: 24px; padding: 26px; border: 1px solid var(--line);
      border-radius: 18px; background: var(--white);
      box-shadow: 0 8px 28px rgba(15, 23, 42, .05);
    }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 18px 0; }}
    .metric {{ min-height: 112px; padding: 20px; border-radius: 14px; background: #f8fafc; border: 1px solid var(--line); }}
    .metric.primary {{ color: #1e40af; background: var(--pale); border-color: #bfdbfe; }}
    .metric strong {{ display: block; font-size: 34px; line-height: 1; margin-bottom: 10px; }}
    .metric span {{ color: var(--muted); font-weight: 650; }}
    .section-heading {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; }}
    .source-file {{ margin: -6px 0 10px; color: var(--muted); font-size: 13px; }}
    h2, h3 {{ margin: 4px 0 12px; }}
    .threshold {{ padding: 7px 11px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 13px; font-weight: 750; }}
    .interpretation {{ padding: 15px 17px; border-left: 4px solid var(--blue); background: var(--pale); border-radius: 8px; }}
    .two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th {{ color: #334155; background: #eef2f7; text-align: left; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
    td:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
    footer {{ margin-top: 24px; color: var(--muted); font-size: 13px; text-align: center; }}
    @media (max-width: 720px) {{
      .metrics, .two-column {{ grid-template-columns: 1fr; }}
      .section-heading {{ display: block; }}
      .threshold {{ display: inline-block; margin-bottom: 10px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <p class="eyebrow">FLOWSENSE</p>
      <h1>Automatic Traffic Results</h1>
      <p>Each moving canonical ID is counted exactly once. Parked and
      stationary tracks are excluded automatically.</p>
    </header>
    <div class="overview">
      <p class="eyebrow">ALL VIDEOS</p>
      <h2>Results at a glance</h2>
      <div class="metrics">
        <div class="metric primary"><strong>{len(videos)}</strong><span>Videos analyzed</span></div>
        <div class="metric primary"><strong>{total_moving}</strong><span>Moving objects counted</span></div>
        <div class="metric"><strong>{total_excluded}</strong><span>Parked or excluded</span></div>
      </div>
      <h3>One-row summary</h3>
      {summary_table}
    </div>
    {''.join(sections)}
    <footer>
      Open object_movement_audit.csv only when you need to check an individual ID.
    </footer>
  </main>
</body>
</html>
"""
    path.write_text(report, encoding="utf-8")


def interval_rows(
    videos: list[str],
    results: list[TrackResult],
    interval_seconds: int,
    durations: dict[str, float],
) -> list[dict[str, object]]:
    moving = [item for item in results if item.counted]
    counts: Counter[tuple[str, int, str, str]] = Counter()
    for item in moving:
        index = math.floor(
            (item.count_time_seconds + 1e-9) / interval_seconds
        )
        counts[(item.video, index, item.direction, item.class_name)] += 1

    rows: list[dict[str, object]] = []
    for video in videos:
        interval_count = (
            math.floor((durations[video] + 1e-9) / interval_seconds) + 1
        )
        groups = sorted(
            {
                (item.direction, item.class_name)
                for item in moving
                if item.video == video
            }
        )
        for index in range(interval_count):
            for direction, class_name in groups:
                rows.append(
                    {
                        "Video": video,
                        "Interval start (s)": index * interval_seconds,
                        "Interval end (s)": (index + 1) * interval_seconds,
                        "Direction": direction,
                        "Class": class_name,
                        "Automatic count": counts[
                            (video, index, direction, class_name)
                        ],
                    }
                )
    return rows


def normalized_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def normalize_direction(name: str) -> str:
    cleaned = " ".join(
        name.strip().lower().replace("_", " ").replace("-", " ").split()
    )
    aliases = {
        "toward": "Toward camera",
        "towards": "Toward camera",
        "toward camera": "Toward camera",
        "towards camera": "Toward camera",
        "away": "Away from camera",
        "away from camera": "Away from camera",
        "cross traffic": "Cross-traffic",
        "mixed": "Mixed/unclear",
        "unclear": "Mixed/unclear",
        "mixed unclear": "Mixed/unclear",
    }
    return aliases.get(cleaned, name.strip())


def load_kelvin_counts(
    path: Path,
) -> dict[tuple[str, str, str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Kelvin count file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header_map = {
            normalized_header(name): name for name in (reader.fieldnames or [])
        }

        def find_header(*options: str) -> str:
            for option in options:
                if option in header_map:
                    return header_map[option]
            raise ValueError(
                f"{path.name} needs Video, Direction, Class, and Kelvin count columns"
            )

        video_header = find_header("video")
        direction_header = find_header("direction")
        class_header = find_header("class", "classname")
        count_header = find_header(
            "kelvin", "kelvincount", "kelvinmanual", "manualcount"
        )
        counts: dict[tuple[str, str, str], int] = {}
        for row_number, row in enumerate(reader, start=2):
            key = (
                required_text(row, video_header, row_number),
                normalize_direction(
                    required_text(row, direction_header, row_number)
                ),
                normalize_class_name(
                    required_text(row, class_header, row_number)
                ),
            )
            count = int(float(required_text(row, count_header, row_number)))
            if count < 0:
                raise ValueError(f"{path.name} row {row_number} is negative")
            if key in counts:
                raise ValueError(f"Duplicate Kelvin row: {key}")
            counts[key] = count
    return counts


def comparison_rows(
    automatic_rows: list[dict[str, object]],
    kelvin_counts: dict[tuple[str, str, str], int] | None,
    evaluation_video: str | None,
) -> list[dict[str, object]]:
    automatic = {
        (str(row["Video"]), str(row["Direction"]), str(row["Class"])): int(
            row["Automatic count"]
        )
        for row in automatic_rows
    }
    keys = set(automatic) | (set(kelvin_counts) if kelvin_counts else set())
    rows: list[dict[str, object]] = []
    for video, direction, class_name in sorted(keys):
        flow_count = automatic.get((video, direction, class_name), 0)
        kelvin = (
            kelvin_counts.get((video, direction, class_name))
            if kelvin_counts
            else None
        )
        error = flow_count - kelvin if kelvin is not None else ""
        absolute_error = abs(error) if isinstance(error, int) else ""
        percent_error: float | str = ""
        if kelvin is not None:
            if kelvin == 0:
                percent_error = 0.0 if flow_count == 0 else ""
            else:
                percent_error = round(absolute_error / kelvin * 100, 2)
        rows.append(
            {
                "Video": video,
                "Dataset role": (
                    "Evaluation"
                    if evaluation_video and video == evaluation_video
                    else "Development"
                ),
                "Direction": direction,
                "Class": class_name,
                "Kelvin count": kelvin if kelvin is not None else "",
                "Automatic count": flow_count,
                "Error (automatic - Kelvin)": error,
                "Absolute error": absolute_error,
                "Percent error": percent_error,
            }
        )
    return rows


def video_alias_map(
    videos: list[str],
    video_settings: dict[str, dict[str, object]],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for video in videos:
        source = Path(str(video_settings[video]["source_csv"]))
        for alias in (video, source.name, source.stem):
            aliases[alias.strip().lower()] = video
    return aliases


def load_kelvin_class_totals(
    path: Path,
    aliases: dict[str, str],
) -> dict[tuple[str, str], int]:
    """Read either Kelvin's wide class columns or a long comparison table."""
    if not path.is_file():
        raise FileNotFoundError(f"Kelvin count file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        header_map = {normalized_header(name): name for name in fieldnames}
        if "video" not in header_map:
            raise ValueError(f"{path.name} needs a Video column")
        video_header = header_map["video"]
        rows = list(reader)

    def resolved_video(raw: str) -> str:
        cleaned = raw.strip()
        return aliases.get(cleaned.lower(), cleaned)

    count_header = next(
        (
            header_map[key]
            for key in (
                "kelvincount",
                "kelvinmanual",
                "manualcount",
                "kelvin",
            )
            if key in header_map
        ),
        None,
    )
    class_header = next(
        (
            header_map[key]
            for key in ("class", "classname")
            if key in header_map
        ),
        None,
    )
    totals: Counter[tuple[str, str]] = Counter()
    if class_header is not None and count_header is not None:
        for row in rows:
            raw_count = (row.get(count_header) or "").strip()
            if not raw_count:
                continue
            count = int(float(raw_count))
            if count < 0:
                raise ValueError("Kelvin counts cannot be negative")
            totals[
                (
                    resolved_video(row[video_header]),
                    normalize_class_name(row[class_header]),
                )
            ] += count
        return dict(totals)

    wide_classes = {
        "car": "Car",
        "cars": "Car",
        "truck": "Truck",
        "trucks": "Truck",
        "bus": "Bus",
        "buses": "Bus",
        "motorcycle": "Motorcycle",
        "motorcycles": "Motorcycle",
        "bicycle": "Bicycle",
        "bicycles": "Bicycle",
        "pedestrian": "Pedestrian",
        "pedestrians": "Pedestrian",
        "person": "Pedestrian",
        "people": "Pedestrian",
        "van": "Van",
        "vans": "Van",
    }
    class_columns = {
        original: wide_classes[normalized]
        for normalized, original in header_map.items()
        if normalized in wide_classes
    }
    if not class_columns:
        raise ValueError(
            f"{path.name} needs either Class/Kelvin count columns or "
            "wide class columns such as Cars, Trucks, and Buses"
        )
    for row in rows:
        video = resolved_video(row[video_header])
        for column, class_name in class_columns.items():
            raw_count = (row.get(column) or "").strip()
            if not raw_count:
                continue
            count = int(float(raw_count))
            if count < 0:
                raise ValueError("Kelvin counts cannot be negative")
            totals[(video, class_name)] += count
    return dict(totals)


def class_comparison_rows(
    videos: list[str],
    results: list[TrackResult],
    kelvin_totals: dict[tuple[str, str], int],
    evaluation_video: str | None,
) -> list[dict[str, object]]:
    automatic = Counter(
        (item.video, item.class_name)
        for item in results
        if item.counted
    )
    keys = set(automatic) | set(kelvin_totals)
    rows: list[dict[str, object]] = []
    for video, class_name in sorted(
        keys,
        key=lambda key: (
            videos.index(key[0]) if key[0] in videos else len(videos),
            key[1],
        ),
    ):
        flow_count = automatic[(video, class_name)]
        kelvin = kelvin_totals.get((video, class_name))
        error = flow_count - kelvin if kelvin is not None else ""
        absolute_error = abs(error) if isinstance(error, int) else ""
        percent_error: float | str = ""
        if kelvin is not None:
            if kelvin == 0:
                percent_error = 0.0 if flow_count == 0 else ""
            else:
                percent_error = round(absolute_error / kelvin * 100, 2)
        rows.append(
            {
                "Video": video,
                "Dataset role": (
                    "Evaluation"
                    if evaluation_video and video == evaluation_video
                    else "Development"
                ),
                "Class": class_name,
                "Kelvin count": kelvin if kelvin is not None else "",
                "Automatic count": flow_count,
                "Error (automatic - Kelvin)": error,
                "Absolute error": absolute_error,
                "Percent error": percent_error,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.movement_threshold_pixels <= 0:
        raise ValueError("--movement-threshold-pixels must be positive")
    if not 0 <= args.cross_traffic_ratio < 1:
        raise ValueError("--cross-traffic-ratio must be from 0 up to 1")
    if args.min_track_frames < 1:
        raise ValueError("--min-track-frames must be at least 1")
    if not 0 < args.passage_line_fraction < 1:
        raise ValueError("--passage-line-fraction must be between 0 and 1")
    if args.passage_hysteresis_pixels < 0:
        raise ValueError("--passage-hysteresis-pixels cannot be negative")
    if args.fragmentation_ids_per_minute <= 0:
        raise ValueError("--fragmentation-ids-per-minute must be positive")

    csv_paths = discover_csvs(args.inputs)
    explicit_numbers = {
        number
        for path in csv_paths
        if (number := inferred_video_number(path)) is not None
    }
    assigned_numbers: dict[Path, int] = {}
    next_available = 1
    for path in csv_paths:
        explicit = inferred_video_number(path)
        if explicit is not None:
            assigned_numbers[path] = explicit
            continue
        while next_available in explicit_numbers or next_available in assigned_numbers.values():
            next_available += 1
        assigned_numbers[path] = next_available
        next_available += 1
    csv_paths.sort(key=lambda path: (assigned_numbers[path], str(path).lower()))

    config = load_config(args.config)
    all_results: list[TrackResult] = []
    videos: list[str] = []
    video_settings: dict[str, dict[str, object]] = {}
    durations: dict[str, float] = {}

    for path in csv_paths:
        (
            video,
            threshold,
            toward,
            requested_counting_mode,
            passage_fraction,
        ) = settings_for_path(
            path,
            config,
            args.movement_threshold_pixels,
            args.toward_camera,
            args.counting_mode,
            args.passage_line_fraction,
            assigned_numbers[path],
        )
        if video in videos:
            raise ValueError(
                f"Duplicate video label {video!r}; give each config entry a unique label"
            )
        detections, id_column = load_detections(path)
        fragmentation_rate = vehicle_id_rate_per_minute(detections)
        selected_counting_method = choose_counting_method(
            requested_counting_mode,
            fragmentation_rate,
            args.fragmentation_ids_per_minute,
        )
        frame_height = max(
            (
                row.frame_bottom
                if row.frame_bottom is not None
                else row.center_y
            )
            for row in detections
        )
        passage_line_y = (
            frame_height * passage_fraction
            if selected_counting_method == "passage"
            else None
        )
        results = analyze_tracks(
            detections=detections,
            video=video,
            source_csv=path,
            movement_threshold=threshold,
            toward=toward,
            cross_traffic_ratio=args.cross_traffic_ratio,
            min_track_frames=args.min_track_frames,
            counting_method=selected_counting_method,
            passage_line_y=passage_line_y,
            passage_hysteresis_pixels=args.passage_hysteresis_pixels,
        )
        videos.append(video)
        all_results.extend(results)
        durations[video] = max(row.time_seconds for row in detections)
        video_settings[video] = {
            "source_csv": str(path.resolve()),
            "id_column_used": id_column,
            "movement_threshold_pixels": threshold,
            "toward_camera_image_direction": toward,
            "requested_counting_mode": requested_counting_mode,
            "selected_counting_method": selected_counting_method,
            "vehicle_ids_per_minute": round(fragmentation_rate, 3),
            "passage_line_fraction": (
                passage_fraction
                if selected_counting_method == "passage"
                else None
            ),
            "passage_line_y": (
                round(passage_line_y, 3)
                if passage_line_y is not None
                else None
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_counts = automatic_count_rows(videos, all_results)
    easy_summary = easy_summary_rows(videos, all_results)
    easy_fields = list(easy_summary[0]) if easy_summary else ["Video"]
    write_csv(
        args.output_dir / "automatic_counts.csv",
        easy_summary,
        easy_fields,
    )
    write_csv(
        args.output_dir / "automatic_counts_detailed.csv",
        detailed_counts,
        ["Video", "Direction", "Class", "Automatic count"],
    )
    write_csv(
        args.output_dir / "READ_ME_FIRST.csv",
        easy_summary,
        easy_fields,
    )
    write_html_report(
        args.output_dir / "FlowSense_report.html",
        videos,
        all_results,
        video_settings,
        easy_summary,
    )
    file_map_rows = [
        {
            "Video label": video,
            "Source filename": Path(
                str(video_settings[video]["source_csv"])
            ).name,
            "Full source path": video_settings[video]["source_csv"],
        }
        for video in videos
    ]
    write_csv(
        args.output_dir / "video_file_map.csv",
        file_map_rows,
        ["Video label", "Source filename", "Full source path"],
    )
    audit = audit_rows(all_results)
    audit_fields = list(audit[0]) if audit else [
        "Video", "Canonical ID", "Class", "Status", "Counted", "Direction"
    ]
    write_csv(
        args.output_dir / "object_movement_audit.csv",
        audit,
        audit_fields,
    )
    volume = interval_rows(
        videos, all_results, args.interval_seconds, durations
    )
    write_csv(
        args.output_dir / "traffic_volume_intervals.csv",
        volume,
        [
            "Video",
            "Interval start (s)",
            "Interval end (s)",
            "Direction",
            "Class",
            "Automatic count",
        ],
    )

    aliases = video_alias_map(videos, video_settings)
    kelvin_class_totals = (
        load_kelvin_class_totals(args.manual_counts, aliases)
        if args.manual_counts
        else {}
    )
    comparison = class_comparison_rows(
        videos,
        all_results,
        kelvin_class_totals,
        args.evaluation_video,
    )
    comparison_name = "comparison_by_video_class.csv"
    write_csv(
        args.output_dir / comparison_name,
        comparison,
        [
            "Video",
            "Dataset role",
            "Class",
            "Kelvin count",
            "Automatic count",
            "Error (automatic - Kelvin)",
            "Absolute error",
            "Percent error",
        ],
    )

    per_video: dict[str, dict[str, object]] = {}
    for video in videos:
        video_results = [item for item in all_results if item.video == video]
        moving = [item for item in video_results if item.counted]
        per_video[video] = {
            **video_settings[video],
            "canonical_ids": len(video_results),
            "moving_ids_counted": len(moving),
            "parked_or_excluded_ids": len(video_results) - len(moving),
            "counts_by_class": dict(
                sorted(Counter(item.class_name for item in moving).items())
            ),
            "counts_by_direction": dict(
                sorted(Counter(item.direction for item in moving).items())
            ),
        }
    summary: dict[str, object] = {
        "method": (
            "Rows are grouped by canonical_id (or track_id fallback). Auto "
            "mode uses one count per moving ID normally, but switches to one "
            "count per horizontal passage-line crossing when vehicle-ID "
            "fragmentation exceeds the configured rate."
        ),
        "settings": {
            "program_version": PROGRAM_VERSION,
            "default_counting_mode": args.counting_mode,
            "default_movement_threshold_pixels": args.movement_threshold_pixels,
            "default_passage_line_fraction": args.passage_line_fraction,
            "passage_hysteresis_pixels": args.passage_hysteresis_pixels,
            "fragmentation_ids_per_minute": args.fragmentation_ids_per_minute,
            "default_toward_camera_image_direction": args.toward_camera,
            "cross_traffic_ratio": args.cross_traffic_ratio,
            "interval_seconds": args.interval_seconds,
            "min_track_frames": args.min_track_frames,
            "evaluation_video": args.evaluation_video,
        },
        "videos": per_video,
        "combined": {
            "videos_analyzed": len(videos),
            "canonical_ids": len(all_results),
            "moving_ids_counted": sum(item.counted for item in all_results),
            "parked_or_excluded_ids": sum(
                not item.counted for item in all_results
            ),
            "counts_by_class": dict(
                sorted(
                    Counter(
                        item.class_name for item in all_results if item.counted
                    ).items()
                )
            ),
            "counts_by_direction": dict(
                sorted(
                    Counter(
                        item.direction for item in all_results if item.counted
                    ).items()
                )
            ),
        },
        "kelvin_comparison_file": comparison_name,
        "kelvin_counts_supplied": bool(kelvin_class_totals),
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def print_summary(summary: dict[str, object], output_dir: Path) -> None:
    combined = summary["combined"]
    assert isinstance(combined, dict)
    print(f"FlowSense automatic traffic counting complete ({PROGRAM_VERSION})")
    print(f"Videos analyzed: {combined['videos_analyzed']}")
    print(f"Moving IDs counted once: {combined['moving_ids_counted']}")
    print(f"Parked/excluded IDs: {combined['parked_or_excluded_ids']}")
    print(f"Counts by class: {combined['counts_by_class']}")
    print(f"Counts by direction: {combined['counts_by_direction']}")
    print(f"Results: {output_dir.resolve()}")
    print("Open FlowSense_report.html first for the easiest explanation.")


def main() -> None:
    args = parse_args()
    try:
        summary = run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
