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
        default=Path("automatic_counter_results"),
        help="Output directory (default: automatic_counter_results)",
    )
    return parser.parse_args()


def discover_csvs(inputs: list[str]) -> list[Path]:
    candidates: list[Path] = []
    if not inputs:
        candidates.extend(
            Path.cwd().rglob("member2_canonical_tracks*.csv")
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
    paths = sorted(unique.values(), key=lambda item: str(item).lower())
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


def default_video_label(path: Path) -> str:
    stem = path.stem
    match = re.search(r"(?:video|vid|v)[ _-]?0*(\d+)", stem, re.IGNORECASE)
    if match:
        return f"Video {int(match.group(1))}"
    return stem


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
) -> tuple[str, float, str]:
    settings: dict[str, object] = {}
    for candidate in (str(path), str(path.resolve()), path.name, path.stem):
        if candidate in config:
            settings = config[candidate]
            break
    label = str(settings.get("video_label", default_video_label(path)))
    threshold = float(
        settings.get("movement_threshold_pixels", default_threshold)
    )
    toward = str(settings.get("toward_camera", default_toward)).lower()
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError(f"{path.name}: movement threshold must be positive")
    if toward not in {"down", "up", "left", "right"}:
        raise ValueError(
            f"{path.name}: toward_camera must be down, up, left, or right"
        )
    return label, threshold, toward


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


def analyze_tracks(
    detections: Iterable[Detection],
    video: str,
    source_csv: Path,
    movement_threshold: float,
    toward: str,
    cross_traffic_ratio: float,
    min_track_frames: int,
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
        status = (
            "Moving"
            if moved
            else "Too few rows"
            if not enough_rows
            else "Parked/stationary"
        )
        direction = (
            direction_from_motion(
                delta_x, delta_y, toward, cross_traffic_ratio
            )
            if moved
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
                status=status,
                counted=moved,
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
            (item.first_time_seconds + 1e-9) / interval_seconds
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


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.movement_threshold_pixels <= 0:
        raise ValueError("--movement-threshold-pixels must be positive")
    if not 0 <= args.cross_traffic_ratio < 1:
        raise ValueError("--cross-traffic-ratio must be from 0 up to 1")
    if args.min_track_frames < 1:
        raise ValueError("--min-track-frames must be at least 1")

    csv_paths = discover_csvs(args.inputs)
    config = load_config(args.config)
    all_results: list[TrackResult] = []
    videos: list[str] = []
    video_settings: dict[str, dict[str, object]] = {}
    durations: dict[str, float] = {}

    for path in csv_paths:
        video, threshold, toward = settings_for_path(
            path,
            config,
            args.movement_threshold_pixels,
            args.toward_camera,
        )
        if video in videos:
            raise ValueError(
                f"Duplicate video label {video!r}; give each config entry a unique label"
            )
        detections, id_column = load_detections(path)
        results = analyze_tracks(
            detections=detections,
            video=video,
            source_csv=path,
            movement_threshold=threshold,
            toward=toward,
            cross_traffic_ratio=args.cross_traffic_ratio,
            min_track_frames=args.min_track_frames,
        )
        videos.append(video)
        all_results.extend(results)
        durations[video] = max(row.time_seconds for row in detections)
        video_settings[video] = {
            "source_csv": str(path.resolve()),
            "id_column_used": id_column,
            "movement_threshold_pixels": threshold,
            "toward_camera_image_direction": toward,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = automatic_count_rows(videos, all_results)
    write_csv(
        args.output_dir / "automatic_counts.csv",
        counts,
        ["Video", "Direction", "Class", "Automatic count"],
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

    kelvin = (
        load_kelvin_counts(args.manual_counts)
        if args.manual_counts
        else None
    )
    comparison = comparison_rows(counts, kelvin, args.evaluation_video)
    comparison_name = (
        "comparison_with_kelvin.csv"
        if kelvin is not None
        else "kelvin_comparison_template.csv"
    )
    write_csv(
        args.output_dir / comparison_name,
        comparison,
        [
            "Video",
            "Dataset role",
            "Direction",
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
            "Rows are grouped by canonical_id (or track_id fallback). Each ID "
            "is counted once when its center trajectory span reaches the "
            "configured movement threshold. Smaller spans are classified as "
            "parked/stationary."
        ),
        "settings": {
            "default_movement_threshold_pixels": args.movement_threshold_pixels,
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
    print("FlowSense automatic traffic counting complete")
    print(f"Videos analyzed: {combined['videos_analyzed']}")
    print(f"Moving IDs counted once: {combined['moving_ids_counted']}")
    print(f"Parked/excluded IDs: {combined['parked_or_excluded_ids']}")
    print(f"Counts by class: {combined['counts_by_class']}")
    print(f"Counts by direction: {combined['counts_by_direction']}")
    print(f"Results: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    try:
        summary = run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
