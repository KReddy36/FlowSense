"""Offline accuracy evaluation for FlowSense short-term position predictions."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .tracking import LearnedMotionCorrector, MotionPredictor, TrackedDetection


@dataclass(frozen=True, slots=True)
class PredictionError:
    """One prediction compared with its later observed canonical center."""

    dataset: str
    track_id: int
    class_name: str
    source_frame: int
    target_frame: int
    horizon_seconds: float
    prediction_error_pixels: float
    stationary_error_pixels: float


def evaluate_tracking_csv(
    path: str | Path,
    *,
    dataset: str | None = None,
    history_points: int = 30,
    velocity_window: int = 5,
    prediction_horizon_frames: int = 15,
    inactive_timeout_frames: int = 30,
    learned_corrector: LearnedMotionCorrector | None = None,
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> list[PredictionError]:
    """Evaluate actual MotionPredictor outputs against future track centers."""
    csv_path = Path(path)
    rows = _load_tracks(csv_path)
    dataset_name = dataset or _dataset_label(csv_path)
    timestamps_by_frame = {
        row.frame_id: row.timestamp
        for frame_rows in rows.values()
        for row in frame_rows
    }
    if len(timestamps_by_frame) < 2:
        return []
    fps = _infer_fps(timestamps_by_frame)
    predictor = MotionPredictor(
        fps=fps,
        history_points=history_points,
        velocity_window=velocity_window,
        prediction_horizon_frames=prediction_horizon_frames,
        inactive_timeout_frames=inactive_timeout_frames,
        learned_corrector=learned_corrector,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    actual_centers = {
        (row.frame_id, row.track_id): row.center
        for frame_rows in rows.values()
        for row in frame_rows
    }
    minimum_history_points = velocity_window + 1
    errors: list[PredictionError] = []
    first_frame = min(rows)
    last_frame = max(rows)
    for frame_id in range(first_frame, last_frame + 1):
        timestamp = timestamps_by_frame.get(frame_id, frame_id / fps)
        snapshots = predictor.update(
            rows.get(frame_id, []),
            frame_id=frame_id,
            timestamp=timestamp,
        )
        for snapshot in snapshots:
            if not snapshot.observed:
                continue
            if len(snapshot.observed_points) < minimum_history_points:
                continue
            target = actual_centers.get(
                (snapshot.predicted_frame, snapshot.track_id)
            )
            if target is None:
                continue
            target_x, target_y = target
            errors.append(
                PredictionError(
                    dataset=dataset_name,
                    track_id=snapshot.track_id,
                    class_name=snapshot.class_name,
                    source_frame=frame_id,
                    target_frame=snapshot.predicted_frame,
                    horizon_seconds=(
                        snapshot.prediction_horizon_frames / fps
                    ),
                    prediction_error_pixels=math.hypot(
                        snapshot.predicted_center_x - target_x,
                        snapshot.predicted_center_y - target_y,
                    ),
                    stationary_error_pixels=math.hypot(
                        snapshot.center_x - target_x,
                        snapshot.center_y - target_y,
                    ),
                )
            )
    return errors


def summarize_errors(
    errors: list[PredictionError],
    *,
    dataset: str,
    prediction_horizon_frames: int,
) -> dict[str, object]:
    """Summarize prediction and stationary-baseline errors."""
    if not errors:
        raise ValueError(f"No eligible prediction samples for {dataset}")
    prediction_errors = [
        sample.prediction_error_pixels for sample in errors
    ]
    stationary_errors = [
        sample.stationary_error_pixels for sample in errors
    ]
    median_prediction = statistics.median(prediction_errors)
    median_stationary = statistics.median(stationary_errors)
    improvement = (
        100.0 * (median_stationary - median_prediction) / median_stationary
        if median_stationary > 0
        else 0.0
    )
    wins = sum(
        prediction < stationary
        for prediction, stationary in zip(
            prediction_errors,
            stationary_errors,
            strict=True,
        )
    )
    return {
        "Dataset": dataset,
        "Samples": len(errors),
        "Track IDs": len(
            {(sample.dataset, sample.track_id) for sample in errors}
        ),
        "Prediction horizon (frames)": prediction_horizon_frames,
        "Prediction horizon (seconds)": round(
            statistics.median(
                sample.horizon_seconds for sample in errors
            ),
            4,
        ),
        "Median prediction error (px)": round(median_prediction, 3),
        "Mean prediction error (px)": round(
            statistics.fmean(prediction_errors),
            3,
        ),
        "P90 prediction error (px)": round(
            _percentile(prediction_errors, 0.90),
            3,
        ),
        "Median stationary baseline error (px)": round(
            median_stationary,
            3,
        ),
        "Mean stationary baseline error (px)": round(
            statistics.fmean(stationary_errors),
            3,
        ),
        "Median improvement vs baseline (%)": round(improvement, 2),
        "Prediction win rate (%)": round(100.0 * wins / len(errors), 2),
    }


def evaluate_datasets(
    paths: list[str | Path],
    *,
    history_points: int = 30,
    velocity_window: int = 5,
    prediction_horizon_frames: int = 15,
    inactive_timeout_frames: int = 30,
) -> tuple[list[dict[str, object]], list[PredictionError]]:
    """Evaluate each dataset and return per-video plus combined summaries."""
    all_errors: list[PredictionError] = []
    summaries: list[dict[str, object]] = []
    for path in paths:
        csv_path = Path(path)
        errors = evaluate_tracking_csv(
            csv_path,
            history_points=history_points,
            velocity_window=velocity_window,
            prediction_horizon_frames=prediction_horizon_frames,
            inactive_timeout_frames=inactive_timeout_frames,
        )
        all_errors.extend(errors)
        summaries.append(
            summarize_errors(
                errors,
                dataset=_dataset_label(csv_path),
                prediction_horizon_frames=prediction_horizon_frames,
            )
        )
    summaries.append(
        summarize_errors(
            all_errors,
            dataset="All videos",
            prediction_horizon_frames=prediction_horizon_frames,
        )
    )
    return summaries, all_errors


def write_accuracy_csv(
    path: str | Path,
    summaries: list[dict[str, object]],
) -> None:
    """Write the final prediction-accuracy table."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_markdown_report(
    path: str | Path,
    summaries: list[dict[str, object]],
    *,
    history_points: int,
    velocity_window: int,
    prediction_horizon_frames: int,
) -> None:
    """Write a readable methodology and headline accuracy table."""
    headers = [
        "Dataset",
        "Samples",
        "Median prediction error (px)",
        "Median stationary baseline error (px)",
        "Median improvement vs baseline (%)",
        "Prediction win rate (%)",
    ]
    separator = ["---", "---:", "---:", "---:", "---:", "---:"]
    table_rows = [
        [str(summary[header]) for header in headers]
        for summary in summaries
    ]
    table = "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(separator) + " |",
            *[
                "| " + " | ".join(row) + " |"
                for row in table_rows
            ],
        ]
    )
    overall = summaries[-1]
    worse_datasets = [
        str(summary["Dataset"])
        for summary in summaries[:-1]
        if float(summary["Median improvement vs baseline (%)"]) < 0
    ]
    limitation = (
        " The stationary baseline was stronger for "
        + ", ".join(worse_datasets)
        + ", so the predictor is not uniformly better on every scene."
        if worse_datasets
        else ""
    )
    report = f"""# FlowSense Prediction Evaluation

This evaluation measures the short-term position predictor against later
observed canonical track centers. Each sample uses only information available
through its source frame. A sample is eligible when the track has at least
`velocity_window + 1` observed points and the same canonical ID is observed
exactly {prediction_horizon_frames} frames later.

The prediction error is Euclidean pixel distance from the predicted center to
the future observed center. The stationary baseline assumes the object remains
at its current observed center. Lower error is better.

## Configuration

- Rolling history: {history_points} points
- Averaged recent velocities: {velocity_window}
- Prediction horizon: {prediction_horizon_frames} frames

## Results

{table}

Across all videos, median prediction error was
{overall["Median prediction error (px)"]} px versus
{overall["Median stationary baseline error (px)"]} px for the stationary
baseline, a {overall["Median improvement vs baseline (%)"]}% reduction in
median error. The predictor beat the baseline on
{overall["Prediction win rate (%)"]}% of eligible samples.{limitation}

## Limitations

Future canonical track centers are used as observed ground truth. This
evaluates the prediction algorithm consistently on the repository datasets,
but detector or tracker localization errors can affect both the prediction
inputs and the future reference centers. It is not an independent
human-annotated position benchmark.

The full table, including mean and 90th-percentile errors, is stored in
`easy_results/prediction_accuracy.csv`.
"""
    Path(path).write_text(report, encoding="utf-8")


def _load_tracks(path: Path) -> dict[int, list[TrackedDetection]]:
    required = {
        "frame",
        "time_seconds",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    tracks: dict[int, list[TrackedDetection]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                track = TrackedDetection(
                    frame_id=int(float(row["frame"])),
                    timestamp=float(row["time_seconds"]),
                    track_id=int(float(row["track_id"])),
                    class_id=int(float(row["class_id"])),
                    class_name=row["class_name"],
                    confidence=float(row["confidence"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid row {line_number} in {path.name}: {exc}"
                ) from exc
            tracks[track.frame_id].append(track)
    if not tracks:
        raise ValueError(f"{path.name} contains no tracks")
    return dict(tracks)


def _infer_fps(timestamps_by_frame: dict[int, float]) -> float:
    ordered = sorted(timestamps_by_frame.items())
    frame_rates = [
        (frame - previous_frame) / (timestamp - previous_timestamp)
        for (previous_frame, previous_timestamp), (frame, timestamp)
        in zip(ordered, ordered[1:])
        if timestamp > previous_timestamp and frame > previous_frame
    ]
    if not frame_rates:
        raise ValueError("Could not infer frame rate from timestamps")
    return statistics.median(frame_rates)


def _dataset_label(path: Path) -> str:
    stem = path.stem
    suffix = stem.removeprefix("member2_canonical_tracks")
    return f"Video {suffix or '1'}"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]
