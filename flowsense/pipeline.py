"""Single-command video-to-results pipeline for FlowSense."""

from __future__ import annotations

import csv
import shutil
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from automatic_counter import run_single_video_report

from .prediction_evaluation import evaluate_tracking_csv, summarize_errors
from .tracking import ByteTrackTracker, IdentityConsolidator, MotionPredictor
from .tracking.render import render_motion_paths, render_tracking_ids
from .tracking.schemas import Detection
from .video_compat import VideoConversionError, convert_to_browser_mp4
from .yolo_detector import YoloDetector


TRACK_COLUMNS = (
    "frame",
    "time_seconds",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "center_x",
    "center_y",
    "x1",
    "y1",
    "x2",
    "y2",
)
MOTION_COLUMNS = (
    "frame",
    "time_seconds",
    "track_id",
    "class_id",
    "class_name",
    "is_observed",
    "frames_since_seen",
    "estimated_center_x",
    "estimated_center_y",
    "velocity_x_pixels_per_second",
    "velocity_y_pixels_per_second",
    "speed_pixels_per_second",
    "direction_degrees",
    "prediction_horizon_frames",
    "predicted_frame",
    "predicted_time_seconds",
    "predicted_center_x",
    "predicted_center_y",
)
ProgressCallback = Callable[[int, int], None]


class FrameDetector(Protocol):
    """Detector interface used by the streaming pipeline."""

    def detect(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
        timestamp: float,
    ) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """User-facing settings for one end-to-end FlowSense run."""

    input_video: Path
    output_dir: Path = Path("results")
    output_video: Path | None = None
    output_report: Path | None = None
    model_name: str = "yolo11n.pt"
    confidence: float = 0.35
    iou: float = 0.50
    device: str | None = None
    history_points: int = 30
    velocity_window: int = 5
    prediction_horizon_frames: int = 15
    inactive_timeout_frames: int = 30
    movement_threshold_pixels: float = 50.0
    toward_camera: str = "down"
    counting_mode: str = "auto"
    maximum_frames: int | None = None
    keep_intermediates: bool = False
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Final artifacts and headline metrics from a completed run."""

    output_video: Path
    output_report: Path
    browser_compatible_video: bool
    video_preview_warning: str | None
    processed_frames: int
    detected_instances: int
    rendered_track_instances: int
    unique_track_ids: int
    suppressed_duplicate_instances: int
    counts_by_class: dict[str, int]
    counts_by_direction: dict[str, int]
    prediction_accuracy_percent: float | None
    prediction_accuracy_samples: int
    intermediate_dir: Path | None


def run_pipeline(
    config: PipelineConfig,
    *,
    detector: FrameDetector | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """Run detection, tracking, prediction, counting, and report generation."""
    _validate_config(config)
    input_video = config.input_video.resolve()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video = _resolve_output_path(
        config.output_video,
        output_dir / f"{input_video.stem}_flowsense.mp4",
        output_dir,
    )
    final_report = _resolve_output_path(
        config.output_report,
        output_dir / f"{input_video.stem}_report.html",
        output_dir,
    )
    _validate_destination(input_video, final_video, config.overwrite)
    _validate_destination(input_video, final_report, config.overwrite)
    if final_video == final_report:
        raise ValueError("The output video and HTML report need different paths")

    workspace_context: AbstractContextManager[str | Path]
    intermediate_dir: Path | None
    if config.keep_intermediates:
        intermediate_dir = output_dir / "intermediates"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        workspace_context = nullcontext(intermediate_dir)
    else:
        intermediate_dir = None
        workspace_context = tempfile.TemporaryDirectory(
            prefix=".flowsense-",
            dir=output_dir,
        )

    with workspace_context as workspace_value:
        workspace = Path(workspace_value)
        staged_video = workspace / "annotated_video.mp4"
        browser_video = workspace / "annotated_video_h264.mp4"
        canonical_tracks = workspace / "canonical_tracks.csv"
        motion_history = (
            workspace / "motion_predictions.csv"
            if config.keep_intermediates
            else None
        )
        counter_outputs = workspace / "counter_outputs"
        summary = _process_video(
            config,
            input_video,
            staged_video,
            canonical_tracks,
            motion_history,
            detector=detector,
            progress_callback=progress_callback,
        )
        if summary["rendered_track_instances"] == 0:
            raise RuntimeError(
                "YOLO did not produce any trackable road-user detections. "
                "Try a lower --confidence value or a different video."
            )

        prediction_errors = evaluate_tracking_csv(
            canonical_tracks,
            dataset=input_video.stem,
            history_points=config.history_points,
            velocity_window=config.velocity_window,
            prediction_horizon_frames=config.prediction_horizon_frames,
            inactive_timeout_frames=config.inactive_timeout_frames,
        )
        prediction_accuracy_percent: float | None = None
        if prediction_errors:
            prediction_summary = summarize_errors(
                prediction_errors,
                dataset=input_video.stem,
                prediction_horizon_frames=(
                    config.prediction_horizon_frames
                ),
            )
            prediction_accuracy_percent = float(
                prediction_summary["Prediction win rate (%)"]
            )

        count_summary = run_single_video_report(
            canonical_tracks,
            counter_outputs,
            video_label=input_video.stem,
            source_video=input_video,
            movement_threshold_pixels=config.movement_threshold_pixels,
            toward_camera=config.toward_camera,
            counting_mode=config.counting_mode,
            prediction_accuracy_percent=prediction_accuracy_percent,
            prediction_accuracy_samples=len(prediction_errors),
            prediction_horizon_frames=config.prediction_horizon_frames,
        )
        staged_report = counter_outputs / "FlowSense_report.html"
        if not staged_report.is_file():
            raise RuntimeError("The counting stage did not create its HTML report")

        browser_compatible_video = True
        video_preview_warning: str | None = None
        video_to_publish = staged_video
        try:
            video_to_publish = convert_to_browser_mp4(
                staged_video,
                browser_video,
            )
            staged_video.unlink(missing_ok=True)
        except VideoConversionError as exc:
            browser_compatible_video = False
            video_preview_warning = (
                "Browser-compatible H.264 conversion was unavailable. "
                "The original annotated MP4 is still available to download. "
                f"{exc}"
            )

        _publish_artifact(video_to_publish, final_video, config.overwrite)
        _publish_artifact(staged_report, final_report, config.overwrite)

        combined = count_summary["combined"]
        if not isinstance(combined, dict):
            raise RuntimeError("The counting stage returned an invalid summary")
        return PipelineResult(
            output_video=final_video,
            output_report=final_report,
            browser_compatible_video=browser_compatible_video,
            video_preview_warning=video_preview_warning,
            processed_frames=int(summary["processed_frames"]),
            detected_instances=int(summary["detected_instances"]),
            rendered_track_instances=int(summary["rendered_track_instances"]),
            unique_track_ids=int(summary["unique_track_ids"]),
            suppressed_duplicate_instances=int(
                summary["suppressed_duplicate_instances"]
            ),
            counts_by_class={
                str(name): int(count)
                for name, count in dict(combined["counts_by_class"]).items()
            },
            counts_by_direction={
                str(name): int(count)
                for name, count in dict(
                    combined["counts_by_direction"]
                ).items()
            },
            prediction_accuracy_percent=prediction_accuracy_percent,
            prediction_accuracy_samples=len(prediction_errors),
            intermediate_dir=intermediate_dir,
        )


def _process_video(
    config: PipelineConfig,
    input_video: Path,
    staged_video: Path,
    canonical_tracks: Path,
    motion_history: Path | None,
    *,
    detector: FrameDetector | None,
    progress_callback: ProgressCallback | None,
) -> dict[str, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is missing. Run: python -m pip install -r requirements.txt"
        ) from exc

    active_detector = detector or YoloDetector(
        config.model_name,
        confidence=config.confidence,
        iou=config.iou,
        device=config.device,
    )
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError("Input video has invalid width, height, or frame rate")
    progress_total = max(0, total_frames)
    if config.maximum_frames is not None and progress_total > 0:
        progress_total = min(progress_total, config.maximum_frames)
    active_progress_callback = _notify_progress(
        progress_callback,
        processed_frames=0,
        total_frames=progress_total,
    )

    writer = cv2.VideoWriter(
        str(staged_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {staged_video}")

    tracker = ByteTrackTracker(frame_rate=max(1, round(fps)))
    consolidator = IdentityConsolidator()
    predictor = MotionPredictor(
        fps=fps,
        history_points=config.history_points,
        velocity_window=config.velocity_window,
        prediction_horizon_frames=config.prediction_horizon_frames,
        inactive_timeout_frames=config.inactive_timeout_frames,
    )
    detected_instances = 0
    rendered_track_instances = 0
    suppressed_duplicate_instances = 0
    unique_track_ids: set[int] = set()
    processed_frames = 0

    motion_handle = (
        motion_history.open("w", newline="", encoding="utf-8")
        if motion_history is not None
        else None
    )
    motion_writer = (
        csv.DictWriter(motion_handle, fieldnames=MOTION_COLUMNS)
        if motion_handle is not None
        else None
    )
    if motion_writer is not None:
        motion_writer.writeheader()

    with canonical_tracks.open("w", newline="", encoding="utf-8") as handle:
        track_writer = csv.DictWriter(handle, fieldnames=TRACK_COLUMNS)
        track_writer.writeheader()
        try:
            while (
                config.maximum_frames is None
                or processed_frames < config.maximum_frames
            ):
                success, frame = capture.read()
                if not success:
                    break
                timestamp = processed_frames / fps
                detections = active_detector.detect(
                    frame,
                    frame_id=processed_frames,
                    timestamp=timestamp,
                )
                detected_instances += len(detections)
                raw_tracks = tracker.update(detections)
                consolidated = consolidator.update(
                    raw_tracks,
                    frame_id=processed_frames,
                )
                visible_tracks = consolidated.visible_tracks
                snapshots = predictor.update(
                    visible_tracks,
                    frame_id=processed_frames,
                    timestamp=timestamp,
                )
                rendered_track_instances += len(visible_tracks)
                suppressed_duplicate_instances += len(
                    consolidated.suppressed_tracks
                )
                unique_track_ids.update(
                    track.track_id for track in visible_tracks
                )
                for track in visible_tracks:
                    center_x, center_y = track.center
                    track_writer.writerow(
                        {
                            "frame": track.frame_id,
                            "time_seconds": track.timestamp,
                            "track_id": track.track_id,
                            "class_id": track.class_id,
                            "class_name": track.class_name,
                            "confidence": track.confidence,
                            "center_x": center_x,
                            "center_y": center_y,
                            "x1": track.x1,
                            "y1": track.y1,
                            "x2": track.x2,
                            "y2": track.y2,
                        }
                    )
                if motion_writer is not None:
                    for snapshot in snapshots:
                        motion_writer.writerow(
                            {
                                "frame": snapshot.frame_id,
                                "time_seconds": snapshot.timestamp,
                                "track_id": snapshot.track_id,
                                "class_id": snapshot.class_id,
                                "class_name": snapshot.class_name,
                                "is_observed": int(snapshot.observed),
                                "frames_since_seen": snapshot.frames_since_seen,
                                "estimated_center_x": snapshot.center_x,
                                "estimated_center_y": snapshot.center_y,
                                "velocity_x_pixels_per_second": (
                                    snapshot.velocity_x
                                ),
                                "velocity_y_pixels_per_second": (
                                    snapshot.velocity_y
                                ),
                                "speed_pixels_per_second": snapshot.speed,
                                "direction_degrees": (
                                    snapshot.direction_degrees
                                ),
                                "prediction_horizon_frames": (
                                    snapshot.prediction_horizon_frames
                                ),
                                "predicted_frame": snapshot.predicted_frame,
                                "predicted_time_seconds": (
                                    snapshot.predicted_time
                                ),
                                "predicted_center_x": (
                                    snapshot.predicted_center_x
                                ),
                                "predicted_center_y": (
                                    snapshot.predicted_center_y
                                ),
                            }
                        )

                annotated = render_tracking_ids(frame, visible_tracks)
                annotated = render_motion_paths(annotated, snapshots)
                cv2.putText(
                    annotated,
                    (
                        f"FlowSense | frame {processed_frames}/"
                        f"{max(0, total_frames - 1)} "
                        f"| active tracks {len(visible_tracks)}"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                processed_frames += 1
                active_progress_callback = _notify_progress(
                    active_progress_callback,
                    processed_frames=processed_frames,
                    total_frames=max(progress_total, processed_frames),
                )
                if processed_frames % 100 == 0:
                    print(
                        f"Processed {processed_frames}/{total_frames} frames"
                    )
        finally:
            capture.release()
            writer.release()
            if motion_handle is not None:
                motion_handle.close()

    if processed_frames == 0:
        raise RuntimeError("The input video contains no readable frames")
    return {
        "processed_frames": processed_frames,
        "detected_instances": detected_instances,
        "rendered_track_instances": rendered_track_instances,
        "unique_track_ids": len(unique_track_ids),
        "suppressed_duplicate_instances": suppressed_duplicate_instances,
    }


def _validate_config(config: PipelineConfig) -> None:
    if not config.input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {config.input_video}")
    if config.maximum_frames is not None and config.maximum_frames <= 0:
        raise ValueError("maximum_frames must be greater than zero")
    if config.history_points < 2:
        raise ValueError("history_points must be at least 2")
    if config.velocity_window < 1:
        raise ValueError("velocity_window must be greater than zero")
    if config.prediction_horizon_frames < 1:
        raise ValueError("prediction_horizon_frames must be greater than zero")
    if config.inactive_timeout_frames < 0:
        raise ValueError("inactive_timeout_frames cannot be negative")
    if config.movement_threshold_pixels <= 0:
        raise ValueError("movement_threshold_pixels must be positive")
    if config.toward_camera not in {"down", "up", "left", "right"}:
        raise ValueError("toward_camera must be down, up, left, or right")
    if config.counting_mode not in {"auto", "movement", "passage"}:
        raise ValueError("counting_mode must be auto, movement, or passage")


def _resolve_output_path(
    configured_path: Path | None,
    default_path: Path,
    output_dir: Path,
) -> Path:
    if configured_path is None:
        return default_path.resolve()
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (output_dir / configured_path).resolve()


def _validate_destination(
    input_video: Path,
    destination: Path,
    overwrite: bool,
) -> None:
    if destination == input_video:
        raise ValueError("An output path cannot overwrite the input video")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Use --overwrite to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)


def _publish_artifact(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {destination}")
        destination.unlink()
    shutil.move(str(source), str(destination))


def _notify_progress(
    callback: ProgressCallback | None,
    *,
    processed_frames: int,
    total_frames: int,
) -> ProgressCallback | None:
    """Call progress reporting without allowing UI failures to stop analysis."""
    if callback is None:
        return None
    try:
        callback(processed_frames, total_frames)
    except Exception as exc:
        print(f"Progress reporting disabled: {exc}")
        return None
    return callback
