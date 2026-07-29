"""Overlay new ByteTrack IDs from Member 1's detection CSV on their video."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from flowsense.csv_detections import load_detection_csv
from flowsense.tracking import (
    ByteTrackTracker,
    IdentityConsolidator,
    LearnedMotionCorrector,
    MotionPredictor,
    default_model_path,
)
from flowsense.tracking.render import render_motion_paths, render_tracking_ids
from flowsense.video_compat import convert_to_browser_mp4, is_h264_mp4


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Paths associated with one of Member 1's detection/video pairs."""

    csv_path: Path
    video_path: Path
    output_path: Path
    tracks_output_path: Path
    motion_output_path: Path


DATASETS = {
    "1": DatasetConfig(
        csv_path=Path("tracking_data/tracking_data.csv"),
        video_path=Path("videos/source_traffic.mp4"),
        output_path=Path("videos/flowsense_hybrid_video_1.mp4"),
        tracks_output_path=Path("outputs/member2_canonical_tracks.csv"),
        motion_output_path=Path("outputs/member2_motion_predictions.csv"),
    ),
    "2": DatasetConfig(
        csv_path=Path("tracking_data/tracking_data2.csv"),
        video_path=Path("videos/source_traffic2.mp4"),
        output_path=Path("videos/flowsense_hybrid_video_2.mp4"),
        tracks_output_path=Path("outputs/member2_canonical_tracks2.csv"),
        motion_output_path=Path("outputs/member2_motion_predictions2.csv"),
    ),
    "3": DatasetConfig(
        csv_path=Path("tracking_data/tracking_data3.csv"),
        video_path=Path("videos/source_traffic3.mp4"),
        output_path=Path("videos/flowsense_hybrid_video_3.mp4"),
        tracks_output_path=Path("outputs/member2_canonical_tracks3.csv"),
        motion_output_path=Path("outputs/member2_motion_predictions3.csv"),
    ),
    "4": DatasetConfig(
        csv_path=Path("tracking_data/tracking_data4.csv"),
        video_path=Path("videos/source_traffic4.mp4"),
        output_path=Path("videos/flowsense_hybrid_video_4.mp4"),
        tracks_output_path=Path("outputs/member2_canonical_tracks4.csv"),
        motion_output_path=Path("outputs/member2_motion_predictions4.csv"),
    ),
}
DEFAULT_TRACKS_OUTPUT = DATASETS["1"].tracks_output_path
DEFAULT_MOTION_OUTPUT = DATASETS["1"].motion_output_path
TRACK_CSV_COLUMNS = (
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
MOTION_CSV_COLUMNS = (
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
    "prediction_scale",
)


def validate_clean_source_path(video_path: str | Path) -> Path:
    """Reject known FlowSense outputs so official videos cannot be double annotated."""
    path = Path(video_path)
    output_markers = (
        "flowsense_tracking",
        "member2_bytetrack_overlay",
        "flowsense_hybrid",
        "_flowsense",
    )
    if any(marker in path.stem.lower() for marker in output_markers):
        raise ValueError(
            f"Official rendering requires clean source footage, not an annotated "
            f"FlowSense output: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Clean source video not found: {path}")
    return path


def load_prediction_corrector(
    *,
    disabled: bool,
    model_path: str | Path | None,
    prediction_horizon_frames: int,
    velocity_window: int,
) -> tuple[LearnedMotionCorrector | None, str]:
    """Load the bundled corrector, with an explicit mathematical fallback."""
    if disabled:
        return None, "constant-velocity baseline (--disable-learned-prediction)"
    selected_path = (
        Path(model_path) if model_path is not None else default_model_path()
    )
    try:
        corrector = LearnedMotionCorrector.from_json(selected_path)
        if not corrector.compatible_with(
            horizon_frames=prediction_horizon_frames,
            velocity_window=velocity_window,
        ):
            raise ValueError(
                "model settings do not match the prediction horizon/velocity window"
            )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return None, f"constant-velocity fallback ({exc})"
    return corrector, f"learned hybrid ({selected_path})"


def build_motion_predictor(
    *,
    fps: float,
    frame_width: int,
    frame_height: int,
    history_points: int = 30,
    velocity_window: int = 5,
    prediction_horizon_frames: int = 15,
    inactive_timeout_frames: int = 30,
    disable_learned_prediction: bool = False,
    learned_model_path: str | Path | None = None,
) -> tuple[MotionPredictor, str]:
    """Build the shared predictor used by the official-video generator."""
    corrector, mode = load_prediction_corrector(
        disabled=disable_learned_prediction,
        model_path=learned_model_path,
        prediction_horizon_frames=prediction_horizon_frames,
        velocity_window=velocity_window,
    )
    return (
        MotionPredictor(
            fps=fps,
            history_points=history_points,
            velocity_window=velocity_window,
            prediction_horizon_frames=prediction_horizon_frames,
            inactive_timeout_frames=inactive_timeout_frames,
            frame_width=frame_width,
            frame_height=frame_height,
            learned_corrector=corrector,
        ),
        mode,
    )


def track_csv_on_video(
    csv_path: str | Path,
    video_path: str | Path,
    output_path: str | Path,
    tracks_output_path: str | Path = DEFAULT_TRACKS_OUTPUT,
    motion_output_path: str | Path = DEFAULT_MOTION_OUTPUT,
    *,
    maximum_frames: int | None = None,
    generate_video: bool = True,
    export_motion_history: bool = True,
    history_points: int = 30,
    velocity_window: int = 5,
    prediction_horizon_frames: int = 15,
    inactive_timeout_frames: int = 30,
    disable_learned_prediction: bool = False,
    learned_model_path: str | Path | None = None,
) -> dict[str, int | float | str]:
    """Render tracks/predictions and export canonical and motion CSV data."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is missing. Run: python -m pip install -r requirements.txt"
        ) from exc

    csv_path = Path(csv_path)
    video_path = validate_clean_source_path(video_path)
    output_path = Path(output_path)
    tracks_output_path = Path(tracks_output_path)
    motion_output_path = Path(motion_output_path)
    detections_by_frame = load_detection_csv(csv_path)
    last_detection_frame = max(detections_by_frame)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError("Source video has invalid width, height, or frame rate")
    if last_detection_frame >= video_frame_count:
        capture.release()
        raise ValueError(
            f"CSV references frame {last_detection_frame}, but the video has only "
            f"{video_frame_count} frames"
        )

    tracks_output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    staging_directory = None
    staged_mp4v = None
    staged_h264 = None
    if generate_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        staging_directory = tempfile.TemporaryDirectory(
            prefix=".flowsense-render-",
            dir=output_path.parent,
        )
        staging_path = Path(staging_directory.name)
        staged_mp4v = staging_path / "annotated_mp4v.mp4"
        staged_h264 = staging_path / "annotated_h264.mp4"
        writer = cv2.VideoWriter(
            str(staged_mp4v),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    if generate_video and (writer is None or not writer.isOpened()):
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    if not generate_video:
        capture.release()

    tracker = ByteTrackTracker(frame_rate=max(1, round(fps)))
    consolidator = IdentityConsolidator()
    motion_predictor, prediction_mode = build_motion_predictor(
        fps=fps,
        frame_width=width,
        frame_height=height,
        history_points=history_points,
        velocity_window=velocity_window,
        prediction_horizon_frames=prediction_horizon_frames,
        inactive_timeout_frames=inactive_timeout_frames,
        disable_learned_prediction=disable_learned_prediction,
        learned_model_path=learned_model_path,
    )
    print(f"Prediction mode: {prediction_mode}")
    processed_frames = 0
    rendered_tracks = 0
    motion_rows = 0
    suppressed_track_instances = 0
    unique_track_ids: set[int] = set()
    tracks_file = tracks_output_path.open("w", newline="", encoding="utf-8")
    tracks_writer = csv.DictWriter(tracks_file, fieldnames=TRACK_CSV_COLUMNS)
    tracks_writer.writeheader()
    motion_file = None
    motion_writer = None
    if export_motion_history:
        motion_output_path.parent.mkdir(parents=True, exist_ok=True)
        motion_file = motion_output_path.open("w", newline="", encoding="utf-8")
        motion_writer = csv.DictWriter(
            motion_file,
            fieldnames=MOTION_CSV_COLUMNS,
        )
        motion_writer.writeheader()

    try:
        while maximum_frames is None or processed_frames < maximum_frames:
            frame = None
            if generate_video:
                success, frame = capture.read()
                if not success:
                    break
            elif processed_frames >= video_frame_count:
                break

            raw_tracks = tracker.update(detections_by_frame.get(processed_frames, ()))
            consolidated = consolidator.update(
                raw_tracks,
                frame_id=processed_frames,
            )
            visible_tracks = consolidated.visible_tracks
            frame_timestamp = processed_frames / fps
            motion_snapshots = motion_predictor.update(
                visible_tracks,
                frame_id=processed_frames,
                timestamp=frame_timestamp,
            )
            unique_track_ids.update(track.track_id for track in visible_tracks)
            rendered_tracks += len(visible_tracks)
            suppressed_track_instances += len(consolidated.suppressed_tracks)
            for track in visible_tracks:
                center_x, center_y = track.center
                tracks_writer.writerow(
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
                for snapshot in motion_snapshots:
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
                            "velocity_x_pixels_per_second": snapshot.velocity_x,
                            "velocity_y_pixels_per_second": snapshot.velocity_y,
                            "speed_pixels_per_second": snapshot.speed,
                            "direction_degrees": snapshot.direction_degrees,
                            "prediction_horizon_frames": (
                                snapshot.prediction_horizon_frames
                            ),
                            "predicted_frame": snapshot.predicted_frame,
                            "predicted_time_seconds": snapshot.predicted_time,
                            "predicted_center_x": snapshot.predicted_center_x,
                            "predicted_center_y": snapshot.predicted_center_y,
                            "prediction_scale": snapshot.prediction_scale,
                        }
                    )
                    motion_rows += 1
            if generate_video and frame is not None and writer is not None:
                annotated = render_tracking_ids(frame, visible_tracks)
                annotated = render_motion_paths(annotated, motion_snapshots)
                cv2.putText(
                    annotated,
                    (
                        f"ByteTrack | frame {processed_frames}/"
                        f"{video_frame_count - 1} "
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
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames}/{video_frame_count} frames")
    finally:
        if generate_video:
            capture.release()
        if writer is not None:
            writer.release()
        tracks_file.close()
        if motion_file is not None:
            motion_file.close()
        if sys.exc_info()[0] is not None and staging_directory is not None:
            staging_directory.cleanup()

    required_frames = min(
        video_frame_count,
        maximum_frames if maximum_frames is not None else video_frame_count,
    )
    if processed_frames != required_frames:
        if staging_directory is not None:
            staging_directory.cleanup()
        raise RuntimeError(
            f"Video decoding stopped at frame {processed_frames}; "
            f"expected {required_frames}"
        )
    if generate_video:
        assert staged_mp4v is not None and staged_h264 is not None
        try:
            convert_to_browser_mp4(staged_mp4v, staged_h264)
            if not is_h264_mp4(staged_h264):
                raise RuntimeError("staged output failed H.264 validation")
            os.replace(staged_h264, output_path)
        finally:
            if staging_directory is not None:
                staging_directory.cleanup()

    summary: dict[str, int | float | str] = {
        "frames": processed_frames,
        "fps": fps,
        "rendered_track_instances": rendered_tracks,
        "motion_history_rows": motion_rows,
        "suppressed_duplicate_instances": suppressed_track_instances,
        "unique_track_ids": len(unique_track_ids),
        "prediction_mode": prediction_mode,
        "frame_width": width,
        "frame_height": height,
        "output_video": (
            str(output_path.resolve()) if generate_video else "disabled"
        ),
        "output_tracks": str(tracks_output_path.resolve()),
        "output_motion": (
            str(motion_output_path.resolve())
            if export_motion_history
            else "disabled"
        ),
    }
    print("Tracking complete")
    for name, value in summary.items():
        print(f"  {name}: {value}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use one or all of Member 1's CSV/video pairs as ByteTrack input "
            "and overlay canonical tracking IDs on the associated video."
        )
    )
    parser.add_argument(
        "--disable-learned-prediction",
        action="store_true",
        help="Use the original constant-velocity predictor for comparison.",
    )
    parser.add_argument(
        "--learned-model",
        type=Path,
        default=None,
        help="Optional learned-corrector JSON (default: bundled production model).",
    )
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="1",
        help="Built-in dataset pair to process (default: 1).",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--tracks-output",
        type=Path,
        default=None,
        help="CSV output containing the canonical IDs visible in each frame.",
    )
    parser.add_argument(
        "--motion-output",
        type=Path,
        default=None,
        help=(
            "CSV output containing active-track motion and prediction history."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional short-run limit for debugging.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "Generate only the canonical tracking CSV; skip video decoding, "
            "annotation, and encoding."
        ),
    )
    parser.add_argument(
        "--no-motion-output",
        action="store_true",
        help="Do not export the frame-by-frame motion/prediction history CSV.",
    )
    parser.add_argument(
        "--history-points",
        type=int,
        default=30,
        help="Recent observed center points retained per active track (default: 30).",
    )
    parser.add_argument(
        "--velocity-window",
        type=int,
        default=5,
        help="Recent point-to-point velocities averaged for smoothing (default: 5).",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=15,
        help="Frames ahead for each short-term position prediction (default: 15).",
    )
    parser.add_argument(
        "--inactive-timeout",
        type=int,
        default=30,
        help="Missing frames retained before an inactive track is deleted (default: 30).",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
    if args.dataset == "all" and any(
        value is not None
        for value in (
            args.csv,
            args.video,
            args.output,
            args.tracks_output,
            args.motion_output,
        )
    ):
        parser.error(
            "--csv, --video, --output, --tracks-output, and --motion-output "
            "cannot be combined with --dataset all"
        )
    if args.no_video and args.output is not None:
        parser.error("--output cannot be used with --no-video")
    if args.no_motion_output and args.motion_output is not None:
        parser.error("--motion-output cannot be used with --no-motion-output")
    if args.history_points < 2:
        parser.error("--history-points must be at least 2")
    if args.velocity_window < 1:
        parser.error("--velocity-window must be greater than zero")
    if args.prediction_horizon < 1:
        parser.error("--prediction-horizon must be greater than zero")
    if args.inactive_timeout < 0:
        parser.error("--inactive-timeout cannot be negative")
    return args


def run_from_args(arguments: argparse.Namespace) -> list[dict[str, int | float | str]]:
    """Run the selected built-in dataset(s), applying individual overrides."""
    dataset_ids = DATASETS if arguments.dataset == "all" else (arguments.dataset,)
    summaries = []
    for dataset_id in dataset_ids:
        config = DATASETS[dataset_id]
        configured_video = config.video_path
        print(f"\nProcessing dataset {dataset_id}")
        summaries.append(
            track_csv_on_video(
                arguments.csv or config.csv_path,
                arguments.video or configured_video,
                arguments.output or config.output_path,
                arguments.tracks_output or config.tracks_output_path,
                arguments.motion_output or config.motion_output_path,
                maximum_frames=arguments.max_frames,
                generate_video=not arguments.no_video,
                export_motion_history=not arguments.no_motion_output,
                history_points=arguments.history_points,
                velocity_window=arguments.velocity_window,
                prediction_horizon_frames=arguments.prediction_horizon,
                inactive_timeout_frames=arguments.inactive_timeout,
                disable_learned_prediction=getattr(
                    arguments, "disable_learned_prediction", False
                ),
                learned_model_path=getattr(arguments, "learned_model", None),
            )
        )
    return summaries


if __name__ == "__main__":
    run_from_args(parse_args())
