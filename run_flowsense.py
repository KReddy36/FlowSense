"""Run the complete FlowSense pipeline with one command."""

from __future__ import annotations

import argparse
from pathlib import Path

from flowsense.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one traffic video with YOLO, ByteTrack, motion "
            "prediction, automatic counting, and an HTML report."
        )
    )
    parser.add_argument("video", type=Path, help="Traffic video to analyze.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for the final MP4 and HTML files (default: results).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Custom MP4 path, relative to --output-dir unless absolute.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Custom HTML path, relative to --output-dir unless absolute.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name or local weights path (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="YOLO confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="YOLO non-max-suppression IoU threshold (default: 0.50).",
    )
    parser.add_argument(
        "--device",
        help="Ultralytics device such as cpu, 0, or cuda:0 (default: automatic).",
    )
    parser.add_argument(
        "--history-points",
        type=int,
        default=30,
        help="Recent observed points retained per active track (default: 30).",
    )
    parser.add_argument(
        "--velocity-window",
        type=int,
        default=5,
        help="Recent velocities averaged for prediction (default: 5).",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=15,
        help="Frames ahead shown by dashed prediction lines (default: 15).",
    )
    parser.add_argument(
        "--inactive-timeout",
        type=int,
        default=30,
        help="Missing frames retained before deleting a track (default: 30).",
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=50.0,
        help="Minimum trajectory span counted as movement (default: 50 px).",
    )
    parser.add_argument(
        "--toward-camera",
        choices=("down", "up", "left", "right"),
        default="down",
        help="Image direction that means toward the camera (default: down).",
    )
    parser.add_argument(
        "--counting-mode",
        choices=("auto", "movement", "passage"),
        default="auto",
        help="Traffic counting strategy (default: auto).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit for a quick test run.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Retain internal canonical/count files for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace final files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_pipeline(
            PipelineConfig(
                input_video=args.video,
                output_dir=args.output_dir,
                output_video=args.output_video,
                output_report=args.report,
                model_name=args.model,
                confidence=args.confidence,
                iou=args.iou,
                device=args.device,
                history_points=args.history_points,
                velocity_window=args.velocity_window,
                prediction_horizon_frames=args.prediction_horizon,
                inactive_timeout_frames=args.inactive_timeout,
                movement_threshold_pixels=args.movement_threshold,
                toward_camera=args.toward_camera,
                counting_mode=args.counting_mode,
                maximum_frames=args.max_frames,
                keep_intermediates=args.keep_intermediates,
                overwrite=args.overwrite,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"FlowSense failed: {exc}") from exc

    print("\nFlowSense complete")
    print(f"  Frames processed: {result.processed_frames}")
    print(f"  Unique track IDs: {result.unique_track_ids}")
    print(f"  Counts by class: {result.counts_by_class}")
    print(f"  Counts by direction: {result.counts_by_direction}")
    if result.prediction_accuracy_percent is None:
        print("  Prediction accuracy: N/A (not enough eligible forecasts)")
    else:
        print(
            "  Prediction accuracy: "
            f"{result.prediction_accuracy_percent:.1f}% "
            f"({result.prediction_accuracy_samples} forecasts)"
        )
    print(f"  Annotated video: {result.output_video}")
    print(f"  HTML report: {result.output_report}")
    if result.intermediate_dir is not None:
        print(f"  Debug intermediates: {result.intermediate_dir}")


if __name__ == "__main__":
    main()
