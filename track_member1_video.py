"""Overlay new ByteTrack IDs from Member 1's detection CSV on their video."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from flowsense.csv_detections import load_detection_csv
from flowsense.tracking import ByteTrackTracker, IdentityConsolidator
from flowsense.tracking.render import render_tracking_ids


DEFAULT_CSV = Path("tracking_data/tracking_data.csv")
DEFAULT_VIDEO = Path("videos/source_traffic.mp4")
DEFAULT_OUTPUT = Path("outputs/member2_bytetrack_overlay.mp4")
DEFAULT_TRACKS_OUTPUT = Path("outputs/member2_canonical_tracks.csv")
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


def track_csv_on_video(
    csv_path: str | Path,
    video_path: str | Path,
    output_path: str | Path,
    tracks_output_path: str | Path = DEFAULT_TRACKS_OUTPUT,
    *,
    maximum_frames: int | None = None,
) -> dict[str, int | float | str]:
    """Render canonical IDs and export the same visible tracks as CSV."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is missing. Run: python -m pip install -r requirements.txt"
        ) from exc

    csv_path = Path(csv_path)
    video_path = Path(video_path)
    output_path = Path(output_path)
    tracks_output_path = Path(tracks_output_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video not found: {video_path}")

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks_output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    tracker = ByteTrackTracker(frame_rate=max(1, round(fps)))
    consolidator = IdentityConsolidator()
    processed_frames = 0
    rendered_tracks = 0
    suppressed_track_instances = 0
    unique_track_ids: set[int] = set()
    tracks_file = tracks_output_path.open("w", newline="", encoding="utf-8")
    tracks_writer = csv.DictWriter(tracks_file, fieldnames=TRACK_CSV_COLUMNS)
    tracks_writer.writeheader()

    try:
        while maximum_frames is None or processed_frames < maximum_frames:
            success, frame = capture.read()
            if not success:
                break

            raw_tracks = tracker.update(detections_by_frame.get(processed_frames, ()))
            consolidated = consolidator.update(
                raw_tracks,
                frame_id=processed_frames,
            )
            visible_tracks = consolidated.visible_tracks
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
            annotated = render_tracking_ids(frame, visible_tracks)
            cv2.putText(
                annotated,
                (
                    f"ByteTrack | frame {processed_frames}/{video_frame_count - 1} "
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
        capture.release()
        writer.release()
        tracks_file.close()

    required_frames = min(
        video_frame_count,
        maximum_frames if maximum_frames is not None else video_frame_count,
    )
    if processed_frames != required_frames:
        raise RuntimeError(
            f"Video decoding stopped at frame {processed_frames}; "
            f"expected {required_frames}"
        )

    summary: dict[str, int | float | str] = {
        "frames": processed_frames,
        "fps": fps,
        "rendered_track_instances": rendered_tracks,
        "suppressed_duplicate_instances": suppressed_track_instances,
        "unique_track_ids": len(unique_track_ids),
        "output_video": str(output_path.resolve()),
        "output_tracks": str(tracks_output_path.resolve()),
    }
    print("Tracking complete")
    for name, value in summary.items():
        print(f"  {name}: {value}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Member 1's CSV detections as ByteTrack input and overlay the "
            "new tracking IDs on the original training video."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tracks-output",
        type=Path,
        default=DEFAULT_TRACKS_OUTPUT,
        help="CSV output containing the canonical IDs visible in each frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional short-run limit for debugging.",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    track_csv_on_video(
        arguments.csv,
        arguments.video,
        arguments.output,
        arguments.tracks_output,
        maximum_frames=arguments.max_frames,
    )
