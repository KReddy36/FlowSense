"""Browser-compatible video conversion for FlowSense outputs."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class VideoConversionError(RuntimeError):
    """Raised when an annotated video cannot be converted to H.264."""


def convert_to_browser_mp4(
    source: str | Path,
    destination: str | Path,
    *,
    ffmpeg_executable: str | Path | None = None,
) -> Path:
    """Convert an MP4 to H.264/yuv420p without invoking a command shell."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise VideoConversionError("The annotated source video is missing.")

    if ffmpeg_executable is None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise VideoConversionError(
                "The H.264 converter is not installed."
            ) from exc
        try:
            ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise VideoConversionError(
                "The bundled FFmpeg executable is unavailable."
            ) from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.unlink(missing_ok=True)
    command = [
        str(ffmpeg_executable),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-an",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError as exc:
        raise VideoConversionError(
            "FFmpeg could not be started."
        ) from exc

    if completed.returncode != 0:
        destination_path.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        suffix = f" ({detail[-1]})" if detail else ""
        raise VideoConversionError(f"H.264 conversion failed{suffix}.")
    if not destination_path.is_file() or destination_path.stat().st_size == 0:
        destination_path.unlink(missing_ok=True)
        raise VideoConversionError(
            "H.264 conversion finished without creating a video."
        )
    if not is_h264_mp4(destination_path):
        destination_path.unlink(missing_ok=True)
        raise VideoConversionError(
            "The converted video does not advertise an H.264 stream."
        )
    return destination_path


def is_h264_mp4(path: str | Path) -> bool:
    """Return whether the MP4 metadata identifies an H.264 video stream."""
    video_path = Path(path)
    if not video_path.is_file():
        return False
    with video_path.open("rb") as handle:
        header = handle.read(2 * 1024 * 1024)
    return b"ftyp" in header[:64] and (
        b"avc1" in header or b"avc3" in header
    )


def browser_video_bytes(path: str | Path) -> bytes:
    """Read an H.264 MP4, converting a temporary preview when necessary."""
    video_path = Path(path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise VideoConversionError("The preview video is missing.")
    if is_h264_mp4(video_path):
        return video_path.read_bytes()

    with tempfile.TemporaryDirectory(prefix="flowsense-preview-") as directory:
        preview_path = Path(directory) / "browser_preview.mp4"
        convert_to_browser_mp4(video_path, preview_path)
        return preview_path.read_bytes()
