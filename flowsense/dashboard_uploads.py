"""Secure local-file helpers for the Streamlit upload workflow."""

from __future__ import annotations

import io
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .pipeline import FrameDetector, PipelineResult, ProgressCallback


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
RUN_MAX_AGE_SECONDS = 24 * 60 * 60
DEFAULT_RUNS_ROOT = (
    Path(tempfile.gettempdir()) / "flowsense-dashboard-runs"
)


class UploadValidationError(ValueError):
    """Raised when an uploaded file is not an acceptable MP4."""


class MissingOutputError(RuntimeError):
    """Raised when a completed pipeline result is missing an artifact."""


@dataclass(frozen=True, slots=True)
class SavedUpload:
    """Paths and display metadata for one isolated dashboard run."""

    original_filename: str
    size_bytes: int
    run_dir: Path
    input_path: Path
    output_dir: Path
    download_stem: str


def validate_mp4_upload(
    filename: str,
    data: bytes | bytearray | memoryview,
    *,
    maximum_bytes: int = MAX_UPLOAD_BYTES,
) -> None:
    """Validate extension, size, and the MP4 file-type signature."""
    if PureWindowsPath(filename).suffix.casefold() != ".mp4":
        raise UploadValidationError("Please select one MP4 video.")
    size = len(data)
    if size == 0:
        raise UploadValidationError("The uploaded video is empty.")
    if size > maximum_bytes:
        limit_mb = maximum_bytes / (1024 * 1024)
        raise UploadValidationError(
            f"The uploaded video exceeds the {limit_mb:g} MB limit."
        )
    header = bytes(data[:64])
    if len(header) < 12 or b"ftyp" not in header:
        raise UploadValidationError(
            "The selected file does not appear to be a readable MP4."
        )


def save_uploaded_mp4(
    filename: str,
    data: bytes | bytearray | memoryview,
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    maximum_bytes: int = MAX_UPLOAD_BYTES,
) -> SavedUpload:
    """Save validated bytes under a fixed filename in a unique run directory."""
    validate_mp4_upload(filename, data, maximum_bytes=maximum_bytes)
    root = Path(runs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix="run-", dir=root)
    ).resolve()
    if run_dir.parent != root:
        raise RuntimeError("Could not create an isolated upload directory.")
    input_path = run_dir / "uploaded_video.mp4"
    try:
        with input_path.open("wb") as handle:
            handle.write(data)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return SavedUpload(
        original_filename=PureWindowsPath(filename).name,
        size_bytes=len(data),
        run_dir=run_dir,
        input_path=input_path,
        output_dir=run_dir / "results",
        download_stem=safe_download_stem(filename),
    )


def safe_download_stem(filename: str) -> str:
    """Return a short display-only name; it is never used for local paths."""
    name = PureWindowsPath(filename).name
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")
    return (cleaned or "traffic-video")[:80]


def remove_run_directory(
    run_dir: str | Path,
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> bool:
    """Remove one verified direct child of the dashboard run root."""
    root = Path(runs_root).resolve()
    candidate = Path(run_dir).resolve()
    if candidate.parent != root or not candidate.name.startswith("run-"):
        raise ValueError("Refusing to remove a path outside the run directory.")
    if not candidate.exists():
        return False
    shutil.rmtree(candidate)
    return True


def cleanup_abandoned_runs(
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    maximum_age_seconds: float = RUN_MAX_AGE_SECONDS,
    exclude: Iterable[str | Path] = (),
    now: float | None = None,
) -> int:
    """Delete inactive run directories older than the configured lifetime."""
    root = Path(runs_root).resolve()
    if not root.exists():
        return 0
    excluded = {Path(path).resolve() for path in exclude}
    cutoff = (time.time() if now is None else now) - maximum_age_seconds
    removed = 0
    for candidate in root.iterdir():
        if (
            candidate in excluded
            or not candidate.is_dir()
            or candidate.is_symlink()
            or not candidate.name.startswith("run-")
        ):
            continue
        try:
            modified = candidate.stat().st_mtime
        except OSError:
            continue
        if modified < cutoff:
            remove_run_directory(candidate, runs_root=root)
            removed += 1
    return removed


def validate_pipeline_outputs(result: PipelineResult) -> None:
    """Confirm the files needed by the dashboard actually exist."""
    missing = [
        label
        for label, path in (
            ("annotated video", result.output_video),
            ("HTML report", result.output_report),
        )
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise MissingOutputError(
            "FlowSense did not create: " + ", ".join(missing) + "."
        )


def analyze_saved_upload(
    saved_upload: SavedUpload,
    *,
    detector: FrameDetector,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """Run the production pipeline for one already validated upload."""
    from .pipeline import PipelineConfig, run_pipeline

    result = run_pipeline(
        PipelineConfig(
            input_video=saved_upload.input_path,
            output_dir=saved_upload.output_dir,
            keep_intermediates=True,
            overwrite=True,
        ),
        detector=detector,
        progress_callback=progress_callback,
    )
    validate_pipeline_outputs(result)
    return result


def build_intermediates_zip(intermediate_dir: str | Path) -> bytes:
    """Package canonical tracks, motion history, and counting CSVs."""
    root = Path(intermediate_dir)
    required = [
        root / "canonical_tracks.csv",
        root / "motion_predictions.csv",
    ]
    counter_dir = root / "counter_outputs"
    counter_csvs = (
        sorted(counter_dir.glob("*.csv"))
        if counter_dir.is_dir()
        else []
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing or not counter_csvs:
        detail = ", ".join(missing or ["counting CSVs"])
        raise MissingOutputError(
            f"Debug download files are incomplete: {detail}."
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in required:
            archive.write(path, arcname=path.name)
        for path in counter_csvs:
            archive.write(path, arcname=f"counting/{path.name}")
    return buffer.getvalue()


def friendly_analysis_error(exc: Exception, *, stage: str) -> str:
    """Translate expected upload and pipeline failures into readable guidance."""
    if isinstance(exc, UploadValidationError):
        return str(exc)
    if isinstance(exc, MissingOutputError):
        return str(exc)
    message = str(exc)
    lowered = message.casefold()
    if stage == "model":
        return (
            "FlowSense could not download or load the YOLO model. Check your "
            "internet connection for the first run, then try again."
        )
    if "could not open input video" in lowered or "invalid width" in lowered:
        return (
            "The uploaded MP4 could not be read. Try exporting it again with "
            "a standard H.264 or MPEG-4 video codec."
        )
    if "no readable frames" in lowered:
        return "The uploaded video is empty or contains no readable frames."
    if "did not produce any trackable" in lowered:
        return (
            "No supported road users were detected. Try a clearer traffic "
            "video with cars, trucks, buses, motorcycles, bicycles, or people."
        )
    return (
        "FlowSense could not finish processing this video. "
        f"Details: {message or type(exc).__name__}"
    )
