"""FlowSense Streamlit dashboard.

Run locally:
    python -m streamlit run Dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from flowsense.dashboard_uploads import (
    DEFAULT_RUNS_ROOT,
    MAX_UPLOAD_BYTES,
    MissingOutputError,
    analyze_saved_upload,
    build_intermediates_zip,
    cleanup_abandoned_runs,
    friendly_analysis_error,
    remove_run_directory,
    save_uploaded_mp4,
    validate_pipeline_outputs,
)
from flowsense.pipeline import PipelineResult
from flowsense.yolo_detector import YoloDetector


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "easy_results"
VIDEOS_DIR = ROOT / "videos"
ANALYSIS_STATE_KEY = "uploaded_analysis"
UPLOADER_VERSION_KEY = "upload_widget_version"

VIDEO_FILES = {
    "Video 1": VIDEOS_DIR / "member2_bytetrack_overlay.mp4",
    "Video 2": VIDEOS_DIR / "flowsense_tracking2.mp4",
    "Video 3": VIDEOS_DIR / "flowsense_tracking3.mp4",
    "Video 4": VIDEOS_DIR / "flowsense_tracking4 (1) (1) (1).mp4",
}

CLASS_COLUMNS = [
    "Cars",
    "Trucks",
    "Buses",
    "Motorcycles",
    "Bicycles",
    "Pedestrians",
]

DIRECTION_COLUMNS = [
    "Toward camera",
    "Away from camera",
    "Cross-traffic",
    "Mixed/unclear",
]


st.set_page_config(
    page_title="FlowSense",
    page_icon="🚦",
    layout="wide",
)


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / filename)


@st.cache_data
def load_summary() -> dict:
    with (RESULTS_DIR / "summary.json").open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource(show_spinner=False)
def load_yolo_detector() -> YoloDetector:
    """Load YOLO once per Streamlit server process."""
    return YoloDetector()


def require_files(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        st.error("Required project files are missing:")
        st.code("\n".join(missing))
        st.stop()


def render_project_results() -> None:
    """Render the original four-video saved-results dashboard."""
    require_files(
        [
            RESULTS_DIR / "automatic_counts.csv",
            RESULTS_DIR / "traffic_volume_intervals.csv",
            RESULTS_DIR / "comparison_by_video.csv",
            RESULTS_DIR / "comparison_by_video_class.csv",
            RESULTS_DIR / "summary.json",
        ]
    )

    counts = load_csv("automatic_counts.csv")
    intervals = load_csv("traffic_volume_intervals.csv")
    comparison = load_csv("comparison_by_video.csv")
    class_comparison = load_csv("comparison_by_video_class.csv")
    summary = load_summary()

    with st.sidebar:
        st.header("Project Results Controls")
        selected_video = st.selectbox(
            "Select traffic footage",
            counts["Video"].tolist(),
        )
        show_video = st.checkbox("Show annotated video", value=True)

        st.divider()
        st.markdown(
            """
            **Pipeline**

            Prerecorded video → YOLO detections → ByteTrack identities →
            movement/line-crossing analysis → traffic dashboard
            """
        )

    selected_counts = counts.loc[counts["Video"] == selected_video].iloc[0]
    selected_comparison = comparison.loc[
        comparison["Video"] == selected_video
    ].iloc[0]
    selected_summary = summary["videos"][selected_video]

    st.subheader(f"{selected_video} overview")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Vehicles counted", int(selected_counts["Vehicles counted"]))
    kpi2.metric("Pedestrians", int(selected_counts["Pedestrians counted"]))
    kpi3.metric("Parked / excluded", int(selected_counts["Parked/excluded"]))
    kpi4.metric(
        "Manual vehicle count",
        int(selected_comparison["Kelvin vehicles"]),
    )
    kpi5.metric(
        "Absolute vehicle error",
        int(selected_comparison["Absolute vehicle error"]),
    )

    method = selected_summary["selected_counting_method"].title()
    role = selected_comparison["Dataset role"]
    st.info(
        f"Dataset role: **{role}** · Counting method: **{method}** · "
        f"Movement threshold: "
        f"**{selected_summary['movement_threshold_pixels']:.0f} px**"
    )

    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("Annotated tracking video")
        video_path = VIDEO_FILES.get(selected_video)
        if show_video and video_path and video_path.exists():
            st.video(str(video_path))
        elif show_video:
            st.warning(
                "The annotated video is not available at the expected "
                "repository path."
            )
        else:
            st.caption("Video display is disabled in the sidebar.")

    with right:
        st.subheader("Counts by class")
        class_values = (
            selected_counts[CLASS_COLUMNS]
            .astype(float)
            .rename_axis("Class")
            .reset_index(name="Count")
        )
        class_values = class_values[class_values["Count"] > 0]
        st.bar_chart(class_values, x="Class", y="Count")

    st.divider()
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Traffic volume by 5-second interval")
        selected_intervals = intervals.loc[
            intervals["Video"] == selected_video
        ].copy()
        selected_intervals["Interval"] = selected_intervals.apply(
            lambda row: (
                f"{int(row['Interval start (s)'])}–"
                f"{int(row['Interval end (s)'])} s"
            ),
            axis=1,
        )
        volume_over_time = (
            selected_intervals.groupby(
                ["Interval start (s)", "Interval"],
                as_index=False,
            )["Automatic count"]
            .sum()
            .sort_values("Interval start (s)")
        )
        st.line_chart(
            volume_over_time,
            x="Interval",
            y="Automatic count",
        )
        if not volume_over_time.empty:
            busiest = volume_over_time.loc[
                volume_over_time["Automatic count"].idxmax()
            ]
            st.caption(
                f"Busiest interval: {busiest['Interval']} "
                f"({int(busiest['Automatic count'])} moving objects)"
            )

    with chart_right:
        st.subheader("Direction of movement")
        direction_values = (
            selected_counts[DIRECTION_COLUMNS]
            .astype(float)
            .rename_axis("Direction")
            .reset_index(name="Count")
        )
        direction_values = direction_values[direction_values["Count"] > 0]
        st.bar_chart(direction_values, x="Direction", y="Count")

    st.divider()
    st.subheader("Manual count vs FlowSense")

    selected_class_comparison = class_comparison.loc[
        class_comparison["Video"] == selected_video,
        [
            "Class",
            "Kelvin count",
            "Automatic count",
            "Error (automatic - Kelvin)",
            "Absolute error",
            "Percent error",
        ],
    ].copy()

    comparison_chart = selected_class_comparison[
        ["Class", "Kelvin count", "Automatic count"]
    ].set_index("Class")

    comparison_left, comparison_right = st.columns([1.1, 1])
    with comparison_left:
        st.bar_chart(comparison_chart)
    with comparison_right:
        st.dataframe(
            selected_class_comparison,
            hide_index=True,
            width="stretch",
        )

    if role == "Evaluation":
        manual_total = float(selected_comparison["Kelvin vehicles"])
        automatic_total = float(selected_comparison["Automatic vehicles"])
        evaluation_agreement = (
            max(
                0.0,
                1.0 - abs(automatic_total - manual_total) / manual_total,
            )
            if manual_total
            else 0.0
        )
        st.metric(
            "Evaluation-set total-count agreement",
            f"{evaluation_agreement:.1%}",
            help=(
                "Calculated as 1 − absolute vehicle-count error / manual "
                "vehicle count. This measures total-count agreement, not "
                "object-level detection accuracy."
            ),
        )

    with st.expander("Method and limitations"):
        st.markdown(
            """
            **Method**

            - A pretrained YOLO model detects road users in each frame.
            - ByteTrack and identity consolidation create more stable IDs.
            - FlowSense excludes stationary objects using trajectory movement.
            - For severely fragmented tracking, it switches to passage-line
              counting.
            - Counts are grouped by class, direction, and five-second interval.

            **Limitations**

            - Occlusion can cause a road user to receive a new tracking ID.
            - Cars and trucks may be confused when vehicles are distant or
              partly hidden.
            - Pixel movement is not the same as real-world speed.
            - Camera perspective affects direction and passage-line
              interpretation.
            - Manual counts are an approximate ground truth.
            """
        )

    with st.expander("All video results"):
        st.dataframe(counts, hide_index=True, width="stretch")

    st.caption(
        "FlowSense · Kellan Reddy · Kelvin Qian · Brayden Chen · Batuhan Akbas"
    )


def render_upload_analysis() -> None:
    """Render secure local upload, analysis, preview, and download controls."""
    st.subheader("Analyze your own traffic video")
    st.write(
        "Upload one MP4 and FlowSense will run the same YOLO, tracking, "
        "prediction, and counting pipeline used by the command-line tool."
    )
    st.warning(
        "Processing may take several minutes on a CPU. Keep this browser tab "
        "open until the analysis finishes."
    )

    analysis = st.session_state.get(ANALYSIS_STATE_KEY)
    active_runs = [analysis["run_dir"]] if analysis else []
    try:
        cleanup_abandoned_runs(exclude=active_runs)
    except OSError:
        pass

    uploader_version = st.session_state.get(UPLOADER_VERSION_KEY, 0)
    uploaded_file = st.file_uploader(
        "Upload an MP4 traffic video",
        type=["mp4"],
        accept_multiple_files=False,
        key=f"traffic_video_upload_{uploader_version}",
        help="Maximum upload size: 100 MB.",
    )

    if uploaded_file is not None:
        size_bytes = int(uploaded_file.size)
        st.caption(
            f"Selected: **{uploaded_file.name}** · "
            f"{size_bytes / (1024 * 1024):.2f} MB"
        )
        if size_bytes > MAX_UPLOAD_BYTES:
            st.error("The selected video exceeds the 100 MB limit.")

    if analysis is not None:
        st.info(
            "An uploaded analysis is already available below. Clear it before "
            "processing another video."
        )

    analyze_clicked = st.button(
        "Analyze video",
        type="primary",
        disabled=(
            uploaded_file is None
            or analysis is not None
            or (
                uploaded_file is not None
                and int(uploaded_file.size) > MAX_UPLOAD_BYTES
            )
        ),
    )
    if analyze_clicked and uploaded_file is not None:
        _analyze_uploaded_file(uploaded_file)

    analysis = st.session_state.get(ANALYSIS_STATE_KEY)
    if analysis is not None:
        _render_uploaded_result(analysis)


def _analyze_uploaded_file(uploaded_file: object) -> None:
    status = st.status("Preparing uploaded video…", expanded=True)
    progress_bar = st.progress(0.0)
    progress_text = st.empty()
    saved_upload = None
    stage = "upload"

    def update_progress(processed_frames: int, total_frames: int) -> None:
        if total_frames > 0:
            fraction = min(1.0, processed_frames / total_frames)
            progress_bar.progress(fraction)
            progress_text.caption(
                f"Processed {processed_frames:,} of "
                f"{total_frames:,} frames"
            )
        else:
            progress_text.caption(f"Processed {processed_frames:,} frames")

    try:
        data = uploaded_file.getvalue()
        saved_upload = save_uploaded_mp4(uploaded_file.name, data)
        status.write("Upload validated and saved in an isolated temporary run.")

        stage = "model"
        status.update(label="Loading the YOLO detector…")
        detector = load_yolo_detector()

        stage = "processing"
        status.update(label="Analyzing video frames…")
        result = analyze_saved_upload(
            saved_upload,
            detector=detector,
            progress_callback=update_progress,
        )

        stage = "outputs"
        validate_pipeline_outputs(result)
        progress_bar.progress(1.0)
        progress_text.caption(
            f"Processed {result.processed_frames:,} frames"
        )
        st.session_state[ANALYSIS_STATE_KEY] = {
            "original_filename": saved_upload.original_filename,
            "download_stem": saved_upload.download_stem,
            "run_dir": str(saved_upload.run_dir),
            "result": result,
        }
        status.update(
            label="FlowSense analysis complete",
            state="complete",
            expanded=False,
        )
    except Exception as exc:
        if saved_upload is not None:
            try:
                remove_run_directory(saved_upload.run_dir)
            except (OSError, ValueError):
                pass
        status.update(
            label="FlowSense could not analyze this video",
            state="error",
            expanded=True,
        )
        st.error(friendly_analysis_error(exc, stage=stage))


def _render_uploaded_result(analysis: dict[str, object]) -> None:
    result = analysis["result"]
    if not isinstance(result, PipelineResult):
        st.error("The saved analysis state is invalid. Clear it and try again.")
        return

    st.divider()
    st.subheader(f"Results for {analysis['original_filename']}")
    try:
        validate_pipeline_outputs(result)
    except MissingOutputError as exc:
        st.error(str(exc))
        _render_clear_button(analysis)
        return

    if result.video_preview_warning:
        st.warning(result.video_preview_warning)

    st.video(str(result.output_video), format="video/mp4")
    metric_frames, metric_ids, metric_prediction = st.columns(3)
    metric_frames.metric("Frames processed", result.processed_frames)
    metric_ids.metric("Unique tracking IDs", result.unique_track_ids)
    with metric_prediction:
        prediction_accuracy = (
            f"{result.prediction_accuracy_percent:.1f}%"
            if result.prediction_accuracy_percent is not None
            else "N/A"
        )
        st.metric("Prediction accuracy", prediction_accuracy)
        if result.prediction_accuracy_percent is None:
            st.caption(
                "Not enough eligible forecasts were available to evaluate."
            )
        else:
            st.caption(
                f"Based on {result.prediction_accuracy_samples:,} forecasts "
                "compared with the stationary baseline."
            )

    count_left, count_right = st.columns(2)
    with count_left:
        st.markdown("#### Counts by class")
        st.dataframe(
            _counts_dataframe(result.counts_by_class, "Class"),
            hide_index=True,
            width="stretch",
        )
    with count_right:
        st.markdown("#### Counts by direction")
        st.dataframe(
            _counts_dataframe(result.counts_by_direction, "Direction"),
            hide_index=True,
            width="stretch",
        )

    st.info(
        "Solid lines show observed center-point paths. Dashed lines are "
        "short-term pixel-space motion predictions, not calibrated real-world "
        "positions or speeds."
    )

    stem = str(analysis["download_stem"])
    video_bytes = result.output_video.read_bytes()
    report_bytes = result.output_report.read_bytes()
    download_video, download_report, download_debug = st.columns(3)
    with download_video:
        st.download_button(
            "Download annotated MP4",
            data=video_bytes,
            file_name=f"{stem}_flowsense.mp4",
            mime="video/mp4",
        )
    with download_report:
        st.download_button(
            "Download HTML report",
            data=report_bytes,
            file_name=f"{stem}_report.html",
            mime="text/html",
        )
    with download_debug:
        if result.intermediate_dir is not None:
            try:
                debug_zip = build_intermediates_zip(result.intermediate_dir)
            except MissingOutputError as exc:
                st.warning(str(exc))
            else:
                st.download_button(
                    "Download analysis CSV ZIP",
                    data=debug_zip,
                    file_name=f"{stem}_analysis_csvs.zip",
                    mime="application/zip",
                )

    _render_clear_button(analysis)


def _render_clear_button(analysis: dict[str, object]) -> None:
    if st.button("Clear uploaded analysis"):
        try:
            remove_run_directory(str(analysis["run_dir"]))
        except (OSError, ValueError):
            pass
        st.session_state.pop(ANALYSIS_STATE_KEY, None)
        st.session_state[UPLOADER_VERSION_KEY] = (
            st.session_state.get(UPLOADER_VERSION_KEY, 0) + 1
        )
        st.rerun()


def _counts_dataframe(values: dict[str, int], label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{label: name, "Count": count} for name, count in sorted(values.items())]
    )


st.title("🚦 FlowSense")
st.caption("AI-Powered Traffic Flow Analysis")

project_results_tab, upload_tab = st.tabs(
    ["Project Results", "Analyze Your Video"]
)
with project_results_tab:
    render_project_results()
with upload_tab:
    render_upload_analysis()
