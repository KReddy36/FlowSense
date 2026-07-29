"""FlowSense Streamlit dashboard.

Run locally:
    python -m streamlit run Dashboard.py
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from flowsense.dashboard_uploads import (
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
from flowsense.video_compat import VideoConversionError, browser_video_bytes
from flowsense.yolo_detector import YoloDetector


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "easy_results"
VIDEOS_DIR = ROOT / "videos"
ANALYSIS_STATE_KEY = "uploaded_analysis"
UPLOADER_VERSION_KEY = "upload_widget_version"

VIDEO_FILES = {
    "Video 1": VIDEOS_DIR / "flowsense_hybrid_video_1.mp4",
    "Video 2": VIDEOS_DIR / "flowsense_hybrid_video_2.mp4",
    "Video 3": VIDEOS_DIR / "flowsense_hybrid_video_3.mp4",
    "Video 4": VIDEOS_DIR / "flowsense_hybrid_video_4.mp4",
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

REQUIRED_RESULTS = [
    "automatic_counts.csv",
    "traffic_volume_intervals.csv",
    "comparison_by_video.csv",
    "comparison_by_video_class.csv",
    "prediction_accuracy.csv",
    "learned_prediction_results.csv",
    "summary.json",
]

PROJECT_VIEWS = [
    "Network Overview",
    "Video Analysis",
    "Prediction & Evaluation",
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


@st.cache_data(show_spinner=False)
def load_browser_video(
    path: str,
    modified_nanoseconds: int,
    size_bytes: int,
) -> bytes:
    """Return cached browser-compatible bytes for one unchanged local video."""
    _ = (modified_nanoseconds, size_bytes)
    return browser_video_bytes(path)


def require_files() -> None:
    paths = [RESULTS_DIR / filename for filename in REQUIRED_RESULTS]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        st.error("Required project files are missing:")
        st.code("\n".join(missing))
        st.stop()


def class_table(row: pd.Series) -> pd.DataFrame:
    values = (
        row[CLASS_COLUMNS]
        .astype(float)
        .rename_axis("Class")
        .reset_index(name="Count")
    )
    return values[values["Count"] > 0]


def direction_table(row: pd.Series) -> pd.DataFrame:
    values = (
        row[DIRECTION_COLUMNS]
        .astype(float)
        .rename_axis("Direction")
        .reset_index(name="Count")
    )
    return values[values["Count"] > 0]


def render_project_results() -> None:
    """Render Kelvin's complete saved-results dashboard."""
    require_files()
    counts = load_csv("automatic_counts.csv")
    intervals = load_csv("traffic_volume_intervals.csv")
    comparison = load_csv("comparison_by_video.csv")
    class_comparison = load_csv("comparison_by_video_class.csv")
    prediction = load_csv("prediction_accuracy.csv")
    learned_prediction = load_csv("learned_prediction_results.csv")
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

            Video → YOLO detection → ByteTrack identities → identity
            consolidation → motion prediction → traffic counting → evaluation
            """
        )

    selected_view = st.radio(
        "Project results view",
        PROJECT_VIEWS,
        horizontal=True,
        label_visibility="collapsed",
    )
    if selected_view == "Network Overview":
        _render_network_overview(counts, learned_prediction, summary)
    elif selected_view == "Video Analysis":
        _render_video_analysis(
            selected_video,
            show_video,
            counts,
            intervals,
            comparison,
            summary,
        )
    else:
        _render_hybrid_prediction_evaluation(
            selected_video,
            prediction,
            comparison,
            class_comparison,
            learned_prediction,
        )

    _render_project_method_and_limitations()
    st.caption(
        "FlowSense · Kellan Reddy · Kelvin Qian · Brayden Chen · Batuhan Akbas"
    )


def _render_hybrid_prediction_evaluation(
    selected_video: str,
    prediction: pd.DataFrame,
    comparison: pd.DataFrame,
    class_comparison: pd.DataFrame,
    learned_prediction: pd.DataFrame,
) -> None:
    """Present stationary, mathematical, and learned-hybrid evaluations."""
    selected_prediction = prediction.loc[
        prediction["Dataset"] == selected_video
    ].iloc[0]
    selected_learned = learned_prediction.loc[
        learned_prediction["Test set"] == selected_video
    ].iloc[0]
    overall = learned_prediction.loc[
        learned_prediction["Test set"] == "All videos"
    ].iloc[0]

    _render_labeled_divider(f"Selected video · {selected_video}")
    st.subheader(f"{selected_video} learned-hybrid position prediction")
    columns = st.columns(5)
    columns[0].metric("Eligible forecasts", f"{int(selected_learned['Samples']):,}")
    columns[1].metric(
        "Stationary median",
        f"{selected_learned['Stationary median (px)']:.1f} px",
    )
    columns[2].metric(
        "Constant-velocity median",
        f"{selected_learned['Current median (px)']:.1f} px",
    )
    columns[3].metric(
        "Learned-hybrid median",
        f"{selected_learned['Hybrid median (px)']:.1f} px",
    )
    columns[4].metric(
        "Hybrid beats constant velocity",
        f"{selected_learned['Hybrid wins (%)']:.1f}%",
    )
    st.caption(
        f"Forecast horizon: {int(selected_prediction['Prediction horizon (frames)'])} "
        f"frames ({selected_prediction['Prediction horizon (seconds)']:.2f} s). "
        "Lower pixel error is better."
    )

    _render_labeled_divider("All videos · Cross-video prediction comparison")
    errors = learned_prediction.loc[
        learned_prediction["Test set"] != "All videos",
        [
            "Test set",
            "Stationary median (px)",
            "Current median (px)",
            "Hybrid median (px)",
        ],
    ].rename(
        columns={
            "Test set": "Dataset",
            "Current median (px)": "Constant velocity (px)",
            "Hybrid median (px)": "Learned hybrid (px)",
        }
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Prediction-method comparison")
        st.bar_chart(
            errors,
            x="Dataset",
            y=[
                "Stationary median (px)",
                "Constant velocity (px)",
                "Learned hybrid (px)",
            ],
            x_label="Video",
            y_label="Median error (pixels)",
            stack=False,
        )
        st.caption("All three methods are grouped side by side for each video.")
    with right:
        st.subheader("Hybrid win rate vs constant velocity")
        st.bar_chart(
            learned_prediction.loc[
                learned_prediction["Test set"] != "All videos",
                ["Test set", "Hybrid wins (%)"],
            ].set_index("Test set")
        )

    st.subheader("Overall leave-one-video-out results")
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": [
                    "Median error (px)",
                    "Mean error (px)",
                    "P90 error (px)",
                ],
                "Constant velocity": [
                    overall["Current median (px)"],
                    overall["Current mean (px)"],
                    overall["Current P90 (px)"],
                ],
                "Learned hybrid": [
                    overall["Hybrid median (px)"],
                    overall["Hybrid mean (px)"],
                    overall["Hybrid P90 (px)"],
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        """
        Across **{samples:,}** held-out forecasts, median error fell from
        **{current_median:.3f} px** to **{hybrid_median:.3f} px**; mean error
        fell from **{current_mean:.3f} px** to **{hybrid_mean:.3f} px**; and
        P90 error fell from **{current_p90:.3f} px** to **{hybrid_p90:.3f} px**.
        The hybrid beat constant velocity on **{hybrid_wins:.3f}%** of samples.

        The Ridge model adjusts only the **length** of the mathematical arrow;
        it does not learn a new turning direction. Evaluation used
        leave-one-video-out testing. The production model was subsequently
        refitted on all four videos, and uploaded videos automatically use that
        hybrid predictor with mathematical fallback.

        These are short-term pixel-space forecasts—not driver-intention,
        collision-risk, real-world position, or speed predictions.
        """.format(
            samples=int(overall["Samples"]),
            current_median=overall["Current median (px)"],
            hybrid_median=overall["Hybrid median (px)"],
            current_mean=overall["Current mean (px)"],
            hybrid_mean=overall["Hybrid mean (px)"],
            current_p90=overall["Current P90 (px)"],
            hybrid_p90=overall["Hybrid P90 (px)"],
            hybrid_wins=overall["Hybrid wins (%)"],
        )
    )
    st.subheader("Learned-prediction evaluation table")
    st.dataframe(learned_prediction, hide_index=True, width="stretch")

    _render_labeled_divider(f"Selected video · {selected_video} count evaluation")
    st.subheader("Manual traffic count vs FlowSense")
    selected_classes = class_comparison.loc[
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
    evaluation_left, evaluation_right = st.columns([1.1, 1])
    with evaluation_left:
        st.bar_chart(
            selected_classes,
            x="Class",
            y=["Kelvin count", "Automatic count"],
            x_label="Road-user class",
            y_label="Count",
            stack=False,
        )
    with evaluation_right:
        st.dataframe(selected_classes, hide_index=True, width="stretch")

    selected_comparison = comparison.loc[
        comparison["Video"] == selected_video
    ].iloc[0]
    if selected_comparison["Dataset role"] == "Evaluation":
        manual = float(selected_comparison["Kelvin vehicles"])
        automatic = float(selected_comparison["Automatic vehicles"])
        agreement = (
            max(0.0, 1.0 - abs(automatic - manual) / manual)
            if manual
            else 0.0
        )
        st.metric(
            "Evaluation-set total-count agreement",
            f"{agreement:.1%}",
            help=(
                "Calculated as 1 - absolute vehicle-count error / manual "
                "vehicle count; this is not object-level detection accuracy."
            ),
        )


def _render_labeled_divider(label: str) -> None:
    """Render a clear scope boundary without changing dashboard data."""
    safe_label = escape(label)
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin: 1.6rem 0 1rem 0;
        ">
            <div style="height: 1px; flex: 1; background: #9ca3af;"></div>
            <div style="
                color: #4b5563;
                font-size: 0.9rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                white-space: nowrap;
            ">{safe_label}</div>
            <div style="height: 1px; flex: 1; background: #9ca3af;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_network_overview(
    counts: pd.DataFrame,
    learned_prediction: pd.DataFrame,
    summary: dict,
) -> None:
    combined = summary["combined"]
    overall_hybrid = learned_prediction.loc[
        learned_prediction["Test set"] == "All videos"
    ].iloc[0]

    st.subheader("Combined system results")
    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    metric1.metric("Videos analyzed", int(combined["videos_analyzed"]))
    metric2.metric("Moving objects", int(combined["moving_ids_counted"]))
    metric3.metric("Parked / excluded", int(combined["parked_or_excluded_ids"]))
    metric4.metric(
        "Hybrid beats constant velocity",
        f"{overall_hybrid['Hybrid wins (%)']:.3f}%",
        help=(
            "Share of held-out forecasts where the learned hybrid had lower "
            "pixel error than the original constant-velocity predictor."
        ),
    )
    metric5.metric(
        "Learned-hybrid median error",
        f"{overall_hybrid['Hybrid median (px)']:.3f} px",
        help="Overall median error from leave-one-video-out evaluation.",
    )

    overview_left, overview_right = st.columns(2)
    with overview_left:
        st.subheader("Moving objects by video")
        st.bar_chart(
            counts[["Video", "Moving objects counted"]].set_index("Video")
        )

    with overview_right:
        st.subheader("Combined class distribution")
        combined_classes = pd.DataFrame(
            {
                "Class": list(combined["counts_by_class"].keys()),
                "Count": list(combined["counts_by_class"].values()),
            }
        )
        st.bar_chart(combined_classes, x="Class", y="Count")

    st.subheader("All traffic-analysis results")
    st.dataframe(counts, hide_index=True, width="stretch")


def _render_video_analysis(
    selected_video: str,
    show_video: bool,
    counts: pd.DataFrame,
    intervals: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: dict,
) -> None:
    selected_counts = counts.loc[counts["Video"] == selected_video].iloc[0]
    selected_comparison = comparison.loc[
        comparison["Video"] == selected_video
    ].iloc[0]
    selected_summary = summary["videos"][selected_video]

    st.subheader(f"{selected_video} traffic overview")
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

    video_column, class_column = st.columns([1.6, 1])
    with video_column:
        st.subheader("Annotated tracking and prediction video")
        video_path = VIDEO_FILES.get(selected_video)
        if show_video and video_path and video_path.exists():
            preview_error = _render_browser_video(video_path)
            if preview_error:
                st.warning(
                    "The annotated video is available to download but its "
                    f"browser preview could not be prepared. {preview_error}"
                )
            else:
                st.caption(
                    "Solid paths show observed trajectories. Dashed extensions "
                    "show short-term predicted positions when available."
                )
        elif show_video:
            st.warning(
                "The annotated video is not available at the expected path."
            )
        else:
            st.caption("Video display is disabled in the sidebar.")

    with class_column:
        st.subheader("Counts by class")
        st.bar_chart(
            class_table(selected_counts),
            x="Class",
            y="Count",
        )

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
        volume = (
            selected_intervals.groupby(
                ["Interval start (s)", "Interval"],
                as_index=False,
            )["Automatic count"]
            .sum()
            .sort_values("Interval start (s)")
        )
        st.line_chart(volume, x="Interval", y="Automatic count")
        if not volume.empty:
            busiest = volume.loc[volume["Automatic count"].idxmax()]
            st.caption(
                f"Busiest interval: {busiest['Interval']} "
                f"({int(busiest['Automatic count'])} moving objects)"
            )

    with chart_right:
        st.subheader("Direction of movement")
        st.bar_chart(
            direction_table(selected_counts),
            x="Direction",
            y="Count",
        )


def _render_project_method_and_limitations() -> None:
    with st.expander("Method and limitations"):
        st.markdown(
            """
            **Method**

            - A pretrained YOLO model detects road users in each frame.
            - ByteTrack and identity consolidation create more stable object IDs.
            - Recent track velocity supplies a mathematical forecast; a Ridge
              model adjusts its distance while preserving direction.
            - Stationary objects are excluded using trajectory movement.
            - Severely fragmented tracking switches to passage-line counting.

            **Limitations**

            - Occlusion can create new tracking identities.
            - Cars and trucks may be confused when distant or partly hidden.
            - Pixel motion is not real-world speed.
            - Prediction ground truth comes from later tracked centers, not
              independent human annotations.
            - Forecasts are pixel-space short-term motion estimates, not driver
              intention, collision risk, or real-world speed.
            """
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

    preview_error = _render_browser_video(result.output_video)
    if preview_error:
        st.warning(
            result.video_preview_warning
            or (
                "The annotated video is available below as a download, but "
                f"its browser preview could not be prepared. {preview_error}"
            )
        )
    elif result.video_preview_warning:
        st.info(
            "FlowSense prepared a temporary browser-compatible preview. The "
            "download remains the original annotated MP4."
        )
    metric_frames, metric_ids, metric_prediction = st.columns(3)
    metric_frames.metric("Frames processed", result.processed_frames)
    metric_ids.metric("Unique tracking IDs", result.unique_track_ids)
    with metric_prediction:
        prediction_accuracy = (
            f"{result.prediction_accuracy_percent:.1f}%"
            if result.prediction_accuracy_percent is not None
            else "N/A"
        )
        st.metric("Prediction accuracy (win rate)", prediction_accuracy)
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


def _render_browser_video(video_path: Path) -> str | None:
    """Render a cached H.264 preview and return a readable failure detail."""
    try:
        metadata = video_path.stat()
        video_data = load_browser_video(
            str(video_path.resolve()),
            metadata.st_mtime_ns,
            metadata.st_size,
        )
    except (OSError, VideoConversionError) as exc:
        return str(exc)
    st.video(video_data, format="video/mp4")
    return None


st.title("🚦 FlowSense")
st.caption("AI-Powered Traffic Flow Analysis and Short-Term Motion Prediction")

project_results_tab, upload_tab = st.tabs(
    ["Project Results", "Analyze Your Video"]
)
with project_results_tab:
    render_project_results()
with upload_tab:
    render_upload_analysis()
