"""FlowSense Streamlit dashboard.

Run locally:
    python -m streamlit run Dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "easy_results"
VIDEOS_DIR = ROOT / "videos"

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

REQUIRED_RESULTS = [
    "automatic_counts.csv",
    "traffic_volume_intervals.csv",
    "comparison_by_video.csv",
    "comparison_by_video_class.csv",
    "prediction_accuracy.csv",
    "summary.json",
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


require_files()

counts = load_csv("automatic_counts.csv")
intervals = load_csv("traffic_volume_intervals.csv")
comparison = load_csv("comparison_by_video.csv")
class_comparison = load_csv("comparison_by_video_class.csv")
prediction = load_csv("prediction_accuracy.csv")
summary = load_summary()


st.title("🚦 FlowSense")
st.caption("AI-Powered Traffic Flow Analysis and Short-Term Motion Prediction")

with st.sidebar:
    st.header("Dashboard Controls")
    selected_video = st.selectbox(
        "Select traffic footage",
        counts["Video"].tolist(),
    )
    show_video = st.checkbox("Show annotated video", value=True)

    st.divider()
    st.markdown(
        """
        **Pipeline**

        Video → YOLO detection → ByteTrack identities → identity consolidation
        → motion prediction → traffic counting → evaluation
        """
    )

overview_tab, video_tab, prediction_tab = st.tabs(
    [
        "Network Overview",
        "Video Analysis",
        "Prediction & Evaluation",
    ]
)


with overview_tab:
    combined = summary["combined"]
    overall_prediction = prediction.loc[
        prediction["Dataset"] == "All videos"
    ].iloc[0]

    st.subheader("Combined system results")
    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    metric1.metric("Videos analyzed", int(combined["videos_analyzed"]))
    metric2.metric("Moving objects", int(combined["moving_ids_counted"]))
    metric3.metric("Parked / excluded", int(combined["parked_or_excluded_ids"]))
    metric4.metric(
        "Prediction win rate",
        f"{overall_prediction['Prediction win rate (%)']:.1f}%",
        help="Share of eligible forecasts that beat a stationary baseline.",
    )
    metric5.metric(
        "Median prediction error",
        f"{overall_prediction['Median prediction error (px)']:.1f} px",
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


with video_tab:
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
            st.video(str(video_path))
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


with prediction_tab:
    selected_prediction = prediction.loc[
        prediction["Dataset"] == selected_video
    ].iloc[0]
    overall_prediction = prediction.loc[
        prediction["Dataset"] == "All videos"
    ].iloc[0]

    st.subheader(f"{selected_video} short-term position prediction")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Eligible forecasts", f"{int(selected_prediction['Samples']):,}")
    p2.metric(
        "Prediction horizon",
        f"{int(selected_prediction['Prediction horizon (frames)'])} frames",
        f"{selected_prediction['Prediction horizon (seconds)']:.2f} s",
    )
    p3.metric(
        "Median error",
        f"{selected_prediction['Median prediction error (px)']:.1f} px",
    )
    p4.metric(
        "Median improvement",
        f"{selected_prediction['Median improvement vs baseline (%)']:.1f}%",
        help=(
            "Improvement in median error compared with assuming the object "
            "does not move."
        ),
    )
    p5.metric(
        "Prediction win rate",
        f"{selected_prediction['Prediction win rate (%)']:.1f}%",
    )

    if selected_prediction["Median improvement vs baseline (%)"] < 0:
        st.warning(
            "For this scene, the stationary baseline has a lower median error "
            "than the motion predictor. This commonly happens when objects "
            "move slowly or detection jitter is large relative to true motion."
        )
    else:
        st.success(
            "For this scene, the motion predictor improves on the stationary "
            "baseline in median error."
        )

    error_chart = prediction.loc[
        prediction["Dataset"] != "All videos",
        [
            "Dataset",
            "Median prediction error (px)",
            "Median stationary baseline error (px)",
        ],
    ].set_index("Dataset")

    prediction_left, prediction_right = st.columns(2)
    with prediction_left:
        st.subheader("Prediction vs stationary baseline")
        st.bar_chart(error_chart)
        st.caption("Lower pixel error is better.")

    with prediction_right:
        st.subheader("Prediction win rate by video")
        win_rate_chart = prediction.loc[
            prediction["Dataset"] != "All videos",
            ["Dataset", "Prediction win rate (%)"],
        ].set_index("Dataset")
        st.bar_chart(win_rate_chart)

    st.subheader("How prediction is evaluated")
    st.markdown(
        """
        The predictor estimates an object's image-center position 15 frames
        into the future using an average of recent velocities. Each forecast is
        compared with the same track's later observed center. The stationary
        baseline assumes the object stays at its current center.

        Across all four videos, the predictor evaluated
        **{samples:,} forecasts**, achieved a median error of
        **{prediction_error:.3f} px**, and beat the stationary baseline on
        **{win_rate:.2f}%** of eligible samples.

        This is short-term image-position prediction—not driver-intention,
        collision-risk, or real-world speed prediction.
        """.format(
            samples=int(overall_prediction["Samples"]),
            prediction_error=overall_prediction[
                "Median prediction error (px)"
            ],
            win_rate=overall_prediction["Prediction win rate (%)"],
        )
    )

    st.subheader("Prediction evaluation table")
    st.dataframe(prediction, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Manual traffic count vs FlowSense")
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

    evaluation_left, evaluation_right = st.columns([1.1, 1])
    with evaluation_left:
        st.bar_chart(
            selected_class_comparison[
                ["Class", "Kelvin count", "Automatic count"]
            ].set_index("Class")
        )
    with evaluation_right:
        st.dataframe(
            selected_class_comparison,
            hide_index=True,
            width="stretch",
        )

    selected_comparison = comparison.loc[
        comparison["Video"] == selected_video
    ].iloc[0]
    if selected_comparison["Dataset role"] == "Evaluation":
        manual_total = float(selected_comparison["Kelvin vehicles"])
        automatic_total = float(selected_comparison["Automatic vehicles"])
        agreement = (
            max(
                0.0,
                1.0 - abs(automatic_total - manual_total) / manual_total,
            )
            if manual_total
            else 0.0
        )
        st.metric(
            "Evaluation-set total-count agreement",
            f"{agreement:.1%}",
            help=(
                "Calculated as 1 − absolute vehicle-count error / manual "
                "vehicle count. This is not object-level detection accuracy."
            ),
        )

with st.expander("Method and limitations"):
    st.markdown(
        """
        **Method**

        - A pretrained YOLO model detects road users in each frame.
        - ByteTrack and identity consolidation create more stable object IDs.
        - Recent track velocity is used for short-term position prediction.
        - Stationary objects are excluded using trajectory movement.
        - Severely fragmented tracking switches to passage-line counting.

        **Limitations**

        - Occlusion can create new tracking identities.
        - Cars and trucks may be confused when distant or partly hidden.
        - Pixel motion is not real-world speed.
        - Prediction ground truth comes from later tracked centers, not
          independent human annotations.
        - The predictor estimates short-term motion, not driver intention.
        """
    )

st.caption(
    "FlowSense · Kellan Reddy · Kelvin Qian · Brayden Chen · Batuhan Akbas"
)
