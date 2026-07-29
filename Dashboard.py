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


def require_files(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        st.error("Required project files are missing:")
        st.code("\n".join(missing))
        st.stop()


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


st.title("🚦 FlowSense")
st.caption("AI-Powered Traffic Flow Analysis")

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
    f"Movement threshold: **{selected_summary['movement_threshold_pixels']:.0f} px**"
)

left, right = st.columns([1.6, 1])

with left:
    st.subheader("Annotated tracking video")
    video_path = VIDEO_FILES.get(selected_video)
    if show_video and video_path and video_path.exists():
        st.video(str(video_path))
    elif show_video:
        st.warning(
            "The annotated video is not available at the expected repository path."
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
        max(0.0, 1.0 - abs(automatic_total - manual_total) / manual_total)
        if manual_total
        else 0.0
    )
    st.metric(
        "Evaluation-set total-count agreement",
        f"{evaluation_agreement:.1%}",
        help=(
            "Calculated as 1 − absolute vehicle-count error / manual vehicle "
            "count. This measures total-count agreement, not object-level "
            "detection accuracy."
        ),
    )

with st.expander("Method and limitations"):
    st.markdown(
        """
        **Method**

        - A pretrained YOLO model detects road users in each frame.
        - ByteTrack and identity consolidation create more stable object IDs.
        - FlowSense excludes stationary objects using trajectory movement.
        - For severely fragmented tracking, it switches to passage-line
          counting.
        - Counts are grouped by class, direction, and five-second interval.

        **Limitations**

        - Occlusion can cause a road user to receive a new tracking ID.
        - Cars and trucks may be confused when vehicles are distant or partly
          hidden.
        - Pixel movement is not the same as real-world speed.
        - Camera perspective affects direction and passage-line interpretation.
        - The reported comparison uses manual counts as an approximate ground
          truth.
        """
    )

with st.expander("All video results"):
    st.dataframe(counts, hide_index=True, width="stretch")

st.caption(
    "FlowSense · Kellan Reddy · Kelvin Qian · Brayden Chen · Batuhan Akbas"
)
