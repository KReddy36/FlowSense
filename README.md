# FlowSense

**AI-powered traffic detection, tracking, motion prediction, counting, and visualization from prerecorded video**

[![Tests](https://github.com/KReddy36/FlowSense/actions/workflows/tests.yml/badge.svg)](https://github.com/KReddy36/FlowSense/actions/workflows/tests.yml)

FlowSense turns traffic footage into an annotated video and a traffic-analysis
report. It uses a pretrained YOLO model to detect road users, ByteTrack plus
identity-consolidation logic to follow them across frames, short-term motion
prediction to estimate where they are moving, and an automatic counter to
summarize traffic by class, direction, and time interval.

The repository also includes a Streamlit dashboard with the team's four
precomputed videos, manual-versus-automatic comparisons, class breakdowns,
direction charts, and traffic-volume graphs.

## What FlowSense does

The complete pipeline is:

**Video → YOLO detection → ByteTrack → identity consolidation → motion
prediction → prediction evaluation → traffic counting → annotated video and
HTML report**

Main features:

- Detects cars, trucks, buses, motorcycles, bicycles, and pedestrians.
- Assigns and stabilizes object identities across video frames.
- Suppresses duplicate tracks and reconnects short tracking gaps.
- Predicts object centers 15 frames into the future.
- Scores eligible predictions against later observed positions.
- Excludes stationary objects and counts moving road users.
- Switches to passage-line counting when tracking is severely fragmented.
- Produces class totals, direction totals, and five-second traffic volumes.
- Compares automatic vehicle counts with manual reference counts.
- Presents saved results in an interactive local dashboard.
- Processes a new traffic video from start to finish with one command.

## Results

Videos 1–3 were used as development data. Video 4 was reserved as the
evaluation video.

| Video | Role | Manual vehicles | FlowSense vehicles | Difference |
| --- | --- | ---: | ---: | ---: |
| Video 1 | Development | 8 | 6 | -2 |
| Video 2 | Development | 72 | 77 | +5 |
| Video 3 | Development | 51 | 47 | -4 |
| Video 4 | Evaluation | 239 | 264 | +25 |

`Difference` means automatic count minus manual count. On the held-out
evaluation video, FlowSense overcounted by 25 vehicles, corresponding to
approximately **89.5% total-count agreement**. This is an aggregate count
comparison, not object-detection accuracy.

### Motion-prediction evaluation

The predictor was evaluated by comparing each predicted center with the same
canonical track's observed center 15 frames later. A stationary baseline
assumed that the object would remain at its current location.

| Metric across all four videos | Result |
| --- | ---: |
| Eligible prediction samples | 61,388 |
| Median prediction error | 10.012 px |
| Median stationary-baseline error | 34.790 px |
| Reduction in median error | 71.22% |
| Samples where prediction beat the baseline | 83.36% |

The predictor did not outperform the stationary baseline on Video 1, where
many objects moved very little. See
[PREDICTION_EVALUATION.md](PREDICTION_EVALUATION.md) for the per-video table,
methodology, and limitations.

## Open the dashboard

The dashboard uses the precomputed files already stored in `easy_results/`.
You do **not** need to rerun YOLO or process the videos before opening it.

### macOS

1. On GitHub, select **Code → Download ZIP**.
2. Expand the ZIP. The example below assumes the folder is named
   `FlowSense-main` and is on the Desktop.
3. Open Terminal and run:

```bash
cd ~/Desktop/FlowSense-main
python3 -m pip install -r requirements.txt
python3 -m streamlit run Dashboard.py
```

Streamlit should open the dashboard automatically. If it does not, open
[http://localhost:8501](http://localhost:8501) in a browser.

If the folder is somewhere else, type `cd ` with a space, drag the expanded
FlowSense folder into Terminal, and press Return. Then run the remaining two
commands.

### Windows

Open PowerShell or Command Prompt inside the expanded repository folder and
run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run Dashboard.py
```

The dashboard is served from `localhost`, so each user runs a private copy on
their own computer. Keep the terminal window open while using it. Press
`Control+C` in the terminal to stop the dashboard.

### Optional virtual environment

Using a virtual environment keeps FlowSense packages separate from other
Python projects:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run Dashboard.py
```

On Windows PowerShell, activate it with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run Dashboard.py
```

## Analyze a new video

After installing the dependencies, run the complete pipeline from the
repository root:

```bash
python3 run_flowsense.py "/path/to/traffic_video.mp4"
```

Windows users can use:

```powershell
py run_flowsense.py "C:\path\to\traffic_video.mp4"
```

The default `results/` folder will contain:

- `<video-name>_flowsense.mp4` — detections, canonical IDs, observed
  trajectories, and dashed short-term predictions.
- `<video-name>_report.html` — traffic counts, a prediction-accuracy
  percentage when enough forecasts are eligible, and analysis in a standalone
  browser report.

The first real run may download the pretrained `yolo11n.pt` weights and
therefore requires an internet connection. No API key or paid service is
required.

Useful options:

```bash
# Process only five frames as a quick integration check
python3 run_flowsense.py video.mp4 --max-frames 5

# Replace an existing output pair
python3 run_flowsense.py video.mp4 --overwrite

# Keep intermediate tracking, prediction, and counting files
python3 run_flowsense.py video.mp4 --keep-intermediates

# See every available setting
python3 run_flowsense.py --help
```

## Reproduce the saved analyses

The repository includes the canonical tracking CSVs used for the final
evaluation, so these commands do not require a new YOLO run.

Regenerate the automatic-count reports:

```bash
python3 automatic_counter.py \
  --manual-counts kelvin_vehicle_counts.csv \
  --evaluation-video "Video 4"
```

Regenerate the motion-prediction evaluation:

```bash
python3 evaluate_motion_prediction.py
```

The main saved outputs are in `easy_results/`:

| File | Purpose |
| --- | --- |
| `FlowSense_report.html` | Standalone traffic-analysis report |
| `automatic_counts.csv` | Automatic totals by video, class, and direction |
| `comparison_by_video.csv` | Manual-versus-automatic vehicle totals |
| `comparison_by_video_class.csv` | Class-level comparison |
| `traffic_volume_intervals.csv` | Five-second traffic volumes |
| `object_movement_audit.csv` | Counted/excluded decision for each track |
| `prediction_accuracy.csv` | Motion-prediction evaluation |
| `summary.json` | Machine-readable settings and results |

For counter-specific options and file formats, see
[README_automatic_counter.md](README_automatic_counter.md).

## Run the tests

Install the dependencies, then run:

```bash
python3 -m unittest discover -v
```

The same test suite runs automatically through GitHub Actions on pushes and
pull requests. It covers detection conversion, tracking, identity
consolidation, motion prediction, counting, prediction evaluation, and the
end-to-end pipeline.

## Team

| Team member | Primary contribution |
| --- | --- |
| **Batuhan Akbas** | YOLO detection foundation, initial detection/tracking notebook, traffic-video datasets, and project coordination |
| **Kellan Reddy** | ByteTrack pipeline, identity consolidation, motion prediction and evaluation, end-to-end pipeline, testing, and repository integration |
| **Brayden Chen** | Automatic traffic counter, hybrid movement/passage logic, generated reports, comparisons, and counter tests |
| **Kelvin Qian** | Manual reference counts, result comparison, Streamlit dashboard, and visualization |

FlowSense is an integrated team project: each stage consumes the output of the
previous stage, and the dashboard presents the combined results.

## Repository guide

| Path | Description |
| --- | --- |
| `run_flowsense.py` | One-command entry point for a new video |
| `Dashboard.py` | Streamlit dashboard for the four saved analyses |
| `automatic_counter.py` | Automatic traffic counting and report generation |
| `evaluate_motion_prediction.py` | Prediction-versus-baseline evaluation |
| `flowsense/` | Detection, tracking, prediction, rendering, and orchestration code |
| `tracking_data/` | Detection and canonical-track CSVs for four videos |
| `videos/` | Source and annotated demonstration videos |
| `easy_results/` | Reproducible result snapshot used by the dashboard |
| `tests/` | Automated regression and integration tests |
| `FILE_MANIFEST.md` | Complete description of production, data, and test files |

## Methods

### Detection and tracking

FlowSense uses a pretrained Ultralytics YOLO model for road-user detections and
ByteTrack for frame-to-frame association. A custom consolidation layer merges
duplicate detections, reconnects short gaps using predicted positions, avoids
merging two established nearby vehicles, and stabilizes temporary class
changes.

### Motion prediction

For each active canonical track, FlowSense stores a bounded history of center
points. It averages recent frame-to-frame velocities and projects the center
forward by 15 frames. Observed trajectories are rendered as solid lines;
predicted motion is rendered as a dashed line ending in an outlined marker.

### Traffic counting

The counter normally counts each moving canonical identity once and excludes
tracks whose total movement is below 50 pixels. If the rate of fragmented IDs
is unusually high, automatic mode switches to horizontal passage-line
counting. Video 4 remained held out while the counting settings were selected
using Videos 1–3.

## Limitations

- FlowSense was evaluated on four prerecorded, fixed-camera videos; the
  results do not establish performance across all roads, cameras, weather, or
  lighting conditions.
- Occlusion, crowded scenes, and long detection gaps can split one vehicle
  into multiple identities and cause overcounting.
- Distant or partially hidden vehicles can be assigned the wrong class.
- The held-out Video 4 result overcounted by 25 vehicles, showing that identity
  fragmentation remains a meaningful limitation.
- The manual counts are an approximate human benchmark, not frame-by-frame
  object annotations.
- Motion and prediction values are measured in image pixels. They are not
  calibrated real-world positions or speeds.
- The predictor is a short-term constant-velocity model and can perform poorly
  for nearly stationary or abruptly turning objects.
- FlowSense analyzes prerecorded footage; it is not currently a live traffic
  control or safety system.

## Requirements and license

- Python 3.12 is recommended.
- Dependencies are listed in `requirements.txt`.
- A real YOLO run can be compute-intensive and may be slow without a supported
  GPU.
- The project is licensed under the
  [GNU Affero General Public License v3.0](LICENSE).
