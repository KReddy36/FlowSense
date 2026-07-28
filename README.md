# FlowSense: AI-Powered Traffic Flow Analysis

FlowSense uses pretrained YOLO detections and ByteTrack to identify, track, and
analyze vehicles and pedestrians in prerecorded traffic footage.

This repository includes Member 2's tracking pipeline. It reads Member 1's
frame-level detection CSV, assigns persistent IDs, consolidates duplicate or
briefly interrupted identities, renders the results over the original video,
and exports canonical tracking data for the analytics and dashboard components.

## Motion prediction

The tracking pipeline maintains a bounded rolling center-point history for
each active canonical track. It averages recent point-to-point velocities to
reduce detection jitter, keeps predicting temporarily missing tracks, and
removes prediction state after a configurable inactivity timeout.

Observed trajectories are solid in the annotated video. Short-term linear
predictions are dashed and end in an outlined marker.

Run a short prediction preview:

```powershell
python track_member1_video.py `
    --dataset 1 `
    --max-frames 300 `
    --history-points 30 `
    --velocity-window 5 `
    --prediction-horizon 15 `
    --inactive-timeout 30
```

The full frame-by-frame motion history is streamed to
`outputs/member2_motion_predictions.csv`; it is not retained in memory. Use
`--motion-output PATH` to choose another file or `--no-motion-output` to
disable it. The existing canonical track CSV format remains unchanged.

The motion CSV contains one row per active track per frame:

```text
frame,time_seconds,track_id,class_id,class_name,is_observed,
frames_since_seen,estimated_center_x,estimated_center_y,
velocity_x_pixels_per_second,velocity_y_pixels_per_second,
speed_pixels_per_second,direction_degrees,prediction_horizon_frames,
predicted_frame,predicted_time_seconds,predicted_center_x,predicted_center_y
```

`is_observed=0` means the row is a temporary estimate during a missed
detection. Statistics and dashboard code can join this file to the canonical
CSV using `frame` and `track_id`.

## Repository inputs and outputs

Required inputs:

```text
tracking_data/tracking_data.csv
tracking_data/tracking_data2.csv
tracking_data/tracking_data3.csv
tracking_data/tracking_data4.csv
videos/source_traffic.mp4
videos/flowsense_tracking2.mp4
videos/flowsense_tracking3.mp4
videos/flowsense_tracking4*.mp4
```

Generated outputs:

```text
outputs/member2_bytetrack_overlay.mp4
outputs/member2_canonical_tracks.csv
outputs/member2_bytetrack_overlay2.mp4
outputs/member2_canonical_tracks2.csv
...
```

A preprocessed presentation fallback is committed as:

```text
videos/member2_bytetrack_overlay.mp4
```

## Environment setup

Python 3.12 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No API key or paid service is required.

## Reproduce Member 2's results

From the repository root:

```powershell
python track_member1_video.py
```

This processes dataset 1 by default. Select either additional pair with:

```powershell
python track_member1_video.py --dataset 2
python track_member1_video.py --dataset 3
python track_member1_video.py --dataset 4
```

Process all four sequentially with:

```powershell
python track_member1_video.py --dataset all
```

This processes all four dataset pairs. Each dataset writes to a different video
and canonical CSV under `outputs/`, so
the results do not overwrite each other. Custom paths are supported for a
single selected dataset:

```powershell
python track_member1_video.py `
    --dataset 1 `
    --csv tracking_data/tracking_data.csv `
    --video videos/source_traffic.mp4 `
    --output outputs/member2_bytetrack_overlay.mp4 `
    --tracks-output outputs/member2_canonical_tracks.csv
```

To generate canonical CSV data without decoding or writing an overlaid video:

```powershell
python track_member1_video.py --dataset 4 --no-video
python track_member1_video.py --dataset all --no-video
```

`--no-video` is the faster option for analytics workflows that only need the
canonical track records.

For dataset 4, the program accepts the canonical `flowsense_tracking4.mp4`
filename or a single browser-uploaded variant such as
`flowsense_tracking4 (1) (1) (1).mp4`.

For a short development check:

```powershell
python track_member1_video.py `
    --max-frames 50 `
    --output outputs/preview.mp4 `
    --tracks-output outputs/preview_tracks.csv
```

## Detection input contract

The detection CSV must contain:

```text
frame,time_seconds,class_name,confidence,x1,y1,x2,y2
```

Additional columns are allowed. Member 1's existing `track_id`, `center_x`, and
`center_y` columns are accepted but are not used as tracker input. Member 2's
pipeline assigns its own IDs.

Supported pretrained COCO classes are:

```text
person, bicycle, car, motorcycle, bus, truck
```

All bounding-box values use source-video pixel coordinates.

## Canonical tracking output

`member2_canonical_tracks.csv` contains one visible canonical object per frame:

```text
frame,time_seconds,track_id,class_id,class_name,confidence,
center_x,center_y,x1,y1,x2,y2
```

Member 3 can read this file directly for unique counts, class distributions,
density, direction, and trajectory statistics. Member 4 can use the annotated
video as a dashboard input or import the Python modules for live processing.
Datasets 2–4 use the corresponding numbered names, such as
`member2_canonical_tracks4.csv`.

Precomputed canonical CSVs for all four datasets are also committed under
`tracking_data/` so the analytics and dashboard components can work without
rerunning ByteTrack.

## Identity consolidation

After ByteTrack assigns raw identities, the consolidation layer:

- merges strongly overlapping detections under one canonical ID;
- hides all but the highest-confidence duplicate in a frame;
- reconnects new raw IDs to recently missing objects using predicted position;
- preserves one ID and color across those matches; and
- stabilizes brief class changes using accumulated confidence.

The default thresholds are defined in
`flowsense/tracking/identity_consolidator.py`.

## Development tests

The following command verifies CSV loading, ByteTrack ID stability, duplicate
suppression, short-gap reconnection, and protection against merging separate
objects:

```powershell
python -m unittest discover -s tests -v
```

Files under `tests/`, `demo_day1.py`, and
`flowsense/tracking/verification.py` are clearly marked **TEST-ONLY**. Keep them
through team integration and regression testing, but exclude them from the
final deployed application if the submission requires runtime files only.

See `FILE_MANIFEST.md` for the complete production/test-only distinction.

## Known limitations

- Identity consolidation uses spatial and motion heuristics rather than visual
  appearance embeddings.
- Pixel motion depends on the fixed camera view and is not real-world speed.
- Heavy occlusion or long detection gaps can still create new identities.
- Closely overlapping real objects can occasionally be mistaken for duplicates.
