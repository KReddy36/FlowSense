# FlowSense: AI-Powered Traffic Flow Analysis

FlowSense uses pretrained YOLO detections and ByteTrack to identify, track, and
analyze vehicles and pedestrians in prerecorded traffic footage.

This repository includes Member 2's tracking pipeline. It reads Member 1's
frame-level detection CSV, assigns persistent IDs, consolidates duplicate or
briefly interrupted identities, renders the results over the original video,
and exports canonical tracking data for the analytics and dashboard components.

## Repository inputs and outputs

Required inputs:

```text
tracking_data/tracking_data.csv
videos/source_traffic.mp4
```

Generated outputs:

```text
outputs/member2_bytetrack_overlay.mp4
outputs/member2_canonical_tracks.csv
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

The defaults use the repository paths listed above. Custom paths are supported:

```powershell
python track_member1_video.py `
    --csv tracking_data/tracking_data.csv `
    --video videos/source_traffic.mp4 `
    --output outputs/member2_bytetrack_overlay.mp4 `
    --tracks-output outputs/member2_canonical_tracks.csv
```

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
