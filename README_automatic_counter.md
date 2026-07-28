# FlowSense Automatic Counter

This is Brayden's assignment deliverable. It converts Kellan's canonical
tracking data into final traffic counts.

It does not rerun YOLO, change Kellan's identities, manually count footage, or
perform prediction.

## Simplest way to run it

1. Put `automatic_counter.py` in the folder containing all files named
   `member2_canonical_tracks*.csv`.
2. Open Command Prompt in that folder.
3. Run:

```powershell
py automatic_counter.py
```

The program automatically finds every matching canonical CSV, including files
inside subfolders.

You can also name files explicitly:

```powershell
py automatic_counter.py member2_canonical_tracks_video1.csv
```

## Results

Open the newly created `automatic_counter_results` folder.

The main deliverable is `automatic_counts.csv`:

```text
Video,Direction,Class,Automatic count
Video 1,Toward camera,Car,18
Video 1,Away from camera,Car,7
```

Other outputs:

- `object_movement_audit.csv` explains whether every canonical ID was counted
  as moving or excluded as parked.
- `traffic_volume_intervals.csv` gives 5- or 10-second volumes.
- `kelvin_comparison_template.csv` is ready for Kelvin's manual totals.
- `summary.json` contains all settings and totals.

## Important settings

The default movement threshold is 50 pixels. An object's complete center
trajectory must span at least 50 pixels to count as moving:

```powershell
py automatic_counter.py --movement-threshold-pixels 50
```

In most road footage, moving downward in the image means moving toward the
camera. If a video's perspective is reversed:

```powershell
py automatic_counter.py video2.csv --toward-camera up
```

If the four videos need different settings or clean labels, copy
`automatic_counter_config.example.json` to `automatic_counter_config.json`,
enter the real filenames, and run:

```powershell
py automatic_counter.py --config automatic_counter_config.json
```

Tune the movement threshold and camera direction using Videos 1–3. Lock those
settings before evaluating Video 4:

```powershell
py automatic_counter.py --config automatic_counter_config.json --evaluation-video "Video 4"
```

## Kelvin comparison

Kelvin's CSV should use:

```csv
Video,Direction,Class,Kelvin count
Video 1,Toward camera,Car,18
Video 1,Away from camera,Car,7
```

Then run:

```powershell
py automatic_counter.py --config automatic_counter_config.json --manual-counts kelvin_counts.csv --evaluation-video "Video 4"
```

This creates `comparison_with_kelvin.csv` with signed error, absolute error,
and percentage error.

## Canonical CSV compatibility

The program prefers a `canonical_id` column. It also accepts the current
`track_id` column as a fallback, so Kellan does not need to rename existing
clean files.

Required columns:

- `frame`
- `time_seconds`
- `canonical_id` or `track_id`
- `class_name`
- `center_x`
- `center_y`

It also uses `class_id` and `confidence` when available.
