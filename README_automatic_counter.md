# FlowSense Automatic Counter

This is Brayden's assignment deliverable. It converts Kellan's canonical
tracking data into final traffic counts.

It does not rerun YOLO, change Kellan's identities, manually count footage, or
perform prediction.

Version `hybrid-v5` automatically detects severe ID fragmentation. Normal
videos count each moving canonical ID once. A heavily fragmented video switches
to counting objects once when they cross a consistent horizontal passage line.
This prevents Video 2's short track fragments from being counted as separate
vehicles.

## Simplest way to run it

The complete bundle already contains the four
`member2_canonical_tracks*.csv` input files.

When running from the shared FlowSense repository, do not copy those inputs
beside this script. The canonical CSVs already live under `tracking_data/`,
and the counter automatically prefers that directory so generated copies
under `outputs/` are not analyzed twice.

1. Extract the ZIP file.
2. Open the extracted folder. The Python program and all four CSVs should be
   directly inside it; there is no additional subfolder.
3. Click the File Explorer address bar, type `cmd`, and press Enter.
4. Run:

```powershell
py automatic_counter.py
```

The program automatically finds every matching canonical CSV, including files
inside subfolders.

Run the automated checks with:

```powershell
py -m unittest -v test_automatic_counter.py
```

You can also name files explicitly:

```powershell
py automatic_counter.py member2_canonical_tracks_video1.csv
```

## Results

Open the newly created `easy_results` folder.

Start with:

- `FlowSense_report.html` — double-click it for a formatted, plain-English
  browser report.
- `automatic_counts.csv` — one simple row for Video 1, Video 2, Video 3, and
  Video 4, with totals by direction and class.
- `video_file_map.csv` — shows exactly which original filename became each
  video label.
- `comparison_by_video_class.csv` — compares class totals with Kelvin while
  deliberately ignoring incompatible direction labels.

The older long-form layout is retained as `automatic_counts_detailed.csv`:

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

The program now accepts Kelvin's wide format:

```csv
Video,Cars,Trucks,Buses,Motorcycles
Video 1,8,0,0,0
Video 2,66,4,2,0
```

It also accepts the older long format. Then run:

```powershell
py automatic_counter.py --config automatic_counter_config.json --manual-counts kelvin_counts.csv --evaluation-video "Video 4"
```

This creates `comparison_by_video_class.csv` with signed error, absolute error,
and percentage error. Direction is intentionally excluded because Kelvin and
Brayden currently use incompatible direction definitions.

The included `reported_vehicle_validation.csv` is a temporary check using the
four manual vehicle totals quoted in the project review. Kelvin's uploaded
comparison template currently has blank `Kelvin count` cells, so replace this
temporary check with a program-generated comparison after the completed manual
counts file is available.

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
