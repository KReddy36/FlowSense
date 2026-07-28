# FlowSense Automatic Counter

This is Brayden's assignment deliverable. It converts Kellan's canonical
tracking data into final traffic counts.

It does not rerun YOLO, change Kellan's identities, manually count footage, or
perform prediction.

## Project contribution ownership

- **Brayden Chen:** automatic-counter owner—movement/passage counting, class
  and direction totals, traffic intervals, Kelvin comparison, and these
  counter tests.
- **Kellan Reddy:** canonical identity-cleanup data and integration work.
- **Kelvin Qian:** manual traffic counts used as the human benchmark.

Later integration commits by another teammate do not change ownership of
Brayden's automatic-counter component.

Version `hybrid-v6.2` automatically detects severe ID fragmentation. Normal
videos count each moving canonical ID once. A heavily fragmented video switches
to counting objects once when they cross a consistent horizontal passage line.
This prevents Video 2's short track fragments from being counted as separate
vehicles.

For a passage-counted video, nearby downstream non-car fragments can refine a
crossing's class without adding another vehicle. This preserves the 72-vehicle
Video 2 total while recovering detected truck and motorcycle evidence.

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
inside subfolders. If one `kelvin_vehicle_counts*.csv` file is beside the
inputs, the program also imports it and creates the Kelvin comparison
automatically. Videos 1–3 are treated as development data and Video 4 is
labelled as the evaluation set by default.

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
- `comparison_by_video.csv` — one row per video showing Kelvin's vehicle total,
  FlowSense's total, total error, and combined class error.
- `FlowSense_report.html` — includes both the overall comparison and a
  side-by-side Kelvin-versus-FlowSense table for every vehicle class.

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

The program accepts Kelvin's actual wide format, including repeated direction
rows and annotated cells such as `3 (only one identified)`:

```csv
Video number,Direction of the road,Cars crossing,Trucks,Buses,Motorcycles
Video 2,down,4,2,0,3
Video 2,up,62,1,0,0
```

It also accepts `Video`, `Cars`, `Trucks`, and similar headers, plus the older
long format. When exactly one Kelvin file is beside the inputs, the normal
command imports it automatically. You may also select it explicitly:

```powershell
py automatic_counter.py --config automatic_counter_config.json --manual-counts kelvin_counts.csv --evaluation-video "Video 4"
```

This creates `comparison_by_video_class.csv` with signed error, absolute error,
and percentage error, plus `comparison_by_video.csv` for the simplest overall
comparison. Direction is intentionally excluded because Kelvin and Brayden
currently use incompatible direction definitions.

Class results remain limited by the classes present in Kellan's canonical CSVs.
For example, the Video 1 CSV contains no truck-labelled track even though
Kelvin counted two trucks, so the counter cannot recover those trucks from
tracking data alone.

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
