# FlowSense file manifest

This manifest distinguishes production code, reproducibility artifacts, and
development-only verification files.

## Production application

### Entry points

- `run_flowsense.py` — one-command video-to-results pipeline
- `Dashboard.py` — Streamlit dashboard
- `automatic_counter.py` — traffic counting and HTML report generator
- `evaluate_motion_prediction.py` — prediction accuracy and baseline evaluator

### Core package

- `flowsense/__init__.py`
- `flowsense/pipeline.py` — unified detection-to-report orchestration
- `flowsense/yolo_detector.py` — in-memory Ultralytics YOLO adapter
- `flowsense/csv_detections.py` — legacy detection CSV loader
- `flowsense/prediction_evaluation.py` — measured future-position evaluation
- `flowsense/dashboard_uploads.py` — secure upload lifecycle and downloads
- `flowsense/video_compat.py` — browser-compatible H.264 conversion
- `flowsense/tracking/__init__.py`
- `flowsense/tracking/bytetrack_tracker.py`
- `flowsense/tracking/identity_consolidator.py`
- `flowsense/tracking/motion_prediction.py`
- `flowsense/tracking/learned_motion.py` — dependency-free hybrid corrector
- `flowsense/models/learned_prediction_corrector.json` — portable Ridge model
- `flowsense/tracking/render.py`
- `flowsense/tracking/schemas.py`

### Configuration and documentation

- `requirements.txt`
- `.streamlit/config.toml` â€” local dashboard upload-size configuration
- `.gitignore`
- `README.md`
- `README_automatic_counter.md`
- `PREDICTION_EVALUATION.md`
- `LEARNED_PREDICTION.md`
- `automatic_counter_config.example.json`
- `FILE_MANIFEST.md`
- `LICENSE`

## Committed result snapshot

The `easy_results/` directory is a reproducible snapshot used by the dashboard
and presentation:

- `easy_results/FlowSense_report.html`
- `easy_results/READ_ME_FIRST.csv`
- `easy_results/automatic_counts.csv`
- `easy_results/automatic_counts_detailed.csv`
- `easy_results/comparison_by_video.csv`
- `easy_results/comparison_by_video_class.csv`
- `easy_results/object_movement_audit.csv`
- `easy_results/prediction_accuracy.csv`
- `easy_results/learned_prediction_results.csv`
- `easy_results/summary.json`
- `easy_results/traffic_volume_intervals.csv`
- `easy_results/video_file_map.csv`

All committed result paths are repository-relative and contain no personal
computer directories.

## Reproducibility inputs

### Detection and canonical tracking data

- `tracking_data/tracking_data.csv`
- `tracking_data/tracking_data2.csv`
- `tracking_data/tracking_data3.csv`
- `tracking_data/tracking_data4.csv`
- `tracking_data/member2_canonical_tracks.csv`
- `tracking_data/member2_canonical_tracks2.csv`
- `tracking_data/member2_canonical_tracks3.csv`
- `tracking_data/member2_canonical_tracks4.csv`
- `tracking_data/README.md`

### Manual observations and templates

- `kelvin_vehicle_counts.csv` — retained manual count reference
- `kelvin_comparison_template.csv`

### Videos

- `videos/source_traffic.mp4`
- `videos/flowsense_tracking.mp4`
- `videos/flowsense_tracking2.mp4`
- `videos/flowsense_tracking3.mp4`
- `videos/flowsense_tracking4*.mp4`
- `videos/member2_bytetrack_overlay.mp4`
- `videos/flowsense_hybrid_video_1.mp4`
- `videos/flowsense_hybrid_video_2.mp4`
- `videos/flowsense_hybrid_video_3.mp4`
- `videos/flowsense_hybrid_video_4.mp4`
- `videos/README.md`

### Legacy reproduction tools

- `track_member1_video.py` — CSV-based Member 2 workflow
- `flowsense_yolo_tracking.ipynb` — original Member 1 notebook

## TEST-ONLY files

These remain in GitHub for regression testing but are not required in a final
runtime-only submission:

- `demo_day1.py`
- `flowsense/tracking/verification.py`
- `test_automatic_counter.py`
- `tests/__init__.py`
- `tests/README.md`
- `tests/fixtures/member1_sample.csv`
- `tests/test_csv_detections.py`
- `tests/test_dataset_configs.py`
- `tests/test_day1_tracking.py`
- `tests/test_dashboard_smoke.py`
- `tests/test_dashboard_uploads.py`
- `tests/test_end_to_end_pipeline.py`
- `tests/test_identity_consolidator.py`
- `tests/test_motion_prediction.py`
- `tests/test_learned_motion.py`
- `tests/test_official_hybrid_generator.py`
- `tests/test_official_dashboard_artifacts.py`
- `tests/test_prediction_evaluation.py`
- `tests/test_video_compat.py`
- `tests/test_yolo_detector.py`
- `.github/workflows/tests.yml` — automated test workflow

## Generated locally — never upload

- `.venv/`, `.venv-old/`, or `.deps/`
- `__pycache__/`, `*.pyc`, `.pytest_cache/`, or `.coverage`
- `.vscode/`
- `fromMember1/`
- `outputs/` previews and diagnostic CSVs
- `results/` generated final videos and reports
- downloaded `*.pt` model weights
