# FlowSense file manifest

This manifest separates production files from development-only verification
files.

## Production files — include in the final project

### Member 2 source code

- `track_member1_video.py`
- `flowsense/__init__.py`
- `flowsense/csv_detections.py`
- `flowsense/tracking/__init__.py`
- `flowsense/tracking/bytetrack_tracker.py`
- `flowsense/tracking/identity_consolidator.py`
- `flowsense/tracking/render.py`
- `flowsense/tracking/schemas.py`

### Reproducibility and shared artifacts

- `requirements.txt`
- `.gitignore`
- `README.md`
- `FILE_MANIFEST.md`
- `tracking_data/tracking_data.csv` — Member 1 detections
- `tracking_data/member2_canonical_tracks.csv` — consolidated Member 2 results
- `videos/source_traffic.mp4` — original footage needed to reproduce the result
- `videos/member2_bytetrack_overlay.mp4` — preprocessed presentation fallback

## TEST-ONLY files — do not include in the final application bundle

- `demo_day1.py`
- `flowsense/tracking/verification.py`
- `tests/README.md`
- `tests/__init__.py`
- `tests/test_csv_detections.py`
- `tests/test_day1_tracking.py`
- `tests/test_identity_consolidator.py`
- `tests/fixtures/member1_sample.csv`

Keep these test-only files in GitHub until integration and final regression
testing are complete. They can be omitted from a deployment or presentation
submission if only runtime files are requested.

## Never upload

- `.venv/`, `.venv-old/`, or `.deps/`
- `__pycache__/` or `*.pyc`
- `.vscode/`
- `fromMember1/` (its relevant files are stored in repository-native folders)
- `outputs/` previews and local environment checks
