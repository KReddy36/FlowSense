# TEST-ONLY files

Everything under `tests/` exists only to verify the Member 2 implementation
during development. These files should remain in GitHub while the team is
developing and merging, but they are not required in the final deployed
application or presentation bundle.

The suite also verifies that datasets 1–4 remain paired with the correct
CSV and video filenames.

Run all tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

`test_motion_prediction.py` verifies bounded rolling histories, velocity
smoothing, prediction through missed detections, and inactive-track deletion.
Like the rest of this directory, it is test-only.
