# TEST-ONLY files

Everything under `tests/` exists only to verify the Member 2 implementation
during development. These files should remain in GitHub while the team is
developing and merging, but they are not required in the final deployed
application or presentation bundle.

Run all tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```
