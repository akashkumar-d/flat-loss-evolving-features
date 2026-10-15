# Campaign V: confirmation of the two cleanest flat-loss settings, and the RMS head at training fraction 0.2 (2026-10-09)

claude/1hl-figures, at Akash's request after campaign U. `PLAN.md` has the design and the confirmation rule (fixed before the
runs); `confirm_V.py` implements the rule (its docstring is the definition).

Files for the workers (registered in `HASHES.json`; `push_V.sh` in `natural_plateau_shared/` checks and pushes them):
`tf_runner.py` (campaign U's 1.2, unchanged), `worker.py` (unchanged), `manifest.json` (written by `make_manifest.py`),
`campaign.json`, `WORKER.md`, `tests/test_tf.py` (12 tests).

Analysis (run where the results are): `confirm_V.py` (the confirmation rule) and `flat_windows.py` (campaign U's window
analysis). Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/`.
