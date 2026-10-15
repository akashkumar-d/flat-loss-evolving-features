# Campaign U: transformer exploration of the flat-loss level and the averaging window (2026-10-08)

claude/1hl-figures, at Akash's request after campaign T. `PLAN.md` has the reasons, the pilot, the arms and the analysis.

Files for the workers (registered in `HASHES.json`; `push_U.sh` in `natural_plateau_shared/` checks them and pushes them):
- `tf_runner.py` 1.2: campaign T's runner (1.1) plus an optional final RMS division before W_U, the training CE recorded in
  float64, and a train-only logistic probe; on campaign-T configurations its observations are byte-identical to 1.1's.
- `manifest.json` (written by `make_manifest.py`, which reads `../campaign_T/manifest.json` for the five crashed runs),
  `campaign.json`, `worker.py` (campaign T's worker, unchanged), `WORKER.md`, `tests/test_tf.py` (12 tests).

Analysis (run where the results are; not needed by the workers): `flat_windows.py` (the key message at each averaging window,
and the flatness profile), `plot_flat_scales.py` (dynamics with several averaging windows).

This campaign includes the five campaign-T runs that crashed (the `campaign_T_rerun` pack is no longer needed); their results
keep their campaign-T ids and are collected into `campaign_U/collected/`.
