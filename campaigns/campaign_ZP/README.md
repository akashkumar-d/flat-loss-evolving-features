# Campaign ZP: other p with the training fraction set per p (2026-10-12)

claude/1hl-figures, at Akash's request after campaign Z's interim results: Z's other-p cells (p = 61, 89, 149) passed in at most
1 of 5 seeds (p = 61 stalled near 0.98 held-out, p = 89 rose too slowly, p = 149 rose before the CE dip). ZP reruns them with
the training fraction set per p and nothing else changed: p = 61 at frac 0.33 and 0.35, p = 89 at 0.25 and 0.275, p = 149 at
0.15 and 0.16; seeds 11-15. `PLAN.md` has the timing rule behind the fractions and the analysis (fixed before the runs).

Files for the workers (registered in `HASHES.json`; `push_ZP.sh` in `natural_plateau_shared/` checks and pushes them):
`tf_runner.py` and `worker.py` (campaign Z's, unchanged), `manifest.json` (written by `make_manifest.py`), `campaign.json`,
`WORKER.md`, `tests/test_tf.py` (31 tests; the manifest test checks every run against Z's run for the same p, lam_v and seed).
30 runs; sessions ZP1-ZP8.

Not for the workers: `analyze_ZP.py`, `plot_ZP.py` (campaign Z's analysis and figures, unchanged, plus the timing check and
the appendix choice; they read Z's and Y's runs as the comparison cells).

Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/`.
