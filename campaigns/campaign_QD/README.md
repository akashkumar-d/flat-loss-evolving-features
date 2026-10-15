# Campaign QD: depth 2 on the 40 evaluation seeds of depths 3-5 (2026-10-13)

claude/1hl-figures, at Akash's request: the paper's depth figure should show the same number of runs in every column. QD runs
the paper's depth-2 setting (p = 31, width 256, r = 0.5, 40k updates) on the 40 seeds of the depth-3-5 cohorts, bias-free
(D2) and with a trained hidden bias (B_D2), 80 runs. `PLAN.md` has the design and the analysis.

Files for the workers (registered in `HASHES.json`; `push_Q.sh` in `natural_plateau_shared/` checks and pushes them):
`bias_runner.py` and `tests/test_bias.py` (campaign I's, unchanged), `worker.py` (worker 2.0, unchanged), `manifest.json`
(written by `make_manifest.py`), `campaign.json`, `WORKER.md`, `tests/test_qd.py` (9 tests). Sessions QD1-QD4.

Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/`.
