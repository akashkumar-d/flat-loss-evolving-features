# Campaign QH: Theorem 4.1's configuration, eta_V = 3e-4, seeds 7-10 (2026-10-13)

claude/1hl-figures, at Akash's request: the paper's figure of Theorem 4.1's configuration should show the same number of runs
in both rows. QH adds seeds 7-10 to campaign H's eta_V = 3e-4 arm (4 runs, session QH1). `PLAN.md` has the design.

Files for the worker (registered in `HASHES.json`; `push_Q.sh` in `natural_plateau_shared/` checks and pushes them):
`t41_runner.py` and `tests/test_t41.py` (campaign H's, unchanged), `worker.py` (worker 2.0, unchanged), `manifest.json`
(written by `make_manifest.py`), `campaign.json`, `WORKER.md`, `tests/test_qh.py` (7 tests).
