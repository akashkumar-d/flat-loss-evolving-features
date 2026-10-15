# Campaign X: Fourier readouts on the transformer settings closest to the paper (2026-10-11)

claude/1hl-figures, at Akash's request after the Fourier-refit replays (`../fourier_refit`). `PLAN.md` has the design and the
analysis (fixed before the runs).

Files for the workers (registered in `HASHES.json`; `push_X.sh` in `natural_plateau_shared/` checks and pushes them):
`tf_runner.py` (1.4: campaign W's 1.3 plus the read-only Fourier readouts, `observers_fourier`; with it off every update and
every record is bit-identical to 1.3), `worker.py` (unchanged), `manifest.json` (written by `make_manifest.py`),
`campaign.json`, `WORKER.md`, `tests/test_tf.py` (22 tests).

Not for the workers: `mlp_side/` (the paper's MLP runs with the same readouts, computed in the analysis session),
`analyze_X.py`, `plot_X.py`.

Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/`.
