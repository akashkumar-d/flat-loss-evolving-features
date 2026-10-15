# Campaign Y: a gain on the RMS head input, so the training loss does not fall during the held-out rise (2026-10-11)

claude/1hl-figures, at Akash's request after campaign X: "make sure that the loss remains flat or is not decreasing in the window
of interest." `PLAN.md` has the design, the pilots and the analysis (fixed before the runs).

Files for the workers (registered in `HASHES.json`; `push_Y.sh` in `natural_plateau_shared/` checks and pushes them):
`tf_runner.py` (1.5: campaign X's 1.4 plus the optional gain, `gain_init`; without it every update and every record is
bit-identical to 1.4), `worker.py` (unchanged), `manifest.json` (written by `make_manifest.py`), `campaign.json`, `WORKER.md`,
`tests/test_tf.py` (28 tests).

Not for the workers: `analyze_Y.py`, `plot_Y.py` (they read campaign X's `analyze_X.py`, `plot_X.py` and its R5 runs as the
no-gain contrast), `pilots/` (the local pilots of PLAN.md section 6).

Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/`.
