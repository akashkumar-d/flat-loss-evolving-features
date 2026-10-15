# Campaign Z: optimizer contrasts with the gain, the one-block recipe, other p, two blocks (2026-10-12)

claude/1hl-figures, at Akash's request after campaign Y: the NGD and AdamW contrasts with the same gain on seeds 11-20, both
held-out definitions (all held-out pairs and the no-mirror subset), a recipe for one block (or a second block), and more p for
the appendix. `PLAN.md` has the design, the pilots and the analysis (fixed before the runs).

Files for the workers (registered in `HASHES.json`; `push_Z.sh` in `natural_plateau_shared/` checks and pushes them):
`tf_runner.py` (1.6: campaign Y's 1.5, except that under AdamW the gain keeps its own sign rule; every other run is 1.5's update
for update), `worker.py` (unchanged), `manifest.json` (written by `make_manifest.py`), `campaign.json`, `WORKER.md`,
`tests/test_tf.py` (31 tests). 143 runs; sessions Z1-Z24 (Z13-Z24 added after the launch as helpers, PLAN.md section 7).

Not for the workers: `analyze_Z.py`, `plot_Z.py` (they read campaign X's and Y's analysis code and their runs as the Polar
cells), `prep/` (the held-out definitions on campaigns X and Y), `pilots/` (PLAN.md section 8).

Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/`.
