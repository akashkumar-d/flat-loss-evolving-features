# Campaign W: transformer settings under the paper's own definitions (2026-10-10)

claude/1hl-figures, at Akash's request after campaign V. `PLAN.md` has the design and the analysis (fixed before the runs);
`paperdefs_tf.py` implements the paper's definitions for the transformer record.

Files for the workers (registered in `HASHES.json`; `push_W.sh` in `natural_plateau_shared/` checks and pushes them):
`tf_runner.py` (1.3: campaign V's 1.2 plus the paper's NGD contrast, `hidden_mode`, and the no-mirror held-out accuracies;
Polar updates and 1.2's observers bit-identical to 1.2), `worker.py` (unchanged), `manifest.json` (written by
`make_manifest.py`), `campaign.json`, `WORKER.md`, `tests/test_tf.py` (16 tests).

Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/`.
