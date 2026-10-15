# Campaign T: transformers on modular addition (development stage)

claude/1hl-figures, 2026-10-07, at Akash's request (arXiv version of the grokking paper). `PLAN.md` fixes the design.

**Question.** Does the paper's coexistence (training cross-entropy relatively flat at a positive level while hidden features
change and held-out accuracy rises) appear in transformers trained with the paper's update rule, and how does AdamW compare?

**This stage** runs 80 development runs at p = 113 (Nanda et al.'s one-layer transformer and a two-layer version): the paper's
rule (supported-Polar hidden steps with decay, normalized unembedding with decay) over a grid of hidden decay, hidden step and
head decay, and AdamW (lr 1e-3, weight decay 1) as the baseline; two development seeds each. The rule in `PLAN.md` then picks one
setting per depth and locks the evaluation window for a confirmation stage on fresh seeds.

| file | what |
|---|---|
| `PLAN.md` | design, qualification and selection rules, window rule (fixed before any run) |
| `manifest.json`, `make_manifest.py` | the 80 runs and their configurations; the session shares (T1-T12) |
| `tf_runner.py` | the transformer, both optimizers, the observers (A_H, train-only refit, margins), checkpoint and resume |
| `tests/test_tf.py` | finite-difference gradients, the last-position shortcut, the Polar and head steps, exact resume, A_H |
| `worker.py`, `campaign.json`, `WORKER.md` | the session worker and its instructions |
| `analyze_T.py` | per-run fields, qualification, selection and the locked window |
| `HASHES.json` | registered SHA-256 of the files above (checked by the selftest and the push script) |

Results arrive on the session branches under `results/<run>/` (run.json.gz with every update's loss and the logged
observations, chain.json with the run's metadata). `../collect_results.sh` copies them to `collected/` on the Mac; then
`python3 analyze_T.py collected/results analysis` writes `analysis/SUMMARY.md` and `analysis/selection.json`.
