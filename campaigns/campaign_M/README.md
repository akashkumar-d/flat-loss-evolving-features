# Campaign M: training-fraction ladder with the last hidden layer's rate re-tuned (development stage)

claude/1hl-figures, 2026-10-11, at Akash's request after campaign L: at training fraction 0.5 the refit and held-out rise came
after the post-knee CE peak. This campaign runs depths 3 and 4 at p = 47 and 61 on nested training sets of 80%, 70%, 60% and 50%
of the pairs, with the last hidden layer's rate r re-tuned per fraction, for the paper's head decay and the linear rule.
`PLAN.md` fixes the design and the analysis.

| file | what |
|---|---|
| `PLAN.md` | design and analysis (fixed before any run) |
| `manifest.json`, `make_manifest.py` | the 216 runs and their configurations; the session shares (M1-M24) |
| `bias_runner.py` | campaigns K's and L's runner, unchanged |
| `worker.py`, `campaign.json`, `WORKER.md` | the session worker (campaigns K's and L's, unchanged) and its instructions |
| `tests/test_m.py` | gradients, Polar and head steps (with a small last-layer rate), resume, nested splits at the four fractions, and that every run is campaign K's configuration of its cell with only the planned changes |
| `endpoints_i.py`, `acrossp_figs.py` | campaign C's per-run endpoint and the paper's renderer (unchanged), for the analysis |
| `HASHES.json` | registered SHA-256 of the files above (checked by the selftest and the push script) |

Results arrive on the session branches under `results/<run>/`; `../collect_results.sh` copies them to `collected/` on the Mac.
Analysis (on the Mac or here): `analyze_M.py`, then `plot_M.py` (PLAN.md section 4).

`design/` (not pushed): the timing extraction (`timing.py`, `timing_LK.json` for campaigns K and L, `timing_D.json` for campaign D)
and the diagnosis figure (`diag_fig.py`, `figs/diag_r_vs_fraction.png`) behind PLAN.md sections 1 and 2; paths as in the
analysis container.
