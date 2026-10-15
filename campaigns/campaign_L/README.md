# Campaign L: campaign K renewed at training fraction 0.5 (development stage)

claude/1hl-figures, 2026-10-09, at Akash's request: campaign K trained on 80% of the pairs; this campaign uses 50%, with the
paper's setting added to every cell and the same plots. `PLAN.md` fixes the design and the analysis.

| file | what |
|---|---|
| `PLAN.md` | design and analysis (fixed before any run) |
| `manifest.json`, `make_manifest.py` | the 120 runs and their configurations; the session shares (L1-L20) |
| `bias_runner.py` | campaign K's runner, unchanged |
| `worker.py`, `campaign.json`, `WORKER.md` | the session worker (campaign K's, unchanged) and its instructions |
| `tests/test_l.py` | gradients, Polar and head steps, resume, nested split, and that every run is campaign K's configuration of its cell with only the planned changes |
| `endpoints_i.py`, `acrossp_figs.py` | campaign C's per-run endpoint and the paper's renderer (unchanged), for the analysis |
| `HASHES.json` | registered SHA-256 of the files above (checked by the selftest and the push script) |

Results arrive on the session branches under `results/<run>/`; `../collect_results.sh` copies them to `collected/` on the Mac.
