# Campaign K: deeper networks at other primes (development stage)

claude/1hl-figures, 2026-10-07, at Akash's request (arXiv version of the grokking paper). `PLAN.md` fixes the design.

**Question.** At p = 23, 47, 61 and 101, can depth-3 and depth-4 networks meet the paper's per-run joint endpoint, as they do at
p = 31? With the paper's settings they pass in 2 of 80 runs: at p >= 47 the training fit relapses inside the band because the
head-norm cap 1/lam_v = 5 is shared by p classes, and at p = 23 the held-out rise is too slow for the loss-only band.

**This stage** runs 96 development runs: per cell (p, depth), two head decays scaled with p (lam_v = 0.2 (31/p)^(1/2) or
0.2 (31/p)) x two first-layer decays (0.015, 0.025) x three development seeds. Everything else is campaign F's configuration
of the cell. The selection rule in `PLAN.md` then picks one setting per cell for a confirmation stage on fresh seeds.

| file | what |
|---|---|
| `PLAN.md` | design, endpoint, selection rule (fixed before any run) |
| `manifest.json`, `make_manifest.py` | the 96 runs and their configurations; the session shares (K1-K20) |
| `bias_runner.py` | campaign I's runner, unchanged (used without its `bias` key; reproduces campaigns C, D and F bit for bit) |
| `worker.py`, `campaign.json`, `WORKER.md` | the session worker and its instructions |
| `tests/test_k.py` | gradients, Polar and head steps, resume, and that every run differs from campaign F only as planned |
| `endpoints_i.py`, `analyze_K.py` | campaign C's per-run endpoint; scoring and the selection rule |
| `HASHES.json` | registered SHA-256 of the files above (checked by the selftest and the push script) |

Results arrive on the session branches under `results/<run>/` (run.json.gz with every update's loss and the logged
observations, chain.json with the run's metadata). `../collect_results.sh` copies them to `collected/` on the Mac; then
`python3 analyze_K.py collected/results analysis` writes `analysis/SUMMARY.md` and `analysis/selection.json`.
