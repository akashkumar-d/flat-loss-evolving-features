# Campaign N: deep networks across p and training fraction, on fresh seeds (2026-10-12)

claude/1hl-figures, at Akash's request ("clean up the deep network results for across p and training split ... so that we can
add them to the appendix"). Depths 3 and 4 at p = 47 and 61, training fractions 0.8, 0.7, 0.6 and 0.5, the linear head decay
lam_v = 0.2 x 31 / p, the last hidden layer's rate r fixed per (depth, fraction) from campaign M's development runs, and 10
fresh seeds per setting (init 9000 + i, split 9500 + i). `PLAN.md` has the design, the selection of r and the analysis, all fixed
before the runs.

Files for the workers (registered in `HASHES.json`; `push_N.sh` in `natural_plateau_shared/` checks and pushes them):
`bias_runner.py`, `endpoints_i.py`, `acrossp_figs.py` and `worker.py` (campaign M's, unchanged), `manifest.json` (written by
`make_manifest.py`), `campaign.json`, `WORKER.md`, `tests/test_n.py` (5 tests; the manifest test checks every run against campaign
M's run of the same cell). 160 runs; sessions N1-N16.

Not for the workers: `analyze_N.py` (campaign M's scorer, `analyze_M.py` here byte for byte, plus per-cell counts, Wilson
intervals and descriptive measures), `plot_N.py` (the paper's renderer for the per-(depth, p) ladders, a scoreboard and a per-run
figure), `monitor/nmon.js`.

Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/`. Then:

    python3 -I analyze_N.py collected/results collected/status analysis/analysis_N.json ../campaign_M/analysis/analysis_M.json
    python3 -I plot_N.py collected/results analysis/analysis_N.json analysis/figures
