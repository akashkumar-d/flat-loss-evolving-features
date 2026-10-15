# Campaign ZM: p = 61 with a mirror-free split (claude/1hl-figures, 2026-10-12)

Does campaign ZP's p = 61 cell fail the all-pairs share only because of mirror pairs? Here the training set holds both orders
of every chosen pair, so no held-out pair has its mirror (b, a) in the training set. Plan, pilot outcome and the rule that
chose the main fraction: `PLAN.md`. Results: `analysis/SUMMARY.md`.

Files
- `tf_runner.py` 1.7: campaign ZP's 1.6 plus `split_mode` ('random' default, unchanged; 'mirror_free').
- `tests/test_zm.py` (6 tests of the new split and of 1.6 equivalence) and `tests/test_tf_16suite.py` (ZP's suite for 1.6, run
  against 1.7 without its manifest test): `python3 -m pytest -q tests/ -k "not manifest"` from this folder (ZP's folder next
  to it is used for the 1.6 comparisons).
- `make_configs.py` (pilot and main configs, from ZP's p = 61 seed-11 configuration), `configs/`, `queue.sh` (runs the
  configs two at a time, one thread each).
- `runs/<id>/run.json.gz` (the runner's records; 2 pilots on development seed 1, 5 main runs on seeds 11-15), `logs/`.
- `analyze_ZM.py` (campaign Z's per-run scoring, unchanged, with the split count taught the mirror-free split; ZP's p = 61
  random-split cells for comparison), `plot_ZM.py` (paper-style figure, version 2: accuracy medians with a slight shadow and a
  strip showing where each accuracy rises; `analysis/figs/ZM_rise_times.json` holds its crossing times; version 1 kept as
  `ZM_mirrorfree_paper_v1`), `analysis/`.
- `readouts_ZM.py` and `plot_readouts_ZM.py` (2026-10-13, post hoc): every logged readout against the network's held-out
  accuracy at p = 61 (both splits) and ZP's p = 89, 113, 149 cells; `analysis/readouts_ZM.json`, `analysis/figs/ZM_readouts`.
  `freqs_ZM.py`: how many frequencies carry held-out signal at the network's 0.5 and 0.9 crossings and where they rank by
  training energy; `analysis/freqs_ZM.json`, `analysis/freqs_ZM.log`.
  Rerun: `python3 -I readouts_ZM.py analysis/readouts_ZM.json`,
  `python3 -I plot_readouts_ZM.py analysis/analysis_ZM.json analysis/readouts_ZM.json analysis/figs`,
  `python3 -I freqs_ZM.py analysis/freqs_ZM.json > analysis/freqs_ZM.log`.

Rerun (from this folder, with campaign_Z, campaign_ZP, campaign_X and campaign_Y next to it):

    python3 make_configs.py pilot; python3 make_configs.py main 0.55
    ./queue.sh ZM_dev_f0.33_s1 ZM_dev_f0.55_s1 ZM_f0.55_s11 ZM_f0.55_s12 ZM_f0.55_s13 ZM_f0.55_s14 ZM_f0.55_s15
    python3 -I analyze_ZM.py analysis/analysis_ZM.json runs ../campaign_ZP/collected/results > analysis/analyze_ZM.log
    python3 -I plot_ZM.py analysis/analysis_ZM.json analysis/figs ZM_f0.55

Environment of these runs: Python 3.11, numpy 2.4.4, torch 2.5.1 (CPU, one thread per run), in Claude's cloud workspace.
