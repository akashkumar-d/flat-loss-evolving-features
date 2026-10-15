# Campaign N (deep MLPs across p and training fraction, fresh seeds): results, all 160 runs (claude/1hl-figures, 2026-10-12)

**State.** 160 of 160 runs finished: 15 sessions report `campaign_done`, no failed runs, no push errors, no run on two
branches; N12 never started and its 10 runs were taken over by the others. All sessions ran numpy 2.5.3 and Python 3.13.16
on the same CPU model; selftests 10:40-10:42Z, last heartbeat 17:28Z. Akash ran `bash collect_results.sh`; the collected
files were read from the Mac and every run.json.gz matches the SHA-256 in `collected/INDEX.json` (160/160). The analysis and
figures are the ones fixed in PLAN.md sections 5 and 6 before any run (`analyze_N.py`, `plot_N.py`; `analysis_N.json`,
`analyze_N.log`, `figures/`), scoring every run with campaign M's scorer unchanged. Seeds: initialization 9000+i, split
9500+i (i = 1-10), disjoint from campaign M's development seeds. Development runs only until the paper cites them.

pass_M = Akash's refit criterion (refit +0.2 inside the paper's post-hoc window W, held-out +0.2 from the start of W to the
end of the run, final held-out accuracy >= 0.99), at least 75% of the held-out and of the refit 10% -> 90% rises inside W, and
Delta A_H of the last hidden layer >= 0.02 across W. pass_M3 adds training accuracy >= 0.97 at every logged step of W. A cell
works if pass_M holds in at least 8 of 10 seeds. Held-out accuracy is on all pairs outside the training set.

## 1. Verdicts (`figures/N_scoreboard.png`)

**All 16 cells work. pass_M 155/160 (Wilson 95% [0.929, 0.987]); pass_M3 = pass_M in every cell.** Depth 3: 78/80, depth 4:
77/80.

| | fraction 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|
| depth 3, p = 47 | 10/10 (r 0.45) | 10/10 (0.30) | 10/10 (0.18) | 8/10 (0.08) |
| depth 3, p = 61 | 10/10 | 10/10 | 10/10 | 10/10 |
| depth 4, p = 47 | 9/10 (r 0.45) | 10/10 (0.28) | 10/10 (0.13) | 9/10 (0.06) |
| depth 4, p = 61 | 9/10 | 10/10 | 10/10 | 10/10 |

r is the last hidden layer's rate factor (same for both p); lam_v = 0.2 x 31/p (linear rule). Campaign M's development seeds
(1-3) gave 3/3 in 15 of the 16 cells and 2/3 at depth 4, p = 47, fraction 0.5 (9/10 here).

The five fails:
- three at fraction 0.5, p = 47 (depth 3 seeds 1 and 3, depth 4 seed 8): the whole held-out rise is inside W (share 1.0),
  but the run reaches the 250k-update budget with final held-out accuracy 0.971, 0.959 and 0.988 (the refit criterion needs
  0.99); the training CE peaks late (184k-250k updates) and W runs to the end;
- two at fraction 0.8, depth 4 (p = 47 seed 2, p = 61 seed 7): the training CE peaks and turns down during the held-out rise,
  W closes early and holds 0.749 and 0.736 of the rise (threshold 0.75); both reach held-out 1.0.

## 2. The loss during the held-out rise (`figures/N_runs.png`, `figures/ladder_allseeds/`)

Training CE change from the refit's 0.2 crossing to the held-out 0.9 crossing (cell median, runs in which it rose), and the
training CE / log p at the held-out 0.5 crossing (cell median):

| | fraction 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|
| depth 3, p = 47 | +10% (10/10); 0.52 | +49% (10/10); 0.36 | +233% (10/10); 0.15 | +1117% (10/10); 0.007 |
| depth 3, p = 61 | +1% (8/10); 0.58 | +13% (10/10); 0.43 | +51% (10/10); 0.21 | +136% (10/10); 0.014 |
| depth 4, p = 47 | -0% (5/10); 0.63 | +29% (10/10); 0.52 | +303% (10/10); 0.17 | +1312% (10/10); 0.044 |
| depth 4, p = 61 | -3% (0/10); 0.67 | +11% (10/10); 0.57 | +69% (10/10); 0.23 | +238% (10/10); 0.045 |

- Fraction 0.8 is the plateau case: after the fit the training CE climbs to 0.52-0.67 log p and stays there (changes of -3% to
  +10% across the rise; drawdown below its running maximum at held-out 0.9 at most 0.06), while held-out accuracy goes from 0
  to 1 and the last hidden layer's A_H gains 0.38-0.53 (cell medians). Training accuracy stays at or near 1: at p = 47 at
  most 2 training pairs flip inside W; at p = 61, 3-18 of the 2,976 training pairs (depth 3) and 4-14 (depth 4).
- At fractions 0.7-0.5 the rise comes earlier relative to the CE's climb: at 0.5 the training CE is still near zero (0.007-0.045
  log p) when held-out accuracy crosses 0.5 and rises through the whole rise (the large percentages are relative to that small
  level).
- In every cell the refit leads the network's held-out accuracy: refit 0.5 crossing 1.2k-9.5k updates before the held-out 0.5
  crossing (cell medians; shortest at p = 61).
- The Fourier structure is in the last hidden layer (A_H 0.5-0.78 at the end); the first hidden layer stays near 0.1.

## 3. Timing (cell medians, thousands of updates: held-out 0.5 / 0.9 crossing)

| | fraction 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|
| depth 3, p = 47 | 16.9 / 19.5 | 20.2 / 24.2 | 26.8 / 34.1 | 38.9 / 48.1 |
| depth 3, p = 61 | 15.5 / 17.6 | 19.0 / 22.0 | 24.7 / 28.3 | 35.5 / 40.7 |
| depth 4, p = 47 | 21.8 / 26.4 | 29.1 / 36.4 | 38.4 / 47.5 | 69.7 / 88.3 |
| depth 4, p = 61 | 20.0 / 23.3 | 26.3 / 31.0 | 35.4 / 41.0 | 59.0 / 67.8 |

Runs stop once held-out accuracy has generalized (median length 30k-156k updates by cell) or at the budget (100k, 150k, 200k
and 250k updates at fractions 0.8, 0.7, 0.6 and 0.5).

## 4. Figures

- `figures/N_scoreboard`: pass_M per cell with pass_M3, campaign M's development count, median share and median CE change.
- `figures/N_runs`: per run against the fraction, the share of the held-out rise inside W and the CE change across the rise.
- `figures/ladder/N_ladder_D{3,4}_p{47,61}`: the paper's renderer (acrossp_figs.py, unchanged), as fixed in PLAN.md: each
  column runs to about 1.15 x the later of the median window end and the 90th percentile of the held-out 0.9 crossings, so in
  some columns the runs that stopped early enter the median with their last value (the flat CE segments at the right edge).
- `figures/ladder_allseeds/` (`plot_N_cut.py`, made after the runs; presentation only): the same ladders with each column's
  x-axis ending at its shortest run, so every seed contributes at every plotted point. In every cell the shortest run ends
  after the last seed's held-out 0.9 crossing, so no part of any rise is cut.
- `paper/N_ladder_D{3,4}_p{47,61}_paper` (`paper_N.py`): the paper versions of the all-seeds ladders: column titles with the
  fraction, r and the seeds that pass, the window in the legend. Draft caption in `paper/CAPTIONS.md`.
