# Campaign N: deep networks across p and training fraction, on fresh seeds (claude/1hl-figures, 2026-10-12)

Fixed before any run. Akash, 2026-10-12: "we need to clean up the deep network results for across p and training split ...
clean results so that we can add them to the appendix." Scope he chose: p = 47 and 61, training fractions 0.8, 0.7, 0.6 and
0.5, 10 fresh seeds per setting; depths 3 and 4 with the linear head-decay rule.

## 1. Why

- The paper's deep-network results pass at p = 31 (fraction 0.8). With the paper's settings, other primes mostly fail the
  per-run endpoint (campaign F: 2/10 at depth 3, p = 47, 0/10 elsewhere), and halving the data (campaign L, fraction 0.5)
  fails the endpoint in all 120 runs: the held-out rise comes 4-6 times later and lands on the CE's drift after its peak.
- Campaign M (development, seeds 1-3) found what moves it: the last hidden layer's rate r must shrink with the training
  fraction (it sets when the post-knee CE peak comes), and the linear head decay lam_v = 0.2 x 31 / p is needed at 0.5.
  pass_M held in 135 of its 216 runs; with the linear rule, one r per (depth, fraction) passes at both primes.
- Campaign N reruns exactly those settings on 10 fresh seeds, so the appendix can report pass counts and seed medians that
  were not used to choose anything.

## 2. Settings

Every run is campaign M's run of the same (depth, p, fraction, r) with the linear rule, with only the seeds changed: campaign
K's configuration of the cell (width 256, no biases, init scales 24 and 8, eta_w = eta_v = 0.03, deep decay 0.07, eta_layers
[0.03, u, ..., r u] with u = 0.03 x 0.015 / 0.07, lam_w 0.015, full-batch CE, supported-Polar hidden steps, normalized head
step), lam_v = 0.2 x 31 / p (0.1319 at p = 47, 0.1016 at p = 61), horizons 100k / 150k / 200k / 250k updates at fractions 0.8 /
0.7 / 0.6 / 0.5, and early stop as in K, L and M (held-out accuracy >= 0.99 kept for max(0.5 t, 10000) more updates after
first reaching it at t).

**r per (depth, fraction).** One rule, applied to campaign M's linear-rule runs pooled over p = 47 and 61 (6 runs per
candidate): the most pass_M, then the most pass_M3, then the higher median q_te, then the earlier median held-out 0.9 crossing.

| depth | fraction 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|
| 3 | **0.45** (6/6) | **0.30** (6/6, q_te 1.00); 0.45 (6/6, 0.97) | **0.18** (6/6, held-out 0.9 at 31k); 0.28 (6/6, 41k); 0.45 (0/6) | **0.08** (6/6); 0.13 (5/6); 0.20 (5/6) |
| 4 | **0.45** (6/6) | **0.28** (6/6); 0.45 (0/6) | **0.13** (6/6, 44k); 0.20 (6/6, 55k); 0.45 (0/6) | **0.06** (5/6, q_te 1.00); 0.09 (5/6, 1.00, later); 0.14 (4/6) |

**Seeds.** init 9000 + i, split 9500 + i, i = 1-10: the fresh seeds campaign K's plan reserved for a confirmation and never
used (campaigns D, F, G used 5000 + i / 6000 + i; K, L, M 7000 + i / 8000 + i). With one split seed the training sets are
nested (0.5 within 0.6 within 0.7 within 0.8), so each seed is a paired ladder across the fractions.

2 depths x 2 primes x 4 fractions x 10 seeds = **160 runs**.

## 3. Code

- `bias_runner.py`, `endpoints_i.py`, `acrossp_figs.py`, `worker.py`: campaign M's files, unchanged (same SHA-256).
  `campaign.json`: M's with the name N.
- `tests/test_n.py`: campaign M's tests unchanged (gradients, the Polar and head steps, resume, nested splits), with the
  manifest test renewed: every run is campaign K's base configuration with only the planned keys changed, r as in the table,
  seeds as above, 16 cells x 10 seeds, and, where campaign M's manifest is present, M's run of the same cell with only
  init_seed and split_seed different.
- `make_manifest.py` writes `manifest.json` and `manifest_test.json` (a miniature for the end-to-end check).

## 4. Sessions and compute

16 sessions N1-N16 of 4 cores, 10 runs each. By campaign M's measured speeds (0.05-0.11 s per update with four runs per
session) and its run lengths at these settings: about 250 core-hours, about 4.3 hours per session; the longest runs (depth
4, p = 47, fraction 0.5) take about 2-5 hours. Fewer sessions also work (the others take over the runs of IDs that never
start); they take longer.

## 5. Analysis (`analyze_N.py`, fixed now)

Each run is scored by campaign M's scorer, unchanged (`analyze_M.py`, a byte-identical copy):
- **pass_M**: the refit criterion (refit +0.2 inside the paper's post-hoc window W; held-out +0.2 from the start of W to the end
  of the run; final held-out >= 0.99), and at least 75% of the held-out and of the refit 10% -> 90% rises inside W, and
  Delta A_H of the last hidden layer >= 0.02 across W.
- **pass_M3**: pass_M with training accuracy >= 0.97 at every logged step of W.
- Campaign C's per-run endpoint is reported, not deciding. Held-out accuracy is on all pairs outside the training set.

Per cell (depth, p, fraction): pass_M and pass_M3 out of 10 with Wilson 95% intervals. **A cell works if pass_M holds in at
least 8 of 10 seeds.** Descriptive: q_te, q_rf, Delta A_H, the most training pairs misclassified in W, the CE change from the
refit's 0.2 crossing to the held-out 0.9 crossing (and in how many runs it rose), the training CE / log p at the held-out 0.5
crossing, the CE drawdown below its running maximum at held-out 0.9, held-out 0.5 and 0.9 crossings, refit 0.5 crossing and the refit's lead, final held-out accuracy, run length; and
campaign M's development counts for the same cells. Dry run on M's runs of these cells (mapped onto N's IDs): the scorer
reproduces M's counts.

## 6. Figures (`plot_N.py`, fixed now)

- `ladder/N_ladder_D{3,4}_p{47,61}`: the paper's renderer (`acrossp_figs.py`, unchanged; the paper's colour key), one column
  per fraction with its r; every seed thin, the median bold; rows: training CE / log p (cycle mean), accuracies (held-out,
  train-only ridge refit, training accuracy), misclassified training pairs, A_H of the last and first hidden layers; grey:
  the median post-hoc window. These are the appendix candidates.
- `N_scoreboard`: pass_M per (depth, p) and fraction, with pass_M3, M's development count, median q_te and median CE change.
- `N_runs`: per run against the fraction, the share of the held-out rise inside W and the CE change across the rise.

## 7. What to expect, and how to read it

- From campaign M's 3 seeds: every cell passed 3/3 except depth 4, p = 47 at fraction 0.5 (2/3; seed 2 ended at held-out
  0.989). That cell is the one most likely to fall short of 8/10.
- The CE from the refit's 0.2 crossing to the held-out 0.9 crossing (M, these settings): -4% to +14% at 0.8 (about flat; the
  post-knee peak comes during the rise), +8% to +47% at 0.7, +56% to +300% at 0.6, +142% to +1543% at 0.5.
- The regime differs with the fraction. The training CE / log p at the held-out 0.5 crossing (M, these settings) is 0.52-0.69
  at 0.8, 0.35-0.57 at 0.7, 0.11-0.27 at 0.6 and 0.005-0.15 at 0.5 (depth 3: 0.005-0.022). At the small fractions the small r
  keeps the last layer large, so the CE first falls to about 0 (its dip is below 0.002 log p) and then climbs slowly as weight
  decay shrinks that layer; the held-out rise happens during that climb, so the large relative changes are small in absolute
  terms. The appendix text should state the CE levels, not only the relative change.
- If a cell does not reach 8/10, the appendix reports its count as it is; no setting is changed after the runs.

## 8. Checks before launch

- tests/test_n.py passes (5 tests; the manifest test against campaign M's manifest).
- The worker's selftest passes and an end-to-end run of manifest_test.json (8 runs, miniature) finishes through two workers
  and a local bare repository.
- analyze_N.py and plot_N.py run on campaign M's runs of these cells mapped onto N's IDs.
