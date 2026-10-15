# Campaign L: campaign K renewed at training fraction 0.5 (plan fixed before any run)

Written by claude/1hl-figures on 2026-10-09, at Akash's request: campaign K (and the paper's deep runs) trained on 80% of the
p^2 pairs; "use at most 50% and see how things turn out", with the same plots. Development runs only; not evidence for the
paper's claims.

## What changes from campaign K

| | campaign K | campaign L |
|---|---|---|
| training fraction | 0.8 | **0.5** |
| horizon (updates) | 120k at p = 23, 100k otherwise | **300k at p = 23, 250k otherwise** |
| settings per cell | K's four | **the paper's (lam_v 0.2, lam_w 0.015) + K's four** |
| seeds | 1-3 (init 7000 + i, split 8000 + i) | **the same** |

Everything else is campaign K's configuration of each cell: depths 3 and 4, p = 23, 47, 61, 101; width 256; no biases;
init scales 24 (first layer) and 8 (deeper layers); eta_w = eta_v = 0.03, eta_layers [0.03, u, ..., r u] with
u = 0.03 x 0.015 / 0.07 and r = 0.45 (0.32 at depth 4, p = 23); lam_layers [lam_w, 0.07, ...]; full-batch CE with the
supported-Polar hidden steps and the normalized head step; logs every 20 updates, heavy logs every 500 (1000 at p = 101);
early stop once the held-out accuracy has stayed >= 0.99 for max(0.5 t, 10000) updates after first reaching it at t.
Runner: campaign K's bias_runner.py, unchanged.

Head decays (lam_v): paper 0.2; square-root rule 0.2 (31/p)^(1/2); linear rule 0.2 (31/p). First-layer decay lam_w:
0.015 or 0.025 (the paper setting uses 0.015). That is 8 cells x 5 settings x 3 seeds = 120 runs.

Data split: with the same split seed, the training set is the first half of the same random permutation of the p^2 pairs
whose first 80% was campaign K's training set; so each run's training set is a subset of campaign K's (same seed), and its
held-out set contains campaign K's held-out pairs plus 30% more. Training pairs: 264 (p = 23), 1104 (47), 1860 (61),
5100 (101).

## Analysis (fixed now; the same as campaign K's, so the figures are comparable)

1. Campaign C's per-run endpoint (`endpoints_i.py`, unchanged): the loss-only peak band holds at least 75% of the held-out
   10% -> 90% rise; final held-out accuracy >= 0.99; the last hidden layer's A_H rises by >= 0.02 across the band; every logged
   training margin in the band is positive. Reported as in campaign K's `analysis/SUMMARY.md` (passes, failing checks, share,
   CE peak / log p, first persistent fit, relapses in the band, t10, t90, final accuracy, Delta A_H), with campaign K's
   scoreboard and per-depth figures (`L_scoreboard`, `L_depth{3,4}_<setting>`), drawn with the paper's renderer
   (`acrossp_figs.py`, unchanged); the paper setting now at fraction 0.5 from this campaign's own runs.
2. Akash's refit criterion, as in `campaign_K/plot_K_refit.py`: W = the paper's post-hoc window; pass = refit +0.2 inside W,
   held-out +0.2 from the start of W to the end, final held-out >= 0.99; pass3 = pass with training accuracy >= 0.97 throughout
   W. Figures `L_refit_scoreboard`, `L_refit_depth{3,4}_<setting>`.
3. Paired with campaign K (same seeds, nested training sets): per run, t10 and t90 of the held-out rise, the CE plateau level,
   the refit's lead over the held-out accuracy, and both verdicts, at fractions 0.8 and 0.5.

A run that does not reach held-out accuracy 0.99 by its horizon fails (1) and (2) on the final accuracy; it is reported with
its final held-out and refit accuracy.

## Pilot (this container; one seed; not part of the results)

Depth 3, p = 47, square-root rule, lam_w 0.015, seed 1, fraction 0.5: the training set is fit by update 5000 (CE at its dip,
0.002 log p), the CE climbs back to 0.33 log p by update 10000, and the held-out accuracy is still 0.00 at update 15000 (at
fraction 0.8 this cell's held-out rise starts near update 13000). The horizons above allow for a rise two to three times
later than at fraction 0.8.

## Compute

120 runs, about 550 core-hours by campaign K's measured speeds (p = 101 runs use 4 threads and take about 60% of it), on
20 sessions L1-L20 of 4 cores: about 7-8 hours per session by this estimate (campaign K's estimate was 6-7 hours and its
sessions finished sooner).
