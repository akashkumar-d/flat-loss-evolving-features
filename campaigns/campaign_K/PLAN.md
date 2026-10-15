# Campaign K: deeper networks at other primes, development stage (plan fixed before any run)

Written by claude/1hl-figures on 2026-10-07, at Akash's request, before any run of this campaign. Purpose: settings at which
the paper's deep networks (depths 3 and 4) meet the per-run joint endpoint at p = 23, 47, 61 and 101, as they do at p = 31
(40/40, 39/40 and 38/40 at depths 3 to 5), for the arXiv version. This stage only selects settings; a second stage, with its
own plan fixed before it starts, confirms each selected setting on fresh seeds.

## Why the current settings fail (campaigns F and G, 10 runs per cell)

At the settings of the paper's Figure 9 (width 256, r = 0.45, or 0.32 at depth 4 and p = 23; head decay lam_v = 0.2, first-layer
decay lam_w = 0.015) the joint endpoint holds in 2 of 80 runs. The failures have two causes:

| p | passes (D3, D4) | failed checks | CE peak / log p | first persistent fit | logged fit relapses |
|---|---|---|---|---|---|
| 23 | 0/10, 0/10 | share; also A_H in 4/10 (D3) and 10/10 (D4) | 0.29, 0.33 | 2.5k, 2.9k | 4, 4 |
| 47 | 2/10, 0/10 | fit in 7/10 and 10/10; share also in 3/10 (D3) | 0.68, 0.75 | 20k, 27k | 264, 661 |
| 61 | 0/10, 0/10 | fit | 0.77, 0.83 | 24k, 33k | 899, 1598 |
| 101 (campaign G) | 0/10, 0/10 | fit; final too in 5/10 (D4) | 0.90, 0.92 | never | 2208, 5000 |

(medians; relapses are logged steps after the first fit with a training pair misclassified. p = 97 behaves like p = 101.)

- **p >= 47: the fit.** The normalized head with decay lam_v keeps ||V||_F near 1/lam_v = 5. Shared by p classes, that is a
  per-class scale of about 5/sqrt(p): 0.90 at p = 31 but 0.50 at p = 101. The plateau then sits near chance (0.68-0.92 log p),
  training margins are tiny, and training pairs keep flipping inside the band, so the strict fit check fails although the
  held-out rise is inside the band (share 0.98-1.0).
- **p = 23: the share.** The fit is early and stable, but the held-out rise is slow (t10 to t90: about 24k to 53k at depth 3, 30k
  to 71k at depth 4), and the loss-only band holds under half of it (median share 0.45-0.48).

## Settings (locked here)

Each cell (p, depth), p in {23, 47, 61, 101}, depth in {3, 4}, runs four settings: the campaign-F configuration of the cell with
two dials changed.

- **Head decay** lam_v in {0.2 (31/p)^(1/2), 0.2 (31/p)}: the first keeps the per-class head scale of p = 31, the second
  over-corrects. Values (rounded to 4 digits): p = 23: 0.2322, 0.2696; p = 47: 0.1624, 0.1319; p = 61: 0.1426, 0.1016;
  p = 101: 0.1108, 0.0614.
- **First-layer decay** lam_w in {0.015 (the paper's), 0.025}: the second speeds the held-out rise (aimed at p = 23).
  `lam_layers[0]` takes the same value; deeper layers keep 0.07.
- **Unchanged** (campaign F): width 256, frac 0.8, eta_w = eta_v = 0.03, init scale 24, deep init scale 8, eta_layers
  [0.03, u, ..., r u] with u = 0.03 * 0.015 / 0.07, r as above, lam_layers [lam_w, 0.07, ...], horizon 120k (p = 23) or 100k,
  early stop `stop_generalized` (0.99, factor 1.5, extra 10k), logs every 20 updates, heavy logs every 500 (1000 at p = 101).
- **Seeds:** development seeds i = 1, 2, 3 with init 7000 + i and split 8000 + i, used by no earlier campaign.
- **Runner:** `bias_runner.py` of campaign I without the `bias` key, which reproduces campaign C, D and F runs bit for bit.

That is 8 cells x 4 settings x 3 seeds = 96 runs.

## Per-run endpoint

Campaign C's, unchanged (`endpoints_i.py`): the loss-only peak band holds at least 75% of the held-out 10% -> 90% rise, final
held-out accuracy at least 0.99, the last hidden layer's A_H rises by at least 0.02 across the band, and every logged training
margin in the band is positive.

## Selection rule (fixed now)

For each cell, among its four settings:
1. the most passes (out of 3);
2. ties: the larger minimum share over the three seeds;
3. then the fewer median relapses inside the band (logged steps with a non-positive training margin);
4. then the square-root head-decay rule over the linear one, then lam_w = 0.015 over 0.025.

A cell where no setting passes any development run gets no confirmation stage; its four settings are reported as they are.

## Second stage (outline; its own PLAN is fixed after this stage and before it starts)

Ten fresh seeds per selected cell (init 9000 + i, split 9500 + i, i = 1-10) at the selected setting, scored with the same
endpoint and reported as k/10 with a Wilson 95% interval, next to campaign F's 10 runs at lam_v = 0.2.

## Report of this stage

Per setting: passes k/3, failed checks, share, CE peak / log p, first persistent fit, relapses in the band, t10, t90, final
held-out accuracy and Delta A_H; the selected setting of every cell and the rule step that decided it.

## Compute

96 runs, about 500 core-hours by campaign F's timings (p = 101 runs use 4 threads), on 20 sessions K1-K20 of 4 cores: about
6-7 hours per session. Development runs are not evidence for the paper's claims; only the second stage is.
