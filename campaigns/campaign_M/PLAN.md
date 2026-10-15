# Campaign M: training-fraction ladder with the last hidden layer's rate re-tuned (plan fixed before any run)

claude/1hl-figures, 2026-10-11, at Akash's request after campaign L: "Even for depth 3 it seems that the refit and held-out
accuracy improves after the peak has been achieved ... we need another iteration over the hyperparameter to capture the
results for smaller training set sizes. One thing we can do is possibly have config with varying training set splits and
then see how the smaller split works out." Scope chosen by Akash: depths 3 and 4, p = 47 and 61, head decay lam_v the
paper's 0.2 and the linear rule, training fractions 0.8, 0.7, 0.6 and 0.5. Development runs only; not evidence for the
paper's claims.

## 1. Why campaign L fails, and the dial that moves it

- After the knee the training CE climbs from its dip to a peak because the hidden layers shrink from their large
  initialization under weight decay. The slowest of them is the last hidden layer, whose rate is r x u (u = 0.03 x 0.015 /
  0.07; the paper's r = 0.45), with decay time 1 / (r x K1), K1 = 0.03 x 0.015. In every campaign L cell (r = 0.45) the
  post-knee CE peak is at 25-30k updates, as at fraction 0.8 when the rise does not interrupt the climb.
- The held-out rise hardly depends on r. Campaign D (fraction 0.8, lam_v 0.2, r from 0.2 to 0.9, two seeds per cell) moves the
  CE peak from about 10k (r = 0.9) to beyond 28k (r = 0.2), while the held-out 0.5 crossing stays at 13-19k at depth 3
  and 18-24k at depth 4 (p >= 47).
- With half the data the rise starts 3-6 times later (campaign L: held-out 0.5 at 59-143k), so it lands on the drift after
  the peak: from the refit's 0.2 crossing to the held-out 0.9 crossing the CE falls 5-21%, and none of the held-out rise of
  any campaign L run at these cells lies inside the paper's post-hoc window (q_te = 0.00 in all 24, section 4), including the
  runs that pass the refit criterion.
- The paper set r with campaign C's recipe at fraction 0.8 (r K1 t10 exp(0.9 (r - r0)) = 3.85, `bias_runner.solve_rate`).
  Re-applied with the measured t10 it gives r of about 0.39-0.46 at 0.7, 0.26-0.35 at 0.6 and 0.16-0.24 at 0.5 (depth 3).
  So the configuration needs r re-tuned per training fraction.

## 2. The r grid

With the CE peak at about 27k x 0.45 / r (depth 3; 29k at depth 4; campaign L), the rate that puts the peak between the
held-out 0.5 and 0.9 crossings predicted at each fraction (geometric interpolation between campaigns K / F at 0.8 and L at
0.5, r = 0.45) is:

| | 0.7 | 0.6 | 0.5 |
|---|---|---|---|
| depth 3 (p = 47, 61; both lam_v) | 0.35-0.53 | 0.18-0.34 | 0.08-0.20 |
| depth 4 (p = 47, 61; both lam_v) | 0.27-0.41 | 0.14-0.25 | 0.07-0.14 |

Grid (the paper's r = 0.45 at every fraction above 0.5 as the control; campaign L is the r = 0.45 control at 0.5):

| | 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|
| depth 3 | 0.45 | 0.45, 0.30 | 0.45, 0.28, 0.18 | 0.20, 0.13, 0.08 |
| depth 4 | 0.45 | 0.45, 0.28 | 0.45, 0.20, 0.13 | 0.14, 0.09, 0.06 |

## 3. Runs (216; `manifest.json` from `make_manifest.py`; runner `bias_runner.py`, campaign K's and L's, unchanged)

Every run is campaign K's configuration of its cell (width 256, no biases, init scales 24 and 8, eta_w = eta_v = 0.03,
eta_layers [0.03, u, ..., r u], lam_layers [0.015, 0.07, ...], full-batch CE, supported-Polar hidden steps, normalized head
step, logs every 20 updates, heavy logs every 500) with three dials: the training fraction, r (grid above) and lam_v (0.2, or
0.2 x 31 / p = 0.1319 at p = 47 and 0.1016 at p = 61); lam_w = 0.015 throughout. Seeds 1-3 of campaigns K and L
(init 7000 + i, split 8000 + i): with the same split seed the training sets are nested (0.5 within 0.6 within 0.7 within
0.8), so every run pairs with campaign K (0.8, linear rule) and campaign L (0.5, r = 0.45). Horizons: 100k (0.8), 150k (0.7),
200k (0.6), 250k (0.5); early stop as in K and L (held-out accuracy >= 0.99 for max(0.5 t, 10000) updates after first
reaching it at t).

2 depths x 2 primes x 2 lam_v x 9 (fraction, r) settings x 3 seeds = 216 runs. The 12 runs at 0.8 with the linear rule
repeat campaign K's runs (same configuration and seeds): the analysis reports the largest loss difference as a
reproducibility check.

## 4. Analysis (fixed now; `analyze_M.py`, `plot_M.py`)

Per run (`analyze_M.py`; campaign L's paired runs scored the same way as the r = 0.45 controls at 0.5):

1. **Timing.** Knee; CE dip; post-knee CE peak (update, CE / log p); first crossings of the ridge refit (0.2, 0.5, 0.9) and
   the held-out accuracy (0.1, 0.5, 0.9); the CE's drawdown below its running maximum since the knee at the held-out 0.5 and
   0.9 crossings; the CE change from the refit's 0.2 crossing to the held-out 0.9 crossing; the most training pairs
   misclassified over that stretch.
2. **Refit criterion** (campaigns K and L, unchanged): W = the paper's post-hoc window (after the knee, the stretch where the
   cycle-mean CE never falls more than 3% below its running maximum, the one with the largest CE rise); pass = refit +0.2
   inside W, held-out +0.2 from the start of W to the end, final held-out >= 0.99; pass3 = pass with training accuracy
   >= 0.97 at every logged step in W.
3. **Rise inside the window** (new; what failed in campaign L): q_te and q_rf = the shares of the held-out and refit
   10% -> 90% rises inside W (the paper's clip formula); Delta A_H of the last hidden layer across W; the least training
   accuracy and the strict fit (every logged margin > 0) in W.
4. **Verdict.** pass_M = refit pass and q_te >= 0.75 and q_rf >= 0.75 and Delta A_H >= 0.02. pass_M3 = pass_M with training
   accuracy >= 0.97 at every logged step of W. Campaign C's per-run endpoint (loss-only band around the peak) is reported,
   not deciding.

On existing runs (checked before launch): campaign K's 12 runs at 0.8 (linear rule, r = 0.45) give pass_M 12/12 (q_te
0.85-1.00, q_rf 1.00); campaign L's 24 runs at 0.5 (r = 0.45, both lam_v) give 0/24 (q_te 0.00 in all 24).

**Settings.** A setting (depth, p, lam_v, fraction, r) is a candidate if pass_M holds in 3/3 seeds. For each (depth, p,
lam_v, fraction) the best r is the one with the most pass_M, then the most pass_M3, then the higher median q_te. Candidates
at 0.6 and 0.5 go to a confirmation on fresh seeds (planned after this campaign).

**Figures** (`plot_M.py`, the paper's renderer `acrossp_figs.py` unchanged): scoreboards of pass_M and of the refit criterion
per (depth, p, lam_v) and (fraction, r), with campaign L's r = 0.45 at 0.5; q_te and the drawdown at the held-out 0.9
crossing against the fraction, per r; per (depth, p, lam_v) the renderer's four rows (CE / log p, accuracies with the refit,
misclassified training pairs, A_H; grey: median post-hoc window) for every (fraction, r); and a ladder (0.8, 0.7, 0.6, 0.5)
with the best r at each fraction.

## 5. Compute

216 runs, about 820 core-hours by campaigns K's and L's measured speeds (0.05-0.13 s per update with four runs per
four-core session; a run at 0.5 that does not generalize runs to 250k updates), on 24 sessions M1-M24 of 4 cores: about
9 hours per session by this estimate. Fewer sessions also work (the others take over unstarted IDs); they take longer.

## 6. Local pilots (this container; not part of the results)

Depth 3, p = 61, linear rule, fraction 0.5, seed 1, r = 0.15 and 0.10 (campaign L's seed-1 run is the r = 0.45 twin), started
2026-10-11 02:04 UTC; at about 0.1 s per update here they need about 4 hours to reach the rise. Reported with the results.
