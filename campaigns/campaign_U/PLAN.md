# Campaign U: transformers, the flat-loss level and the averaging window (exploration)

claude/1hl-figures, 2026-10-08, at Akash's request after campaign T:
"the flat loss level is quite low, should be higher" and "how does the notion of different flatness come into the picture?
In some sense the first column for AdamW also seems flat, but we need to consider a much larger moving window."

The key message under test (Akash, 2026-10-08): in a flat or rising training loss, the refit gets better, and either within the
same window or later the held-out accuracy goes up.

## 1. What campaign T showed (75 of 80 runs)

- The smoothed training CE stays flat while the refit and the held-out accuracy rise in two places only: one layer,
  lam_w 0.02, lam_v 0.2 (all four runs; CE about 3-4e-7, confirmed in float64), and two layers, lam_w 0.01, eta_w 0.1,
  lam_v 1 (seed 1 only; CE about 2.4e-2, training accuracy dipping to 0.81). Everywhere else the CE falls by 3x to 1000x
  while the features are learned.
- Flatness depends on the averaging window (flat_windows.py, figure T_flat_scales). Over the feature-learning interval I (from
  the first time the refit or held-out accuracy reaches 0.1 to the later of their first times at 0.9), the drawdown of the
  centred geometric moving average (its largest relative fall below its running maximum) is, for one layer at lam_v 0.2,
  0.37 at 100 updates, 0.10 at 500 and 0.05 at 1000 (|I| = 2000 updates): flat on windows well below |I|. For AdamW (one
  layer, wd 1) the CE oscillates over three to four decades with a period of about 1000 updates; the drawdown is 1.0 up to
  1000 updates, 0.79 at 5000 and 0.71 at 10000 (|I| = 5100) and small only at 20000, a window four times longer than the
  event. Its 5000-update average falls about fivefold across I; on a log axis spanning eight decades that looks flat.
- Why the level was low: without normalization the residual stream's norm grows through every matrix (Frobenius caps 1/lam_w
  of 50-100 per matrix), so the logit scale is bounded only loosely and the caps bind at a very small CE. The paper's MLPs sit
  at a CE of 0.5-0.8 log p in their flat window (campaign K, paper setting, depths 3 and 4).

## 2. Pilot (one seed, this container, not part of the results)

One layer with a final RMS division before W_U (the head's input then has norm sqrt(d), so lam_v alone bounds the logits),
lam_w 0.01, frac 0.3, seed 1:

| lam_v | eta_w | CE / log p | what happens |
|---|---|---|---|
| 4 | 0.1 | 0.95 | underfits: training accuracy stuck at 0.45 |
| 2 | 0.1 | 0.89 | underfits: training accuracy stuck at 0.72 |
| 1 | 0.1 | 0.82 -> 0.80 (updates 500 -> 1500) | held-out 0.53 -> 0.98, refit 0.36 -> 0.92, training 0.79 -> 0.99 |
| 0.5 | 0.1 | 0.70 -> 0.61 (500 -> 1500) | held-out 0.42 -> 1.0, refit 0.36 -> 1.0, training 0.96 -> 1.0 |
| 0.5 | 0.03 | 0.77 (update 500), then 0.62 -> 0.60 (1500 -> 5500) | held-out 0.17 -> 0.85 -> 0.99, refit 0.24 -> 0.41 -> 0.91, training 0.77 -> 0.95 -> 1.0 |

The same layout with AdamW (wd 1): CE about 1e-2, oscillating (campaign T's AdamW, without the RMS division: 1e-6 to 1e-3);
held-out 0.31 -> 1.0 between updates 1000 and 2000.

Scored with flat_windows.py (delta 0.1; the pilots were cut at 2000-9000 updates): lam_v 0.5 with eta_w 0.03 passes the key
message at every window from 1 to 2000 updates (stretch 1000-7000, CE 2.87 = 0.61 log p, refit +0.63, held-out +0.61; not
pass3, training accuracy 0.87 at the stretch's start); lam_v 1 with eta_w 0.1 has refit +0.70 and held-out +0.83 on a stretch
at CE 3.8 but was cut before its held-out accuracy reached 0.99; lam_v 0.5 with eta_w 0.1 generalizes before its flat stretch
(held-out +0.04 after it); AdamW with the RMS division has no flat stretch at windows up to 2000 (drawdown over I 0.76-1.0).
In these runs the held-out accuracy rises largely together with the training fit; the refit lags it.

## 3. Questions

- Q1 (level). Does the key message hold at a CE level comparable to the MLPs' (CE / log p of 0.05-0.8) rather than 4e-7?
- Q2 (window). For each run, at which averaging windows (and with arithmetic or geometric averaging) does it hold? Is a flat
  stretch found only at windows comparable to the event itself (the AdamW case)?

## 4. Arms (71 runs, frac 0.3, p = 113; make_manifest.py)

| arm | layers | layout | rule | settings | seeds |
|---|---|---|---|---|---|
| N1 | 1 | final RMS | Polar | lam_w 0.01: eta_w 0.1 with lam_v 0.25, 0.5, 1, and eta_w 0.03 with lam_v 0.5, 1; lam_w 0.02, eta_w 0.1: lam_v 0.5, 1 | 1-3 |
| N2 | 2 | final RMS | Polar | lam_w 0.01, eta_w 0.1: lam_v 0.5, 1 | 1-3 |
| H1 | 1 | campaign T | Polar | lam_w 0.01, eta_w 0.1: lam_v 3 (seeds 3-5; T has 1-2), lam_v 10 | as stated, else 1-3 |
| H2 | 2 | campaign T | Polar | lam_w 0.01, eta_w 0.1: lam_v 1 (seeds 3-5; T has 1-2), lam_v 2 | as stated, else 1-3 |
| L1 | 1 | campaign T | Polar | lam_w 0.02, eta_w 0.1, lam_v 0.2 (the 4e-7 plateau) | 3-5 |
| A1 | 1 | campaign T and final RMS | AdamW | wd 0.3, 1, 3 | 1-3 |
| A2 | 2 | campaign T and final RMS | AdamW | wd 1 | 1-3 |
| TR | 1-2 | campaign T | Polar | the five campaign-T runs that crashed (their ids, configurations and seeds) | - |

Common: d = 128, 4 heads, MLP 512, eta_v 0.03, AdamW lr 1e-3 with betas 0.9/0.98, 60000 updates at most; a run stops once the
held-out accuracy has stayed >= 0.99 for max(0.5 t, 5000) updates after first reaching it at t (AdamW: max(0.5 t, 20000), so
that long averaging windows have room). Seeds: init 7100 + i, split 8100 + i (seeds 1 and 2 are campaign T's).

## 5. Records (tf_runner.py 1.2)

Identical to campaign T's runner (1.1) except: the optional final RMS division; the recorded training CE is the float64 CE of
the float32 logits (the trajectory is unchanged: same observations, byte for byte, on campaign-T configurations); the
train-only logistic probe of the head's input. Training CE at every update; accuracies, margins and held-out loss every 20
updates; A_H, norms and the ridge refit every 250; the logistic probe every 500.

## 6. Analysis (exploratory; fixed before the runs)

flat_windows.py, for every run, windows w = 1, 100, 500, 2000, 5000, 10000 updates, centred arithmetic and geometric averages:
- key message at scale w: a stretch of at least max(2w, 500) updates in which the average never falls more than delta = 0.1
  below its running maximum, the refit's held-out accuracy rises by >= 0.2, the held-out accuracy rises by >= 0.2 from the
  stretch's start to the end of the run, and the final held-out accuracy is >= 0.99 ("pass"); "pass3" also requires training
  accuracy >= 0.97 throughout the stretch. Repeated with the logistic probe in place of the ridge refit, and with delta = 0.03
  and 0.3;
- flatness profile: the drawdown and the end/start ratio of each average over I, and w*, the smallest window with drawdown
  <= 0.3;
- the CE level (and CE / log p) of each passing stretch.

Figures: per setting, the dynamics with averages at 100, 1000 and 5000 updates; a runs x windows map of pass / pass3 with
the CE level; the level of passing stretches against lam_v. A confirmation stage, if any, fixes its rule from these results.

## 7. Cost

About 139 core-hours; 12 sessions (U1-U12) of 4 cores, 3-6 hours each (the longest single run, a two-layer campaign-T rerun
of 60000 updates, takes about 5 hours).
