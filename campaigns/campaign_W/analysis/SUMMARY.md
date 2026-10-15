# Campaign W (transformers under the paper's own definitions): results, all 42 runs (claude/1hl-figures, 2026-10-11)

**State.** 42 of 42 runs finished on the 10 sessions W1-W10 (all `campaign_done`, last heartbeats 2026-10-10 23:00-23:03Z, no
failed runs). The results were read from the session branches through Akash's browser and laid out as collect_results.sh does
(`collected/results`, `collected/status`, `collected/INDEX.json`); every run.json.gz matches the SHA-256 in its chain.json.
The analysis is the one fixed in PLAN.md (`analyze_W.py`, log `analyze_W.log`, per run `runs_W.json`); the relaxed-fit
numbers below (`relaxed_W.py`, `relaxed_W.json`) were chosen after seeing the results and are exploratory. Held-out accuracy
is on the pairs whose mirror (b, a) is also held out (runner 1.3), except the final accuracy (all held-out pairs).
Development runs only; not evidence for the paper's claims.

## 1. Pre-registered verdict: no candidate

A run meets the paper's definitions if a window with raw CE within 10% at every update and every training pair correct at
every 20-update log holds >= 75% of the held-out 10% -> 90% rise, the ridge refit gains >= 0.2 inside it, Delta A_H (head
input) >= 0.02 and the final held-out accuracy is >= 0.99. A setting is a candidate if 3/3 of its runs meet them.

| arm | setting (p = 113 unless noted; RMS head unless noted) | meets | final >= 0.99 | what happens |
|---|---|---|---|---|
| R5 | frac 0.2, eta_w 0.1, lam_v 0.275 | **2/3** | 3/3 | fit first (held-out 0.00-0.12 at the fit); the third run has a few training pairs misclassified during the rise |
| R4 | frac 0.2, eta_w 0.1, lam_v 0.25 (seeds 3-5) | 1/3 | 3/3 | the same; two runs lose the strict fit at some logs |
| R1 | frac 0.2, eta_w 0.03, lam_v 0.3 | 1/3 | 2/3 | slower (held-out 0.9 at 12-18k); one run unfinished at 40k (0.85) |
| R1 | frac 0.2, eta_w 0.03, lam_v 0.25 | 0/3 | 2/3 | one run does not generalize by 40k (0.44) |
| R2 | frac 0.15, lam_v 0.3, 0.4, 0.5 | 0/9 | 0/9 | no generalization in 40k (no-mirror held-out <= 0.08) |
| R3 | frac 0.1, lam_v 0.4, 0.5 | 0/6 | 0/6 | no generalization in 60k (<= 0.001) |
| S1 | p = 31, frac 0.5, lam_v 0.5, 1 | 0/6 | 0/6 | all-pairs held-out about 0.5 is the mirror pairs; no-mirror 0.03-0.06 at 30k |
| N1 | NGD, frac 0.2, lam_v 0.25 | 0/3 | 0/3 | never fits; held-out 0.96 |
| N1 | NGD, frac 0.2, lam_v 0.5 | 0/3 | 0/3 | no generalization (0.26-0.53 at 30k) |
| P1 | C1 in float64 (plain head, frac 0.3, seeds 6-8) | 0/3 | 3/3 | no 10% window (section 4) |

## 2. The fraction-0.2 settings with a relaxed fit (exploratory)

During the rise (from the first full fit to held-out 0.9) at most 4-9 of the 2553 training pairs are misclassified at any log
in R4 and R5. Allowing up to 10 misclassified pairs (0.4%) at every log, other conditions unchanged:

| setting | meets | share of the rise in the window | CE / log p in it | ridge gain | probe gain | Delta A_H |
|---|---|---|---|---|---|---|
| R5, lam_v 0.275 | **3/3** | 0.85, 1.00, 1.00 | 0.33-0.34 (about 1.6 nats) | +0.27 to +0.55 | +0.60 to +0.85 | 0.31-0.34 |
| R4, lam_v 0.25 | **3/3** | 0.91, 0.87, 0.96 | 0.28 (about 1.3 nats) | +0.35 to +0.40 | +0.65 to +0.73 | 0.28-0.34 |
| R1, lam_v 0.3 (eta_w 0.03) | 2/3 | 1.00, 0.91 (unfinished), 0.99 | 0.34-0.36 | +0.43 to +0.46 | +0.78 to +0.83 | 0.32-0.35 |

The CE does not stop falling in these windows: in R4 and R5 it falls 1.7-2.4% per 1000 updates while the held-out
accuracy rises, 6.5-9.1% from held-out 0.1 to 0.9 (4-6% from 0.2 to 0.9); training accuracy 0.996-1.000 throughout.
Section 6b has the rates and a zoom.

## 3. Contrast: NGD (the paper's Appendix F contrast) on the same configuration

NGD hidden steps, lam_v 0.25: the training set is never fit (training accuracy 0.58-0.96 during the held-out rise, at most
0.99 later), held-out accuracy rises earlier (0.9 at 4.6-6.6k) to 0.96, and the CE falls 17-23% across the rise. With lam_v
0.5 NGD does not generalize in 30k. Polar on the same configuration (R4): fit first, CE within 10% (-5 to -6%) across the rise.

## 4. C1 in float64: the flat 3e-7 CE was a float32 floor

Same seeds (6-8), same configuration, float32 (campaign V) against float64 (P1): in float32 the post-fit CE sits at
2.2-3.4e-7 nats; in float64 it keeps falling, to 6e-9 to 9e-13 (500-update medians) by the held-out rise and lower after, with
single-update spikes above 10x the median at 36-40% of updates (float32: 5-11%) and a misclassified training pair at 2-12
logs (float32: 0-1). Generalization timing is similar (no-mirror held-out 0.9 at 5.3-8.2k against 4.7-7.2k). So C1 is not a
flat-loss example; its plateau is the float32 resolution of the softmax.

## 5. Readouts

In every Polar run with an RMS head the ridge refit (the paper's canonical refit on the head input) crosses 0.5 1.4-4.9k
updates after the held-out accuracy; the logistic probe crosses within -260 to +780 updates of it. So for transformers only
the "same window" part of the key message holds, and with the probe rather than the ridge refit.

## 6. Reading

- The transformer example that comes closest to the paper's definitions is the RMS head at training fraction 0.2 with
  eta_w 0.1 and lam_v 0.25-0.275: training set fit first, then the held-out accuracy (no mirror) rises 0 -> 1 while the raw
  CE stays within 10% at about 1.3-1.6 nats and A_H rises 0.3. Under the strict fit it holds in 2/3 runs (lam_v 0.275); with
  0.4% training error allowed, in 6/6 (lam_v 0.25 and 0.275).
- Fewer training pairs (fractions 0.15, 0.1) or p = 31 do not generalize within the horizons.
- C1 should not be used as a flat-loss example (section 4).
- Next step if wanted: a confirmation on 10 fresh seeds of R5 (and R4), with NGD and AdamW contrasts on the same seeds, the
  fit condition (strict, or a fixed tolerance) decided before the runs.

## 6b. How fast the CE falls while the held-out accuracy rises (added 2026-10-11, after Akash's question)

Training CE as the paper's cycle mean averaged over 201 updates; slope = log-linear fit over held-out 0.1 -> 0.9 (no-mirror
pairs for the transformer); `zoom/slopes.py`, `zoom/slopes.json`, figures `zoom/figs/zoom_runs` (every R4 and R5 run, CE in
nats) and `zoom/figs/zoom_compare` (CE relative to its value at held-out 0.5, runs aligned there, against the paper's runs).

| setting | CE at held-out 0.5 | % per 1000 updates | total over held-out 0.1 -> 0.9 |
|---|---|---|---|
| transformer R5, lam_v 0.275 (3 runs) | 1.57 nats | -1.7 to -2.4 | -6.5 to -8.2% |
| transformer R4, lam_v 0.25 (3 runs) | 1.33 nats | -2.1 to -2.3 | -7.3 to -9.1% |
| transformer R1, lam_v 0.3, eta_w 0.03 (2 runs) | 1.69-1.71 nats | -0.9 to -1.3 | -7.1 to -8.4% |
| transformer NGD, lam_v 0.25 (3 runs) | 2.22-2.27 nats | -6.8 to -24.7 | -26 to -30% |
| C1 float32 (3 runs; 201-update running median) | about 3e-7 nats | -2.8 to -4.0 | -6.8 to -14% |
| paper, one hidden layer, p = 31, frac 0.8 (campaign F, 8 runs) | 1.56 nats | -0.4 to -0.9 | -1.2 to -3.2% |
| paper, depth 3, p = 61, frac 0.8 (campaign F, 8 runs) | 3.17 nats | 0.0 to +0.2 | +0.1 to +1.5% |

- In the transformer the CE falls steadily from the fit on and keeps falling after the rise (R4 and R5: -3.1 to -5.4% per
  1000 updates before held-out 0.1, -0.8 to -2.2% after 0.9): a slow relaxation, not something the rise causes. It passes the
  10% test only because the rise is short (3-4.5k updates).
- The paper's one-hidden-layer runs peak about 2k updates before held-out 0.5 and then fall gently (about 3x slower than the
  transformer); the depth-3 runs are flat or rising. In both the ridge refit rises before the held-out accuracy.
- C1 (float32) is not a positive-loss case: its CE is about 3e-7 nats, and the float64 rerun shows that this level is the
  float32 floor (section 4).

## 7. Files

`runs_W.json` (fixed analysis), `analyze_W.log`, `relaxed_W.py` and `relaxed_W.json` (exploratory), `draw_W.py` and
`figs/W_frac02` (R5, R4, R1 lam_v 0.3 and NGD: CE, raw CE, misclassified training pairs, accuracies with refit and probe,
A_H; grey: the strict 10% window, median over the runs that have one), `figs/W_C1_precision` (C1 float32 against float64).
