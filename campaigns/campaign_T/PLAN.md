# Campaign T: transformers on modular addition, development stage (plan fixed before any run)

Written by claude/1hl-figures on 2026-10-07, at Akash's request, before any run of this campaign. Purpose: does the paper's
coexistence (training cross-entropy relatively flat at a positive level while hidden features change and held-out accuracy
rises) appear in transformers trained with the paper's update rule, and how does AdamW, the usual grokking optimizer, compare?
This stage explores settings; a second stage, with its own plan fixed before it starts, confirms the selected settings on
fresh seeds with a locked window, as the paper's one-hidden-layer headline does.

## Model and task

- **Task.** a + b mod p at p = 113 (the usual grokking benchmark), all p^2 pairs, a random split keeping a fraction `frac` for
  training (numpy permutation with split_seed, as in the MLP runner). Input [a, b, =]; logits read at "=". Full-batch mean
  cross-entropy.
- **Model** (Nanda et al. 2023 layout): token and position embeddings (d = 128), `n_layers` blocks of causal 4-head attention
  and a ReLU MLP (512), no LayerNorm and no biases, unembedding at the last position. One layer (P1, A1) and two layers
  (P2, A2). float32 on CPU (`tf_runner.py`; tests in `tests/test_tf.py`).
- **The paper's update rule** (P1, P2): every matrix except the unembedding takes a supported-Polar step with decay,
  W <- W + eta_w (Polar(-G)/sqrt(rank G) - lam_w W); the unembedding takes a normalized gradient step with decay,
  W_U <- W_U + eta_v (-G/||G||_F - lam_v W_U), eta_v = 0.03. No momentum, schedules or clipping.
- **AdamW** (A1, A2): lr 1e-3, weight decay 1.0, betas (0.9, 0.98), all parameters (Nanda et al.'s settings).

## Settings (locked here)

| arm | layers | grid | seeds | runs |
|---|---|---|---|---|
| P1 | 1 | frac 0.3: lam_w in {0.01, 0.02, 0.03, 0.05} x eta_w in {0.03, 0.1} x lam_v in {0.2, 1, 3}; frac 0.5: lam_w in {0.01, 0.02, 0.03}, eta_w 0.1, lam_v in {1, 3} | 2 | 60 |
| P2 | 2 | frac 0.3: lam_w in {0.01, 0.03}, eta_w 0.1, lam_v in {0.2, 1, 3} | 2 | 12 |
| A1 | 1 | AdamW, frac in {0.3, 0.5} | 2 | 4 |
| A2 | 2 | AdamW, frac in {0.3, 0.5} | 2 | 4 |

Development seeds i = 1, 2: init 7100 + i, split 8100 + i. Horizon 60k updates, stopped 1.5x after held-out accuracy first stays
at or above 0.99 (at least 5k updates later). Logs: the training cross-entropy at every update; accuracies, margins and held-out
loss every 20 updates; A_H (MLP activations at "=" in every block, final residual stream, logits), weight norms and the
train-only ridge refit (tau = 1e-3 n, on the final residual stream and on the last MLP) every 500.

The grid was centred by short calibration runs (one layer, frac 0.3, one seed, init 11, split 12, up to 8k updates):
- AdamW fits by about 250 updates, its cross-entropy stays near 1e-4, and held-out accuracy goes from 0.35 to 1.0 between 3.9k
  and 5.4k updates.
- The paper's rule with eta_w = 0.03: lam_w = 0.015 fits by about 1.2k and the cross-entropy is 0.002 by 1.75k; lam_w = 0.03 fits
  by about 2k and the cross-entropy keeps falling (0.69 at 2k, 0.01 at 6k) with held-out accuracy still 0; lam_w = 0.05 reaches
  half the training pairs by 6k; lam_w = 0.1 and 0.3 stay near chance for 2k updates. A Polar step of Frobenius norm 0.03 moves
  a 128 x 128 matrix about four times less per update than AdamW at lr 1e-3, hence the larger step eta_w = 0.1 in the grid.
- eta_w = 0.1, lam_w = 0.01, lam_v = 0.2: fits by 500 updates; the cross-entropy then falls to about 1e-7 and rises again from
  about 1.7k (to about 3e-3 at 2.25k, with a few training pairs relapsing between 2.2k and 3.5k) while held-out accuracy rises
  slowly (0.02 at 2k, 0.10 at 3.6k, 0.25 at 4.75k): the shape the deep networks show after their knee.
- The same with lam_v = 1.0: fits by about 750 updates; from 1k to 2.75k the cross-entropy stays between 0.013 and 0.025 (flat at
  a positive level), training accuracy stays at 0.998-1 with a few pairs relapsing, and held-out accuracy is still below 0.003.
  lam_v = 3 is in the grid to see whether a lower head cap lifts the plateau further and lets the held-out rise happen on it.

## Per-run fields and qualification (development)

For each run: t_fit, the first logged update from which every training margin stays positive; t10 and t90, held-out accuracy
first at 0.1 and from then on at 0.9; v = max/min - 1 of the per-update training cross-entropy on [t_fit, t90]; Delta A_H of
the last block's MLP activations from t_fit to t90; the refit's first 0.9.

A run **qualifies** when: the fit comes before the held-out rise (t_fit <= t10 and held-out accuracy <= 0.1 at t_fit), the final
held-out accuracy is at least 0.99, v <= 0.10, and Delta A_H >= 0.02.

## Selection and the locked window (fixed now)

For each Polar arm (P1, P2): among the settings whose runs all qualify, the smallest maximum v over the seeds; ties: the larger
minimum Delta A_H, then frac 0.3. The window for the second stage is I = [a, b], a = 1.2 x the largest t_fit and b = 1.2 x the
largest t90 over the selected setting's development runs, each rounded up to a multiple of 500. If no setting qualifies, the arm
gets no confirmation stage and its runs are reported as they are.

## Second stage (outline; its own PLAN is fixed after this stage and before it starts)

Twenty fresh seeds per selected Polar arm (init 9100 + i, split 9600 + i) with the locked window, and ten AdamW runs per depth on
the first ten of the same seeds. A fresh Polar run passes on I when: every logged training margin in I is positive, the
cross-entropy varies by at most 10% on I, held-out accuracy is at most 0.1 at a and at least 0.9 at b, and A_H of the last MLP
rises by at least 0.02 across I. Reported as k/20 with a Wilson 95% interval; the AdamW runs are reported with the same fields
(their cross-entropy is expected to fall by orders of magnitude while they generalize).

## Report of this stage

`analyze_T.py`: every run's fields and checks, every setting's qualification, the selected setting and window per arm.

## Compute

80 runs, about 190 core-hours by the calibration timings (0.15 s per one-layer update and 0.30 s per two-layer update on one
core; about 255 if no Polar run stops early), on 12 sessions T1-T12 of 4 cores: about 4-6 hours per session.
Development runs are not evidence for the paper's claims; only the second stage is.
