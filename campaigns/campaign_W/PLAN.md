# Campaign W: transformer settings under the paper's own definitions (plan fixed before any run)

claude/1hl-figures, 2026-10-10, at Akash's request ("let's try to find some configurations"; "make sure the optimizer we are
using for these experiments is same as the one used in paper otherwise we need to clarify this").

## Why

Campaigns T, U and V scored transformer runs with campaign V's rule (a smoothed CE, training accuracy >= 0.97). Re-scored with
the paper's definitions (Section 3.3: raw CE within 10% at every update; "fitted" = every training pair has a positive margin;
Appendix D.3: logged every 20 updates), no setting passes:

- C1 (plain head, frac 0.3, lam_w 0.02, eta_w 0.1, lam_v 0.2): the training set stays fit, but the raw CE (about 3e-7 nats)
  varies about 4x (median) during the held-out rise, with single-update spikes up to 0.8 nats in 10 of 15 runs. No 10% window
  captures any of the rise.
- RMS head, frac 0.2, lam_v 0.5: a 10% raw-CE window holds 92-99% of the rise (CE about 0.6 log p), but 0.2-7% of the
  training pairs stay misclassified throughout it.

Local pilots (float64, seed 1, frac 0.2, eta_w 0.1, runner 1.3 = 1.2): lam_v 0.25 fits every training pair from update 860
while held-out accuracy is 0.12, then held-out rises to 0.98 (0.9 at 6820), but the raw CE falls 16% from held-out 0.2 to 0.9
(best fit-and-10% window: 69% of the rise). lam_v 0.3 has a flat CE (7.6% over the rise; 92% of the rise in a 10% window) but
its fit relapses (47 logs with a misclassified training pair inside the rise). A tighter cap (larger lam_v) flattens the CE and
makes a full fit harder; this campaign looks for settings where both hold.

## Optimizer: the paper's, with the differences stated

Every Polar run uses the paper's update (eq. 3) on every matrix except the head, one matrix at a time:
W <- (1 - eta_w lam_w) W + eta_w Polar(F)/sqrt(rank F), supported Polar over singular values above 1e-7 s_max (the same
relative cutoff as the paper's deep runs), and the head W_U <- (1 - eta_v lam_v) W_U + eta_v G/||G||_F; simultaneous, full-batch,
no momentum. Now float64, as in the paper. Remaining differences, to be stated in the paper: one (eta_w, lam_w) for all
matrices (embeddings, attention, MLP); the attention matrices W_Q, W_K, W_V, W_O are each one 128 x 128 matrix across the four
heads; the final RMS division (no parameters) is a model change; the paper's NGD contrast (Appendix F) is arm N1.

## Held-out accuracy without mirror pairs

With a shared token embedding the transformer gets a held-out pair (a, b) right while memorizing when its mirror (b, a) is in the
training set. Checked at update 600 (seed 1, RMS head): p = 113, frac 0.2: 0.58 on such pairs (20% of the held-out set) against
0.003 on the others; p = 31, frac 0.5: 1.00 against 0.004. This is the early held-out plateau at about frac x 0.6 in the RMS runs.
Runner 1.3 also logs the held-out accuracy on the pairs whose mirror is held out too (`test_acc_nm`, and `refit_test_acc_nm`,
`probe_test_acc_nm`); training is unchanged (bit-identical to 1.2). The analysis uses the no-mirror accuracy for the crossings,
the share of the rise, fit-first and the readouts; the final accuracy (>= 0.99) is on all held-out pairs.

## Arms (42 runs; manifest.json from make_manifest.py; runner tf_runner.py 1.3)

Common: p = 113 unless noted, one block, d_model 128, 4 heads, MLP 512, final RMS division unless noted, eta_v 0.03,
lam_w 0.01, float64, full batch, early stop as in campaign V (held-out >= 0.99 from t; stop at max(1.5 t, t + 5000)), logs every
20 updates, ridge refit, A_H and norms every 250, logistic probe every 500. Seeds 1-3 (init 7100 + i, split 8100 + i) unless noted.

| arm | what changes | settings | runs | horizon |
|---|---|---|---|---|
| R1 | slower feature rate | frac 0.2, eta_w 0.03, lam_v 0.25, 0.3 | 6 | 40k |
| R2 | fewer training pairs | frac 0.15, eta_w 0.1, lam_v 0.3, 0.4, 0.5 | 9 | 40k |
| R3 | fewer still | frac 0.1, eta_w 0.1, lam_v 0.4, 0.5 | 6 | 60k |
| R4 | the pilot setting, more seeds | frac 0.2, eta_w 0.1, lam_v 0.25, seeds 3-5 | 3 | 30k |
| R5 | between the two pilots | frac 0.2, eta_w 0.1, lam_v 0.275 | 3 | 30k |
| N1 | the paper's contrast: NGD hidden steps | as R4, lam_v 0.25, 0.5 | 6 | 30k |
| S1 | p = 31, the paper's prime (exploratory) | frac 0.5, eta_w 0.1, lam_v 0.5, 1 | 6 | 30k |
| P1 | C1 in float64: are its CE spikes a float32 effect? | plain head, frac 0.3, lam_w 0.02, eta_w 0.1, lam_v 0.2, seeds 6-8 | 3 | 30k |

## Analysis (fixed now; `paperdefs_tf.py`, `analyze_W.py`; held-out accuracy as above)

Per run, with the paper's definitions:

1. **Fit.** T_fit = first log with every training margin > 0. Fit first: held-out accuracy <= 0.2 at T_fit. Fit kept: no
   logged margin <= 0 from T_fit through the first held-out 0.9 crossing.
2. **Flat.** The best window I on the 20-update log grid with raw-CE v(I) = max L / min L - 1 <= 0.10 at every update and every
   logged training margin in I > 0; its share q(I) of the held-out 10% -> 90% rise (the paper's clip formula).
3. **Readouts.** Ridge refit (the paper's canonical: tau = 1e-3 n, targets e_y - 1/p, the head's input) gain inside I; logistic
   probe gain; first 0.5 and 0.9 crossings of ridge, probe and network.
4. **A_H.** Delta A_H(I) of the head's input (and of the MLP activations, reported).
5. **Final.** Last logged held-out accuracy.

A run **meets the paper's definitions** if q(I) >= 0.75 with the strict fit, the ridge refit gains >= 0.2 inside I,
Delta A_H(I) >= 0.02 and the final held-out accuracy is >= 0.99. A setting is a **candidate** for a confirmation campaign on
10 fresh seeds if 3/3 of its runs meet them; among candidates, the one with the highest median CE / log p inside I. Also
reported, not deciding: the paper's post-hoc non-falling window (Appendix D.3; knee = first heavy log with ||W_U|| within 1% of
1/lam_v, since W_U starts above its cap here), the paper's deep primary endpoint, v from held-out 0.2 to 0.9, and the CE level.

N1 and P1 are descriptive: N1 against Polar on the same seeds (CE change over the rise, generalization time, fit); P1 against the
float32 C1 runs of the same seeds (V_C1 s6-s8): spike counts (updates with CE > 10x the median after the fit) and q(I).
