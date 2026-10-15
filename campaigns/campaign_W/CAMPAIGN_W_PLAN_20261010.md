# Transformers under the paper's own definitions, and campaign W (claude/1hl-figures, 2026-10-10)

Request (Akash): clean up the three transformer issues (C1's near-zero CE, the exploratory high-loss case, the refit not
leading), find configurations, and "make sure the optimizer we are using for these experiments is same as the one used in paper
otherwise we need to clarify this". The training splits stay as they are.

## 1. Optimizer: the paper's rule, with differences to state

tf_runner applies the paper's update (eq. 3, Section 3.2) to every matrix except the head, one matrix at a time: supported Polar
divided by sqrt(rank), decay outside the normalization, the same 1e-7 relative cutoff as the deep runs; the head W_U takes the
normalized-gradient step with decay; simultaneous full-batch steps, no momentum. Differences:

- float32 in campaigns T, U, V (the paper: float64). Campaign W is float64.
- One (eta_w, lam_w) for every matrix (embeddings, position embedding, W_Q/W_K/W_V/W_O, W_in/W_out); the paper's deep nets use
  per-layer decays and a tuned last-layer rate. C1 uses eta_w 0.1 (the paper 0.03).
- Each attention matrix is one 128 x 128 matrix across the four heads.
- The paper's contrast optimizer is NGD feature updates (Appendix F), not AdamW. tf_runner 1.3 adds `hidden_mode: 'ngd'`
  (Polar updates and 1.2's observers bit-identical to 1.2; checked on 25 updates, float32 and float64, plain and RMS).
- The final RMS division (no parameters) is a model change, outside the paper's model.
- Mirror pairs: with a shared token embedding the transformer gets a held-out pair (a, b) right while memorizing when its
  mirror (b, a) is in the training set (at update 600, seed 1, RMS head: 0.58 on such pairs vs 0.003 on the rest at p = 113,
  frac 0.2; 1.00 vs 0.004 at p = 31, frac 0.5). This is the early held-out plateau (about 0.12-0.2) of the RMS runs; the
  paper's MLP (separate W_a, W_b) has no such floor. Runner 1.3 also logs the held-out accuracy on pairs whose mirror is held
  out too (`test_acc_nm`, `refit_test_acc_nm`, `probe_test_acc_nm`); campaign W's analysis and figures use it.
- W_U starts above its cap (norm about sqrt(p)), so the paper's knee (first log with ||V|| >= 0.99/lam_v) is update 0 here;
  `paperdefs_tf.knee_of` uses "first heavy log with ||W_U|| within 1% of 1/lam_v" instead.

## 2. The existing 196 runs, scored the paper's way (`paperdefs_tf.py`, `score_existing.py`, `existing_paperdefs.json`)

Campaigns T-V used campaign V's rule (smoothed CE, training accuracy >= 0.97). With the paper's definitions (raw CE within 10%
at every update; every training pair correct at every 20-update log), no setting passes:

- **C1** (15 runs): the training set stays fit, but the raw CE (about 3e-7 nats) varies about 4x (median 99th/1st percentile
  ratio) between held-out 10% and 90%, with single-update spikes of 30x to 7e6x the median in 10 of 15 runs (up to 0.8 nats).
  No 10% window captures any of the rise (best share 0.00-0.02). The other plain-head settings: same.
- **RMS head, frac 0.2, lam_v 0.5** (6 runs): a 10% window holds 92-99% of the rise at about 0.6 log p, but 0.2-7% of the
  training pairs stay misclassified throughout (negative minimum margin at 81-100% of logs from held-out 0.2 to 0.9).
- The paper's deep primary endpoint (loss-selected band around the post-knee CE peak) fails everywhere: after the knee the
  transformer's CE falls slowly instead of rising, so the band sits before the rise.
- AdamW: 0/38 (its CE cycles between 1e-7 and spikes, slingshot-like).
- Readouts: in no transformer run does the ridge refit lead the network by more than the 250-500-update log grid; in RMS runs it
  trails by 800-3400 updates while the logistic probe tracks or slightly leads. The ridge refit on the MLP activations (the
  literal "last hidden table") does not lead either, and in RMS runs it hardly rises.

Figure: `campaign_W/figs/existing_paperdefs.png` (raw CE on a log scale, misclassified training pairs, accuracies, A_H).

## 3. Local pilots (float64, runner 1.3, RMS head, one block, frac 0.2, eta_w 0.1, lam_w 0.01, 8000 updates)

| pilot | first full fit (held-out then) | held-out 0.2 / 0.5 / 0.9 | raw CE over held-out 0.2 -> 0.9 | best fit + 10% window | readouts at held-out 0.5 |
|---|---|---|---|---|---|
| lam_v 0.25, seed 1 | 860 (0.12); 0 misclassified from ~2200 | 3220 / 5060 / 6820 | 0.300 -> 0.270 log p, v 16% | 69% of the rise | ridge 1440 later; probe within one probe log (500) |
| lam_v 0.25, seed 2 | 500 (0.06) | 3960 / 6360 / not by 8000 (0.79 at 8000) | 0.295 -> 0.277 by 8000 | 65% (truncated) | ridge 1640 later; probe within one probe log |
| lam_v 0.3, seed 1 | 2020 (0.13), relapses | 3460 / 5260 / 6860 | 0.393 -> 0.368, v 7.6% | 10% (fit breaks); 92% without the fit | ridge 1240 later; probe within one probe log |
| NGD, lam_v 0.25, seed 1 | never (training accuracy about 0.92) | 2400 / 2740 / 4040 | 0.53 -> 0.40, v 37% | none | ridge 760 later; probe 260 later |

Two more pilots at p = 31 (frac 0.5, eta_w 0.1, 8000 updates): with lam_v 1 and 0.5 the training set is fit by update 120-500
with large margins and the CE is flat to 1% (0.45 and 0.113 log p), but held-out accuracy jumps to about 0.5 at once and then
only creeps (0.53-0.57 at 8000; A_H of the head input 0.56 -> 0.64). The jump is the swap symmetry: about half of the held-out
pairs have their mirror (b, a) in the training set, and with a shared token embedding the transformer gets those right while
memorizing (the same effect gives the 0.12-0.15 early held-out plateau at p = 113, frac 0.2, where 20% of mirrors are in the
training set). Campaign W's S1 runs these to 30k updates.

Reading: a looser head cap (lam_v 0.25) gives a clean fit before the rise at about 0.28 log p (1.3 nats, close to the paper's
1HL plateau of about 1.47 nats), but the CE drifts down about 15% across the rise; a tighter cap (0.3) keeps the CE within
10% but training pairs keep flipping. The paper's contrast, NGD, never fits the training set, generalizes during fitting and
its CE falls 37% across the rise. Figure: `campaign_W/figs/pilots_rms_frac0.2.png` (misclassified training pairs on a
symlog scale).

## 4. Campaign W (42 runs, all float64; plan fixed before any run: `campaign_W/PLAN.md`)

| arm | what changes | settings | runs |
|---|---|---|---|
| R1 | slower feature rate | frac 0.2, eta_w 0.03, lam_v 0.25, 0.3 | 6 |
| R2 | fewer training pairs | frac 0.15, eta_w 0.1, lam_v 0.3, 0.4, 0.5 | 9 |
| R3 | fewer still | frac 0.1, eta_w 0.1, lam_v 0.4, 0.5 | 6 |
| R4 | the pilot setting, more seeds | frac 0.2, eta_w 0.1, lam_v 0.25, seeds 3-5 | 3 |
| R5 | between the pilots | frac 0.2, eta_w 0.1, lam_v 0.275 | 3 |
| N1 | the paper's NGD contrast | as R4, lam_v 0.25 and 0.5 | 6 |
| S1 | p = 31 (exploratory) | frac 0.5, eta_w 0.1, lam_v 0.5, 1 | 6 |
| P1 | C1 in float64 (are the spikes a float32 effect?) | plain head, frac 0.3, seeds 6-8 | 3 |

A run meets the paper's definitions if a window with raw CE within 10% and every training pair correct at every log holds
>= 75% of the held-out 10% -> 90% rise, the ridge refit gains >= 0.2 inside it, Delta A_H (head input) >= 0.02 and the final
held-out accuracy is >= 0.99. A setting with 3/3 such runs is a candidate for a 10-fresh-seed confirmation (with NGD and AdamW
contrasts on the same seeds), which would give the paper's transformer figure (median and thin runs, `tf_figs.py`).

**Where.** Mac: `~/Downloads/modular arithmetic/natural_plateau_shared/campaign_W/` (code, plan, manifest, tests, analysis,
pilots, figures; the six registered files checked against HASHES.json after writing), `push_W.sh`, `START_PROMPTS_W.md`,
`collect_results.sh` (now also collects W). Launch: `bash push_W.sh`, then sessions W1-W10 (about 2 hours). Analysis:
`python3 analyze_W.py campaign_W/collected/results`.
