# Fourier-structured refits for transformers and MLPs (claude/1hl-figures, 2026-10-11)

Akash: "to be consistent with what we have for the MLP, we need some version of ridge refit where we are using the Fourier
features in a certain way. We can try a few different ideas there." Exploratory; development runs only, one run per setting;
not evidence for the paper's claims.

## 1. Premise

The paper's refit (eq. paper:expridge; bias_runner `v1_ridge_acc_te`) uses no Fourier features. It is a ridge on the raw
last hidden layer: no intercept, targets e_y - 1/p, tau = 1e-3 n. The transformer refit (tf_runner 1.3 `refit_*`) is the
same formula on the head input. Fourier structure enters the MLP pipeline only through the observers (A_H, AGOP) and the
theory's analysis of that ridge.

## 2. The refit as a sum over frequencies, and the readouts tried

Since e_y - 1/p = (2/p) sum_k cos(w_k (y - c)) over k = 1..(p-1)/2 (as a function of the class c), the paper's refit is
exactly

    f_c(x) = (2/p) sum_k [C_k(x) cos(w_k c) + S_k(x) sin(w_k c)],

with C_k, S_k the ridge predictions (same features, same tau) of cos(w_k y) and sin(w_k y). Every frequency gets the same
weight, whether or not the features carry it (`test_refits.test_identity` checks the identity to 1e-10). The readouts
(`refits.py`) change only the frequency sum:

| readout | definition | uses |
|---|---|---|
| ridge | the paper's refit (all frequencies) | training rows |
| filt_top{K} | the sum over the top K frequencies by training addition-Fourier energy, K fixed (2, 4, 8, 16) | training rows |
| phase_top{K} | the same, each (C_k, S_k) normalized to a unit vector (one equal vote per frequency) | training rows |
| filt_sig | frequencies above a 95% label-permutation null of the training energy | training rows |
| filt_cv | top K, K by 5-fold CV accuracy on the training rows (folds keep (a, b) and (b, a) together) | training rows |
| proj_top{K} | features projected onto the training class-mean Fourier directions of the top K, then the paper's refit | training rows |
| ridge_cv | the paper's refit with tau by the same CV (control for plain regularization) | training rows |
| probe | multinomial logistic readout (tf_runner's probe_lr) | training rows |
| net_filt_top{K} | the network's own logits restricted (in class space) to the same top-K frequencies; no fitting | weights only |
| oracle_filt | the refit's sum over the frequencies whose held-out R^2 > 0 | held-out labels (diagnostic only) |

Train-only readouts never see held-out labels (`test_no_leak`, and the same check on a real transformer snapshot). On a
snapshot the plain readout here equals the runners' own refit exactly (`test_matches_tf_runner`, `test_matches_bias_runner`).

## 3. Runs

`replay.py` reruns a configuration with the campaigns' runners, imported unchanged (tf_runner.py sha256 df9c53ae...,
bias_runner.py 16bb2ebd...), saving the parameters every 200 updates (100 for the MLP); `score.py` scores every readout at
every snapshot (every 200 updates); `analyze.py` gives the first crossings of held-out 0.5 and 0.9 (readouts interpolated
between snapshots; network from its 20-update log; transformer: the no-mirror held-out pairs).

- MLP: the paper's one-hidden-layer configuration (campaign F H_p31, seed 10: p = 31, width 512, frac 0.8). The replay is
  identical to campaign F's archived run (loss to the stored precision).
- Transformers: campaign W's R5 (frac 0.2, eta_w 0.1, lam_v 0.275, RMS head, float64) seed 1 to 8000 updates, and R4
  (lam_v 0.25) seed 3 to 8600. The replays match the archived runs to the stored precision up to update 3320 (R5) and 2896
  (R4), then drift (torch 2.5.1 here, 2.14.1 on the sessions); the network's held-out 0.5 and 0.9 crossings are within
  40-140 updates of the archive's.

## 4. Results: lead over the network at held-out 0.5 / 0.9 (updates; positive = the readout crosses first)

| | MLP, 1HL, p = 31 | transformer R5, seed 1 | transformer R4, seed 3 |
|---|---|---|---|
| network crosses 0.5 / 0.9 at | 8212 / 10514 | 5447 / 6834 | 5830 / 7677 |
| paper refit (all frequencies) | +2747 / +4252 | -1331 / not by 8000 (0.77) | -1615 / not by 8600 (0.80) |
| Fourier-filtered refit, top 8 | +2368 / +3811 | **+1208 / +822** | **+1606 / +1446** |
| Fourier-filtered refit, top 16 | (15 frequencies: = paper refit) | +717 / +637 | +883 / +855 |
| Fourier-filtered refit, top 4 | +2030 / +3022 | +946 / caps at 0.81 | +1814 / caps at 0.81 |
| phase vote, top 8 | +2244 / +3622 | +969 / +384 | +1331 / +1090 |
| CV-chosen K | +2747 / +4018 | +545 / +919 | +1262 / +1253 |
| logistic probe | +1256 / +2566 | +197 / +314 | +164 / +314 |
| network's logits restricted to top 8 | -1716 / never (0.53) | +1160 / +809 | +1638 / +1112 |
| oracle filter (diagnostic) | +2448 / +4252 | +1253 / +754 | +1812 / +1541 |

Frequencies with held-out signal (held-out R^2 of the refit's prediction of e^{i w_k y} above 0.05): MLP, all 15 of 15
already at update 7000 and at the network's 0.5 crossing; R5, 6 of 56 at update 4200 and 7 at 5400 (6 and 7 of them among
the top 8 by training energy); R4, 5 at 4600 and 10 at 5800 (5 and 8 among the top 8).

Figure: `figs/fourier_refit_runs.png` (top: held-out accuracy of the network and readouts; bottom: held-out R^2 per
frequency, with the top-8 training-energy frequencies marked).

## 5. Reading

- In the transformer the paper's refit lags because it weights all 56 frequencies equally while only 5-10 carry held-out
  signal at the rise; the others carry the memorized training labels. Keeping the top 8 frequencies by training energy (a
  train-only choice, nearly the oracle set) removes them, and the refit then leads the network by 1.2-1.6k updates at
  held-out 0.5 and 0.8-1.4k at 0.9.
- The network's own logits restricted to the same 8 frequencies lead by the same amount. So the generalizing Fourier circuit
  is in place 1-2k updates before the network's held-out accuracy rises; the rise is the decay of the remainder at the other
  frequencies (Nanda et al.'s restricted-loss picture).
- In the paper's one-hidden-layer MLP every frequency carries held-out signal, so the filter changes little (top 8 leads by
  2.4k against 2.7k for the paper's refit) and restricting the network's logits hurts it.
- One definition, the paper's ridge refit with its frequency sum restricted to the top K frequencies by training energy,
  K = 8, gives a refit-first chronology in both architectures. K = 8-16 works in both transformer runs; K = 4 is too few there
  (caps at 0.81).
- The logistic probe tracks the network in the transformer (as in campaign W) and trails the ridge in the MLP.

## 6. What does not work

- Choosing frequencies from training data by significance or by cross-validation: once the training set is fit, every
  training pair's features carry its own label at every frequency. The permutation null flags all 56 (all 15) frequencies
  (filt_sig = paper refit), and CV accuracy on training folds is near 1 for any K (about 0.999 with all 56 at update 4200
  of R5), so CV picks K of about 20-50 before the rise and becomes informative (K of about 5-7) only once the memorized part
  fades. Hence K is fixed in advance.
- The class-mean projection (proj) tracks or trails the network; the CV-chosen ridge penalty gains nothing in the transformer.

## 7. Caveats

- One run per setting. K = 8 as the headline was chosen after seeing the first scored transformer snapshot (R5, update
  4200); a confirmation should fix K and the readout before new runs.
- Deep MLPs (the paper's depth runs, campaigns K and M) not checked yet; they need snapshots or the readout as a runner
  observer.

## 8. Files

`refits.py` (readouts), `replay.py`, `score.py`, `analyze.py`, `plot_refits.py`, `test_refits.py` (all pass),
`tf_runner.py` and `bias_runner.py` (unchanged copies), `cfgs/`, `scores/*.json` (per-snapshot scores and the logs),
`scores/analysis.json` (crossings), `figs/fourier_refit_runs.png`. The parameter snapshots (about 70 MB per transformer run)
are not included; `replay.py` regenerates them.
