# Campaign X: Fourier readouts on the transformer settings closest to the paper (plan fixed before any run)

claude/1hl-figures, 2026-10-11, at Akash's request: "run a new campaign for transformers where we are trying to include these
computations for the different ridge refit accuracies and note how they really turn out against the held-out accuracies ...
highlight plots that can complement the current paper nicely ... plot the frequencies of the Fourier feature". Scope chosen by
Akash: R5 and R4 only, 10 seeds each, stop after generalizing as in campaign W. Development runs; not evidence for the paper's
claims.

## 1. Why

- In campaign W the paper's ridge refit trails the transformer's held-out accuracy. In two replays with parameter snapshots
  (`natural_plateau_shared/fourier_refit`, project doc claude/FOURIER_REFIT_20261011.md), restricting the refit's frequency sum
  to the 8 frequencies with the largest training energy makes it lead the network by 1.2-1.6k updates at held-out 0.5, and the
  network's own logits restricted the same way lead by as much: only 5-10 of the 56 frequencies carry held-out signal, the
  others carry the memorized training labels. In the paper's one-hidden-layer MLP every frequency carries held-out signal and
  the plain refit already leads.
- Campaign X measures this inside the runs (tf_runner 1.4) on 10 fresh seeds per setting, so the figures can show medians, and
  records the per-frequency spectra for the frequency figures.

## 2. Runs (20; `manifest.json` from `make_manifest.py`; runner `tf_runner.py` 1.4)

Campaign W's R5 and R4 configurations, unchanged except for the seeds and the logging: one block, d_model 128, 4 heads, MLP
512, RMS division before the head, float64, p = 113, training fraction 0.2, eta_w 0.1, lam_w 0.01, eta_v 0.03, lam_v 0.275
(R5) and 0.25 (R4); supported-Polar steps with decoupled decay on every matrix except W_U, normalized head step. Seeds 11-20
(init 7100 + i, split 8100 + i; campaigns T-W used 1-8). Light logs every 20 updates, heavy logs (A_H, ridge refits, Fourier
readouts) every 100, the logistic probe every 200. Early stop as in W: held-out accuracy >= 0.99 held for max(1.5 t, t + 5000)
updates after first reaching it at t; 30000 at most.

## 3. Readouts (tf_runner 1.4, `observers_fourier`; definitions in the runner's docstring)

With w_k = 2 pi k / p and K = 56, the paper's ridge refit of the head input is exactly (2/p) sum_k [C_k(x) cos(w_k c) +
S_k(x) sin(w_k c)], with C_k, S_k the ridge predictions of cos(w_k y), sin(w_k y). Recorded at every heavy log, held-out
accuracies on all held-out pairs and on the no-mirror pairs:
- `fref_all` (= the paper's refit), `fref_top{1,2,4,8,16}` (the sum over the top-K frequencies by training energy; train-only),
  `fphase_top8` (one equal vote per frequency), `fnet_top{4,8,16}` (the network's own logits restricted in class space to the
  top-K; no fitting), the probe (every 200);
- diagnostics using held-out labels: `fspec_hor2` (held-out R^2 per frequency), `fref_oracle`, `fnet_oracle`, and
  `ffam_{all,ho,tr}` (Janati et al.'s "family alone": the head input replaced by its mean over the pairs with the same a + b,
  over all, held-out or training pairs, decoded by W_U);
- spectra (56 numbers each): `fspec_train` (training energy), `fspec_grid` (full-grid A_H per frequency), `fspec_head`,
  `fspec_embed`; `ftop8`.

## 4. Analysis (fixed now; `analyze_X.py`, `plot_X.py`)

Per run:
1. Crossings of held-out accuracy (no-mirror pairs) 0.1, 0.5 and 0.9: the network on its 20-update log (first logged value
   at or above); every readout on its own log (100 updates; probe 200), linearly interpolated. Lead = network crossing minus
   readout crossing (positive: the readout crosses first).
2. Primary readout: the Fourier-filtered refit with K = 8 (`fref_top8`), fixed now; it was chosen in the exploratory replays.
   Secondary: K = 4 and 16, the paper's refit, the restricted logits (K = 8), the phase vote, the probe.
3. Frequencies at the heavy log nearest the network's 0.5 crossing: the number of frequencies with held-out R^2 > 0.05, how
   many of them are among the top 8 by training energy, and the overlap of that top 8 with the head's and the embedding's
   top 8.
4. The training loss over the band of the paper's Figure 2, adapted: from the filtered refit's onset (its last value below
   0.1 before crossing 0.5) to the network's 0.9 crossing: the change of the cycle-mean CE (relative), the slope (% per 1000
   updates), and the least training accuracy and most misclassified training pairs in the band. Campaign W's window test
   (raw CE within 10% at every update, every training pair correct at every log, >= 75% of the held-out 10% -> 90% rise
   inside), strict and with at most 10 misclassified pairs, reported as in W.
5. Janati diagnostic: crossings of `ffam_all`, `ffam_ho` and `ffam_tr` against the network.

Reported per setting (counts out of 10, medians and ranges; descriptive, no pass threshold):
- the filtered refit (K = 8) crosses held-out 0.5 (and 0.9) before the network;
- the paper's refit crosses 0.5 after the network;
- the restricted logits (K = 8) cross 0.5 before the network;
- at the network's 0.5 crossing at most 12 of the 56 frequencies carry held-out R^2 > 0.05, and at least 5 of the top 8 by
  training energy are among them;
- W's window test, strict and with the 10-pair tolerance.

Figures (medians over the 10 seeds, paper colour key):
- `X_main`: per setting, (a) training CE (cycle mean; seeds thin, median bold), (b) held-out accuracy of the network, the
  paper's refit, the Fourier-filtered refit (K = 8) and the restricted logits (K = 8), medians with the interquartile band,
  grey band from the median filtered-refit onset to the median network 0.9 crossing, (c) A_H of the head input.
- `X_freq`: for the seed with the median lead in each setting, the held-out R^2 per frequency over time with the top 8 by
  training energy marked, the spectra at the network's 0.5 crossing (training energy, head, embedding, held-out R^2), and
  the paper's MLP for contrast.
- `X_leads`: leads at 0.5 and 0.9 per seed and readout, R5, R4 and the MLP.
- `X_family`: the family-alone diagnostics against the network.

The MLP side (for the contrast panels; not part of the worker runs): the paper's one-hidden-layer configuration (campaign F:
p = 31, frac 0.8, width 512, seeds 3-10), trained by bias_runner.py unchanged (the archived runs replay exactly), with
tf_runner 1.4's `fourier_observers` on the last hidden layer every 100 updates (`mlp_side/`).

## 5. Compute

20 runs, about 14.5k updates each at about 0.2 s per update on one core: 18.5 core-hours with contention, on 5 sessions X1-X5
of 4 cores, about 1 hour each. Fewer sessions also work (the others take over unstarted IDs); they take longer.

## 6. Checks before launch

- tf_runner 1.4 with `observers_fourier` off reproduces 1.3 bit for bit (loss and every record) on three configurations
  (Polar float64 with the RMS head, Polar float32 with the plain head, AdamW with two blocks); with it on, the loss and every
  1.3 record are unchanged.
- On campaign-W replay snapshots (R5 seed 1, updates 2000, 4200, 5400) the 1.4 readouts equal `fourier_refit/refits.py`'s.
- `tests/test_tf.py`: 22 tests (16 from 1.3, 5 for the Fourier readouts, 1 for the manifest).
