# Why the ridge refit trails the transformer, and why the Fourier refit is the right readout (claude/1hl-figures, 2026-10-14)

Akash, 2026-10-13: "the only thing is refit for almost all the cases comes after the held-out accuracy. Can you try to find
some perturbations where the refit at least aligns with the held-out or at least explain cleanly as to why the refit is
coming after the held-out accuracy and why the refit used here is not correct and we should consider the Fourier refit
instead."

Exploratory, development data only: the logs of all 352 transformer runs of campaigns V to Q (`scan_runs.py`), and six
replays with parameter snapshots every 200 updates (`replay2.py`, the campaigns' runners unchanged): the paper's Figure 5
setting (campaign Y's G2, seed 11) and Figure 15 setting (campaign Z's two-block cell, seed 11), campaign W's R5 (seed 1)
and R4 (seed 3), and the two settings where the plain refit comes first (campaign W's P1, seed 8; campaign V's E2, seed 1).
The replays reproduce the archived crossings (Figure 5 seed 11: network 0.5 at 5457 against 5480 archived, refit 7252
against 7400, Fourier refit 4090 against 4100). Held-out = the no-mirror pairs.

## 1. The explanation

1. **The refit's logits are an equal-weight sum over all frequencies.** The target e_y - 1/p has the same weight at every
   one of the K = (p-1)/2 frequencies, and ridge is linear in its targets, so (paper, revision 4: Appendix D.1, eq. (30))
   f_c(x) = (2/p) sum_k [C_k(x) cos(w_k c) + S_k(x) sin(w_k c)], with C_k, S_k the ridge predictions of cos(w_k y) and
   sin(w_k y). Every frequency gets weight 2/p whether or not the features carry it.
2. **The transformer's head input carries only about 8 frequencies at the rise.** At the network's 0.5 crossing, the
   median run of every paper setting has 8 or 9 frequencies with held-out R^2 above 0.05 (range 6 to 13), out of 30
   (p = 61) to 74 (p = 149); the top 8 by training energy contain all but 1 or 2 of them (`paper_cells_freq.json`).
3. **The ridge fits the other frequencies anyway, from memorized features, and they are noise on held-out pairs.** In the
   Figure 5 replay at the network's 0.5 crossing (update 5400), 6 of 56 frequencies are carried. For the other 48 by
   training energy, the ridge's predictions fit the training rows (training R^2 0.38), are as large on held-out rows as on
   training rows (mean amplitude 0.62 and 0.54, against 0.80 for the top 8 on held-out rows), and are worse than predicting
   zero there (held-out R^2 -0.42).
   Across the 10 runs of each paper setting, the held-out R^2 of the frequencies outside the top 8 is -0.27 to -0.75
   (medians).
4. **The features are as decodable for the refit as for the network; the refit adds noise because its target asks for
   it.** At the same update, the refit restricted to the top 8 frequencies is 0.72 accurate and the network restricted to
   them 0.77. But the other 48 frequencies hold 80% of the refit's held-out logit variance against 51% of the network's,
   and the full refit is at 0.18 while the network is at 0.48. A logistic probe of the same features looks like the
   network (0.46 of its variance on the top 8, accuracy 0.49). The Figure 15 replay (two blocks, seed 11) shows the same at
   its 0.5 crossing (update 6400): 9 of 56 frequencies carried; the other 48 fitted on the training rows (training R^2 0.53)
   and noise on held-out rows (held-out R^2 -0.49); top-8 part 0.96 for both the refit and the network; the other 48 hold
   78% of the refit's held-out logit variance against 51% of the network's; full refit 0.27 against the network's 0.48.
5. **Why later rather than never.** The memorized part forms early and fades slowly. In the R5 replay the other 48
   frequencies have training R^2 0.39 and held-out R^2 -0.47 at update 800 and still 0.32 and -0.20 at 8000, while the top-8
   part grows from held-out R^2 -0.39 to 0.73 and its accuracy alone tracks the network's top-8 part (0.44/0.41 at 4000,
   0.86/0.86 at 5600, 0.98/0.98 at 7200). The refit becomes accurate only when the clock frequencies outweigh a 48-term
   noise floor. The network, trained with CE, never has to reproduce the one-hot target: only the true class has to win,
   and its head moves its logit variance onto the clock frequencies as they grow (21% at update 800, 57% at 6400 in R5).
6. **The MLP is different.** In the paper's one-hidden-layer network (p = 31, replay of campaign F's run) all 15
   frequencies are carried at the network's 0.5 crossing (held-out R^2 at least 0.58, median 0.75), and the refit's top-8 part and
   its other 7 frequencies each classify alone (1.00 and 0.99). The one-hot target is reachable without memorization, so
   the plain refit leads (by 2.7k updates) and the Fourier filter changes little (+2.4k).

So the plain refit is not wrong; it answers a different question: can a ridge fitted on training rows reproduce the exact
one-hot label from these features? For a few-frequency code (the transformer's clock) that is true only once the
memorized part has faded, which is after the network generalizes. The paper's question, whether the features already
support the rule before the network's head uses it, is answered by the same ridge, on the same features and training rows
with the same penalty, with its target restricted to the frequencies the features carry; the training-energy ranking makes
that choice from training data only.

## 2. Perturbations

### 2.1 Readout (replays; lead at held-out 0.5, thousands of updates, positive = readout first)

| readout | Figure 5, seed 11 | Figure 15 (two blocks), seed 11 | R5 seed 1 | R4 seed 3 | MLP, p = 31 |
|---|---|---|---|---|---|
| paper's refit, tau = 1e-3 n | -1.79 | -0.54 | -1.33 | -1.62 | +2.75 |
| same, tau = 1e-1 n | -1.23 | -0.34 | -0.79 | -0.98 | +2.70 |
| same, tau = 10 n | -0.56 | -0.25 | -0.26 | -0.41 | -1.81 |
| same, tau -> infinity (class centroids) | -0.57 | -0.36 | -0.31 | -0.40 | never |
| Fourier refit, top 8 (tau = 1e-3 n) | +1.37 | +1.48 | +1.21 | +1.61 | +2.37 |
| logistic probe | +0.02 | +0.13 | +0.20 | +0.16 | - |

- A larger penalty shrinks the memorized directions (near-OLS ridge boosts weak directions to fit every frequency) and cuts
  the transformer lag by 50 to 80%, but the refit stays behind in every replay and never reaches 0.9 by update 8000 in the
  Figure 5 replay.
  The same penalty stops the MLP refit from generalizing at all: there the decodable Fourier directions are the weak ones
  (A_H of the hidden layer about 0.12). No single penalty works for both; the frequency filter does.
- The logistic probe (a CE target, like the network's) aligns with the network.
- How many frequencies the refit keeps (logged, 10 runs per paper setting; `refit_lag_dose`): the top 4, 8 or 16 lead in
  every setting at held-out 0.5, and keeping all of them lags in every setting. At p = 61 the top 8 is too few for 0.9
  (paper, revision 4: Table 9); the top 16 leads at both crossings.

### 2.2 Training (logs of 352 runs)

| family | runs with both crossings | plain refit first |
|---|---|---|
| RMS head with the decaying gain: the paper's recipe and its variations (lam_V, eta_W, lam_W, gain value, time constant or fixed, training fraction, p = 61 to 149, one and two blocks, NGD, AdamW) | 221 | 1 (a tie, +6 updates) |
| RMS head, no gain (campaigns V, W, X, Z) | 66 | 3, all in V's E2 (two blocks, lam_V 0.5) |
| plain head, no RMS step (campaigns V C1, C2, W P1) | 19 | 12, by +0.01k to +0.21k |

The two exceptions fit the same mechanism (replays, `refit_lag_cases`):

- **Plain head** (W P1, seed 8; fraction 0.3, lam_W 0.02, training CE falls to about 1e-7): at the network's 0.5 crossing the
  features carry 19 of 56 frequencies, and the others are close to harmless (held-out R^2 -0.04, training R^2 0.23). The
  one-hot target is nearly reachable, and the plain refit leads by 0.15k (probe 0.00k; the top 8 lead at 0.5 by 0.26k but
  are too few for 0.9). Outside the paper's regime: there is no positive CE plateau.
- **Two blocks, head decay 0.5, no gain** (V E2, seed 1): the network lags its own features. A logistic probe leads it by
  1.26k, and the network's top-8 part is 0.88 accurate while its full logits are at 0.50. Every readout leads (plain refit
  +1.14k, Fourier refit +3.71k). The runs are unstable later and fail the pass rule.

Within the paper's regime no perturbation makes the plain refit align: the gap is set by the representation (few
frequencies plus a memorized part), which every variant of the recipe keeps.

## 3. Suggested text for the paper (not yet in the draft)

Main text, Section 5.3, after the readout paragraph:

> The cause is the refit's target, not the features: the one-hot label has equal weight on all 56 frequencies, while at
> the network's $50\%$ crossing the head input carries only about 8 of them; on held-out pairs the ridge's predictions at
> the others are noise, and the refit restricted to the top 8 is about as accurate as the network restricted to them
> (Appendix~\ref{paper:app:tf-readouts}).

Appendix D.5 (`paper:app:tf-readouts`), after "...carry held-out signal in any run.":

> **Why the refit trails.** By \eqref{paper:fourier-refit} the refit's logits sum the ridge's predictions of
> $(\cos\omega_k y,\sin\omega_k y)$ with equal weight over all $K$ frequencies. At the network's $50\%$ crossing, $\hat r$
> carries 8 or 9 of them in the median run of each setting (held-out $R^2$ above $0.05$), and the top 8 by training energy
> contain all but one or two. The ridge fits the other frequencies on the training rows as well, from features that encode
> the training labels, and on held-out pairs those predictions are noise (median held-out $R^2$ between $-0.75$ and
> $-0.27$). In a replay of a Figure 5 run at the network's $50\%$ crossing, the refit restricted to the top 8 is $0.72$
> accurate and the network restricted to them $0.77$, but the other frequencies hold $80\%$ of the refit's held-out logit
> variance against $51\%$ of the network's, and the full refit is at $0.18$ against the network's $0.48$. The refit thus asks
> the features to reproduce the one-hot target at every frequency, which a few-frequency code does only after its memorized
> part fades; the network, trained with CE, needs only the true class to win. In the one-hidden-layer networks every
> frequency carries held-out signal (all 15 at the network's $50\%$ crossing at $p=31$), so the target is reachable and the
> refit leads. A larger ridge penalty is no substitute: with $\tau_t=10\,n$ the transformer refit's lag at the $50\%$
> crossing falls from $1.8$k to $0.6$k updates, while the same penalty keeps the one-hidden-layer refit from generalizing.

## 4. Files

`natural_plateau_shared/refit_lag/` (Mac): this note; `scan_runs.py` and `scan_runs.json` (352 runs), `summarize.py`;
`replay2.py`, `cfgs/` (the four new replays' configurations; R5, R4 and the MLP are in `fourier_refit/cfgs`), `replays/` (their
loss and light logs); `lag_tau.py` (per snapshot: the refit at 7 penalties, the centroid limit,
the Fourier refit, the probe, per-frequency diagnostics and the logit decomposition) and its outputs `lag_*.json`;
`analyze_lag.py` (`analysis_lag.json`), `paper_cells_freq.json`; `plot_lag.py`, `plot_cases.py` and `figs/`
(`refit_lag_mechanism`, `refit_lag_dose`, `refit_lag_cases`). The snapshots (70 to 190 MB per run) stay in the cloud
workspace; `replay2.py` regenerates them. Earlier work: `fourier_refit/README.md` (2026-10-11).

## 5. Figures (`figs/`)

- `refit_lag_mechanism`: the Figure 5 replay. (a) held-out accuracy of the network and four train-only readouts of the same
  features; (b) held-out R^2 of the refit's per-frequency predictions at the network's 0.5 crossing, frequencies ranked by
  training energy (top 8 blue); (c) at the same update, the share of each readout's held-out logit variance on the top 8,
  with the accuracy of the top-8 part and of the full logits.
- `refit_lag_dose`: (a) lead at held-out 0.5 against the number of frequencies the refit keeps, 10 runs per paper setting
  (medians and quartiles, from the runners' logs); (b) lead against the ridge penalty for the transformer replays and the
  MLP replay.
- `refit_lag_cases`: the Figure 5 and Figure 15 settings and the two settings where the plain refit comes first, one replay
  each: accuracy curves with the leads, and the logit-variance split at the network's 0.5 crossing.

The three PNG files gained a content-credentials (C2PA `caBX`) chunk when they were written to the Mac; their pixels are
unchanged, and SHA256SUMS lists the Mac copies.

## 6. Follow-up (2026-10-14): MLPs at larger p, and how many frequencies to keep

Akash, 2026-10-14: "would the problem persist even for MLPs for higher p values, i.e., the ridge refit will lag behind?" and
"do we have an idea how many Fourier features to fit, 8 or more? Is there an easy way to guess? Does the answer depend on
the current count of the Fourier features to recreate the phenomena?"

Data: the logs of 870 MLP runs (`mlp/scan_mlp.py`; p = 23 to 101, depth 1 to 4, training fraction 0.5 to 0.8), the logged
spectra of the 260 transformer runs with the Fourier observers (`k_rules.py`; campaigns X to Q), two new MLP replays (the
paper's one-hidden-layer setting, lambda_W = 0.02, at p = 61 seed 1 and p = 101 seed 7, snapshots every 200 updates) and the
seven earlier replays. `k_curve.py` scores, per snapshot, the top-K Fourier refit for every K; `k_val.py` picks the
frequencies on a validation split of the held-out pairs. The MLP replays reproduce the archived crossings (p = 61: network
0.5 at 6592 against 6593 archived, refit 4698 against 4683; p = 101: 6295 against 6289, refit 4892 against 4792 from the
archive's 1000-update logs).

### 6.1 MLPs: no lag at any p we have

- The paper's refit crosses held-out 0.5 before the network in 839 of the 840 MLP runs where both cross (the exception: one
  depth-4 run at p = 61, 68 updates behind at a 500-update log) and 0.9 first in 820 of 826.
- Mechanism: the MLP's features carry every frequency before the network rises, and all frequencies rise together. At
  p = 101 the held-out R^2 of the refit's 50 frequency predictions moves up as a block: 7 of 50 above 0.05 at update 4800, 41
  at 5200, all 50 at 5600; the network crosses 0.5 at 6295. Same at p = 61 (all 30 by 5200; network 6592) and p = 31 (all 15).
  There is no memorized remainder for the refit to weight, so its lead grows with K and is largest with every frequency
  (lead at 0.5: p = 61, K = 8 +1.3k, K = 16 +1.7k, all +1.9k; p = 101, K = 8 +0.7k, K = 16 +1.1k, all +1.4k).
- The transformer differs because a few frequencies carry held-out R^2 of 0.5 to 0.8 while the others stay memorized (held-out
  R^2 near -0.5) through the rise. The training fraction is not the reason: the transformer at p = 61 with fraction 0.55
  still lags (-0.7k, 11 runs), and the depth-3/4 MLPs at fraction 0.5 lead.
- One trend: at fixed weight decay (lambda_W = 0.02) the one-hidden-layer lead shrinks with p, from 0.40 of the network's
  crossing time at p = 23 to 0.24 at p = 101 (3.1k to 1.5k updates); at lambda_W = 0.005 it is flat (0.24 to 0.30). All
  frequencies are carried at p = 101, so this is not the transformer's mechanism.

### 6.2 How many frequencies

- **The right K is the number of frequencies the features carry when you read them.** It grows during training, at about the
  same values for every p: at the network's held-out 0.1 / 0.5 / 0.9 crossings, median 3 / 7 / 16 at p = 61, 3 / 9 / 13 at
  89, 5 / 9 / 12 at 113, 5 / 9 / 16 at 113 with two blocks, 5 / 9 / 13 at 149 (260 runs; 22 to 29 at the end of one-block
  runs). In the replays the best K for the 0.5 lead is 4 to 6 (Figure 5, Figure 15, R5, R4 and V's E2 replays), about the
  count when the top-8 refit itself crosses 0.5 (4 to 6, 1.2 to 3.8k updates before the network); in the plain-head run the
  features carry 19 frequencies at the network's crossing and the best K is 20; in the MLPs it is all of them.
- **Why 8 works, and when 16 is safer.** Median leads over the network (thousands of updates, held-out 0.5 / 0.9):

  | setting (runs) | K = 4 | K = 8 | K = 16 | all (paper) | oracle set |
  |---|---|---|---|---|---|
  | p = 61 (37) | -1.6 / never* | +0.9 / -2.2 | +0.9 / +0.7 | -0.7 / -0.7 | +0.7 / +0.8 |
  | p = 89 (25) | +0.3 / -0.8 | +0.6 / +0.4 | +0.4 / +0.3 | -0.8 / -1.1 | +0.6 / +0.5 |
  | p = 113 (133) | +0.6 / -0.3 | +0.7 / +0.8 | +0.5 / +0.5 | -1.1 / -1.9 | +0.8 / +0.8 |
  | p = 113, two blocks (40) | +1.2 / +1.3 | +1.2 / +2.3 | +0.9 / +1.6 | -0.5 / -0.5 | +1.2 / +2.4 |
  | p = 149 (25) | +0.5 / -0.1 | +0.8 / +1.1 | +0.5 / +0.5 | -1.2 / -1.6 | +0.8 / +1.0 |

  (*2 of 37 cross.) K = 8 matches the oracle (frequencies with positive held-out R^2) at 0.5 in every setting. At 0.9, K = 8
  fails at p = 61, where 16 frequencies are carried by then; K = 16 leads everywhere but by less at 0.5. K <= 2 lags everywhere.
  In the replays the accuracy is flat near the best K: the lead at 0.5 stays within 200 updates of the best for K = 4-5
  (Figure 5 replay), 4-9 (Figure 15), 5-10 (R5), 3-10 (R4).
- **Guessing K from the training rows does not work.** The order is easy: by training energy, the carried frequencies are
  the top ones (at p >= 89 the smallest top-n holding all of them equals their count). The count is not: once the training set
  is fit, every training row carries its label at every frequency, so cross-validation on training rows and the permutation
  null pick nearly all frequencies (`fourier_refit` section 6). Thresholding the energy spectrum (frequencies above twice the
  uniform share) counts 8 at p = 113 and 9 at p = 149 (9 carried) but 6 at p = 89 (9), 7 with two blocks (9) and 0 at p = 61
  (7), where memorization flattens the spectrum (the top frequency holds 1.3 times the uniform share). The unembedding's
  spectrum gives the same counts; the embedding's is still flat at the rise (count 0 except at p = 149).
- **A validation split does work.** Taking the frequencies whose R^2 on 10% of the held-out pairs exceeds 0.05 (816 pairs at
  p = 113, 75 at p = 61) and scoring on the other 90%: it matches the oracle set (within 0.1k of its lead) in all six
  replays tested and adapts by itself (7 to 10 frequencies at the network's 0.5 crossing in the transformer, all 30 or 50 in
  the MLP). Leads at 0.5 / 0.9: Figure 5 replay +1.6 / +0.7 (top 8: +1.4 / +0.6), Figure 15 +1.5 / +2.3 (+1.5 / +2.3), R5
  +1.2 / +0.8, R4 +1.8 / +1.6, MLP p = 61 +1.7 / +3.0 (all: +1.9 / +3.0), MLP p = 101 +1.2 / +1.7 (all: +1.4 / +1.8). It
  uses held-out labels (one choice per snapshot), so it is a model-selection readout, not a train-only one.
- **Recommendation.** Fixed K = 8 for the train-only readout at the rise (it was picked on one snapshot of campaign W's R5;
  the 260 runs above came later and confirm it), K = 16 if the readout must also reach 0.9 first at small p, all frequencies
  for the MLP; the validation-split rule when one adaptive definition is wanted for both architectures.

Files: `k_curve.py`, `k_summary.py` (`kcurve/summary.json`), `k_val.py`, `k_val_summary.py` (`kcurve/val_summary.json`),
`k_rules.py` (`k_rules.json`), `mlp/scan_mlp.py` (`mlp/scan_mlp.json`), `mlp/H_p61_s1.json` and `mlp/H_p101_s7.json` (the
replays' configurations; `fourier_refit/replay.py mlp CFG OUT 9000 200`), `kcurve/*.json` (per-snapshot curves), `plot_k.py`
and `figs/refit_k_choice.{png,pdf}`: (a) share of frequencies with held-out signal against updates from the network's 0.5
crossing; (b) lead at held-out 0.5 against K, dot = all frequencies; (c) carried count at the network's crossings, all 260
transformer runs.
