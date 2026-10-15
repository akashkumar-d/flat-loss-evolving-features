# Campaign Z: optimizer contrasts with the gain, the one-block recipe, other p, two blocks (plan fixed before any run)

claude/1hl-figures, 2026-10-12, at Akash's request after campaign Y: run the contrasts (NGD, and AdamW) with the same gain on
seeds 11-20, and also (i) make clear which held-out accuracy is plotted (all pairs outside the training set, or the no-mirror
subset), (ii) one more block, or a clear recipe for one block that gives these plots, (iii) more p for the appendix; "the next
campaign should include a thorough understanding for all these things". Development runs; not evidence for the paper's claims
until confirmed.

## 1. Where campaign Y left things

Campaign Y (one block, p = 113, frac 0.2, RMS head input with a scalar gain g that relaxes from 2 or 3 to 1 under weight
decay; the paper's update rule): in 20/20 runs the training CE rises across the whole held-out rise (+33% with g0 = 2, +111%
with g0 = 3; without the gain, campaign X on the same seeds, -8%), the paper's window holds the whole rise, no training pair is
misclassified in it, A_H of the head input rises 0.07 -> 0.88, the Fourier-filtered refit leads in 20/20, the paper's refit
trails in 20/20, and the timing of the rise is that of the no-gain runs.

Open: the contrast optimizers with the gain; what in the setting is needed (the recipe); whether it carries to other p and to
two blocks; and the held-out definition.

## 2. Held-out accuracy: two definitions (both used everywhere from now on)

The runner logs, for the network and for every readout, the accuracy on
- **all held-out pairs**: every (a, b) outside the training set (10,216 of 12,769 at p = 113, frac 0.2); and
- **the no-mirror subset**: the held-out pairs whose mirror (b, a) is also held out (about 8.2k; `*_nm`).
The other held-out pairs (about 2.0k, "mirror in training") can be answered by memorizing (b, a): the transformer reads a and b
through one shared token embedding. Campaigns W-Y plotted and scored the no-mirror subset. On campaign X and Y's runs
(`prep/heldout_defs.py`, `prep/heldout_defs.png`): at the first full fit the mirror-in-training pairs are 20-63% correct and
the no-mirror pairs 0-1%, which gives the all-pairs curve an early step to about 0.1; then both curves rise together and end
at >= 0.998. With the gain, every conclusion holds on all pairs too (G2 / G3: the paper's window holds 0.93-1.00 / 0.90-1.00
of the rise; the Fourier-filtered refit leads in 10/10, +0.74k / +0.64k at 0.5; the paper's refit trails, -1.2k / -1.0k). The
share of held-out pairs whose mirror is in training is about frac, so the all-pairs step grows with frac (about 0.4 at
frac 0.4: pilot, section 8).

In campaign Z every crossing, lead, window and verdict is computed on both definitions, every accuracy panel shows both
(solid: all held-out pairs; dashed: no-mirror), and the analysis reports the mirror-in-training pairs at the first fit.

## 3. Runs (manifest.json from make_manifest.py; runner tf_runner.py 1.6; seeds as in X and Y: init 7100 + i, split 8100 + i)

Base = campaign Y's G2: one block, d_model 128, 4 heads, MLP 512, causal attention, no LayerNorm or biases in the block, final
RMS division with the gain g (g0 2, eta_g 2e-4, lam_g 1: time constant 5000 updates), float64, p = 113, frac 0.2, full batch;
supported-Polar feature steps (every matrix but W_U; eta_w 0.1, lam_w 0.01), normalized head step (eta_v 0.03, lam_v 0.275),
sign step for g; Fourier readouts (heavy log every 100), probe every 200; early stop once held-out accuracy (all pairs) >= 0.99
has held for max(1.5 t, t + 5000) updates; 30000 updates at most.

### 3a. Part C: optimizer contrasts (the user's request), same seeds as X and Y

| cell | change from the base | seeds |
|---|---|---|
| C_NGD_G2 | feature matrices take the Frobenius-normalized gradient step (the paper's Appendix F contrast), same eta_w, lam_w | 11-20 |
| C_ADAMW_G2 | AdamW on every matrix including W_U (lr 1e-3, wd 1.0, betas 0.9 / 0.98: the usual grokking baseline); g keeps its sign rule | 11-20 |
| C_NGD_NOGAIN | NGD, no gain | 11-15 |
| C_ADAMW_NOGAIN | AdamW, no gain | 11-15 |

With campaign Y's G2 (Polar with the gain) and campaign X's R5 (Polar without it) this is the 3 x 2 table optimizer x gain on
the same initializations and splits.

### 3b. Part R: the one-block recipe (what is needed and how much), seeds 11-13

The mechanism to test. With the RMS division, the logit scale is g ||W_U|| (times an alignment); W_U reaches its norm
equilibrium within about 1 / (eta_v lam_v) = 120 updates. After the training set is fit, two things move the training CE:
the features improve (margins grow, CE falls) and the gain decays (logit scale shrinks, CE rises). The CE rises over the held-out
rise when the gain's relative decay rate, (g - 1) / (g tau), exceeds the margins' relative growth rate throughout the rise. So:
a fixed gain should behave like no gain (CE falls); a time constant much shorter than the rise brings g to about 1 before the
rise ends (CE falls late in the rise); a much longer one decays too slowly; a larger g0 widens the band that works.

| cells | what changes | runs |
|---|---|---|
| R_G{1.5, 2, 3}_tau{2.5k, 5k, 10k, 20k} | g0 and the time constant (eta_g = 1 / tau, lam_g = 1); G2 tau 5k is campaign Y's | 11 cells x 3 |
| R_G{1.5, 2, 3}_fixed | the gain held at g0 (eta_g = 0): the mechanism's control | 3 cells x 3 |
| R_lam_v0.25, R_lam_v0.3 | head decay (the cap of the logit scale; the base is 0.275) | 2 x 3 |
| R_eta_w0.05, R_eta_w0.2 | feature rate (base 0.1) | 2 x 3 |
| R_lam_w0.005, R_lam_w0.02 | feature decay (base 0.01; in campaigns T-U lam_w >= 0.03 stalled or did not learn) | 2 x 3 |
| R_frac0.3 | training fraction (base 0.2; 0.15 and 0.1 did not generalize in campaign W) | 1 x 3 |

### 3c. Part P: other p (appendix), one block, the recipe (g0 2, tau 5k), seeds 11-15

lam_v sets the cap of the logit scale and so the CE level; for other p it is rescaled so that the margin, which grows like
||W_U|| sqrt(d / p), keeps pace with log p: lam_v(p) = 0.275 sqrt(113) ln 113 / (sqrt(p) ln p) (0.43 at p = 61, 0.33 at 89,
0.23 at 149), with a second, larger value per p because the p = 61 pilot came out at about half of p = 113's CE level (section 8).

| cell | p | frac | lam_v | training pairs |
|---|---|---|---|---|
| P_p61_f0.3_lam_v0.45, _lam_v0.6 | 61 | 0.3 | 0.45, 0.6 | 1,116 |
| P_p89_f0.2_lam_v0.33, _lam_v0.4 | 89 | 0.2 | 0.33, 0.4 | 1,584 |
| P_p149_f0.2_lam_v0.23, _lam_v0.275 | 149 | 0.2 | 0.23, 0.275 | 4,440 |

(p = 113, frac 0.2, lam_v 0.275 is campaign Y's G2.) frac 0.3 at p = 61: at 0.4 the held-out rise starts at about 550
updates, before the CE's dip (about 800), and 40% of the held-out pairs have their mirror in training (pilot).

### 3d. Part B: two blocks, seeds 11-15 (two threads per run)

Earlier two-block runs (campaigns T-V, float32, no gain): with the RMS head at frac 0.3 and lam_v 0.5 the held-out accuracy
rose fast (0.5 at 1.1-1.5k) while the training set was being fit; at frac 0.2 and lam_v 0.5 the training set was fit at 3-5k and
the held-out accuracy reached 0.5 at about 10k but only 0.92-0.95 by 40k; at frac 0.2 and lam_v 1, no generalization in 40k.
With the gain the training set is fit much sooner (one block: 200 updates against 1,190), so frac 0.3 may now fit first.

| cell | frac | lam_v | gain (g0 2) | why |
|---|---|---|---|---|
| B_f0.3_lam_v0.5_tau5k | 0.3 | 0.5 | time constant 5k | campaign U's two-block setting, where the rise came while fitting; the gain fits sooner |
| B_f0.3_lam_v0.275_tau5k | 0.3 | 0.275 | 5k | campaign Y's head decay |
| B_f0.25_lam_v0.4_tau10k | 0.25 | 0.4 | 10k | in between |
| B_f0.2_lam_v0.275_tau10k | 0.2 | 0.275 | 10k | campaign Y's setting; the rise came at about 10k in campaign V, so a slower gain |

Every B run: up to 30000 updates, two threads. Two blocks cost about five times one block per update (block 1 runs at all three
positions), so this part is about half of the campaign's compute. Seeds 14 and 15 of every B cell were added on 2026-10-12,
after the first sessions had started and before any result, when Akash added sessions (section 7): each B cell then has 5
seeds, so a passing cell is usable without a second confirmation.

## 4. Runner 1.6

1.5 plus one change: with optimizer 'adamw' the gain is left out of AdamW and takes its own sign rule (as with 'polar' and
'ngd'), so the gain relaxes the same way under every optimizer. Every run without a gain, and every 'polar' or 'ngd' run with
one, is 1.5's update for update (checked bit for bit on 40 updates of six configurations, and in the test suite when campaign Y's
runner is present). The observers are 1.5's: both held-out definitions for the network and every readout; A_H of each block's
MLP neurons, of the head input and of the logits.

## 5. Analysis (fixed now; analyze_Z.py)

Per run, on each held-out definition (all, nm):
1. Crossings of 0.1 / 0.5 / 0.9 (linear interpolation between logs) of the network and the readouts (primary: the
   Fourier-filtered refit, top 8 by training energy; also the paper's refit, the network's logits restricted to the top 8, the
   probe); leads = network crossing minus readout crossing.
2. The paper's post-hoc window on the training CE averaged over 201 updates: after the first full fit, the stretch where the CE
   never falls more than 3% below its running maximum, chosen to hold the largest share q_te of the held-out 10% -> 90% rise
   (the paper's clip formula). Inside it: the gain of the Fourier-filtered refit and of the paper's refit, Delta A_H of the head
   input, the most misclassified training pairs.
3. ok = q_te >= 0.75, Fourier-filtered refit gain >= 0.2, Delta A_H >= 0.02, and at most 10 misclassified training pairs at
   every log inside the window (strict: none).
Per run: **pass = ok on both definitions and final held-out accuracy (all pairs) >= 0.99**; also reported per definition. The loss
across the rise (the no-mirror 0.1 -> 0.9 crossings): the CE's relative change, its largest drop below its running maximum, the
CE at the 0.5 crossing (nats and / log p). At the first full fit: held-out accuracy on all, no-mirror and mirror-in-training
pairs. A_H at the end (head input and each block's neurons); the gain at the crossings.

Per cell: counts (pass strict and loose), medians and ranges. Reading (recipe): a cell "works" when >= 2 of 3 seeds pass
strictly; for P and B (5 seeds each) a setting is a candidate for the paper when all its seeds pass.

## 6. Figures (plot_Z.py; medians with interquartile bands, seeds thin for cells of 3; paper colour key)

- `Z_contrast_gain`: Polar (campaign Y's G2) / NGD / AdamW with the gain; rows: training CE with the median window, accuracies
  (network and Fourier-filtered refit on both definitions, the paper's refit, training accuracy), A_H, the gain.
- `Z_contrast_nogain`: the same without the gain (Polar = campaign X's R5).
- `Z_recipe`: the CE change across the rise and the share of the rise in the window over (g0, time constant), with passing
  seeds; the one-factor cells.
- `Z_p`: one column per p cell (CE / log p), with campaign Y's G2 at p = 113.
- `Z_blocks`: one column per two-block cell, with A_H of each block's neurons.

## 7. Compute

143 runs in 35 cells, about 274 core-hours by the estimate in make_manifest.py (C 41, R 47, P 23, B 163; one block at p = 113
about 0.19 s per update, as in X and Y). The campaign was launched with 12 sessions Z1-Z12 of 4 cores (135 runs; each session
starts with its two-block run, about 3-5 hours on two threads, and runs its one-block runs beside it). Akash then added 12
sessions (2026-10-12, before any result): the manifest keeps every run of Z1-Z12 exactly as launched and adds Z13-Z24; Z13-Z20
each own one new run (seeds 14 and 15 of the B cells), Z21-Z24 own none, and all of them take runs from the back of the busiest
queues (worker.py's takeover rule). One run per helper keeps the sessions started with the 12-worker manifest safe: they never
see a helper list two runs of its own. Simulated with the estimates (helpers starting half an hour after Z1-Z12): the one-block
parts (C, R, P) finish after about 3.4 hours instead of 4.9, the two-block runs after about 5.2 hours (4.7 with the 135 runs
alone). The two-block runs set the end either way.

## 8. Pilots (local, seed 1: init 7101, split 8101; Fourier readouts off, heavy logs every 500; `pilots/`)

| pilot | updates so far | what it shows |
|---|---|---|
| p = 61, frac 0.4, lam_v 0.275, gain 2 -> 1 (5k) | 3,500 (stopped) | fit by 500; the CE dips to 0.013 nats at 1k and doubles by 3k while the no-mirror held-out accuracy goes 0.1 -> 0.99 (600 -> 2,500): the phenomenon, at a negligible CE level |
| p = 61, frac 0.4, lam_v 0.45 | 3,000 (stopped) | fit by 200; CE dip 0.27 nats at 800, then +35% by 2,400 while the no-mirror accuracy goes 0.12 -> 0.98; but the rise starts (0.1 at about 550) before the dip, and the all-pairs accuracy is already 0.33 at 200 (40% of held-out pairs have their mirror in training), so the window would hold about 0.9 of the no-mirror rise and about 0.5 of the all-pairs rise. CE at the no-mirror 0.5 crossing 0.28 nats (0.07 log p, half of p = 113's G2): hence frac 0.3 and lam_v 0.45 / 0.6 in part P |
| two blocks, frac 0.2, lam_v 0.275, gain 2 -> 1 (10k) | 3,000 (running) | fit by 500; CE 0.51 -> 0.43 nats by 2.5k, still falling slowly; no held-out rise yet (no-mirror 0.003), A_H 0.08 |
| two blocks, frac 0.3, lam_v 0.5, gain 2 -> 1 (5k) | 1,000 (running) | training accuracy 0.986 at 500 at a CE of 2.2 nats; no held-out rise yet |

Speed: two blocks take 0.78 s per update here (2.8 GHz, readouts off) against about 0.2 s for one block. The two-block pilots go
on; their outcome will be added here and to the project doc, and does not change the runs above.

## 9. Checks before launch

- tf_runner 1.6 = 1.5 bit for bit where it should be (section 4); tests/test_tf.py: 31 tests (27 of 1.5, 3 for 1.6, the manifest
  test: every run is campaign Y's G2 with only its cell's declared changes, seeds as listed, and each helper owns at most one run).
- The worker's selftest and an end-to-end run of manifest_test.json (one run of six cells, miniature).
