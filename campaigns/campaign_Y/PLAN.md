# Campaign Y: a transformer whose training loss does not fall while the features and held-out accuracy rise (plan fixed before any run)

claude/1hl-figures, 2026-10-11, at Akash's request after launching campaign X: "we need to make sure that the loss remains flat
or is not decreasing in the window of interest." Development runs; not evidence for the paper's claims.

## 1. Why

- In campaign W's R4 and R5 the training CE falls 1.7-2.4% per 1000 updates while the held-out accuracy rises, 6.5-9.1% from
  held-out 0.1 to 0.9 (W section 6b). Campaign X (the same settings on seeds 11-20, finished) confirms it on 20 runs: averaged
  over 201 updates, the CE falls 6.7-10.2% (R5) and 7.8-12.4% (R4) from held-out 0.1 to 0.9, so the paper's post-hoc window (CE
  never more than 3% below its running maximum) holds only 43-61% (R5) and 38-54% (R4) of the rise; by section 4's verdict
  0/20 runs qualify.
- With the RMS division before the head, the logit scale is set by W_U alone, and W_U reaches its norm equilibrium
  (cos / lam_v, about 3.6) within a few hundred updates (time constant 1 / (eta_v lam_v) = 121 updates). After that nothing
  shrinks the logits, so every improvement of the features lowers the CE. In the paper's MLP the CE climbs from its dip because
  the first layer, initialized large (scale 24), shrinks slowly under weight decay while the head re-equilibrates; the features
  improve during that climb.
- Pilots (seed 1 of R5, 12000 updates; `pilots/`, section 6):
  - Slowing the head instead (eta_v 0.0025 and 0.0012 from its initial norm 10.7): the CE climbs, but the head's own Fourier
    structure develops 10-20x more slowly, and the held-out rise waits until the head has reached its equilibrium; the climb is
    over before the rise starts. Rejected.
  - A scalar gain g on the RMS head input (logits = g r_hat W_U), started at 2 or 3 and relaxing to 1 under weight decay with
    the head's rule for a scalar, while W_U keeps eta_v = 0.03: see section 6.

## 2. Runs (20; `manifest.json` from `make_manifest.py`; runner `tf_runner.py` 1.5)

Campaign X's R5 runs (one block, d_model 128, 4 heads, MLP 512, RMS division before the head, float64, p = 113, training
fraction 0.2, eta_w 0.1, lam_w 0.01, eta_v 0.03, lam_v 0.275; supported-Polar steps with decoupled decay on every matrix except
W_U, normalized head step; seeds 11-20, init 7100 + i, split 8100 + i; light logs every 20, heavy logs with the Fourier
readouts every 100, probe every 200; early stop when held-out accuracy >= 0.99 has held for max(1.5 t, t + 5000) updates; 30000
at most), with the gain:

| arm | gain_init | eta_g | lam_g | time constant 1 / (eta_g lam_g) |
|---|---|---|---|---|
| G2 | 2 | 2e-4 | 1 | 5000 updates |
| G3 | 3 | 2.5e-4 | 1 | 4000 updates |

g <- g + eta_g (-sign(dL/dg) - lam_g g): once the training set is fit, g relaxes to 1 (the plain RMS head of campaign X). Every
run pairs with campaign X's R5 run of the same seed (same initialization and split; X is the no-gain contrast).

## 3. Runner 1.5

`gain_init` (default None: off) adds the scalar gain; everything else is 1.4. Without it every update and every record is
1.4's, bit for bit (checked on a 60-update run with the RMS head, Fourier readouts and probe). The observers read the
normalized head input r_hat (before the gain), so A_H, the refits and the Fourier readouts are defined exactly as in
campaign X; the gain is logged at every light log (`gain`).

## 4. Analysis (fixed now; `analyze_Y.py`, `plot_Y.py`)

Applied the same way to campaign Y's runs and to campaign X's R5 runs on the same seeds. Held-out accuracy is on the no-mirror
pairs (campaigns W, X); the CE is the training cross-entropy averaged over 201 updates (W section 6b).

Per run:
1. **Crossings and leads** (campaign X's `analyze_X.per_run`, unchanged): held-out 0.1 / 0.5 / 0.9, the readouts' crossings and
   leads (primary readout: the Fourier-filtered refit, top 8 by training energy; also the paper's refit, the restricted logits,
   the probe), the frequencies at the network's 0.5 crossing.
2. **The loss across the rise**, from held-out 0.1 to 0.9: the CE's relative change, its largest drawdown below its running
   maximum since held-out 0.1, the drawdown at held-out 0.9, and the most misclassified training pairs at any log.
3. **The paper's post-hoc window** on the averaged CE: after the first full fit, the stretch where the CE never falls more than
   3% below its running maximum, chosen to hold the largest share of the held-out 10% -> 90% rise (q_te, the paper's clip
   formula). Inside it: the gain of the Fourier-filtered refit (top 8) and of the paper's refit, Delta A_H of the head input,
   the most misclassified training pairs.
4. **Verdicts.** pass_Y = q_te >= 0.75, Fourier-filtered refit gain >= 0.2 inside the window, Delta A_H >= 0.02, final held-out
   accuracy (all held-out pairs) >= 0.99, and at most 10 misclassified training pairs at every log inside the window;
   pass_Y_strict = the same with every training pair correct at every log inside. Also reported: campaign W's raw-CE 10%
   window (strict), as in campaigns W and X.

Reported per arm (counts out of 10, medians and ranges), with X's R5 runs on the same seeds alongside: pass_Y and
pass_Y_strict; the CE change across the rise; the drawdowns; q_te; the leads at 0.5 and 0.9; the gain at the 0.1, 0.5 and 0.9
crossings; the CE level at held-out 0.5 (nats and / log p).

Figures (medians over the 10 seeds with the interquartile band, seeds thin; paper colour key):
- `Y_main`: one column per setting (X R5 without the gain, G2, G3): (a) training CE (201-update average) with the median paper
  window shaded, (b) held-out accuracy of the network, the Fourier-filtered refit (top 8), the paper's refit and the restricted
  logits, and training accuracy, (c) A_H of the head input, (d) the gain g.
- `Y_rise`: the CE relative to its value at held-out 0.1, against the held-out accuracy, per seed: X R5 against G2 and G3.
- `Y_freq`: campaign X's frequency figure on the seed with the median lead of each arm.

## 5. Compute

20 runs, about 12k updates each at about 0.19 s per update on one core: about 15 core-hours with contention, on 5 sessions Y1-Y5
of 4 cores, about 45 minutes each. Fewer sessions also work (the others take over unstarted IDs); they take longer.

## 6. Pilots (before this plan's runs; seed 1 of R5, 12000 updates, local)

Seed 1 of R5 (init 7101, split 8101), float64, Fourier readouts every 200 updates, no early stop; `pilots/` (runs, `analyze_G.py`,
`pilots_analysis.json`, figure `pilots.png`). Held-out accuracy on the no-mirror pairs; CE averaged over 201 updates.

| pilot | first full fit | held-out 0.1 / 0.5 / 0.9 | CE from held-out 0.1 to 0.9 | paper's window: share of the rise | W's raw-CE 10% window | Fourier top-8 refit lead at 0.5 | paper's refit lead at 0.5 | gain at held-out 0.1 / 0.9 |
|---|---|---|---|---|---|---|---|---|
| no gain (W's R5 seed 1, replayed) | 1680 | 3720 / 5460 / 6840 | 1.65 -> 1.53 nats (-7.4%) | 0.53 | 0.80 | +1.2k | -1.3k | - |
| slow head, eta_v 0.0025 | 200 | no rise by 7000 (0.04) | - | - | - | - | - | - |
| slow head, eta_v 0.0012 | 240 | no rise by 7000 (0.02) | - | - | - | - | - | - |
| gain 2 -> 1, time constant 5000 | 220 | 3280 / 4960 / 6560 | 0.64 -> 0.93 nats (+44%) | 1.00 (window from 820 to the end) | 0.37 | +1.4k | -1.4k | 1.52 / 1.27 |
| gain 3 -> 1, time constant 4000 | 160 | 3040 / 4720 / 6000 | 0.24 -> 0.64 nats (+162%) | 1.00 (from 760 to the end) | 0.15 | +1.3k | -1.2k | 1.93 / 1.44 |

- With the gain, the training set stays fit (no misclassified pair during the rise; at most one at any log after the first
  fit), the CE climbs from its dip (about 1000 updates) to the end of the run, never more than 0.05% below its running maximum,
  and the held-out rise comes slightly earlier than without the gain; A_H of the head input rises 0.07 -> 0.87-0.88 in both.
  The CE level is lower than without the gain while g > 1 (median over the rise 0.17 and 0.09 log p, against 0.33), and it
  climbs toward the no-gain level as g -> 1. W's raw-CE 10% window fails by design: the CE is not flat, it rises.
- The slow head: the CE climbs as the head norm decays (0.35 -> 1.68 nats with eta_v 0.0025), but the head's top-8 Fourier
  share is still 0.21 at 6800 updates (no gain: 0.39 by 4000), A_H only 0.13, and held-out accuracy is 0.04 at 7000.

## 7. Checks before launch

- tf_runner 1.5 without `gain_init` equals 1.4 bit for bit (loss and every record, including the Fourier readouts and the probe).
- `tests/test_tf.py`: 28 tests (21 from 1.4, 6 for the gain: off is 1.4, the gain scales the logits and needs the RMS head,
  dL/dg against finite differences and the update rule, the relaxation to 1 / lam_g once the training set is fit, the observers
  read the normalized input, exact resume with the gain; and the manifest test: every run is campaign X's R5 run of its seed
  plus `gain_init`, `eta_g`, `lam_g`, nothing else).
- The worker's selftest (without the git step) passes, and the worker ran a run of `manifest_test.json` end to end (runner
  1.5, the gain logged).
