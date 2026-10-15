# Campaign Z: results, all 143 runs (claude/1hl-figures, 2026-10-12)

**State.** 143 of 143 runs finished, none failed, no push errors, no run on two branches (24 sessions Z1-Z24; last run
finished before 08:31Z). The results were read from the session branches through Akash's browser and laid out as
collect_results.sh does (every run.json.gz matches its chain.json). On the Mac, `bash collect_results.sh` in
natural_plateau_shared/ writes the same files to `../collected/`, which the analysis and figure scripts read. The analysis is
PLAN.md section 5 (`analyze_Z.py`, `plot_Z.py`; `analysis_Z.json`, `analyze_Z.log`, `figs/`), with campaign Y's G2/G3 and
campaign X's R5 (same seeds, initialization and split) as reference cells. Every measure is computed on both held-out
definitions, all held-out pairs and the no-mirror subset, and a run passes only if it passes on both (and its final held-out
accuracy on all pairs is at least 0.99); in the figures solid lines are all held-out pairs, dashed lines the no-mirror pairs.
Development runs only; not evidence for the paper's claims.

## 1. Optimizer contrasts (one block, p = 113, frac 0.2; `figs/Z_contrast_gain.png`, `figs/Z_contrast_nogain.png`)

| | Polar (paper's rule) | NGD feature steps | AdamW (lr 1e-3, wd 1) |
|---|---|---|---|
| with the gain 2 -> 1 (seeds 11-20) | **10/10** (campaign Y G2); CE +33%, up in 10/10 | 0/10: the training set is never fitted (2/10 reach a full fit, at about 19k), so there is no window; held-out rises anyway (0.1 -> 0.9 at 2.1k-4.3k), final 0.67-0.995; CE -31% | 0/10: fits by 120, held-out jumps at 3.0k-3.7k (no-mirror 0.1 -> 0.9 in about 700 updates); CE -63% (median), spiky; inside the window the Fourier refit gains nothing and more than 10 training pairs are misclassified (10/10) |
| without the gain (seeds 11-15; Polar: X R5, seeds 11-20) | 0/10: CE -8%, window share 0.35-0.61 | 0/5: never fits; final 0.61-0.92 | 0/5: as with the gain (CE -39%) |

The gain alone does not produce the effect: with the same gain, only the Polar rule gives a held-out rise that is gradual,
starts after the CE minimum and happens while the training CE rises.

## 2. The one-block recipe (seeds 11-13; `figs/Z_recipe.png`)

| | time constant 2.5k | 5k | 10k | 20k | fixed gain |
|---|---|---|---|---|---|
| g0 1.5 | 1/3 strict (3/3 loose), CE +10% | 3/3, +13% | 2/3, +6% | 2/3, -4% | 0/3, -17% |
| g0 2 | 3/3, +28% | 3/3, +38% (Y's G2) | 3/3, +24% | 2/3, -1% | 0/3, -27% |
| g0 3 | 3/3, +84% | 3/3, +104% | 3/3, +58% | 2/3, +9% | 0/3, -44% |

- g0 2-3 with time constant 2.5k-10k: 18/18 strict passes, CE +24% to +104% across the rise (cell medians); window share
  0.96-0.99. A fixed gain fails in all 9 runs (the CE keeps falling through the rise): the relaxation of the gain is what lifts
  the CE. With time constant 20k the CE is about flat (6/9 pass).
- One factor at a time around G2: lam_v 0.25 and 0.3 3/3 each; eta_w 0.05 and 0.2 3/3 each (0.2: CE only +10%); lam_w 0.005
  1/3 (two runs too slow, final 0.53 and 0.98); lam_w 0.02 2/3 (CE flat); frac 0.3 0/3 (rise at about 450, before the CE dip at
  1.3k).
- Recipe: gain g0 2-3 relaxing to 1 with a time constant comparable to the time of the held-out rise (2.5k-10k updates), Polar
  feature steps (eta_w 0.05-0.2, lam_w 0.01), head lam_v 0.25-0.3, and a training fraction that puts the rise after the CE dip.

## 3. Other p (one block, gain 2 -> 1; `figs/Z_p.png`)

| cell | strict | what went wrong |
|---|---|---|
| p = 61, frac 0.3, lam_v 0.45 / 0.6 | 1/5 / 1/5 | rise well after the CE dip, CE +87% / +58%, but held-out accuracy stalls at 0.96-0.99 by 30k |
| p = 89, frac 0.2, lam_v 0.33 / 0.4 | 1/5 / 0/5 | too slow: 3 of 5 seeds without a rise by 30k (final 0.39-0.76) |
| p = 149, frac 0.2, lam_v 0.23 / 0.275 | 0/5 / 0/5 | rise at 1.0k-1.9k, before the CE dip at about 2.3k (window share under 0.3) |

Campaign ZP (`../../campaign_ZP/`) reruns these with the training fraction set per p.

## 4. Two blocks (seeds 11-15; `figs/Z_blocks.png`)

0/20. At frac 0.2 (lam_v 0.275, time constant 10k) the effect is there during the rise: the CE rises +9% to +26% (5/5), the
window holds the whole no-mirror rise (share 1.0) and the Fourier refit leads by about 1.6k updates; but after about 18k
updates the fit degrades (training pairs flip, held-out 0.93-1.0), so more than 10 training pairs are misclassified inside the
window in 5/5 and 3/5 end below 0.99. At frac 0.25 and 0.3 the rise comes before the CE dip (window share fails in 14/15).
A second block needs a way to keep the late fit stable (for example stopping the gain's relaxation, or a smaller eta_w, once
the rise is over); not tested here.
