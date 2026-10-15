# Campaign ZM plan: p = 61 with a mirror-free split (claude/1hl-figures, 2026-10-12)

Fixed before any ZM run. Akash, 2026-10-12, agreeing to "a p = 61 run with a split that keeps both orders of each pair on the
same side, so there are no mirror pairs".

## 1. Question

At p = 61 (campaign ZP, training fraction 0.33, lam_v 0.45, seeds 11-15) the one-block transformer passes on the no-mirror
held-out pairs in 5/5 runs but on all held-out pairs in only 3/5: about a third of the held-out pairs have their mirror (b, a)
in the training set, are answered correctly before the window opens (90% by update 240-520; the window opens at 580-700), and
cap the all-pairs share of the held-out rise inside the window at (0.9 - a0) / 0.8 with a0 about 0.30. If that is the whole
story, removing mirror pairs from the held-out set should give an all-pairs share near 1 with the same settings.

## 2. Design

- Runner: tf_runner 1.7 = campaign ZP's 1.6 plus `split_mode` ('random', the default, is 1.6 split for split and update for
  update; 'mirror_free': unordered pairs {a, b} in a seeded random order, both orders taken, until int(frac p^2) ordered
  training pairs). With 'mirror_free' every held-out pair's mirror is held out, so all held-out pairs are no-mirror pairs.
- Everything else is campaign ZP's p = 61 cell: one block, d_model 128, 4 heads, d_mlp 512, float64, Polar feature steps
  (eta_w 0.1, lam_w 0.01), normalized head step (eta_v 0.03, lam_v 0.45), RMS head input with the gain 2 -> 1 (eta_g 2e-4,
  lam_g 1), observers as in ZP, 30k updates with ZP's early stop.
- Fraction. A mirror-free split at fraction f has about f p^2 / 2 distinct unordered training pairs, a random split at f
  about (2f - f^2) p^2 / 2. Two candidates: f = 0.33 (the same 1,227 training pairs as ZP, 620-629 distinct unordered pairs
  for the seeds used here) and f = 0.55 (2,046 training pairs, 1,038-1,043 distinct unordered pairs, close to the random
  split's 1,026-1,031 at 0.33).
- Pilot (development seed 1: init 7101, split 8101; one run per fraction; at most 15k updates). Rule for the main fraction:
  0.33 if its pilot's held-out rise starts (accuracy 0.1) at least 1.2 times the time of the training CE's minimum and reaches
  held-out 0.99 within 15k updates; otherwise 0.55 under the same conditions; otherwise no main runs (pilot 0.45 next).
- Main: the chosen fraction, seeds 11-15 (init 7111-7115, split 8111-8115: ZP's initializations), 30k updates. Run in the
  cloud workspace (numpy 2.4.4, torch 2.5.1, one thread each): a 200-update replay of ZP's p = 61 seed 11 here matches the
  worker's record to a relative 5e-7 in the loss and within 5e-8 in all 246 logged accuracies (the records' rounding), not
  bit for bit (different torch).

## 3. Analysis (fixed now)

Each run scored by campaign Z's per-run analysis (analyze_Z.per_run, unchanged, with its split count taught the
mirror-free split): the paper's window, shares on both held-out definitions (identical here), refit and A_H gains, the
misclassified training pairs inside the window, final held-out accuracy; pass (strict) as in ZP. Reported next to ZP's
random-split cell (same seeds): the all-pairs accuracy when the window opens (a0), the all-pairs share, the rise timing, the
CE change across the rise and the pass counts.

Prediction if the mirror pairs are the whole story: a0 at most 0.05 and an all-pairs share of at least 0.95 in every run whose
held-out accuracy rises after the CE minimum; the all-pairs criterion then passes in at least 4 of 5 runs unless the rise is
too slow to finish (final below 0.99 by 30k).

Development runs only; not evidence for the paper's claims until cited.

## 4. Pilot outcome and the main fraction (added 2026-10-12, after the pilots; sections 1-3 unchanged)

| pilot (seed 1) | CE minimum | held-out 0.1 / 0.5 / 0.9 / 0.99 | rise start / CE minimum | final (15k) | rule |
|---|---|---|---|---|---|
| mirror-free, fraction 0.33 (1,227 training pairs) | 541 | never | - | 0.001 | fails (no rise in 15k) |
| mirror-free, fraction 0.55 (2,046 training pairs) | 709 | 2,060 / 3,980 / 6,440 / 10,320 | 2.91 | 0.996 | qualifies |

Main fraction by the rule: **0.55**. Main runs ZM_f0.55_s11-s15 (configs/). The 0.33 pilot is reported as it is: with the
same number of training pairs as ZP's cell but only about 625 distinct unordered pairs (against about 1,030 in the random
split), held-out accuracy stays at 0 for 15k updates while the training CE rises from 0.04 to 0.26 log p.
