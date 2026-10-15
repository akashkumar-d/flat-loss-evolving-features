# Campaign ZM: p = 61 with a mirror-free split, results (claude/1hl-figures, 2026-10-12)

**Question.** Campaign ZP's p = 61 cell (one block, gain 2 -> 1, training fraction 0.33, lam_v 0.45) passes on the no-mirror
held-out pairs in 5/5 runs but on all held-out pairs in only 3/5. Is that only because a third of the held-out pairs have their
mirror (b, a) in the training set? Here every training pair comes with its mirror, so no held-out pair does (`split_mode
'mirror_free'`, tf_runner 1.7; everything else is ZP's cell). Plan, pilot rule and pilot outcome: `../PLAN.md`.

**State.** 2 pilot runs (development seed 1) and 5 main runs (seeds 11-15, ZP's initializations), all finished, in Claude's
cloud workspace (Python 3.11, numpy 2.4.4, torch 2.5.1, one thread per run; a 200-update replay of a ZP worker run matches it
to a relative 5e-7). Scored by campaign Z's per-run analysis, unchanged (`analyze_ZM.py`; `analysis_ZM.json`,
`analyze_ZM.log`). Development runs only; not evidence for the paper's claims until cited.

## 1. Result: with no mirror pairs, p = 61 passes 5/5 (`figs/ZM_mirrorfree_paper`)

Same seeds (initializations 7111-7115); medians, ranges in brackets.

| | random split, fraction 0.33 (ZP) | mirror-free split, fraction 0.55 (ZM) |
|---|---|---|
| training pairs (ordered / distinct unordered) | 1,227 / about 1,030 | 2,046 / about 1,040 |
| held-out pairs with their mirror in training | 798-816 of 2,494 | 0 of 1,675 |
| CE minimum (update, CE / log p) | 731, 0.059 | 726, 0.055 |
| no-mirror held-out 0.1 / 0.5 / 0.9 (updates) | 1.9k / 3.6k / 5.4k | 1.8k / 3.9k / 6.2k |
| all-pairs held-out accuracy when the window opens (a0) | 0.30 [0.30, 0.33] | 0.025 [0.007, 0.033] |
| all-pairs share of the rise inside the window | 0.75 [0.71, 0.75] | 1.00 [1.00, 1.00] |
| training CE across the rise (no-mirror 0.1 -> 0.9) | +102% [+87, +133] | +145% [+118, +166] |
| misclassified training pairs inside the window | 0 | 0 |
| final held-out accuracy (all pairs) | [0.995, 0.998] | [0.992, 0.996] |
| refit gain / Fourier refit gain / A_H gain inside the window | 0.69 / 0.69 / 0.58 | 0.98 / 0.84 / 0.49 |
| pass (strict, both definitions) | **3/5** (2 fail on the share) | **5/5** |

- The prediction fixed in the plan holds: a0 at most 0.05 and an all-pairs share of at least 0.95 in every run, and 5/5 pass.
  The p = 61 failure of the all-pairs criterion was entirely the mirror pairs' head start.
- The timing is unchanged: the CE minimum, the start and the middle of the held-out rise and the CE's climb across it match the
  random split, and in both the Fourier-filtered refit leads the network's held-out accuracy at 0.5 (by 0.85k and 0.59k
  updates) while the paper's refit lags (by 0.70k in both).
- Four of the five mirror-free runs reach the 30k-update budget: held-out accuracy hovers just above 0.99 near the end, and the
  early stop (held-out accuracy at least 0.99 without a break from its first time t until max(1.5 t, t + 5000)) keeps
  restarting. Final accuracy is at least 0.992 in all five.

## 2. What sets the timing: distinct unordered pairs (pilots, development seed 1)

| mirror-free fraction | training pairs (ordered / distinct unordered) | held-out 0.1 / 0.5 / 0.9 / 0.99 | final |
|---|---|---|---|
| 0.33 | 1,227 / 622 | never in 15k updates (the training CE rises from 0.04 to 0.26 log p) | 0.001 |
| 0.55 | 2,046 / 1,038 | 2.0k / 4.0k / 6.4k / 10.3k | 0.996 |

With the same 1,227 training pairs as ZP's cell but only about 625 distinct unordered pairs, the network does not generalize
in 15k updates; with about 1,040 distinct unordered pairs (the random split at 0.33 has about 1,030) it generalizes on ZP's
schedule. So the information that matters is the number of distinct unordered pairs: a training pair's mirror adds little,
and in a random split the held-out pairs whose mirror is in the training set are effectively training pairs. That is why the
no-mirror definition is the right held-out measure for a random split; ZP's timing rule (rise start against
x = n_train / (p ln p)) may be better stated in distinct unordered pairs (not tested beyond these two pilots).

## 3. For the paper

At p = 61 the appendix can use the mirror-free split (5/5 on all held-out pairs) or keep the random split and report the
no-mirror definition (5/5); either way the reason for the gap is shown by `campaign_ZP/analysis/paper/ZP_mirror_paper` and
this figure.

## 4. Why the refits do not lead held-out accuracy here (added 2026-10-13; `figs/ZM_readouts`, `readouts_ZM.json`)

Descriptive and post hoc (`readouts_ZM.py`, `plot_readouts_ZM.py`, `freqs_ZM.py`): every logged readout against the network's
held-out accuracy on the no-mirror pairs, at p = 61 (both splits) and at ZP's p = 89, 113 and 149 cells. Lead = the network's
crossing minus the readout's (positive: the readout crosses first); medians over seeds, with the number of seeds in which it
leads. (`figs/ZM_mirrorfree_paper` draws the two refits on all held-out pairs, which in the random split include the mirror
pairs; here every curve is on the no-mirror pairs.)

| readout | p61 mirror-free: lead at 0.5 / 0.9 | p61 random | p89 | p113 | p149 |
|---|---|---|---|---|---|
| paper refit (ridge on the head input, all frequencies) | -0.70k / -1.09k (0/5, 0/5) | -0.70k / -0.81k (0/5, 0/5) | -0.83k / -1.41k | -1.31k / -2.15k | -1.51k / -2.04k |
| Fourier refit, top 8 (the transformer figures' readout) | +0.59k / -2.34k (4/5, 0/5) | +0.85k / -1.08k (5/5, 0/5) | +0.64k / +0.42k | +0.86k / +0.91k | +0.77k / +1.19k |
| Fourier refit, top 16 | +0.84k / +0.71k (5/5, 4/5) | +0.81k / +0.66k (5/5, 5/5) | +0.39k / +0.31k | +0.50k / +0.47k | +0.50k / +0.53k |

- The paper's refit lags the network in every transformer cell (0 of 30 runs at 0.5 and at 0.9), not only in ZM: it weights all
  frequencies equally, and most of them carry the memorized training labels (campaign X's finding).
- The top-8 Fourier refit leads at the 0.5 crossing in every cell (29 of 30 runs), and at the 0.9 crossing at p = 89, 113 and
  149 (20/20), but at p = 61 it stops near 0.95 (final 0.950 random, 0.961 mirror-free) and crosses 0.9 after the network (by
  1.1k and 2.3k updates). At the network's 0.9 crossing, 15-19 of p = 61's 30 frequencies carry held-out signal (held-out R^2
  above 0.05, tf_runner's diagnostic `fspec_hor2`), against 9-15 of 44-74 at the larger p; the top 8 keep 8 of them, the top 16
  keep 12-16 at p = 61 and all of them at the larger p (training-energy ranks up to 15 there, up to 17-22 at p = 61 in 7 of 10
  runs; `freqs_ZM.py`, `freqs_ZM.log`).
- The top-16 Fourier refit leads at both crossings in every cell (all seeds except one at p = 61 mirror-free, 0.9). So do the
  network's own logits restricted to the top 16 frequencies; the held-out-label oracles (diagnostics only) lead in their
  medians. The logistic probe tracks the network (median leads 0.01-0.15k).
- K = 8 was fixed by campaign X at p = 113 before these runs; K = 16 is chosen here after seeing p = 61, so it is a post hoc
  readout until confirmed on fresh seeds.
