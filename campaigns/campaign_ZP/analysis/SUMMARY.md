# Campaign ZP: results, all 30 runs (claude/1hl-figures, 2026-10-12)

**State.** 30 of 30 runs finished: 8 sessions ZP1-ZP8 report `campaign_done`, no failed runs, no push errors, last heartbeat
2026-10-12T11:16Z (ZP4 ran none; its runs were taken over by the others). Akash ran `bash collect_results.sh`; the collected
files (`collected/results`, `collected/status`, `collected/INDEX.json`) were read from the Mac and every run.json.gz matches the
SHA-256 in INDEX.json (30/30). The analysis is the one fixed in PLAN.md (`analyze_ZP.py`, `plot_ZP.py`; `analysis_ZP.json`,
`analyze_ZP.log`, `figs/`), with campaign Z's other-p cells and campaign Y's G2 (p = 113) as references. A run passes if it
passes on both held-out definitions (all pairs outside the training set, and the no-mirror subset) and its final held-out
accuracy on all pairs is at least 0.99; strict: no training pair misclassified inside the window, loose: at most 10.
Development runs only; not evidence for the paper's claims.

## 1. Pass counts (one block, gain 2 -> 1, seeds 11-15; `figs/ZP_p61.png`, `figs/ZP_p89.png`, `figs/ZP_p149.png`)

CE change: training CE across the no-mirror held-out rise (0.1 -> 0.9), cell median (runs in which it rose).

| p | frac (campaign) | strict | loose | strict, no-mirror definition alone | CE change | what decided it |
|---|---|---|---|---|---|---|
| 61 | 0.3 (Z; lam_v 0.45 / 0.6) | 1/5 / 1/5 | 1/5 / 1/5 | 1/5 / 1/5 | +87% / +58% | too slow: final held-out 0.96-0.99 by 30k |
| 61 | 0.33 (ZP) | 3/5 | 3/5 | 5/5 | +102% (5/5) | all-pairs share 0.71 and 0.73 in the two fails (passes: 0.751-0.753) |
| 61 | 0.35 (ZP) | 0/5 | 0/5 | 5/5 | +87% (5/5) | all-pairs share 0.65-0.73 |
| 89 | 0.2 (Z; lam_v 0.33 / 0.4) | 1/5 / 0/5 | 1/5 / 0/5 | 1/5 / 0/5 | +33% / +22% | too slow: 3 of 5 without a rise by 30k |
| 89 | 0.25 (ZP) | **5/5** | 5/5 | 5/5 | +37% (5/5) | |
| 89 | 0.275 (ZP) | 2/5 | 2/5 | 5/5 | +4% (5/5) | rise starts before the CE dip; all-pairs share 0.67-0.70 in the fails |
| 113 | 0.2 (Y G2, seeds 11-20) | **10/10** | 10/10 | 10/10 | +33% (10/10) | |
| 149 | 0.15 (ZP) | 0/5 | 4/5 | 0/5 | +28% (4/4) | 1-6 training pairs misclassified inside the window in 4 seeds; one seed never rose (final 0.12) |
| 149 | 0.16 (ZP) | 3/5 | **5/5** | 3/5 | +16% (5/5) | 4 and 7 training pairs misclassified inside the window in the two strict fails |
| 149 | 0.2 (Z; lam_v 0.23 / 0.275) | 0/5 / 0/5 | 0/5 / 0/5 | 0/5 / 0/5 | -23% / -18% | rise before the CE dip |

- The plateau problem of campaign Z at p = 61 is fixed: at frac 0.33 and 0.35 the final held-out accuracy is 0.995-0.999
  (Z at frac 0.3: 0.96-0.99 by 30k).
- In every passing ZP run the whole no-mirror rise lies inside the window (share 1.0) and the training CE rises across it.

## 2. Mirror pairs bound the all-pairs share at p = 61

About a share frac of the held-out pairs have their mirror in the training set (0.32 at p = 61, frac 0.33), and these are
learned soon after the fit. At p = 61 they are learned before the CE dip that opens the window: the all-pairs held-out
accuracy at the window start is 0.30-0.33 (frac 0.33) and 0.31-0.38 (frac 0.35). The share counts the 10% -> 90% rise,
(clip(acc at the window end) - clip(acc at the window start)) / 0.8 with clip to [0.1, 0.9], so a window that opens at 0.30
caps the share at 0.75: the three passes at frac 0.33 open at 0.298-0.299 (shares 0.751-0.753), the two fails at 0.318 and
0.334 (0.727, 0.707). At larger p the mirrors are only partly learned when the window opens (0.17-0.23 at p = 89, frac 0.25;
0.06-0.16 at p = 113; 0.03-0.11 at p = 149) and the cap does not bind. The timing rule (section 3) needs a larger training
fraction at smaller p (frac = x ln p / p with x about 4.8), which puts p = 61 at the cap: every p = 61 run at frac 0.33 and
0.35 passes on the no-mirror definition (10/10) while the all-pairs shares sit at 0.65-0.75. At p = 61 the all-pairs criterion
cannot be met reliably in this setup; reporting p = 61 needs the no-mirror definition or a mirror-free split (both orders of a
pair on the same side). Direct test, campaign ZM (`../../campaign_ZM`): with a mirror-free split at fraction 0.55 (about the
same number of distinct unordered training pairs as the random split at 0.33) the same cell passes 5/5 on all held-out pairs
(a0 0.007-0.033, share 1.0); at fraction 0.33 (half as many distinct pairs) it does not generalize in 15k updates.

## 3. When the rise starts (`figs/ZP_timing.png`)

The rule fixed in PLAN.md, rise start = 2300 / (x - 3.9) updates with x = n_train / (p ln p), predicted the ZP cells' median
rise start within a factor 0.63-1.34 (observed / predicted). At the cell level the outcome follows the ratio of the rise start
to the CE dip:

| median rise start / CE dip | cells | outcome |
|---|---|---|
| below 1 | p89 f0.275 (0.75), p149 f0.2 (0.45, 0.47), p113 f0.3 (Z recipe, 0.33) | fail on the share (rise before the window) |
| 1.5-3.2 | p89 f0.25 (1.52), p113 f0.2 (Y G2, 2.60), p61 f0.33 (2.71), p149 f0.16 (3.18) | pass (p61 limited by mirrors, section 2; p149 strict 3/5, loose 5/5) |
| 1.8 | p61 f0.35 (1.83) | fails only on the mirror cap |
| 4.7 and above | p149 f0.15 (4.76), p61 f0.3 (5.6, 6.2), p89 f0.2 (10.4, 10.8) | too slow: not finished, or a seed never rises, within 30k |

Individual runs overlap across these bands (strict passes range from 0.81 to 9.4), so the ratio is a guide for choosing frac,
not a criterion.

## 4. Appendix choice (fixed rule: most strict passes, ties by loose passes, then the smaller frac; `figs/ZP_appendix.png`)

| p | cell | strict | loose | reaches 4/5 |
|---|---|---|---|---|
| 61 | frac 0.33, lam_v 0.45 (ZP) | 3/5 | 3/5 | no (5/5 on the no-mirror definition) |
| 89 | frac 0.25, lam_v 0.33 (ZP) | 5/5 | 5/5 | yes |
| 113 | frac 0.2, lam_v 0.275 (Y G2) | 10/10 | 10/10 | yes |
| 149 | frac 0.16, lam_v 0.23 (ZP) | 3/5 | 5/5 | strict no, loose yes |

The one-block recipe carries to p = 89 and p = 149 once frac is set by the timing rule; p = 149 needs the loose fit criterion
(at most 10 of 3,552 training pairs misclassified inside the window), and p = 61 needs the no-mirror definition.

## 5. Paper figures (`paper/`, `paper_ZP.py`; draft captions in `paper/CAPTIONS.md`)

- `ZP_acrossp_paper`: the four appendix cells in the style of the Z contrast figure (medians, interquartile bands, the paper's
  colour key), rows CE / accuracies / misclassified training pairs / A_H, each column ending at its shortest run.
- `ZP_mirror_paper`: how the mirror pairs set the all-pairs share. The held-out accuracy on the mirror pairs is derived exactly
  from the two logged accuracies and the pair counts (n_ho acc_all = n_nm acc_nm + n_mirror acc_mirror; it matches
  analyze_ZP's own value at the first fit in all 45 runs). At p = 61 (fraction 0.33) 90% of the mirror pairs are correct by
  update 240-520, before the window opens at 580-700; when the window opens, the mirror pairs are 90-95% correct at p = 61,
  63-88% at p = 89, 29-74% at p = 113 and 19-67% at p = 149. Every run's all-pairs share equals (0.9 - max(a0, 0.1)) / 0.8,
  with a0 the all-pairs accuracy when the window opens, so the share fails exactly when a0 > 0.3.
- `ZP_p61_fraction_paper`: at p = 61 no training fraction passes on all pairs: at 0.30 the window opens below 0.3 (share
  0.79-0.86) but held-out accuracy stalls below 0.99 by 30k in 4 of 5 runs; at 0.33 the share sits at the threshold (0.71-0.75,
  3/5); at 0.35 it is below (0.65-0.73, 0/5). The no-mirror share is 1.0 in all 15 runs.
- `mirror_ZP.json`: the per-run numbers behind the two mirror figures.
