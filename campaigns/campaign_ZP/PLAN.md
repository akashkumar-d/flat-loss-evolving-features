# Campaign ZP: other p with the training fraction set per p (claude/1hl-figures, 2026-10-12)

## 1. Why

Campaign Z's part P ran the one-block recipe that works at p = 113 (campaign Y's G2: gain 2 -> 1 with time constant 5000 on
the RMS head input, Polar feature steps, frac 0.2) at p = 61, 89 and 149, with the head decay lam_v from Z's p-rule. None of
its cells passed in 4 of 5 seeds, for three different reasons:

| Z cell | strict passes | what went wrong |
|---|---|---|
| p = 61, frac 0.3, lam_v 0.45 | 1/5 | the held-out rise came well after the CE dip (CE +58% to +136% across it), but held-out accuracy (all pairs) stalled at 0.979-0.994 by 30k updates |
| p = 89, frac 0.2, lam_v 0.33 | 1/5 | too slow: 3 of 5 seeds had not risen by 30k updates (held-out 0.39-0.76) |
| p = 149, frac 0.2, lam_v 0.23 | 0/5 | the rise (updates 980-1,920) came before the CE dip (about 2,300), so the paper's window held under 25% of it |

(lam_v 0.6 at p = 61, 0.4 at p = 89 and 0.275 at p = 149 did no better.) The training fraction is the common lever: it moves
the start of the held-out rise. Campaign ZP reruns the three p with the fraction set per p and nothing else changed.

## 2. The fractions

**Timing rule.** Across campaigns Y and Z (one block, gain 2 -> 1), the start of the held-out rise (no-mirror accuracy crosses
0.1, median over seeds) follows one curve in x = n_train / (p ln p):

    rise start ~ 2300 / (x - 3.9) updates

| cell | x | rule | measured |
|---|---|---|---|
| p = 89, frac 0.2 (Z) | 3.97 | 35,000 | 2 of 5 rose (about 8,500); 3 had not by 30k |
| p = 61, frac 0.3 (Z) | 4.45 | 4,180 | 4,420 |
| p = 113, frac 0.2 (Y, G2) | 4.78 | 2,620 | 2,660 |
| p = 149, frac 0.2 (Z) | 5.96 | 1,120 | 1,050 |
| p = 113, frac 0.3 (Z, R_frac0.3) | 7.17 | 700 | 460 |

The fraction alone (n_train / p^2) does not order these cells (p = 61 at frac 0.3 rises later than p = 113 at frac 0.2), nor does
n_train / p^1.5; n_train / p and n_train / (p ln p) both do, and the second gives the smoother curve.

**When a rise is late enough.** Over the one-block Polar runs of campaigns X, Y and Z that rise, a run passes only when its
rise starts near or after the CE dip (the minimum of the 201-update CE average). Strict passes by rise start / dip: below 0.5,
1 of 31; 0.5-0.75, 2 of 8; 0.75-1, 6 of 7; above 1, 56 of 70 (most failures there are the final-accuracy failures at p = 61
and 89). The dips at frac 0.2-0.3 are about 700 updates (p = 61), 800 (p = 89), 1,070 (p = 113) and 2,300 (p = 149).

**The mirror bound at p = 61.** Held-out pairs whose mirror (b, a) is in training (a share of about frac of all held-out pairs)
are mostly right by the CE dip. At p = 61, frac 0.3 the all-pairs accuracy at the start of the window was 0.21-0.27 (about
0.73 x frac), and the window share on that definition, q_te = (0.9 - clip(start)) / 0.8, needs the start to be at most 0.3. That
caps frac near 0.35 at p = 61; Z's pilot at frac 0.4 (seed 1) had a third of the held-out pairs right by update 200 and its
no-mirror rise started at about 550, before the dip at 800.

**The choice.** Two fractions per p, both predicted to rise after the dip, the larger one faster:

| p | lam_v | frac | n_train | x | predicted rise start | rise start / dip | all-pairs accuracy at the dip (p = 61) |
|---|---|---|---|---|---|---|---|
| 61 | 0.45 | 0.33 | 1,227 | 4.89 | 2,300 (1,400 on p = 61's own two points) | 2-3 | about 0.24 (q_te about 0.82) |
| 61 | 0.45 | 0.35 | 1,302 | 5.19 | 1,800 (1,000) | 1.4-2.5 | about 0.26 (q_te about 0.80; the worst Z seed would give 0.73) |
| 89 | 0.33 | 0.25 | 1,980 | 4.96 | 2,200 | about 2.6 | |
| 89 | 0.33 | 0.275 | 2,178 | 5.45 | 1,500 | about 1.8 | |
| 149 | 0.23 | 0.15 | 3,330 | 4.47 | 4,100 | about 1.8-2 | |
| 149 | 0.23 | 0.16 | 3,552 | 4.76 | 2,700 | about 1.2-1.3 | |

For p = 61 the bet is that the earlier, faster rise leaves time for the held-out accuracy to pass 0.99 within 30k updates (at
frac 0.3 it reached 0.98 only after about 20k; Z's pilot at frac 0.4 reached 0.98 on the no-mirror pairs by 2,400). p = 89 at
frac 0.3 is left out: on p = 61's own curve it would rise at about 540, before its dip.

## 3. Runs

6 cells x 5 seeds = 30 runs. Seeds 11-15 (init 7100 + seed, split 8100 + seed), as in X, Y and Z. Every run is Z's other-p run
for the same p, lam_v and seed with only frac changed (the manifest test checks this against Z's manifest): one block,
d_model 128, 4 heads, MLP 512, float64; RMS head input with the gain 2 -> 1 (eta_g 2e-4, lam_g 1); Polar feature steps eta_w 0.1,
lam_w 0.01; head eta_v 0.03; Fourier readouts on, heavy logs every 100, probe every 200; early stop once held-out accuracy reaches 0.99
(reached at update t, the run stops at max(1.5 t, t + 5000)); at most 30k updates. Run IDs `ZP_P_p<p>_f<frac>_lam_v<lam_v>_s<seed>`; cells as in Z
(`P_p61_f0.33_lam_v0.45`, ...).

## 4. Code

- `tf_runner.py` 1.6 and `worker.py`: campaign Z's files, unchanged (same SHA-256). `campaign.json`: Z's with the name `ZP`.
- `tests/test_tf.py`: Z's 30 runner tests unchanged; the manifest test is replaced by ZP's (every run is Y's G2 with p, frac and
  lam_v changed; the six cells, seeds 11-15, one thread; against Z's manifest, only frac differs; at most 4 runs per worker).
- `make_manifest.py` writes `manifest.json` (and `manifest_test.json`, a miniature for the end-to-end check).

## 5. Analysis (fixed before the runs)

`analyze_ZP.py OUT.json ZP_RESULTS [Z_RESULTS Y_RESULTS]` runs campaign Z's analysis unchanged (analyze_Z.py: Y's thresholds,
q_te >= 0.75, Fourier top-8 refit gain >= 0.2, dA_H >= 0.02, at most 10 misclassified training pairs in the window (strict: 0),
final held-out accuracy on all pairs >= 0.99; pass = both held-out definitions) on the ZP runs, with Z's P cells, Z's
R_frac0.3 and Y's G2 alongside. It adds:
- **timing**: per cell x, the rule's prediction, the measured rise start, the CE dip and their ratio (the checks of section 2);
- **appendix choice**: per p, the cell with the most strict passes (ties: more loose passes, then the smaller frac); the recipe
  works at that p if that cell passes strictly in at least 4 of 5 seeds.

## 6. Figures (`plot_ZP.py ANALYSIS.json OUT_DIR`)

- `ZP_p61.png`, `ZP_p89.png`, `ZP_p149.png`: per p, Z's old fraction next to ZP's two (rows: training CE / log p, accuracy on
  both held-out definitions with the readouts, A_H, gain; median and interquartile band over seeds; the paper's window shaded).
- `ZP_appendix.png`: one column per p (61, 89, 113, 149) with the appendix choice (p = 113: Y's G2, 10 seeds).
- `ZP_timing.png`: (a) rise start against x, one point per run, with the rule; (b) rise start / CE dip, filled = strict pass.

## 7. Sessions

8 sessions ZP1-ZP8 of 4 cores, 3 or 4 runs each, so every run starts at once. One-block cost as in X, Y and Z (about 0.19 s per
update at p = 113, frac 0.2, scaled by p^2 and frac): about 25 core-hours if the runs stop near their expected length, about
50 if all run to 30k. Wall time about 1-1.5 hours, at most about 2.5 (p = 149 runs to 30k). Fewer sessions also work: after
15 minutes the others take over the runs of IDs that never started.

## 8. Reading the outcome

- A p whose best cell passes in at least 4 of 5 seeds goes into the appendix figure with that fraction, stated per p ("frac set
  so the rise starts after the CE dip").
- If p = 61 fails again on the final accuracy only, the stall is a small-p effect at this lam_v, not timing: the next lever
  would be the head decay (Z: lam_v 0.6 was worse than 0.45) or a larger p near 61.
- If a cell fails on the window share, the timing figure shows whether its rise came before the dip (rule wrong for that p)
  or the all-pairs start was too high (mirror bound).

## 9. Checks before launch

- tests/test_tf.py (31 tests) pass; the worker's selftest passes; an end-to-end run of manifest_test.json (2 cells x 2 seeds,
  miniature) finishes and pushes through the worker to a local bare repository.
- analyze_ZP.py and plot_ZP.py run on Z's and Y's results (the comparison cells), so only the ZP cells are new at analysis time.
