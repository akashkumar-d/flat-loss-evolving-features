# Paper figures for campaign ZP: files and draft captions (claude/1hl-figures, 2026-10-12)

Drawn by `paper_ZP.py` (from campaign_ZP: `python3 -I paper_ZP.py analysis/analysis_ZP.json analysis/paper`), from
`analysis_ZP.json` without rescoring. Colour key of the paper's deep-network figures (refit solid purple); the transformer's
Fourier refit is sky blue. PDF and PNG, 6.75 in wide, text 8 pt. Draft captions below; "campaign Z" and "campaign Y" are
internal names to replace.

## ZP_acrossp_paper (6.75 x 5.6 in)

One-block transformer across p. Polar feature steps; the gain on the RMS-normalized head input relaxes from 2 to 1 (time
constant 5k updates); at each p the training fraction (0.33, 0.25, 0.2, 0.16) puts the start of the held-out rise after the
training CE's minimum. Medians over seeds (5; 10 at p = 113) with interquartile bands; misclassified training pairs: seeds
thin, median bold. Held-out accuracy on all held-out pairs (solid) and on the no-mirror pairs, those (a, b) whose mirror
(b, a) is not a training pair (dashed). Refit: the train-only ridge refit of the head input; Fourier refit (top 8): the same
ridge with its sum over frequencies restricted to the 8 frequencies with the most training energy. Grey: the paper's window
(median over seeds). Each column ends at its shortest run. A run passes if it passes on both held-out definitions with no
training pair misclassified inside the window; at p = 149 all 5 pass if up to 10 of the 3,552 training pairs may be
misclassified, and at p = 61 all 5 pass on the no-mirror pairs (Figure [mirror]).

## ZP_mirror_paper (6.75 x 5.0 in)

Mirror pairs and the all-pairs share. Top: held-out accuracy at the four settings of Figure [across p] on the held-out pairs
(a, b) whose mirror (b, a) is a training pair (dotted), on the other held-out pairs (dashed) and on all of them (solid);
medians, log time axis; dashed black: training accuracy; grey: the window of the all-pairs definition (median); dot: the
median all-pairs accuracy when it opens. Mirror pairs are answered correctly soon after the training set is fitted. At
p = 61, 90% of them are correct by update 240-520, before the window opens (580-700), so the all-pairs accuracy starts the
window at 0.30; when the window opens, 63-88% of the mirror pairs are correct at p = 89, 29-74% at p = 113 and 19-67% at
p = 149. Bottom left: all-pairs accuracy when the window opens against the share of held-out pairs that are mirrors, one
point per run (p = 61 at training fractions 0.30, 0.33 and 0.35; p = 89 at 0.25 and 0.275; p = 113 at 0.2; p = 149 at 0.15
and 0.16). Bottom right: share of the held-out 10% -> 90% rise inside the window against that accuracy $a_0$. Every run lies
on $(0.9 - \max(a_0, 0.1))/0.8$, the largest share a window opening at $a_0$ can hold, so the share falls below 0.75 exactly
when $a_0 > 0.3$. Crosses: the no-mirror pairs, whose share is at least 0.96 in every run. One p = 149 seed that never
generalized is left out.

## ZP_p61_fraction_paper (6.75 x 2.35 in)

p = 61 at three training fractions ($\lambda_V = 0.45$, 5 seeds each; 0.30 from campaign Z). (a) All-pairs held-out accuracy
when the window opens; dotted: the share of held-out pairs that are mirrors. (b) Share of the held-out rise inside the
window, all pairs (circles) and no-mirror pairs (crosses). (c) Final held-out accuracy on all pairs (at 0.30 the runs reach
the 30k-update budget). Filled: the run passes. At 0.30 the window opens below 0.3, but held-out accuracy stalls below 0.99 in
4 of 5 runs; at 0.33 and 0.35 the runs finish, and the mirror pairs, learned before the window, hold the all-pairs share at
the 0.75 threshold (0.33: 3 of 5 pass) or below it (0.35: none). On the no-mirror pairs every rise lies inside the window.

## Per-run numbers

`mirror_ZP.json`: for every run in the two mirror figures, the mirror share, first fit, the updates at which the mirror pairs
reach 50% and 90%, both windows, the all-pairs accuracy when the all-pairs window opens (a0) and its mirror / no-mirror parts,
the bound, both shares and the pass flags. Checks: the derived mirror-pair accuracy at the first fit equals analyze_ZP's
independent value in all 45 runs; every included run's all-pairs share equals the bound to 0.002.
