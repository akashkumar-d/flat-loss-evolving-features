# Paper figures for campaign N: files and a draft caption (claude/1hl-figures, 2026-10-12)

`N_ladder_D{3,4}_p{47,61}_paper.{pdf,png}` (6.75 x 5.6 in), drawn by `paper_N.py` (from campaign_N:
`python3 -I paper_N.py collected/results analysis/analysis_N.json analysis/paper`) with the paper's renderer for the
deep-network figures (acrossp_figs.py: colour key, fonts, seeds thin and median bold), each column ending at its shortest run,
and counts and windows from `analysis_N.json`. Draft caption (one per figure; fill in depth and p):

Deep network of depth D at p = P across training fractions. Ten seeds per fraction, with initialization and split seeds that
were not used to choose any setting; thin lines: seeds, bold: median. Rows: training cross-entropy divided by log p (mean over
the two-update cycle); accuracy on held-out pairs (vermillion), of the train-only ridge refit of the last hidden layer
(purple) and on the training set (dashed); misclassified training pairs (symmetric log scale); Fourier concentration $A_H$ of
the last (solid) and first (dash-dot) hidden layer. Grey: the paper's post-hoc window (median over seeds). r is the rate
factor of the last hidden layer; head decay $\lambda_V = 0.2 \cdot 31/p$. Each column ends at its shortest run, so every seed
contributes at every plotted point. A seed passes if the refit gains at least 0.2 inside the window, held-out accuracy gains at
least 0.2 from the window's start and ends at least 0.99, at least 75% of both the held-out and the refit 10% -> 90% rises lie
inside the window, and $A_H$ of the last hidden layer rises by at least 0.02 across it.

Pass counts (also in the column titles): depth 3, p = 47: 10, 10, 10, 8 of 10 (fractions 0.8, 0.7, 0.6, 0.5); depth 3,
p = 61: 10, 10, 10, 10; depth 4, p = 47: 9, 10, 10, 9; depth 4, p = 61: 9, 10, 10, 10. In all, 155 of 160.
