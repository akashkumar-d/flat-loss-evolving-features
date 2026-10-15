# Campaign M (training-fraction ladder, last hidden layer's rate re-tuned): results, all 216 runs (claude/1hl-figures, 2026-10-11)

**State.** 216 of 216 runs finished, no failed runs, no push errors. 23 sessions report `campaign_done` (last heartbeat
2026-10-11T11:49Z); M23 never pushed a status file and its 9 runs were taken over by other sessions. The results were read
from the session branches through Akash's browser and laid out as collect_results.sh does (`collected/results`,
`collected/status`, `collected/INDEX.json`); every run.json.gz matches the SHA-256 in its chain.json. The analysis is the one
fixed in PLAN.md section 4 (`analyze_M.py`, `plot_M.py`, unchanged; log `analyze_M.log`, per run `analysis_M.json`), plus a
descriptive per-cell table (`cells_M.json`). Development runs only; not evidence for the paper's claims.

**Reproducibility.** The 12 runs at fraction 0.8 with the linear rule repeat campaign K's runs: the loss is identical to the
last digit (largest difference 0 over the 31-48k updates both have).

## 1. Verdicts

pass_M = Akash's refit criterion (campaigns K and L) and >= 75% of both the held-out and the refit 10% -> 90% rises inside the
paper's post-hoc window W and Delta A_H >= 0.02; pass_M3 adds training accuracy >= 0.97 at every logged step of W.
Overall: **pass_M 135/216**, pass_M3 128/216, refit criterion 169/216. Campaign L (fraction 0.5, r = 0.45): 0/24.

| fraction | r (depth 3 / depth 4) | depth 3 (12 runs) | depth 4 (12 runs) |
|---|---|---|---|
| 0.8 | 0.45 | 12/12 | 12/12 |
| 0.7 | 0.45 | 12/12 | **0/12** |
| 0.7 | 0.30 / 0.28 | 12/12 | 12/12 |
| 0.6 | 0.45 | **0/12** | **0/12** |
| 0.6 | 0.28 / 0.20 | 10/12 | 11/12 |
| 0.6 | 0.18 / 0.13 | 10/12 | 10/12 |
| 0.5 | 0.45 (campaign L) | 0/12 | 0/12 |
| 0.5 | 0.20 / 0.14 | 5/12 | 4/12 |
| 0.5 | 0.13 / 0.09 | 5/12 | 6/12 |
| 0.5 | 0.08 / 0.06 | 8/12 | 6/12 |

By head decay: at fraction 0.5 the linear rule (lam_v = 0.2 x 31 / p) passes **30/36** and the paper's lam_v = 0.2 4/36; at
0.6 with re-tuned r, 24/24 and 17/24. With the linear rule every re-tuned setting at p = 61 passes 3/3 at every fraction
(depths 3 and 4). Figures: `M_scoreboard` (pass_M, with the refit criterion, pass_M3 and median q_te in small print),
`M_refit_scoreboard`, `M_share` (q_te per run), `M_drawdown`.

## 2. Why runs fail

- **r = 0.45 below fraction 0.8** (0.6 at both depths, 0.7 at depth 4; 36 runs, 0 pass): the post-knee CE peak comes at
  21-27k updates, before the held-out rise (held-out 0.5 at 28-62k). The CE then falls 5-18% from the refit's 0.2 crossing to
  the held-out 0.9 crossing, and W holds 0-67% of the held-out rise. The refit criterion alone still passes in 33 of the 36.
- **The paper's lam_v = 0.2 at fraction 0.5**: the rise lies inside W (q_te 0.45-1.00, mostly 1.00) but does not finish:
  final held-out 0.84-0.997, below 0.99 in 32 of 36 runs at the 250k horizon. The same final-accuracy failure (0.973-0.990) takes
  7 of the 24 re-tuned paper-rule runs at 0.6, always seed 2 or 3.
- **The largest r at 0.5** (0.20 at depth 3, 0.14 at depth 4): the CE peak (56-94k) falls inside a slow rise (held-out 0.9 at
  72-247k), so part of the rise comes after W (q_te 0.45-1.00; 9/24 pass).
- With the linear rule at 0.5, 6 failures, all at p = 47 (seed 2 in 5 of them): final held-out 0.974-0.989 in 5 (one of them
  also q_te 0.54), and q_te 0.49 in one depth-4 run at r = 0.14 (seed 3).

## 3. The loss across the rise (passing runs)

CE (cycle mean) from the refit's 0.2 crossing to the held-out 0.9 crossing, and the CE's drawdown below its running maximum
(since the knee) at the held-out 0.9 crossing:

| fraction | passing runs | CE change, range (median) | runs where the CE rose | drawdown at held-out 0.9 |
|---|---|---|---|---|
| 0.5 | 34 | +4% to +1230% (+157%) | 34/34 | median 0, at most 6.7% |
| 0.6 | 41 | +3% to +300% (+32%) | 41/41 | median 0, at most 2.1% |
| 0.7 | 36 | -3% to +47% (+8%) | 29/36 | median 0, at most 4.3% |
| 0.8 | 24 | -4% to +14% (-0.2%) | 10/24 | median 1.6%, at most 4.1% |

At the smaller fractions the re-tuned (smaller) r moves the CE peak after the rise, so the held-out accuracy and the refit rise
while the training CE is still climbing from its dip. The very large relative changes at small r start from the CE dip (a few
hundredths of log p). Training fit in W: pass_M3 fails in 7 of the 135 passing runs (all r = 0.45 with the paper's lam_v at
p = 61, fractions 0.8 and 0.7: least training accuracy 0.94-0.97); at most 10 training pairs misclassified at any log of W at
0.5, 66 at 0.6.

**Timing.** The refit's 0.5 crossing precedes the held-out 0.5 crossing in all 216 runs, by 1.0k to 63k updates (2-18k at the
smallest r at fraction 0.5, 22-63k with the paper's lam_v at the larger r there).

## 4. The r that works per fraction

`plot_M.py`'s choice (most pass_M, then pass_M3, then the higher median q_te), used in `ladder/`:

| group | 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|
| depth 3, p 47, lam_v 0.2 | 0.45 (3/3) | 0.45 (3/3) | 0.28 (2/3) | 0.13 (0/3) |
| depth 3, p 47, linear | 0.45 (3/3) | 0.45 (3/3) | 0.28 (3/3) | 0.08 (3/3) |
| depth 3, p 61, lam_v 0.2 | 0.45 (3/3) | 0.30 (3/3) | 0.28 (2/3) | 0.08 (2/3) |
| depth 3, p 61, linear | 0.45 (3/3) | 0.30 (3/3) | 0.28 (3/3) | 0.20 (3/3) |
| depth 4, p 47, lam_v 0.2 | 0.45 (3/3) | 0.28 (3/3) | 0.20 (2/3) | 0.09 (1/3) |
| depth 4, p 47, linear | 0.45 (3/3) | 0.28 (3/3) | 0.20 (3/3) | 0.09 (2/3) |
| depth 4, p 61, lam_v 0.2 | 0.45 (3/3) | 0.28 (3/3) | 0.20 (3/3) | 0.09 (0/3) |
| depth 4, p 61, linear | 0.45 (3/3) | 0.28 (3/3) | 0.20 (3/3) | 0.09 (3/3) |

So the PLAN's recipe holds: r has to shrink with the training fraction, roughly 0.45 -> 0.3 -> 0.2-0.28 -> 0.06-0.2 from 0.8
to 0.5, and at 0.5 the linear head-decay rule is needed for the held-out accuracy to finish within 250k updates.

## 5. Files

`analysis/`: this summary, `analysis_M.json`, `analyze_M.log`, `cells_M.json`, `plot_M.log`, `figures/` (scoreboards,
`M_share`, `M_drawdown`, `groups/` with every setting per group, `ladder/` with the best r per fraction). `collected/`: the
216 runs, the status files and INDEX.json. `unpack_M.py` turned the browser transfer into that layout.
