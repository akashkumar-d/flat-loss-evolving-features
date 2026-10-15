# Campaign Y (transformer R5 with a gain on the RMS head input, seeds 11-20): results, all 20 runs (claude/1hl-figures, 2026-10-11)

**State.** 20 of 20 runs finished (stopped by the early-stop rule at 9.1k-14.0k updates, 31-45 minutes each; final held-out
accuracy on all held-out pairs 0.9993-1.000), no failed runs, no push errors; all 5 sessions report `campaign_done` (last
heartbeat 22:31Z). The results were read from the session branches through Akash's browser and laid out as
collect_results.sh does; every run.json.gz matches its chain.json. The analysis is PLAN.md section 4 (`analyze_Y.py`,
`plot_Y.py`; `analysis_Y.json`, `analyze_Y.log`, `figs/`), applied the same way to campaign X's R5 runs on the same seeds (same
initialization and split, no gain). Held-out accuracy is on the no-mirror pairs; the CE is the training cross-entropy averaged
over 201 updates. Development runs only; not evidence for the paper's claims.

## 1. The loss across the held-out rise (the point of the campaign)

| | X R5, no gain | G2: gain 2 -> 1, time constant 5k | G3: gain 3 -> 1, time constant 4k |
|---|---|---|---|
| CE change, held-out 0.1 -> 0.9 | -8.4% (-10.2% to -6.7%), up in 0/10 | **+33% (+18% to +54%), up in 10/10** | **+111% (+85% to +161%), up in 10/10** |
| largest drop below the CE's running max since held-out 0.1 | 6.7-10.2% | at most 0.04% | at most 0.03% |
| CE at held-out 0.5 | 1.60 nats (0.34 log p) | 0.68 nats (0.14 log p) | 0.28 nats (0.06 log p) |
| misclassified training pairs during the rise (most at any log) | 0-6 | 0 | 0 |
| first full fit of the training set (median) | 1190 updates | 200 | 160 |
| paper's window (CE never > 3% below its running max): share of the held-out rise q_te | 0.43-0.61 | 1.00 in 10/10 (window from 720-800 updates to the end of the run) | 1.00 in 10/10 (same) |
| inside the window: Fourier top-8 refit gain / paper's refit gain / Delta A_H (medians) | 0.15 / 0.17 / 0.12 | 0.97 / 0.995 / 0.81 | 0.97 / 0.996 / 0.81 |
| **pass_Y (strict: no misclassified training pair inside the window)** | **0/10 (0)** | **10/10 (10)** | **10/10 (10)** |
| campaign W's raw-CE 10% window: met strict / at most 10 misclassified; share of the rise | 5/10 / 10/10; 0.09-1.00 | 0/10 / 0/10; 0.24-0.52 | 0/10 / 0/10; 0.12-0.17 |

With the gain, the training CE reaches its minimum at 0.9-1.1k updates (median 0.47 nats in G2, 0.10 in G3) and then rises
without interruption to the end of every run (median 1.15 and 1.04 nats; from held-out 0.1 on, never more than 0.04% below its
running maximum), while the held-out accuracy goes 0 -> 1 and A_H of the head input goes from 0.07-0.11 at the CE's minimum to
0.84-0.90 at the end. W's raw-CE 10% window fails by design: it asks for a flat CE, and here the CE rises.
The rise of the CE is the gain relaxing toward 1 under weight decay (a scale effect), the transformer counterpart of the
paper's MLP, where the large first layer shrinks under weight decay while the features improve.

## 2. Readouts against the network's held-out accuracy (lead = network crossing minus readout crossing, updates)

| readout | X R5, lead at 0.5 / 0.9 | G2, lead at 0.5 / 0.9 | G3, lead at 0.5 / 0.9 |
|---|---|---|---|
| Fourier-filtered refit, top 8 (primary) | 10/10 lead, +838 [+512, +1396] / 10/10, +970 | 10/10, +868 [+324, +1442] / 10/10, +909 [+236, +2057] | 10/10, +754 [+364, +948] / 10/10, +820 [+250, +1151] |
| network's logits restricted to the top 8 | 10/10, +767 / 10/10, +903 | 10/10, +821 / 10/10, +749 | 10/10, +681 / 10/10, +731 |
| logistic probe | 10/10, +104 / 10/10, +270 | 8/10, +42 / 10/10, +50 | 6/10, +26 / 10/10, +46 |
| the paper's refit (all frequencies) | 0/10, -1339 / 0/10, -2321 | 0/10, -1294 [-1971, -669] / 0/10, -2145 | 0/10, -1062 [-1569, -694] / 0/10, -1503 |

The gain leaves the readout picture of campaign X unchanged: the Fourier-filtered refit leads in all 20 runs by 0.3-1.4k
updates at 0.5, the paper's refit trails in all 20.

**Timing.** Network held-out crossings at 0.1 / 0.5 / 0.9 (medians): X R5 2240 / 3350 / 4500, G2 2670 / 3940 / 4900, G3
2500 / 3570 / 4630. Paired by seed (arm minus X): G2 +420 / +590 / +460 (earlier in 3, 4, 4 of 10 seeds), G3 +240 / +70 /
-300 (earlier in 3, 5, 6 of 10); the duration of the rise from 0.1 to 0.9 changes by -70 (G2) and -480 (G3) updates. So the
seed-1 pilots' "slightly earlier" does not hold up: the rise comes at about the same time, within seed-to-seed variation.
The gain at the 0.1 / 0.5 / 0.9 crossings: G2 1.58 / 1.45 / 1.37, G3 2.07 / 1.82 / 1.63.

**Frequencies at the network's 0.5 crossing.** 7-13 (G2) and 8-11 (G3) of the 56 frequencies carry held-out R^2 > 0.05
(X R5: 6-10); of the top 8 by training energy, 6-8 (G2) and 7-8 (G3) are among them.

## 3. Figures (`figs/`)

`Y_main` (one column per setting, X R5 / G2 / G3; medians over 10 seeds with interquartile bands, seeds thin: training CE with
the median paper window shaded; accuracies with the readouts; A_H of the head input; the gain), `Y_rise` (CE relative to its
value at held-out 0.1 against held-out accuracy, every seed), `Y_freq` (campaign X's frequency figure on the seed with each
arm's median lead).

## 4. Where

Mac: `~/Downloads/modular arithmetic/natural_plateau_shared/campaign_Y/collected/` (20 runs, status, INDEX.json;
collect_results.sh's layout) and `campaign_Y/analysis/` (this summary, `analysis_Y.json`, `analyze_Y.log`, `plot_Y.log`,
`figs/`). `plot_Y.py` there now has the import-path fix (`sys.path.insert(0, HERE)`; the figures do not depend on it).
