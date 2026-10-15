# Campaign X (Fourier readouts on transformer R5 and R4, seeds 11-20): results, all 20 runs (claude/1hl-figures, 2026-10-11)

**State.** 20 of 20 runs finished (stopped by the early-stop rule at 8.6k-15.8k updates; final held-out accuracy 0.999-1.000),
no failed runs, no push errors; all 5 sessions report `campaign_done` (last heartbeat 14:46Z). The results were read from the
session branches through Akash's browser and laid out as collect_results.sh does; every run.json.gz matches its chain.json.
The analysis is PLAN.md section 4 (`analyze_X.py`, `plot_X.py`, unchanged; `analysis_X.json`, `analyze_X.log`, `figs/`), plus
campaign Y's loss-in-the-window measures applied to these runs (`rise_paperwindow_X.json`, from `campaign_Y/analyze_Y.py`).
Held-out accuracy is on the no-mirror pairs. Development runs only; not evidence for the paper's claims.

## 1. Readouts against the network's held-out accuracy (lead = network crossing minus readout crossing, updates)

| readout | R5, lead at 0.5 | R5, lead at 0.9 | R4, lead at 0.5 | R4, lead at 0.9 |
|---|---|---|---|---|
| Fourier-filtered refit, top 8 (primary) | 10/10 lead, +838 [+512, +1396] | 10/10, +970 [+519, +1724] | 10/10, +895 [+558, +1244] | 10/10, +913 [+608, +2177] |
| Fourier-filtered refit, top 16 | 10/10, +474 | 10/10, +531 | 10/10, +599 | 10/10, +546 |
| Fourier-filtered refit, top 4 | 9/10, +680 | 3/10, -117 | 10/10, +605 | 4/10, -197 |
| phase vote, top 8 | 10/10, +705 | 10/10, +707 | 10/10, +726 | 10/10, +732 |
| network's logits restricted to the top 8 | 10/10, +767 | 10/10, +903 | 10/10, +845 | 10/10, +768 |
| logistic probe | 10/10, +104 | 10/10, +270 | 10/10, +103 | 10/10, +194 |
| the paper's refit (all frequencies) | **0/10**, -1339 [-2201, -769] | 0/10, -2321 | **0/10**, -1371 [-2404, -808] | 0/10, -2624 |

So on 20 fresh seeds the exploratory replays hold: the paper's ridge refit trails the transformer by 0.8-2.4k updates, and
restricting the same refit to the 8 frequencies with the largest training energy (train-only) makes it lead in every run, by
about 0.5-1.4k at 0.5 and 0.5-2.2k at 0.9. The network's own logits restricted the same way lead by as much. Janati et al.'s
"family alone" diagnostics lead by 1.9-3.4k at 0.5 (medians; they use the full grid and are not readouts).

**Frequencies at the network's 0.5 crossing.** 6-10 of the 56 frequencies carry held-out R^2 > 0.05 (R5: 6-10, R4: 7-10); of the
top 8 by training energy, 6-8 are among them (all 8 in 16 of 20 runs), and the top 8 overlaps the head's top 8 in 7-8 and the
embedding's in 6-8. In the paper's MLP (p = 31, 8 runs) all 15 frequencies carry held-out signal and the plain refit already
leads (+2.6k at 0.5).

## 2. The training loss in the window (the point raised when X was launched)

| | R5 | R4 | MLP 1HL (p = 31) |
|---|---|---|---|
| CE change, held-out 0.1 -> 0.9 (201-update average) | -6.7% to -10.2% (median -8.4%) | -7.8% to -12.4% (median -10.1%) | |
| CE change over the band [filtered-refit onset, network 0.9] | -11% to -17% | -12% to -19% | +1.6% to +8.5% |
| slope over the band, % per 1000 updates | -2.2 to -8.8 | -3.0 to -8.9 | -0.4 to +0.6 |
| CE at held-out 0.5 | 1.58-1.63 nats (0.33-0.35 log p) | 1.34-1.39 nats (0.28-0.29 log p) | about 0.45 log p |
| the paper's window (CE never > 3% below its running max): share of the held-out rise | 0.43-0.61 | 0.38-0.54 | |
| campaign W's raw-CE 10% window, strict / at most 10 misclassified | 5/10 / 10/10 | 5/10 / 6/10 | |
| most misclassified training pairs during the rise | 0-6 | 0-3 | 0 |

In every X run the training CE falls while the held-out accuracy rises (20/20), by 7-12% from held-out 0.1 to 0.9 (medians
8% for R5 and 10% for R4), so
under the paper's window definition none of the 20 runs qualifies (pass_Y 0/20 in campaign Y's terms). The fall is not
something the rise causes: the head reaches its norm equilibrium within a few hundred updates and the CE relaxes slowly from
then on (campaign W, section 6b). Campaign Y (prepared) adds a gain on the RMS head input that relaxes from 2 or 3 to 1 under
weight decay, which in the seed-1 pilots turns the CE across the rise into +44% and +162%, with the rise slightly earlier and the
Fourier-filtered refit leading by 1.3-1.4k (1.2k without the gain).

## 3. Figures (`figs/`)

`X_main` (CE, accuracies with the readouts, A_H; medians over 10 seeds with interquartile bands; grey: the median band from
the filtered refit's onset to the network's 0.9), `X_freq` (held-out R^2 per frequency over time, spectra at the network's 0.5,
share of frequencies with held-out signal; MLP column for contrast), `X_leads` (leads per seed and readout), `X_family`
(Janati diagnostics).

## 4. Where

Mac: `~/Downloads/modular arithmetic/natural_plateau_shared/campaign_X/collected/` (20 runs, status, INDEX.json; collect_results.sh's
layout) and `campaign_X/analysis/` (this summary as SUMMARY.md, `analysis_X.json`, `analyze_X.log`, `rise_paperwindow_X.json`,
`figs/`, `unpack_X.py`).
