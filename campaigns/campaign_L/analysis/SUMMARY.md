# Campaign L (training fraction 0.5): results, all 120 runs (claude/1hl-figures, 2026-10-10)

**State.** 120 of 120 runs finished. All 20 sessions report `campaign_done` (last heartbeat 2026-10-10T12:43Z), no failed
runs, no push errors. The results were read from the 20 session branches through Akash's browser and laid out as
collect_results.sh does (`collected/results`, `collected/status`, `collected/INDEX.json`); every run.json.gz matches the
SHA-256 in its chain.json. The analysis is the one fixed in PLAN.md (campaign K's, unchanged): `analyze_L.py` (log:
`analyze_L.log`, per run: `runs_L.json`), `plot_L.py`, plus two descriptive scripts, `summarize_L.py` (the table below) and
`extra_L.py` (`extra_L.json`: CE change and training fit while the refit leads). Development runs only; not evidence for the
paper's claims.

## 1. Verdicts

| | fraction 0.5 (L) | fraction 0.8 (K, same seeds; paper setting: F/G, 10 seeds) |
|---|---|---|
| campaign C's per-run endpoint | **0/120** | 8/176 |
| refit criterion (post-hoc window, refit +0.2 inside, held-out +0.2 to the end, final >= 0.99) | **14/120** | 157/176 |
| the same with training accuracy >= 0.97 throughout the window (pass3) | 8/120 | - |

Endpoint failures at 0.5: share 120, final accuracy 89, training fit inside the band 89, A_H rise 23. Refit-criterion
outcomes at 0.5: pass 14; final accuracy only 42; refit gain only 17; refit and final 17; all three 30 (the 30 p = 23 runs).
The 14 passes are all depth 3: p = 47 (square-root and linear rules, lam_w 0.015: 2/3 each), p = 61 (linear, lam_w 0.015:
3/3; square-root 0.015: 2/3; linear 0.025: 1/3), p = 101 (square-root 0.015 and linear 0.025: 2/3 each). Depth 4: 0/60.
Figures: `L_scoreboard`, `L_refit_scoreboard`.

## 2. Paired with campaign K (same seeds; L's training set is a subset of K's)

| depth, p | setting (rule lam_v, lam_w) | endpoint 0.5 (0.8) | refit 0.5 (0.8) | t10 at 0.5 (0.8), k | t90 at 0.5 (0.8), k | refit lead at 0.5 (0.8), k | CE peak / log p at 0.5 (0.8) | final held-out at 0.5 |
|---|---|---|---|---|---|---|---|---|
| D3 p23 | paper 0.2, 0.015 | 0/3 (0/10) | 0/3 (10/10) | never (20-28) | never (48-64) | - (16.0-20.5) | 0.21 (0.29-0.31) | 0.000-0.004 |
| D3 p23 | sqrt 0.2322, 0.015 | 0/3 (0/3) | 0/3 (3/3) | never (23-30) | never (56-68) | - (17.2-24.6) | 0.28 (0.36-0.38) | 0.000-0.004 |
| D3 p23 | sqrt 0.2322, 0.025 | 0/3 (0/3) | 0/3 (0/3) | never (18-25) | never (71-112) | - (25.6-28.5) | 0.51 (0.58-0.59) | 0.000-0.004 |
| D3 p23 | linear 0.2696, 0.015 | 0/3 (0/3) | 0/3 (3/3) | never (23-32) | never (71-75) | - (21.8-26.9) | 0.35 (0.44-0.45) | 0.000-0.004 |
| D3 p23 | linear 0.2696, 0.025 | 0/3 (0/3) | 0/3 (0/3) | never (20-27) | never (83-106) | - (27.9-33.0) | 0.57 (0.63-0.64) | 0.004-0.008 |
| D3 p47 | paper 0.2, 0.015 | 0/3 (2/10) | 0/3 (10/10) | 64-72 (14-16) | 152-180 (19-23) | 43.5-57.1 (2.2-3.2) | 0.71-0.72 (0.68-0.69) | 0.957-0.982 |
| D3 p47 | sqrt 0.1624, 0.015 | 0/3 (0/3) | 2/3 (3/3) | 61-67 (14-15) | 126-154 (19-21) | 37.0-47.8 (2.0-2.9) | 0.65-0.66 (0.61-0.62) | 0.973-0.995 |
| D3 p47 | sqrt 0.1624, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 53-56 (12) | 182-250, 1 never (16-17) | 51.8-56.9 (1.2-1.6) | 0.77-0.78 (0.75) | 0.888-0.943 |
| D3 p47 | linear 0.1319, 0.015 | 0/3 (0/3) | 2/3 (3/3) | 59-62 (14-15) | 110-124 (19-20) | 32.0-37.7 (2.1-2.4) | 0.58-0.59 (0.53-0.54) | 0.986-1.000 |
| D3 p47 | linear 0.1319, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 52-54 (11-12) | 144-177 (15-17) | 42.6-46.6 (1.1-1.7) | 0.73 (0.70) | 0.948-0.985 |
| D3 p61 | paper 0.2, 0.015 | 0/3 (0/10) | 0/3 (10/10) | 52-60 (13-14) | 115-164 (18-21) | 26.3-30.0 (1.3-2.4) | 0.81-0.82 (0.77-0.78) | 0.930-0.987 |
| D3 p61 | sqrt 0.1426, 0.015 | 0/3 (0/3) | 2/3 (3/3) | 48-57 (13-14) | 92-105 (18-19) | 20.3-21.7 (1.2-1.7) | 0.74-0.75 (0.68-0.69) | 0.984-0.996 |
| D3 p61 | sqrt 0.1426, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 41-47 (11) | 93-242 (14-15) | 19.0-24.5 (0.5-1.2) | 0.83-0.84 (0.80) | 0.903-0.983 |
| D3 p61 | linear 0.1016, 0.015 | 0/3 (0/3) | 3/3 (3/3) | 46-49 (14) | 76-78 (17-18) | 16.1-17.6 (1.0-1.6) | 0.65 (0.58) | 0.998-0.999 |
| D3 p61 | linear 0.1016, 0.025 | 0/3 (0/3) | 1/3 (3/3) | 38-47 (11) | 85-104 (14-15) | 17.2-18.0 (0.6-0.8) | 0.77-0.78 (0.71-0.72) | 0.974-0.996 |
| D3 p101 | paper 0.2, 0.015 | 0/3 (0/10) | 0/3 (10/10) | 51-52 (12-14) | 123-155, 1 never (17-20) | 15.5-16.5 (0.0-1.2) | 0.93 (0.89-0.90) | 0.899-0.945 |
| D3 p101 | sqrt 0.1108, 0.015 | 0/3 (0/3) | 2/3 (3/3) | 42-53 (13) | 70-93 (17-18) | 9.2-10.4 (-0.3-0.4) | 0.87 (0.82-0.83) | 0.980-0.996 |
| D3 p101 | sqrt 0.1108, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 37-43 (10-11) | 73-76, 1 never (13-14) | 8.8-9.5 (-0.2-0.3) | 0.91-0.92 (0.87-0.88) | 0.899-0.955 |
| D3 p101 | linear 0.0614, 0.015 | 0/3 (0/3) | 0/3 (3/3) | 39-50 (13) | 59-73 (17-18) | 5.2-6.2 (-0.2-0.2) | 0.77-0.78 (0.72-0.73) | 0.998-1.000 |
| D3 p101 | linear 0.0614, 0.025 | 0/3 (0/3) | 2/3 (3/3) | 34-40 (10-11) | 52-70 (13-14) | 5.0-6.3 (-0.3-0.1) | 0.85 (0.77-0.78) | 0.978-0.996 |
| D4 p23 | paper 0.2, 0.015 | 0/3 (0/10) | 0/3 (10/10) | never (25-38) | never (63-86) | - (19.2-30.0) | 0.25 (0.33-0.34) | 0.000 |
| D4 p23 | sqrt 0.2322, 0.015 | 0/3 (0/3) | 0/3 (2/3) | never (28-35) | never (75-89) | - (22.8-27.3) | 0.32 (0.40-0.41) | 0.000 |
| D4 p23 | sqrt 0.2322, 0.025 | 0/3 (0/3) | 0/3 (1/3) | never (25-28) | never (87-108) | - (26.9-36.6) | 0.55 (0.61-0.62) | 0.000 |
| D4 p23 | linear 0.2696, 0.015 | 0/3 (0/3) | 0/3 (1/3) | never (30-39) | never (76-114) | - (25.0-36.4) | 0.39-0.40 (0.47-0.48) | 0.000-0.004 |
| D4 p23 | linear 0.2696, 0.025 | 0/3 (0/3) | 0/3 (0/3) | never (25-30) | never (97) | - (33.4-34.1) | 0.60 (0.66-0.67) | 0.000 |
| D4 p47 | paper 0.2, 0.015 | 0/3 (0/10) | 0/3 (10/10) | 98-112 (18-22) | 182-197 (26-31) | 53.1-58.5 (3.2-4.9) | 0.75 (0.74-0.77) | 0.963-0.985 |
| D4 p47 | sqrt 0.1624, 0.015 | 0/3 (3/3) | 0/3 (3/3) | 97-107 (18-20) | 177-188 (27-29) | 48.0-54.1 (3.3-3.5) | 0.70 (0.69-0.71) | 0.971-0.996 |
| D4 p47 | sqrt 0.1624, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 81-102 (15-17) | 210-232, 1 never (23-25) | 61.7-67.7 (2.0-2.4) | 0.81 (0.78-0.79) | 0.891-0.953 |
| D4 p47 | linear 0.1319, 0.015 | 0/3 (3/3) | 0/3 (3/3) | 89-97 (18-20) | 150-163 (25-28) | 41.2-43.5 (2.3-3.0) | 0.63-0.64 (0.63-0.65) | 0.991-1.000 |
| D4 p47 | linear 0.1319, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 78-87 (15-16) | 177-202 (22-23) | 50.4-51.7 (1.7-1.9) | 0.77 (0.74) | 0.945-0.981 |
| D4 p61 | paper 0.2, 0.015 | 0/3 (0/10) | 0/3 (10/10) | 77-83 (16-18) | 143-197 (23-27) | 31.4-34.8 (1.7-2.9) | 0.84 (0.81-0.83) | 0.937-0.987 |
| D4 p61 | sqrt 0.1426, 0.015 | 0/3 (0/3) | 0/3 (3/3) | 76-82 (17-19) | 129-149 (24-25) | 26.5-31.8 (1.7-2.0) | 0.78 (0.76-0.77) | 0.987-1.000 |
| D4 p61 | sqrt 0.1426, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 63-83 (14-15) | 152-234, 1 never (19-21) | 29.0-33.8 (0.8-1.5) | 0.86 (0.82-0.83) | 0.884-0.976 |
| D4 p61 | linear 0.1016, 0.015 | 0/3 (0/3) | 0/3 (3/3) | 65-79 (17-18) | 108-117 (23-25) | 20.5-22.1 (1.5-1.9) | 0.70 (0.68-0.69) | 0.999-1.000 |
| D4 p61 | linear 0.1016, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 60-66 (14-15) | 112-164 (18-21) | 22.3-25.2 (0.6-0.9) | 0.81 (0.75-0.77) | 0.949-0.995 |
| D4 p101 | paper 0.2, 0.015 | 0/3 (0/10) | 0/3 (5/10) | 74-88 (15-17) | never (23-31) | 21.6-29.8 (0.2-1.3) | 0.94 (0.92-0.93) | 0.798-0.877 |
| D4 p101 | sqrt 0.1108, 0.015 | 0/3 (0/3) | 0/3 (3/3) | 65-84 (16) | 118-158 (23) | 13.7-14.3 (0.0-0.2) | 0.89 (0.87) | 0.933-0.974 |
| D4 p101 | sqrt 0.1108, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 61-65 (13-14) | 205, 2 never (18-19) | 14.7-17.5 (-0.1-0.1) | 0.93 (0.90) | 0.846-0.908 |
| D4 p101 | linear 0.0614, 0.015 | 0/3 (0/3) | 0/3 (3/3) | 58-66 (16-17) | 93-97 (21-22) | 7.2-8.8 (-0.5-0.4) | 0.81 (0.79) | 0.993-0.999 |
| D4 p101 | linear 0.0614, 0.025 | 0/3 (0/3) | 0/3 (3/3) | 52-63 (13-14) | 90-101 (17-18) | 7.6-9.0 (-0.4-0.0) | 0.87-0.88 (0.83-0.84) | 0.982-0.988 |

t10, t90: first update with held-out accuracy >= 0.1, first persistent >= 0.9 (in thousands; "never": not within the horizon,
300k at p = 23, 250k otherwise). Refit lead: update at which the held-out accuracy first reaches 0.5 minus the same for the
train-only ridge refit. CE peak: campaign C's CE peak in its band (the post-knee CE peak).

## 3. What it shows

1. **With half the data, generalization comes about 4-6 times later, and at p = 23 not at all.** At p >= 47 the held-out
   rise starts at 34-112k updates instead of 10-22k, and t90 is 52-250k instead of 13-31k (paired with the same seeds at
   0.8: t10 2.9-6.2 times later, median 4.3; t90 3.4-16.9 times, median 5.9); 31 of the 90 runs reach a final held-out
   accuracy >= 0.99 within 250k updates. At p = 23 none of the 30 runs gets past held-out 0.008 in 300k updates
   (the same seeds at 0.8 generalize at 48-114k), while the ridge refit creeps up to 0.16-0.48.
2. **The refit leads by tens of thousands of updates.** At p >= 47 the train-only refit reaches 0.5 between 5k (p = 101,
   linear rule) and 68k updates (p = 47) before the network, against -0.5 to 5k at fraction 0.8; it reaches >= 0.99 in all
   90 runs, the network in 31. The lead grows as p falls and as lam_v rises.
3. **But the CE is not flat while the refit leads.** After the knee the CE rises to its peak (0.58-0.94 log p) and then
   drifts down: from the refit's 0.2 crossing to the network's 0.9 crossing it falls by 5-21% (median 11%; the 80 of 90 runs
   that reach 0.9). At fraction 0.8 it changes between a 14% rise and a 4% fall over the same stretch, so it stays within
   10% or rises. The paper's post-hoc window sits on the CE rise after the knee, before the refit rises, so the refit
   criterion fails on the refit gain or on the final accuracy.
4. **The training fit relapses after the knee.** While the refit leads, up to 50 (p = 47), 317 (p = 61) and 1,762 (p = 101)
   training pairs are misclassified (at 0.8: up to 12, 156 and 1,296; so at p = 101 this is not specific to fraction 0.5).
   This is the endpoint's "fit" failure (89/120).
5. **The CE plateau level hardly moves**: at p >= 47 the post-knee CE peak is within 0.08 log p of the 0.8 level (mostly a
   little above); at p = 23 it is 0.06-0.10 below.

For the key message ("in a flat or rising training loss, the refit gets better, and in the same window or later the held-out
accuracy goes up"): at fraction 0.5 the refit gets better first, by a much wider margin than at 0.8, but the training CE
drifts down by 5-21% while it does, and the training set is not fully fit; at 0.8 the CE stays within 10% or rises over the
same stretch.

## 4. Figures

`L_scoreboard`, `L_refit_scoreboard` (verdicts per cell and setting, with the 0.8 counts); `figures/L_depth{3,4}_<setting>`
and `figures_refit/L_refit_depth{3,4}_<setting>` (the paper's renderer: training CE / log p, accuracies, misclassified training
pairs, A_H; grey: median band or post-hoc window); `L_vs_K_D{3,4}_p{23,47,61,101}` (medians at 0.5 against the same seeds at
0.8, or F/G's 10 seeds for the paper setting).
