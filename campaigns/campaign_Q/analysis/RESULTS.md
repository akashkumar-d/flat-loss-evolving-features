# Campaign Q results (claude/1hl-figures, 2026-10-13)

40 runs, all finished (collected with collect_results.sh). Scored by campaign Z's per-run analysis, unchanged
(`../analyze_Q.py` -> `analysis_Q.json`), with seeds 11-15 of campaigns ZP, ZM and Z (and Y's G2 at p = 113) pooled in.

| Cell | Runs | Strict | Loose | Final, all pairs |
|---|---|---|---|---|
| p = 61, mirror-free split 0.55 (ZM + Q) | 10 | 9 | 9 | 0.9899-0.997 |
| p = 61, random split 0.33 (ZP + Q) | 10 | 5 (10 on no-mirror pairs) | 5 | 0.994-0.999 |
| p = 89, 0.25 (ZP + Q) | 10 | 10 | 10 | 1.000 |
| p = 113, 0.2 (Y G2) | 10 | 10 | 10 | 0.999-1.000 |
| p = 149, 0.16 (ZP + Q) | 10 | 7 | 10 | 0.992-1.000 |
| Two blocks, 0.2 (Z + Q) | 10 | 0 | 0 | 0.941-0.994 |
| Two blocks, gain frozen at 12k (F12) | 3 | 2 | 2 | 0.983-0.997 |
| Two blocks, eta_W 0.1 -> 0.03 at 12k (E12) | 3 | 2 | 2 | 0.986-0.997 |
| Two blocks, eta_V 0.03 -> 0.01 at 12k (V12) | 3 | 0 | 0 | 0.950-0.970 |
| Two blocks, fraction 0.175 | 3 | 0 | 0 | 0.82-0.91 |
| Two blocks, fraction 0.225 | 3 | 0 | 0 | 0.989-0.998 |

Two blocks, base: the first misclassified training pair after 12k comes at 17.8k-22.0k in all 10 runs (45-438 at some
log), 1.4k-5.7k updates after all-pairs held-out accuracy first reaches 0.99; 6 of 10 end below 0.99.
- F12: the CE falls from 12k, so the window closes at 13.5k-13.7k; pairs still flip at 18.8k-19.7k, after the window (the
  two passes lose their fit too).
- V12: the flips come later (21.6k-23.0k), inside the window.
- E12: no misclassified training pair after 12k in 3/3. The averaged CE dips about 8% at the change and then rises, so the
  window closes at 12.0k; held-out accuracy reaches 0.99 later (20.1k, 23.8k; the third ends at 0.986). Exploratory: 3
  seeds, designed after the base failures; confirming it needs fresh seeds.
- Fraction 0.175: the no-mirror rise ends at 14.3k-16.2k and pairs flip from 17.7k-22.4k. Fraction 0.225: the rise
  (2.6k-3.3k) starts before the CE minimum (4.9k-5.2k).
- The three changed runs of a seed are identical up to 12k. They part from Z's runs of the same seed after 2117-2988
  updates (4 threads here, 2 in Z), so the comparison with the base runs is statistical, not paired.

`../zm10/`: the 10-seed versions of campaign ZM's mirror-pair and readout analyses (`build_zm10.py`, `plot_ZM10.py` = plot_ZM.py
with the mirror-pair count shown as a range). Paper use: arXiv v2 revision 3, Figures 12-15 and Tables 6-8.
