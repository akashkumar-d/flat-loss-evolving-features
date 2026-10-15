# Campaign QH results (claude/1hl-figures, 2026-10-13)

4 runs, all finished: campaign H's eta_V = 3e-4 arm on seeds 7-10. With campaign H's seeds 1-6 and its endpoints J1-J4
(analyze.fields from the code release, unchanged; `../analyze_QH.py` -> `analysis_QH.json`):

| Arm | Runs | J1 | J2 | J3 | J4 | Joint | v([1000, T]) | Held-out at 40k |
|---|---|---|---|---|---|---|---|---|
| eta_V = 3e-4 | 10 | 10 | 10 | 10 | 10 | 10 | 0.40% (0.396-0.402) | 0.25-0.41 |
| eta_V = 1e-3 (primary) | 10 | 0 | 10 | 10 | 10 | 0 | 1.36% (1.34-1.37) | 0.28-0.47 |

The refit reaches 0.9 at 10.25k-11.5k updates in the eta_V = 3e-4 arm and is at 1.0 by 40k in every run. Paper use: arXiv v2
revision 3, Figure 8 (10 runs per row) and Appendix E.6.
