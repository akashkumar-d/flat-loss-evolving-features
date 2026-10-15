# Campaign QD results (claude/1hl-figures, 2026-10-13)

80 runs, all finished: depth 2 at p = 31 on the 40 seeds of depths 3-5 (init 1000 + i, split 2000 + i), bias-free
(QD_D2) and with hidden biases (QD_B_D2), observers_v1 on. Scored with campaign C's primary endpoint
(`../endpoints_i.py`, campaign I's implementation, unchanged): `../analyze_QD.py` -> `analysis_QD.json`.

| Arm | Primary endpoint | Wilson 95% | Misses |
|---|---|---|---|
| Bias-free | 34/40 | 0.71-0.93 | 6 on the share (one also ends at 0.984) |
| With biases | 18/40 | 0.31-0.60 | 20 on the share, 3 on the final accuracy (one both) |

Paired: 16 seeds lose the endpoint with biases, none gains it (exact McNemar p = 3.1e-5). Bias-free medians: held-out
10% at 16.6k, 90% at 26.2k updates, share 0.89. Depths 3-5 meet a Wilson lower end of 0.80; depth 2 does not.
A replay of an archived run diverges after 61 updates between numpy 2.4.4 and 2.5.3 (floating-point chaos); the code is
the same by hash. Paper use: arXiv v2 revision 3, Figure 3 (depth-2 column, canonical refit), Sections 5.2, F.3, F.6.
