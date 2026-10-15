# Campaign G: the p = 97 settings at p = 101, 10 fresh seeds each

30/30 new runs finished (0 failed). Settings, seeds and endpoint: PLAN.md (fixed before any run). Pass = campaign C's per-run endpoint: the loss-only band holds >= 75% of the held-out 10% -> 90% rise, final held-out >= 0.99, last-layer ΔA_H >= 0.02 across the band, every logged training margin positive in the band. Every seed is fresh.

## Per setting

| family | p | width | setting | fresh seeds: pass | Wilson 95% | all seeds: pass | median share | median CE / log p | median t10 → t90 | median final | failed checks (all seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 101 | 512 | λ_W 0.02 | 0/10 | [0.00, 0.28] | 0/10 | 93% | 0.94 | 5760 → 7230 | 0.994 | final 1, fit 10 |
| D3 | 101 | 256 | r 0.45 | 0/10 | [0.00, 0.28] | 0/10 | 100% | 0.90 | 13330 → 18850 | 0.997 | fit 10 |
| D4 | 101 | 256 | r 0.45 | 0/10 | [0.00, 0.28] | 0/10 | 100% | 0.92 | 15980 → 25270 | 0.990 | final 5, fit 10 |

## Next to p = 97 (campaign F, all 10 seeds of the same setting)

| setting | p = 101: pass | p = 97: pass | median share 101 / 97 | median CE / log p 101 / 97 | median t90 101 / 97 | median final 101 / 97 |
|---|---|---|---|---|---|---|
| H_p101_lw0.02 | 0/10 | 0/10 | 93% / 91% | 0.94 / 0.93 | 7230 / 7250 | 0.994 / 0.994 |
| D3_p101_r0.45 | 0/10 | 0/10 | 100% / 100% | 0.90 / 0.89 | 18850 / 18170 | 0.997 / 0.998 |
| D4_p101_r0.45 | 0/10 | 0/10 | 100% / 100% | 0.92 / 0.92 | 25270 / 24180 | 0.990 / 0.993 |

## Every run

| run | source | seed | status | pass | share | band | CE / log p | t10 → t90 | final | ΔA_H last | first fit | why not |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H_p101_lw0.0200_s1 | campaign G | 1 | done (stopped early) | no | 89% | 2386–6935 | 0.94 | 5760 → 7380 | 0.993 | 0.08 | – | fit |
| H_p101_lw0.0200_s2 | campaign G | 2 | done (stopped early) | no | 99% | 2364–7015 | 0.94 | 5720 → 7060 | 0.997 | 0.08 | – | fit |
| H_p101_lw0.0200_s3 | campaign G | 3 | done (stopped early) | no | 95% | 2360–7023 | 0.94 | 5740 → 7200 | 0.996 | 0.08 | – | fit |
| H_p101_lw0.0200_s4 | campaign G | 4 | done | no | 93% | 2355–7021 | 0.94 | 5780 → 7320 | 0.988 | 0.08 | – | final, fit |
| H_p101_lw0.0200_s5 | campaign G | 5 | done | no | 93% | 2385–6976 | 0.94 | 5800 → 7240 | 0.990 | 0.08 | 24560 | fit |
| H_p101_lw0.0200_s6 | campaign G | 6 | done (stopped early) | no | 93% | 2379–7004 | 0.94 | 5740 → 7300 | 0.994 | 0.08 | – | fit |
| H_p101_lw0.0200_s7 | campaign G | 7 | done (stopped early) | no | 100% | 2364–7075 | 0.94 | 5780 → 7060 | 0.994 | 0.08 | 18460 | fit |
| H_p101_lw0.0200_s8 | campaign G | 8 | done (stopped early) | no | 99% | 2360–6904 | 0.94 | 5680 → 6920 | 0.997 | 0.09 | 8500 | fit |
| H_p101_lw0.0200_s9 | campaign G | 9 | done (stopped early) | no | 93% | 2372–7007 | 0.94 | 5760 → 7220 | 0.996 | 0.08 | – | fit |
| H_p101_lw0.0200_s10 | campaign G | 10 | done (stopped early) | no | 93% | 2389–7037 | 0.94 | 5780 → 7300 | 0.992 | 0.08 | – | fit |
| D3_p101_r0.45_s1 | campaign G | 1 | done (stopped early) | no | 100% | 8474–44459 | 0.90 | 13320 → 18460 | 0.997 | 0.53 | – | fit |
| D3_p101_r0.45_s2 | campaign G | 2 | done (stopped early) | no | 100% | 8232–32199 | 0.89 | 12340 → 16860 | 0.999 | 0.54 | – | fit |
| D3_p101_r0.45_s3 | campaign G | 3 | done (stopped early) | no | 100% | 8609–44759 | 0.90 | 13840 → 19740 | 0.997 | 0.50 | – | fit |
| D3_p101_r0.45_s4 | campaign G | 4 | done (stopped early) | no | 100% | 8414–42819 | 0.90 | 13080 → 19260 | 0.998 | 0.54 | – | fit |
| D3_p101_r0.45_s5 | campaign G | 5 | done (stopped early) | no | 100% | 8564–46839 | 0.90 | 13680 → 19640 | 0.996 | 0.52 | – | fit |
| D3_p101_r0.45_s6 | campaign G | 6 | done (stopped early) | no | 100% | 8425–39959 | 0.90 | 13580 → 18400 | 0.999 | 0.49 | – | fit |
| D3_p101_r0.45_s7 | campaign G | 7 | done (stopped early) | no | 100% | 8622–44739 | 0.90 | 13600 → 18920 | 0.997 | 0.50 | – | fit |
| D3_p101_r0.45_s8 | campaign G | 8 | done (stopped early) | no | 100% | 8444–37959 | 0.90 | 13040 → 18180 | 0.999 | 0.51 | – | fit |
| D3_p101_r0.45_s9 | campaign G | 9 | done (stopped early) | no | 100% | 8438–43859 | 0.90 | 13340 → 18780 | 1.000 | 0.51 | – | fit |
| D3_p101_r0.45_s10 | campaign G | 10 | done (stopped early) | no | 100% | 8458–48339 | 0.89 | 12780 → 19140 | 0.994 | 0.52 | – | fit |
| D4_p101_r0.45_s1 | campaign G | 1 | done | no | 100% | 9488–73064 | 0.92 | 15960 → 27640 | 0.970 | 0.51 | – | final, fit |
| D4_p101_r0.45_s2 | campaign G | 2 | done | no | 100% | 9257–99999 | 0.92 | 15120 → 25280 | 0.987 | 0.47 | – | final, fit |
| D4_p101_r0.45_s3 | campaign G | 3 | done | no | 100% | 9255–92739 | 0.92 | 15620 → 31200 | 0.987 | 0.45 | – | final, fit |
| D4_p101_r0.45_s4 | campaign G | 4 | done | no | 100% | 9735–36915 | 0.93 | 17040 → 25800 | 0.993 | 0.48 | – | fit |
| D4_p101_r0.45_s5 | campaign G | 5 | done (stopped early) | no | 100% | 9667–39632 | 0.93 | 16760 → 26100 | 0.994 | 0.48 | – | fit |
| D4_p101_r0.45_s6 | campaign G | 6 | done | no | 100% | 9436–48094 | 0.92 | 16040 → 24740 | 0.987 | 0.51 | – | final, fit |
| D4_p101_r0.45_s7 | campaign G | 7 | done (stopped early) | no | 100% | 9644–44218 | 0.93 | 16420 → 25260 | 0.997 | 0.51 | – | fit |
| D4_p101_r0.45_s8 | campaign G | 8 | done (stopped early) | no | 100% | 9505–54127 | 0.92 | 16000 → 24580 | 0.995 | 0.48 | – | fit |
| D4_p101_r0.45_s9 | campaign G | 9 | done (stopped early) | no | 100% | 9351–42704 | 0.92 | 15780 → 23220 | 0.997 | 0.55 | – | fit |
| D4_p101_r0.45_s10 | campaign G | 10 | done | no | 100% | 9329–97839 | 0.92 | 15180 → 24020 | 0.982 | 0.54 | – | final, fit |
