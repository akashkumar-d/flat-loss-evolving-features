# Campaign F: the campaign-D and depth-2 settings at 10 seeds

137/137 new runs finished (0 failed). Settings, seeds and endpoint: PLAN.md (fixed before any run). Pass = campaign C's per-run endpoint: the loss-only band holds >= 75% of the held-out 10% -> 90% rise, final held-out >= 0.99, last-layer ΔA_H >= 0.02 across the band, every logged training margin positive in the band. Fresh = run in this campaign.

## Per setting

| family | p | width | setting | fresh seeds: pass | Wilson 95% | all seeds: pass | median share | median CE / log p | median t10 → t90 | median final | failed checks (all seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 23 | 512 | λ_W 0.01 | 7/8 | [0.53, 0.98] | 9/10 | 100% | 0.09 | 9140 → 12620 | 0.995 | final 1 |
| H | 31 | 512 | λ_W 0.015 | 2/8 | [0.07, 0.59] | 4/10 | 100% | 0.46 | 7150 → 10240 | 0.987 | final 6, share 1 |
| H | 47 | 512 | λ_W 0.02 | 0/8 | [0.00, 0.32] | 0/10 | 52% | 0.78 | 5920 → 8960 | 0.962 | final 10, fit 10, share 9 |
| H | 61 | 512 | λ_W 0.02 | 0/8 | [0.00, 0.32] | 0/10 | 55% | 0.85 | 5800 → 7990 | 0.985 | final 8, fit 10, share 10 |
| H | 97 | 512 | λ_W 0.02 | 0/8 | [0.00, 0.32] | 0/10 | 91% | 0.93 | 5760 → 7250 | 0.994 | final 2, fit 10 |
| HW | 97 | 2048 | λ_W 0.015 | 0/9 | [0.00, 0.30] | 0/10 | 100% | 0.89 | 7560 → 8780 | 1.000 | fit 10 |
| D3 | 23 | 256 | r 0.45 | 0/8 | [0.00, 0.32] | 0/10 | 45% | 0.29 | 23660 → 52970 | 1.000 | F02 4, share 10 |
| D3 | 47 | 256 | r 0.45 | 2/8 | [0.07, 0.59] | 2/10 | 98% | 0.68 | 14880 → 21040 | 1.000 | fit 7, share 3 |
| D3 | 61 | 256 | r 0.45 | 0/8 | [0.00, 0.32] | 0/10 | 100% | 0.77 | 13740 → 19280 | 1.000 | fit 10 |
| D3 | 97 | 256 | r 0.45 | 0/9 | [0.00, 0.30] | 0/10 | 100% | 0.89 | 12850 → 18170 | 0.998 | fit 10 |
| D4 | 23 | 256 | r 0.32 | 0/8 | [0.00, 0.32] | 0/10 | 48% | 0.33 | 30490 → 70880 | 1.000 | F02 10, share 10 |
| D4 | 47 | 256 | r 0.45 | 0/8 | [0.00, 0.32] | 0/10 | 100% | 0.75 | 18970 → 28600 | 1.000 | fit 10 |
| D4 | 61 | 256 | r 0.45 | 0/8 | [0.00, 0.32] | 0/10 | 100% | 0.83 | 17620 → 25600 | 1.000 | fit 10 |
| D4 | 97 | 256 | r 0.45 | 0/9 | [0.00, 0.30] | 0/10 | 100% | 0.92 | 15940 → 24180 | 0.993 | final 3, fit 10 |
| D2 | 23 | 256 | λ_W 0.015, r 0.45 | 0/8 | [0.00, 0.32] | 0/10 | 59% | 0.26 | 21070 → 39080 | 1.000 | share 10 |
| D2 | 31 | 256 | λ_W 0.015, r 0.5 | 5/5 | [0.57, 1.00] | 9/10 | 86% | 0.47 | 16870 → 26560 | 1.000 | final 1 |
| D2 | 47 | 256 | λ_W 0.015, r 0.5 | 0/9 | [0.00, 0.30] | 0/10 | 100% | 0.68 | 13460 → 20600 | 1.000 | fit 10 |

## Every run

| run | source | seed | status | pass | share | band | CE / log p | t10 → t90 | final | ΔA_H last | first fit | why not |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H_p23_lw0.0100_s1 | campaign D | 1 | done (stopped early) | PASS | 97% | 8928–27019 | 0.09 | 8620 → 12840 | 0.991 | 0.05 | 120 |  |
| H_p23_lw0.0100_s2 | campaign D | 2 | done (stopped early) | PASS | 100% | 7564–13099 | 0.09 | 9320 → 12980 | 1.000 | 0.06 | 280 |  |
| H_p23_lw0.0100_s3 | campaign F | 3 | done (stopped early) | PASS | 91% | 7697–11852 | 0.09 | 9160 → 12440 | 0.991 | 0.05 | 180 |  |
| H_p23_lw0.0100_s4 | campaign F | 4 | done (stopped early) | PASS | 85% | 9408–19154 | 0.09 | 8560 → 12300 | 0.991 | 0.05 | 200 |  |
| H_p23_lw0.0100_s5 | campaign F | 5 | done (stopped early) | PASS | 100% | 7671–13949 | 0.09 | 9380 → 13240 | 1.000 | 0.07 | 540 |  |
| H_p23_lw0.0100_s6 | campaign F | 6 | done (stopped early) | PASS | 100% | 8211–20669 | 0.09 | 9260 → 12720 | 1.000 | 0.07 | 160 |  |
| H_p23_lw0.0100_s7 | campaign F | 7 | done | no | 100% | 8033–29491 | 0.09 | 9320 → 13160 | 0.981 | 0.07 | 140 | final |
| H_p23_lw0.0100_s8 | campaign F | 8 | done (stopped early) | PASS | 100% | 8089–15342 | 0.09 | 9120 → 12520 | 1.000 | 0.06 | 180 |  |
| H_p23_lw0.0100_s9 | campaign F | 9 | done (stopped early) | PASS | 100% | 7900–26139 | 0.08 | 9020 → 12380 | 0.991 | 0.07 | 180 |  |
| H_p23_lw0.0100_s10 | campaign F | 10 | done (stopped early) | PASS | 100% | 8130–25919 | 0.08 | 8800 → 12060 | 1.000 | 0.07 | 200 |  |
| H_p31_lw0.0150_s1 | campaign D | 1 | done (stopped early) | PASS | 100% | 4961–21979 | 0.46 | 7140 → 9900 | 0.995 | 0.10 | 340 |  |
| H_p31_lw0.0150_s2 | campaign D | 2 | done | PASS | 100% | 5104–9990 | 0.46 | 7120 → 9580 | 0.995 | 0.08 | 660 |  |
| H_p31_lw0.0150_s3 | campaign F | 3 | done (stopped early) | PASS | 100% | 5063–16284 | 0.46 | 7140 → 10740 | 1.000 | 0.09 | 720 |  |
| H_p31_lw0.0150_s4 | campaign F | 4 | done | no | 100% | 4967–39999 | 0.46 | 7160 → 10200 | 0.984 | 0.10 | 280 | final |
| H_p31_lw0.0150_s5 | campaign F | 5 | done | no | 73% | 5127–8937 | 0.47 | 7320 → 10920 | 0.990 | 0.07 | 420 | share, final |
| H_p31_lw0.0150_s6 | campaign F | 6 | done | no | 100% | 5087–39999 | 0.46 | 7180 → 9840 | 0.959 | 0.09 | 400 | final |
| H_p31_lw0.0150_s7 | campaign F | 7 | done (stopped early) | PASS | 100% | 5049–23299 | 0.46 | 7080 → 9300 | 1.000 | 0.10 | 420 |  |
| H_p31_lw0.0150_s8 | campaign F | 8 | done | no | 89% | 4975–9425 | 0.46 | 7100 → 10280 | 0.953 | 0.08 | 480 | final |
| H_p31_lw0.0150_s9 | campaign F | 9 | done | no | 100% | 5182–11118 | 0.46 | 7320 → 10560 | 0.984 | 0.09 | 700 | final |
| H_p31_lw0.0150_s10 | campaign F | 10 | done | no | 76% | 5007–8867 | 0.47 | 7200 → 10520 | 0.974 | 0.07 | 240 | final |
| H_p47_lw0.0200_s1 | campaign D | 1 | done | no | 50% | 3503–6736 | 0.78 | 5960 → 9220 | 0.980 | 0.07 | 6720 | share, final, fit |
| H_p47_lw0.0200_s2 | campaign D | 2 | done | no | 48% | 3498–6724 | 0.78 | 5940 → 9620 | 0.937 | 0.07 | 20980 | share, final, fit |
| H_p47_lw0.0200_s3 | campaign F | 3 | done | no | 46% | 3483–6770 | 0.78 | 5940 → 9380 | 0.962 | 0.08 | 6460 | share, final, fit |
| H_p47_lw0.0200_s4 | campaign F | 4 | done | no | 53% | 3450–6793 | 0.77 | 5880 → 9020 | 0.962 | 0.08 | 8920 | share, final, fit |
| H_p47_lw0.0200_s5 | campaign F | 5 | done | no | 69% | 3478–6967 | 0.78 | 5960 → 8360 | 0.982 | 0.08 | 5760 | share, final, fit |
| H_p47_lw0.0200_s6 | campaign F | 6 | done | no | 49% | 3494–6675 | 0.78 | 5940 → 8560 | 0.950 | 0.07 | 6460 | share, final, fit |
| H_p47_lw0.0200_s7 | campaign F | 7 | done | no | 79% | 3456–6865 | 0.77 | 5800 → 7580 | 0.986 | 0.08 | 5500 | final, fit |
| H_p47_lw0.0200_s8 | campaign F | 8 | done | no | 57% | 3449–6849 | 0.77 | 5900 → 8980 | 0.964 | 0.08 | 8740 | share, final, fit |
| H_p47_lw0.0200_s9 | campaign F | 9 | done | no | 65% | 3487–6860 | 0.77 | 5820 → 8080 | 0.952 | 0.08 | 4980 | share, final, fit |
| H_p47_lw0.0200_s10 | campaign F | 10 | done | no | 48% | 3447–6656 | 0.77 | 5900 → 8940 | 0.962 | 0.07 | 5560 | share, final, fit |
| H_p61_lw0.0200_s1 | campaign D | 1 | done | no | 35% | 3133–6409 | 0.85 | 5900 → 8140 | 0.969 | 0.07 | – | share, final, fit |
| H_p61_lw0.0200_s2 | campaign D | 2 | done | no | 65% | 3097–6620 | 0.85 | 5760 → 7640 | 0.988 | 0.07 | 22620 | share, final, fit |
| H_p61_lw0.0200_s3 | campaign F | 3 | done (stopped early) | no | 55% | 3110–6547 | 0.85 | 5800 → 7760 | 0.992 | 0.07 | 8440 | share, fit |
| H_p61_lw0.0200_s4 | campaign F | 4 | done (stopped early) | no | 60% | 3069–6605 | 0.85 | 5720 → 8040 | 0.992 | 0.07 | 9020 | share, fit |
| H_p61_lw0.0200_s5 | campaign F | 5 | done | no | 51% | 3153–6555 | 0.85 | 5880 → 7900 | 0.987 | 0.07 | 9700 | share, final, fit |
| H_p61_lw0.0200_s6 | campaign F | 6 | done | no | 55% | 3074–6669 | 0.85 | 5800 → 8240 | 0.983 | 0.07 | 17720 | share, final, fit |
| H_p61_lw0.0200_s7 | campaign F | 7 | done | no | 55% | 3092–6594 | 0.85 | 5820 → 8660 | 0.956 | 0.07 | 17640 | share, final, fit |
| H_p61_lw0.0200_s8 | campaign F | 8 | done | no | 55% | 3106–6556 | 0.85 | 5760 → 7980 | 0.985 | 0.07 | 9020 | share, final, fit |
| H_p61_lw0.0200_s9 | campaign F | 9 | done | no | 58% | 3102–6566 | 0.85 | 5780 → 7840 | 0.976 | 0.07 | 15020 | share, final, fit |
| H_p61_lw0.0200_s10 | campaign F | 10 | done | no | 64% | 3083–6666 | 0.85 | 5800 → 8000 | 0.985 | 0.07 | 13880 | share, final, fit |
| H_p97_lw0.0200_s1 | campaign D | 1 | done (stopped early) | no | 93% | 2440–6945 | 0.93 | 5760 → 7180 | 0.991 | 0.08 | 18020 | fit |
| H_p97_lw0.0200_s2 | campaign D | 2 | done (stopped early) | no | 91% | 2456–6918 | 0.93 | 5760 → 7240 | 0.994 | 0.08 | 21060 | fit |
| H_p97_lw0.0200_s3 | campaign F | 3 | done (stopped early) | no | 98% | 2436–6933 | 0.93 | 5700 → 7040 | 0.996 | 0.08 | 7960 | fit |
| H_p97_lw0.0200_s4 | campaign F | 4 | done (stopped early) | no | 87% | 2473–6918 | 0.93 | 5820 → 7340 | 0.997 | 0.08 | – | fit |
| H_p97_lw0.0200_s5 | campaign F | 5 | done | no | 89% | 2472–6916 | 0.93 | 5820 → 7280 | 0.989 | 0.08 | – | final, fit |
| H_p97_lw0.0200_s6 | campaign F | 6 | done (stopped early) | no | 95% | 2430–6932 | 0.93 | 5720 → 7100 | 0.994 | 0.08 | – | fit |
| H_p97_lw0.0200_s7 | campaign F | 7 | done (stopped early) | no | 91% | 2443–6919 | 0.93 | 5780 → 7260 | 0.994 | 0.08 | – | fit |
| H_p97_lw0.0200_s8 | campaign F | 8 | done | no | 84% | 2428–6903 | 0.93 | 5720 → 7440 | 0.989 | 0.08 | 25780 | final, fit |
| H_p97_lw0.0200_s9 | campaign F | 9 | done (stopped early) | no | 95% | 2436–6997 | 0.93 | 5780 → 7160 | 0.995 | 0.08 | 14900 | fit |
| H_p97_lw0.0200_s10 | campaign F | 10 | done (stopped early) | no | 90% | 2405–6937 | 0.93 | 5680 → 7280 | 0.994 | 0.09 | – | fit |
| HW_p97_w2048_lw0.0150_s1 | campaign D | 1 | done (stopped early) | no | 100% | 4308–9528 | 0.89 | 7500 → 8780 | 1.000 | 0.09 | 8160 | fit |
| HW_p97_w2048_lw0.0150_s2 | campaign F | 2 | done (stopped early) | no | 100% | 4334–9443 | 0.89 | 7520 → 8740 | 1.000 | 0.07 | 8060 | fit |
| HW_p97_w2048_lw0.0150_s3 | campaign F | 3 | done (stopped early) | no | 100% | 4360–9485 | 0.90 | 7560 → 8700 | 1.000 | 0.07 | 8160 | fit |
| HW_p97_w2048_lw0.0150_s4 | campaign F | 4 | done (stopped early) | no | 100% | 4361–9516 | 0.90 | 7600 → 8840 | 0.999 | 0.09 | 8600 | fit |
| HW_p97_w2048_lw0.0150_s5 | campaign F | 5 | done (stopped early) | no | 100% | 4345–9493 | 0.89 | 7560 → 8780 | 1.000 | 0.07 | 8700 | fit |
| HW_p97_w2048_lw0.0150_s6 | campaign F | 6 | done (stopped early) | no | 100% | 4329–9571 | 0.89 | 7500 → 8680 | 1.000 | 0.09 | 8800 | fit |
| HW_p97_w2048_lw0.0150_s7 | campaign F | 7 | done (stopped early) | no | 100% | 4358–9430 | 0.90 | 7560 → 8780 | 1.000 | 0.07 | 8900 | fit |
| HW_p97_w2048_lw0.0150_s8 | campaign F | 8 | done (stopped early) | no | 100% | 4342–9518 | 0.89 | 7580 → 8840 | 1.000 | 0.09 | 9300 | fit |
| HW_p97_w2048_lw0.0150_s9 | campaign F | 9 | done (stopped early) | no | 100% | 4330–9482 | 0.89 | 7580 → 8700 | 0.999 | 0.07 | 8240 | fit |
| HW_p97_w2048_lw0.0150_s10 | campaign F | 10 | done (stopped early) | no | 100% | 4307–9628 | 0.89 | 7540 → 8820 | 1.000 | 0.09 | 16520 | fit |
| D3_p23_r0.45_s1 | campaign D | 1 | done (stopped early) | no | 39% | 23877–34221 | 0.30 | 24260 → 51060 | 1.000 | 0.02 | 2420 | share, F02 |
| D3_p23_r0.45_s2 | campaign D | 2 | done (stopped early) | no | 50% | 23759–34376 | 0.29 | 23760 → 55400 | 1.000 | 0.02 | 2440 | share |
| D3_p23_r0.45_s3 | campaign F | 3 | done (stopped early) | no | 26% | 23489–33368 | 0.31 | 28200 → 57720 | 1.000 | 0.03 | 2220 | share |
| D3_p23_r0.45_s4 | campaign F | 4 | done (stopped early) | no | 48% | 23878–34628 | 0.29 | 21620 → 47840 | 1.000 | 0.02 | 2280 | share |
| D3_p23_r0.45_s5 | campaign F | 5 | done (stopped early) | no | 46% | 23596–34336 | 0.30 | 25560 → 49580 | 1.000 | 0.02 | 2200 | share |
| D3_p23_r0.45_s6 | campaign F | 6 | done (stopped early) | no | 46% | 23818–34654 | 0.30 | 21980 → 50220 | 1.000 | 0.02 | 2360 | share, F02 |
| D3_p23_r0.45_s7 | campaign F | 7 | done | no | 44% | 23899–34957 | 0.29 | 23560 → 64260 | 1.000 | 0.02 | 2280 | share, F02 |
| D3_p23_r0.45_s8 | campaign F | 8 | done (stopped early) | no | 29% | 23584–34377 | 0.30 | 25520 → 55940 | 1.000 | 0.02 | 2200 | share |
| D3_p23_r0.45_s9 | campaign F | 9 | done | no | 29% | 23906–34803 | 0.29 | 21860 → 54880 | 1.000 | 0.02 | 2440 | share, F02 |
| D3_p23_r0.45_s10 | campaign F | 10 | done (stopped early) | no | 59% | 24199–34948 | 0.29 | 20160 → 49120 | 1.000 | 0.02 | 2420 | share |
| D3_p47_r0.45_s1 | campaign D | 1 | done (stopped early) | no | 51% | 16324–38099 | 0.68 | 13920 → 20280 | 1.000 | 0.22 | 2820 | share, fit |
| D3_p47_r0.45_s2 | campaign D | 2 | done (stopped early) | no | 97% | 15142–35843 | 0.68 | 14880 → 21040 | 1.000 | 0.30 | 2880 | fit |
| D3_p47_r0.45_s3 | campaign F | 3 | done (stopped early) | PASS | 100% | 14767–33596 | 0.69 | 15300 → 21280 | 1.000 | 0.31 | 2740 |  |
| D3_p47_r0.45_s4 | campaign F | 4 | done (stopped early) | no | 90% | 15538–34679 | 0.68 | 14800 → 21300 | 1.000 | 0.24 | 2880 | fit |
| D3_p47_r0.45_s5 | campaign F | 5 | done (stopped early) | no | 100% | 14792–31772 | 0.69 | 15460 → 21860 | 1.000 | 0.29 | 2940 | fit |
| D3_p47_r0.45_s6 | campaign F | 6 | done (stopped early) | no | 100% | 15049–31290 | 0.69 | 15560 → 22660 | 1.000 | 0.27 | 3020 | fit |
| D3_p47_r0.45_s7 | campaign F | 7 | done (stopped early) | no | 53% | 16209–33239 | 0.68 | 14080 → 18900 | 1.000 | 0.23 | 2660 | share |
| D3_p47_r0.45_s8 | campaign F | 8 | done (stopped early) | no | 100% | 14679–37105 | 0.68 | 14880 → 21020 | 1.000 | 0.32 | 2940 | fit |
| D3_p47_r0.45_s9 | campaign F | 9 | done (stopped early) | no | 65% | 16092–36262 | 0.68 | 14320 → 20780 | 1.000 | 0.23 | 2760 | share, fit |
| D3_p47_r0.45_s10 | campaign F | 10 | done (stopped early) | PASS | 100% | 14801–34467 | 0.68 | 15480 → 21040 | 1.000 | 0.30 | 3040 |  |
| D3_p61_r0.45_s1 | campaign D | 1 | done (stopped early) | no | 100% | 12210–38496 | 0.78 | 14300 → 21360 | 1.000 | 0.45 | 3800 | fit |
| D3_p61_r0.45_s2 | campaign D | 2 | done (stopped early) | no | 100% | 11557–34139 | 0.77 | 13140 → 18540 | 1.000 | 0.45 | 3820 | fit |
| D3_p61_r0.45_s3 | campaign F | 3 | done (stopped early) | no | 100% | 11877–34799 | 0.78 | 14120 → 19420 | 1.000 | 0.46 | 4320 | fit |
| D3_p61_r0.45_s4 | campaign F | 4 | done (stopped early) | no | 100% | 11899–39899 | 0.77 | 13600 → 19960 | 1.000 | 0.43 | 3800 | fit |
| D3_p61_r0.45_s5 | campaign F | 5 | done (stopped early) | no | 100% | 11751–35199 | 0.77 | 13780 → 18980 | 1.000 | 0.47 | 3700 | fit |
| D3_p61_r0.45_s6 | campaign F | 6 | done (stopped early) | no | 100% | 11341–37473 | 0.78 | 14500 → 20600 | 0.999 | 0.46 | 3560 | fit |
| D3_p61_r0.45_s7 | campaign F | 7 | done (stopped early) | no | 100% | 12146–48659 | 0.77 | 13700 → 19840 | 0.997 | 0.42 | 4000 | fit |
| D3_p61_r0.45_s8 | campaign F | 8 | done (stopped early) | no | 100% | 11729–35439 | 0.77 | 13220 → 18560 | 0.999 | 0.46 | 3660 | fit |
| D3_p61_r0.45_s9 | campaign F | 9 | done (stopped early) | no | 100% | 11849–35379 | 0.77 | 13800 → 19140 | 1.000 | 0.43 | 23880 | fit |
| D3_p61_r0.45_s10 | campaign F | 10 | done (stopped early) | no | 100% | 11546–32579 | 0.77 | 13480 → 18360 | 0.999 | 0.47 | 4100 | fit |
| D3_p97_r0.45_s1 | campaign D | 1 | done (stopped early) | no | 100% | 8527–37239 | 0.88 | 12740 → 18240 | 0.999 | 0.53 | – | fit |
| D3_p97_r0.45_s2 | campaign F | 2 | done (stopped early) | no | 100% | 8381–53319 | 0.88 | 12340 → 18120 | 0.998 | 0.50 | 48160 | fit |
| D3_p97_r0.45_s3 | campaign F | 3 | done (stopped early) | no | 100% | 8529–34319 | 0.89 | 13320 → 17600 | 0.999 | 0.51 | 28940 | fit |
| D3_p97_r0.45_s4 | campaign F | 4 | done (stopped early) | no | 100% | 8389–60159 | 0.89 | 12940 → 18660 | 0.996 | 0.49 | – | fit |
| D3_p97_r0.45_s5 | campaign F | 5 | done (stopped early) | no | 100% | 8890–52259 | 0.90 | 13900 → 19680 | 0.996 | 0.52 | – | fit |
| D3_p97_r0.45_s6 | campaign F | 6 | done (stopped early) | no | 100% | 8366–37319 | 0.89 | 12740 → 18220 | 0.998 | 0.53 | 29800 | fit |
| D3_p97_r0.45_s7 | campaign F | 7 | done (stopped early) | no | 100% | 8387–40959 | 0.88 | 12580 → 17140 | 0.997 | 0.56 | – | fit |
| D3_p97_r0.45_s8 | campaign F | 8 | done (stopped early) | no | 100% | 8629–43659 | 0.89 | 13260 → 19100 | 0.997 | 0.51 | – | fit |
| D3_p97_r0.45_s9 | campaign F | 9 | done (stopped early) | no | 100% | 8494–39839 | 0.89 | 13080 → 17900 | 0.999 | 0.51 | – | fit |
| D3_p97_r0.45_s10 | campaign F | 10 | done (stopped early) | no | 100% | 8357–41579 | 0.88 | 12760 → 17960 | 0.998 | 0.53 | – | fit |
| D4_p23_r0.32_s1 | campaign D | 1 | done | no | 58% | 34411–51158 | 0.33 | 37860 → 73540 | 1.000 | 0.02 | 2700 | share, F02 |
| D4_p23_r0.32_s2 | campaign D | 2 | done | no | 45% | 34675–51617 | 0.33 | 25960 → 79800 | 1.000 | 0.01 | 2200 | share, F02 |
| D4_p23_r0.32_s3 | campaign F | 3 | done | no | 32% | 34952–52556 | 0.34 | 37460 → 85620 | 1.000 | 0.01 | 2560 | share, F02 |
| D4_p23_r0.32_s4 | campaign F | 4 | done | no | 57% | 34618–51503 | 0.33 | 31000 → 64020 | 1.000 | 0.01 | 2680 | share, F02 |
| D4_p23_r0.32_s5 | campaign F | 5 | done | no | 37% | 34542–51985 | 0.34 | 38340 → 75380 | 1.000 | 0.02 | 2740 | share, F02 |
| D4_p23_r0.32_s6 | campaign F | 6 | done | no | 47% | 34516–52420 | 0.33 | 25360 → 70040 | 1.000 | 0.01 | 2620 | share, F02 |
| D4_p23_r0.32_s7 | campaign F | 7 | done | no | 45% | 35117–52711 | 0.33 | 28000 → 71720 | 0.991 | 0.01 | 2620 | share, F02 |
| D4_p23_r0.32_s8 | campaign F | 8 | done (stopped early) | no | 59% | 34218–51543 | 0.33 | 30140 → 64180 | 1.000 | 0.02 | 2760 | share, F02 |
| D4_p23_r0.32_s9 | campaign F | 9 | done | no | 50% | 33924–50593 | 0.33 | 30840 → 67620 | 1.000 | 0.01 | 2680 | share, F02 |
| D4_p23_r0.32_s10 | campaign F | 10 | done (stopped early) | no | 62% | 34711–51360 | 0.33 | 28960 → 63080 | 1.000 | 0.01 | 2780 | share, F02 |
| D4_p47_r0.45_s1 | campaign D | 1 | done (stopped early) | no | 100% | 15926–30043 | 0.75 | 19060 → 29200 | 1.000 | 0.23 | 2660 | fit |
| D4_p47_r0.45_s2 | campaign D | 2 | done (stopped early) | no | 100% | 15090–26963 | 0.75 | 18880 → 25980 | 1.000 | 0.25 | 2620 | fit |
| D4_p47_r0.45_s3 | campaign F | 3 | done (stopped early) | no | 77% | 15696–28626 | 0.77 | 22300 → 31320 | 1.000 | 0.19 | 2440 | fit |
| D4_p47_r0.45_s4 | campaign F | 4 | done (stopped early) | no | 100% | 15942–28843 | 0.76 | 20500 → 28880 | 1.000 | 0.21 | 2620 | fit |
| D4_p47_r0.45_s5 | campaign F | 5 | done (stopped early) | no | 93% | 15983–29484 | 0.76 | 20900 → 30680 | 1.000 | 0.18 | 2580 | fit |
| D4_p47_r0.45_s6 | campaign F | 6 | done (stopped early) | no | 100% | 15375–28407 | 0.74 | 18380 → 27120 | 0.998 | 0.25 | 2540 | fit |
| D4_p47_r0.45_s7 | campaign F | 7 | done (stopped early) | no | 100% | 16227–30122 | 0.75 | 18500 → 27100 | 1.000 | 0.24 | 2540 | fit |
| D4_p47_r0.45_s8 | campaign F | 8 | done (stopped early) | no | 100% | 15685–29079 | 0.74 | 18100 → 26880 | 1.000 | 0.25 | 2660 | fit |
| D4_p47_r0.45_s9 | campaign F | 9 | done (stopped early) | no | 100% | 15406–28799 | 0.74 | 18800 → 28320 | 1.000 | 0.27 | 2680 | fit |
| D4_p47_r0.45_s10 | campaign F | 10 | done (stopped early) | no | 97% | 15739–28675 | 0.75 | 19200 → 29180 | 1.000 | 0.23 | 2500 | fit |
| D4_p61_r0.45_s1 | campaign D | 1 | done (stopped early) | no | 100% | 13166–27685 | 0.83 | 18220 → 26100 | 1.000 | 0.36 | 3280 | fit |
| D4_p61_r0.45_s2 | campaign D | 2 | done (stopped early) | no | 100% | 12918–26877 | 0.83 | 17820 → 24780 | 1.000 | 0.38 | 3260 | fit |
| D4_p61_r0.45_s3 | campaign F | 3 | done (stopped early) | no | 100% | 12876–28012 | 0.83 | 18400 → 26560 | 1.000 | 0.32 | 3140 | fit |
| D4_p61_r0.45_s4 | campaign F | 4 | done (stopped early) | no | 100% | 12650–29710 | 0.83 | 17380 → 25760 | 0.999 | 0.41 | 3080 | fit |
| D4_p61_r0.45_s5 | campaign F | 5 | done (stopped early) | no | 100% | 12863–27323 | 0.83 | 17960 → 25020 | 1.000 | 0.36 | 3020 | fit |
| D4_p61_r0.45_s6 | campaign F | 6 | done (stopped early) | no | 100% | 12719–28285 | 0.82 | 17420 → 25440 | 0.999 | 0.41 | 2980 | fit |
| D4_p61_r0.45_s7 | campaign F | 7 | done (stopped early) | no | 100% | 13062–28630 | 0.82 | 17400 → 27480 | 1.000 | 0.38 | 3180 | fit |
| D4_p61_r0.45_s8 | campaign F | 8 | done (stopped early) | no | 100% | 12983–30220 | 0.82 | 15880 → 23820 | 1.000 | 0.43 | 3360 | fit |
| D4_p61_r0.45_s9 | campaign F | 9 | done (stopped early) | no | 100% | 12761–32374 | 0.81 | 15740 → 22660 | 0.999 | 0.46 | 3600 | fit |
| D4_p61_r0.45_s10 | campaign F | 10 | done (stopped early) | no | 100% | 13400–29431 | 0.83 | 18060 → 25900 | 1.000 | 0.34 | 3160 | fit |
| D4_p97_r0.45_s1 | campaign D | 1 | done (stopped early) | no | 100% | 9733–42913 | 0.92 | 16300 → 25000 | 0.992 | 0.48 | – | fit |
| D4_p97_r0.45_s2 | campaign F | 2 | done (stopped early) | no | 100% | 9733–30694 | 0.92 | 16500 → 22980 | 0.996 | 0.50 | – | fit |
| D4_p97_r0.45_s3 | campaign F | 3 | done (stopped early) | no | 100% | 9860–36163 | 0.92 | 16740 → 24180 | 0.994 | 0.49 | – | fit |
| D4_p97_r0.45_s4 | campaign F | 4 | done (stopped early) | no | 100% | 9651–47148 | 0.92 | 16460 → 26520 | 0.994 | 0.46 | – | fit |
| D4_p97_r0.45_s5 | campaign F | 5 | done | no | 100% | 9573–84638 | 0.92 | 15660 → 26240 | 0.985 | 0.44 | – | final, fit |
| D4_p97_r0.45_s6 | campaign F | 6 | done (stopped early) | no | 100% | 9315–52599 | 0.91 | 14760 → 21300 | 0.997 | 0.53 | – | fit |
| D4_p97_r0.45_s7 | campaign F | 7 | done | no | 100% | 9572–53053 | 0.91 | 15600 → 22400 | 0.989 | 0.57 | – | final, fit |
| D4_p97_r0.45_s8 | campaign F | 8 | done (stopped early) | no | 100% | 9293–70719 | 0.91 | 14820 → 22460 | 0.993 | 0.58 | – | fit |
| D4_p97_r0.45_s9 | campaign F | 9 | done (stopped early) | no | 100% | 9680–64207 | 0.92 | 16220 → 24180 | 0.996 | 0.48 | – | fit |
| D4_p97_r0.45_s10 | campaign F | 10 | done | no | 100% | 9612–70602 | 0.91 | 15660 → 27360 | 0.990 | 0.52 | – | final, fit |
| T16_D2_p23_lw0.015_r0.45_s0_T80k | development runs | 0 | done | no | 66% | 21387–30511 | 0.26 | 21100 → 37740 | 1.000 | 0.04 | 1680 | share |
| T16_D2_p23_lw0.015_r0.45_s1_T80k | development runs | 1 | done | no | 64% | 21366–30373 | 0.26 | 21040 → 43360 | 1.000 | 0.04 | 1500 | share |
| T16_D2_p23_lw0.015_r0.45_s2_T80k | campaign F | 2 | done | no | 57% | 20504–28903 | 0.26 | 21700 → 42280 | 0.991 | 0.04 | 1580 | share |
| T16_D2_p23_lw0.015_r0.45_s3_T80k | campaign F | 3 | done | no | 61% | 20711–30023 | 0.25 | 19000 → 36880 | 1.000 | 0.04 | 1500 | share |
| T16_D2_p23_lw0.015_r0.45_s4_T80k | campaign F | 4 | done | no | 50% | 21115–29588 | 0.26 | 20700 → 41680 | 1.000 | 0.04 | 1700 | share |
| T16_D2_p23_lw0.015_r0.45_s5_T80k | campaign F | 5 | done | no | 50% | 20470–28518 | 0.25 | 23280 → 38820 | 1.000 | 0.04 | 1480 | share |
| T16_D2_p23_lw0.015_r0.45_s6_T80k | campaign F | 6 | done | no | 51% | 20568–28478 | 0.26 | 20960 → 36240 | 1.000 | 0.04 | 1600 | share |
| T16_D2_p23_lw0.015_r0.45_s7_T80k | campaign F | 7 | done | no | 36% | 20448–29006 | 0.26 | 22920 → 46180 | 1.000 | 0.05 | 1680 | share |
| T16_D2_p23_lw0.015_r0.45_s8_T80k | campaign F | 8 | done | no | 62% | 21062–30373 | 0.25 | 18700 → 37240 | 1.000 | 0.04 | 1400 | share |
| T16_D2_p23_lw0.015_r0.45_s9_T80k | campaign F | 9 | done | no | 64% | 20250–29128 | 0.25 | 21300 → 39340 | 1.000 | 0.05 | 1780 | share |
| seedD2_r0.5_s0 | development runs | 0 | done | PASS | 91% | 16621–25258 | 0.46 | 15320 → 25900 | 1.000 | 0.07 | 2060 |  |
| seedD2_r0.5_s1 | development runs | 1 | done | PASS | 85% | 15947–24472 | 0.47 | 16760 → 28920 | 0.995 | 0.08 | 1900 |  |
| seedD2_r0.5_s2 | development runs | 2 | done | PASS | 96% | 15914–25160 | 0.46 | 16020 → 26380 | 1.000 | 0.08 | 2020 |  |
| seedD2_r0.5_s3 | development runs | 3 | done | no | 87% | 15496–24128 | 0.46 | 16400 → 26260 | 0.990 | 0.08 | 2020 | final |
| seedD2_r0.5_s4 | development runs | 4 | done | PASS | 77% | 15707–24160 | 0.47 | 18140 → 28600 | 1.000 | 0.08 | 2040 |  |
| seedD2_r0.5_s5 | campaign F | 5 | done | PASS | 78% | 15673–24595 | 0.47 | 16980 → 28220 | 0.995 | 0.08 | 2000 |  |
| seedD2_r0.5_s6 | campaign F | 6 | done | PASS | 92% | 16247–25051 | 0.46 | 16480 → 26620 | 1.000 | 0.07 | 1780 |  |
| seedD2_r0.5_s7 | campaign F | 7 | done | PASS | 83% | 15681–23758 | 0.47 | 17160 → 26540 | 1.000 | 0.08 | 1940 |  |
| seedD2_r0.5_s8 | campaign F | 8 | done | PASS | 93% | 15859–24406 | 0.46 | 17020 → 25160 | 1.000 | 0.08 | 2300 |  |
| seedD2_r0.5_s9 | campaign F | 9 | done | PASS | 82% | 15221–23912 | 0.47 | 17120 → 26580 | 1.000 | 0.09 | 1960 |  |
| T3_D2_p47_r0.5_s0 | development runs | 0 | done | no | 99% | 12977–27187 | 0.67 | 12900 → 19860 | 1.000 | 0.16 | 3300 | fit |
| T3_D2_p47_r0.5_s1 | campaign F | 1 | done | no | 80% | 13693–28316 | 0.67 | 12380 → 20940 | 1.000 | 0.13 | 3300 | fit |
| T3_D2_p47_r0.5_s2 | campaign F | 2 | done | no | 90% | 12866–27965 | 0.67 | 12240 → 19980 | 1.000 | 0.15 | 3140 | fit |
| T3_D2_p47_r0.5_s3 | campaign F | 3 | done | no | 100% | 11661–23824 | 0.68 | 13980 → 21660 | 1.000 | 0.19 | 3140 | fit |
| T3_D2_p47_r0.5_s4 | campaign F | 4 | done | no | 100% | 11847–24922 | 0.68 | 13600 → 20760 | 0.998 | 0.18 | 3400 | fit |
| T3_D2_p47_r0.5_s5 | campaign F | 5 | done | no | 100% | 12367–23421 | 0.69 | 14300 → 21480 | 1.000 | 0.16 | 3260 | fit |
| T3_D2_p47_r0.5_s6 | campaign F | 6 | done | no | 100% | 11559–24641 | 0.68 | 13900 → 19840 | 1.000 | 0.21 | 3600 | fit |
| T3_D2_p47_r0.5_s7 | campaign F | 7 | done | no | 100% | 11884–24416 | 0.68 | 13700 → 20440 | 1.000 | 0.19 | 3380 | fit |
| T3_D2_p47_r0.5_s8 | campaign F | 8 | done | no | 100% | 12244–26836 | 0.67 | 13320 → 22660 | 1.000 | 0.19 | 3600 | fit |
| T3_D2_p47_r0.5_s9 | campaign F | 9 | done | no | 100% | 12315–26605 | 0.67 | 13280 → 19840 | 1.000 | 0.17 | 3360 | fit |
