# Campaign Z pilots (local, 2026-10-12): seed 1 (init 7101, split 8101), tf_runner 1.5, Fourier readouts off, heavy logs every 500

| config | what |
|---|---|
| `pZ_B_p61_f0.4_g2_tau5k_s1.json` | p = 61, frac 0.4, lam_v 0.275, gain 2 -> 1 (time constant 5k); stopped at 3,500 updates |
| `pZ_C_p61_f0.4_lv0.45_g2_tau5k_s1.json` | the same with lam_v 0.45; stopped at 3,000 |
| `pZ_A_2L_f0.2_g2_tau10k_s1.json` | two blocks, frac 0.2, lam_v 0.275, gain 2 -> 1 (10k); running |
| `pZ_D_2L_f0.3_lv0.5_g2_tau5k_s1.json` | two blocks, frac 0.3, lam_v 0.5, gain 2 -> 1 (5k); running |

`status_*.txt`: every 250 updates the training CE (201-update average), training accuracy, held-out accuracy on all pairs and
on the no-mirror subset, the gain and A_H (head input, each block's MLP neurons), from `peek.py`. PLAN.md section 8 reads them.
