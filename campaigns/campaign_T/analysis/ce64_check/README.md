# Float64 check of the flat training loss (campaign T, 1 layer, lam_w 0.02, eta_w 0.1, lam_v 0.2; 2026-10-08)

`ce64.py` reruns both development seeds for 6,500 updates with tf_runner 1.1 (float32 training, one thread, unchanged) and records
at every update the training CE computed in float64 from the same float32 logits (`ce64`), next to the float32 CE (`ce32`).
The rerun ran on a different CPU than the campaign session, so it is not bit-identical, but it tracks it (held-out accuracy at
2k/3k/4k/5k updates: seed 1 0.04/0.31/0.89/0.99 vs 0.03/0.27/0.86/0.99 in the campaign run; seed 2 0.04/0.23/0.78/0.98 vs
0.04/0.24/0.79/0.99).

Result: while held-out accuracy and the refit rise from about 0 to about 1 (2.5k-4.5k updates), the float64 training CE stays flat
at 3.3-4.5e-7 (geometric mean per 500 updates; the float32 value is about 10% lower and equally flat), with brief upward spikes;
the smallest training margin stays positive throughout (mostly 7-12; lowest 0.7, seed 1 near 4.7k). The flat level is therefore a real, decay-set floor of the loss
(CE about exp(-margin)), not float32 rounding.
