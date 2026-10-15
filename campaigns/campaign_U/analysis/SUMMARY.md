# Campaign U: results (exploration, claude/1hl-figures, 2026-10-09)

71 of 71 runs finished on 12 sessions (no failures, no push errors); all files match `collected/INDEX.json`. The five
campaign-T runs that used to crash completed 60000 updates under tf_runner 1.2. Analysis as fixed in PLAN.md section 6
(`flat_windows.py`): the key message at averaging windows w = 1, 100, 500, 2000, 5000, 10000 updates, geometric and
arithmetic averages, tolerance delta = 0.1 (0.03 and 0.3 as sensitivity), with the ridge refit and with the logistic probe.
Exploration only; not evidence for the paper.

Key message at window w: a stretch of at least max(2w, 500) updates in which the averaged training CE never falls more than
10% below its running maximum, the refit rises by >= 0.2 inside it, the held-out accuracy rises by >= 0.2 from its start to
the end of the run, and the final held-out accuracy is >= 0.99 ("pass"); "pass3" also needs training accuracy >= 0.97 at
every logged step of the stretch.

## 1. Results by setting (delta 0.1, ridge refit; seeds passing at some window, pass3 in brackets)

| setting | CE / log p over I | pass (pass3) | windows |
|---|---|---|---|
| **Polar, campaign-T layout** | | | |
| 1 layer, lam_w 0.02, eta_w 0.1, lam_v 0.2 (L1, seeds 3-5) | 7e-8 (CE about 3e-7) | 3/3 (3/3) | 100-2000 |
| 2 layers, lam_w 0.01, eta_w 0.1, lam_v 1 (H2, seeds 3-5) | 0.0047 (CE about 2e-2) | 3/3 (0/3) | 500-5000 |
| 1 layer, lam_w 0.01, lam_v 10 | 0.044 (CE about 0.2) | 1/3 (1/3) | 500 |
| 1 layer, lam_w 0.01, lam_v 3 (seeds 3-5) | 0.005 | 0/3 | - |
| 2 layers, lam_w 0.01, lam_v 2 | no generalization (final held-out 0.05-0.74) | 0/3 | - |
| campaign-T reruns (lam_w 0.03 with lam_v 3; two layers lam_w 0.03, lam_v 0.2) | 0.58-0.64 for lam_v 3 | 0/5 | held-out stalls at 0.73-0.89 (lam_v 3) |
| **Polar, final RMS division** | | | |
| 1 layer, lam_v 1, eta_w 0.03 | 0.79 | 3/3 (0/3) | every window, 1-10000 |
| 1 layer, lam_v 0.5, eta_w 0.03 | 0.62 | 3/3 (0/3) | 2 of 3 seeds at most windows |
| 1 layer, lam_v 0.5, eta_w 0.1 | 0.66 | 3/3 (0/3); probe 2/3 | short and long windows |
| 2 layers, lam_v 0.5 | 0.65 | 3/3 (1/3) | every window |
| 2 layers, lam_v 1 | 0.80 | 2/3 (0/3) | every window |
| 1 layer, lam_w 0.02, lam_v 0.5 or 1 | 0.68, 0.83 | 1/3, 1/3 | - |
| 1 layer, lam_v 1, eta_w 0.1 | 0.80 | 0/3 (unstable: held-out falls back to 0.45-0.89) | - |
| 1 layer, lam_v 0.25 | 0.39 | 0/3 (the CE is still falling during the held-out rise) | - |
| **AdamW** (wd 0.3, 1, 3; one and two layers; with and without RMS) | 1e-4 to 0.05 | 0/24 at every window, both readouts | - |

With the logistic probe in place of the ridge refit the counts are the same or within one seed. At delta 0.3, AdamW passes
in 1 (ridge) and 2 (probe) of 24 runs; at delta 0.03 in none. Figure: `U_scoreboard` (per setting and window).

## 2. What it shows

1. **The key message belongs to the Polar rule's capped regime, at whatever CE level the caps set.** It holds at CE about
   3e-7 (L1, 3/3 with the training set fit throughout), about 2e-2 (two layers, lam_v 1, 3/3) and 0.6-0.8 log p (with the
   final RMS division). Across the feature-learning interval I (first time the refit or held-out accuracy reaches 0.1 to the
   later of their first times at 0.9), the 500-update geometric average of the CE ends at a median 0.87 of its starting value
   for both Polar groups, and at 0.045 for AdamW (a 22-fold fall).
2. **Window.** The Polar plateaus pass over a range of windows (L1: 100-2000; two layers lam_v 1: 500-5000; RMS: every
   window, even the raw CE). AdamW fails at every window up to 10000 updates: its CE falls through the event at every scale,
   with slingshot oscillations on top. The flat look of AdamW's curve came from the log axis and long averaging.
3. **A higher level changes the sequence of events.** With the RMS division the CE is pinned near the head cap's level from
   about update 500; training accuracy, held-out accuracy and the ridge refit rise together under it. In one layer with
   eta_w 0.03, the held-out accuracy reaches 0.9 before the training accuracy reaches 0.99 (lam_v 1: updates 2900-10400 vs
   20400-31600). The logistic probe of the head's input is at 1.0 by update 2000, while the ridge refit keeps improving for
   20000 updates (lam_v 1, eta_w 0.03: 0.40-0.61 at update 2000, 0.87-0.96 at 20000, CE 3.78 -> 3.71). So at a high CE the
   flat stretch includes the training fit, and pass3 is rare (1 of 27 RMS runs).
4. **The cleanest higher-level example in the original layout:** two layers, lam_w 0.01, eta_w 0.1, lam_v 1 (now 4 of 5
   seeds with campaign T's seed 1). Seed 5, updates 12000 -> 16000: CE 0.0175 -> 0.0264 (rising), held-out 0.11 -> 0.91,
   ridge refit 0.09 -> 0.94, probe 0.12 -> 0.96, A_H of the last MLP 0.23 -> 0.76. Training accuracy averages 0.98-0.99 in
   the stretches but dips to 0.88-0.91 at loss spikes, which is why pass3 fails.
5. **Readouts.** For AdamW without the RMS division, the ridge refit ends at 0.37-0.88 while the logistic probe ends at
   0.996-1.0: the ridge (MSE) readout misses features that AdamW's head uses.

## 3. Files

- `key_message_by_window_U_d{0.03,0.1,0.3}.json`: per run, the key-message check at every window, kind and readout, and
  the flatness profile over I.
- Figures: `U_scoreboard`, `U_flat_scales_overview` (six representative runs: CE with 100/1000/5000 averages, accuracies
  with ridge refit and probe, drawdown against window), `settings/U_dyn_<setting>` (every setting's seeds).
- Code: `flat_windows.py` (now with `--readout` and an exact faster stretch search; identical results on campaign T's 75
  runs), `plot_flat_scales.py`, `plot_U.py`.

## 4. Possible next steps

- Confirm the two cleanest settings (L1; two layers lam_v 1) on fresh seeds, with the rule fixed beforehand (for example
  pass3 on a 500-update average of the training accuracy, so loss spikes do not decide it).
- For a high-level plateau with a fit-first phase: the RMS division with a smaller training fraction (0.2) or a slower
  hidden step, so that the held-out rise starts after the training fit.
