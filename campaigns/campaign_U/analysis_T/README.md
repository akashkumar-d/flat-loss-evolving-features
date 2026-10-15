# Campaign T seen through averaging windows (claude/1hl-figures, 2026-10-08)

- `T_flat_scales.{pdf,png,json}` (plot_flat_scales.py): four campaign-T runs (AdamW one layer wd 1; Polar one layer lam_w 0.02,
  eta_w 0.1, lam_v 0.2; Polar two layers lam_w 0.01, eta_w 0.1, lam_v 1; Polar one layer lam_w 0.01, eta_w 0.1, lam_v 1).
  Rows: training CE with centred geometric averages over 100, 1000 and 5000 updates and the arithmetic average over 5000;
  accuracies; drawdown of each average over the feature-learning interval I against the window (dotted line: |I|).
- `key_message_by_window_T.json` (flat_windows.py, delta 0.1): for each of the 75 analysed runs and each window (1, 100, 500,
  2000, 5000, 10000 updates; arithmetic and geometric), the key-message check and its stretch, plus the flatness profile.

Result: the key message holds at some window in 10 of the 75 runs, all with the Polar rule, and in none at the raw CE or
at a 100-update window:
- seven one-layer lam_v 0.2 runs (lam_w 0.01 or 0.02): CE 2.6e-7 to 6e-7 on the stretch at the shortest passing window, up
  to 2e-5 at the longest;
- one one-layer lam_v 1 run (lam_w 0.02, eta_w 0.03): arithmetic 2000-update window only, CE 8e-6;
- two two-layer runs (lam_w 0.01, eta_w 0.1): lam_v 0.2 at CE 2e-6 to 3e-4, and lam_v 1 at CE 2.3e-2 to 5.1e-2, the only
  one above 1e-3, which fails the 3%-training-error condition.
It holds for none of the 8 AdamW runs at any window up to 10000 updates.
