# Campaign X, MLP side: the paper's one-hidden-layer runs with the Fourier readouts (2026-10-11)

claude/1hl-figures. For the contrast panels of campaign X's figures; not part of the worker runs.

- Configuration: campaign F's one-hidden-layer runs at the paper's headline recipe (p = 31, training fraction 0.8, width 512,
  init scale 24, eta 0.03, lambda_W 0.015, lambda_V 0.2), seeds 3-10 (`cfgs/`).
- Training: `mlp_runner/bias_runner.py` (campaigns I-M, sha256 16bb2ebd..., unchanged), the loop of its run() without the early
  stop, to 13000 updates. The runs equal campaign F's archived runs to the stored precision (checked on seed 3: loss within
  5e-7 relative, held-out accuracy identical; the readout `fref_all` equals the archived refit `v1_ridge_acc_te`).
- Readouts: tf_runner 1.4's `fourier_observers` (campaign X's runner, `../tf_runner.py`) on the last hidden layer every 100
  updates, with the head V and the first layer's token columns; the logistic probe every 200. No mirror pairs here (the MLP's
  two operands have separate input weights), so held-out accuracy is on all held-out pairs.
- `mlp_fourier.py CFG OUT T_END EVERY` runs one; `run_all.sh` runs the eight in two lanes (about 6 minutes each).
