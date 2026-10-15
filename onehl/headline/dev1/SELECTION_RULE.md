# Hero dev sweep: selection rule (declared before launch, 2026-09-29)

Purpose: pick one p = 17 recipe and one fixed window for the paper's headline figure, then lock both for a fresh 20-seed confirmation. The hero figure will show confirmation pair 1.

Grid (see SWEEP.json):

- Fixed for every run: p = 17, width 512, c = 24, η_W = 0.03, supported-Polar W step (relative cutoff 1e-10), Frobenius-normalized head step, 50k updates, protocol-v1 observers (cheap every 100, full every 500).
- Head decay λ_V = 0.2 (4 dev pairs per config): train fraction {0.5, 0.8} × λ_W {0.005, 0.0075, 0.01, 0.015} × η_V {0.01, 0.03}.
- No head decay, λ_V = 0 (2 dev pairs per config): train fraction {0.5, 0.8} × λ_W {0.005, 0.01} × η_V {0.001, 0.003, 0.01, 0.03}.

Candidate windows are [t0, t1] on the 500-update grid, with t0 ≥ 1000 and t1 − t0 ≥ 2000.

A (config, window) pair qualifies only if every dev run of that config satisfies all of the following on the window:

1. The training-CE range (max − min)/min, taken over every update, is ≤ 8%.
2. The window-minimum training CE is ≥ 0.12.
3. ΔA_H = A_H(t1) − A_H(t0) ≥ 0.07.
4. The canonical q = 4 AGOP endpoints are valid, and ΔAGOP − Δnull ≥ 0.02 (null = median of 8 random frames).
5. Held-out accuracy is ≤ 20% at t0, and first reaches 90% at an update in (t0, t1 − 500].
6. Strict training accuracy is 100% at every update in the window.

Thresholds 1–3 carry a margin over the headline targets (range ≤ 10%, CE ≥ 0.1, ΔA_H ≥ 0.06), to absorb seed variation in the confirmation.

Choice among qualifying pairs:

- Take the pair with the largest minimum ΔA_H across dev runs.
- If two pairs are within 0.005 on that, break the tie in this order:
  1. larger minimum window-minimum CE;
  2. train fraction 0.5;
  3. λ_V = 0.2;
  4. shorter window.

If nothing qualifies:

1. Rerun the rule with the bare targets instead (range ≤ 10%, CE ≥ 0.10, ΔA_H ≥ 0.06; conditions 4–6 unchanged).
2. If nothing qualifies under those either, launch no confirmation and report per-config diagnostics instead.

`select_hero.py` implements this rule; its sha256 is recorded in SOURCE.sha256 before launch.
