# Hero dev sweep v2: selection rule (declared before launch, 2026-09-29)

Same rule as dev sweep v1 (unchanged `select_hero.py`), applied to the v2 grid:

- p ∈ {23, 31} × train fraction {0.5, 0.8} × λ_W {0.01, 0.015, 0.02}.
- Fixed: λ_V = 0.2, η_V = 0.03, η_W = 0.03, c = 24, width 512.
- 4 dev pairs per config, 40k updates, protocol-v1 observers (cheap every 100, full every 500).

Candidate windows are [t0, t1] on the 500-update grid, with t0 ≥ 1000 and t1 − t0 ≥ 2000.

A (config, window) pair qualifies only if every dev run satisfies all of:

1. CE range ≤ 8% over every update.
2. Window-minimum training CE ≥ 0.12.
3. ΔA_H ≥ 0.07.
4. ΔAGOP − Δnull ≥ 0.02, with valid q = 4 endpoints.
5. Held-out ≤ 20% at t0, and first 90% crossing in (t0, t1 − 500].
6. 100% strict training accuracy at every window update.

Choice among qualifying pairs:

- Take the largest minimum ΔA_H.
- Break ties (within 0.005) in this order:
  1. larger minimum window-minimum CE;
  2. train fraction 0.5;
  3. shorter window.

If nothing qualifies, apply the bare targets (range ≤ 10%, CE ≥ 0.10, ΔA_H ≥ 0.06). If still nothing qualifies, launch no confirmation and report diagnostics instead.

The chosen recipe and window are then locked for a fresh 20-seed confirmation (seed rule 20260930). The hero figure is confirmation pair 1.
