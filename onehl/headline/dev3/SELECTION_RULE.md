# Hero dev sweep v3 (p = 31): rule (declared before launch, 2026-09-29)

This sweep does not replace the locked hero (p = 31, 80% data, λ_W = 0.015, λ_V = 0.2), whose 20-seed confirmation has already finished. It tests two variants of that hero.

**Arm A: eq. (4) exactly (no head decay, λ_V = 0) with small head steps.**

- Random 80% split.
- λ_W {0.01, 0.015} × η_V (= χs) {3e-5, 1e-4, 3e-4, 1e-3}.

**Arm B: the theory's quadratic-residue half-grid split (Sec. 1.1, 465/961 training pairs).**

- λ_W = 0.015.
- (λ_V, η_V) ∈ {(0.2, 0.03), (0, 1e-4), (0, 3e-4)}.

Both arms use 4 dev pairs, 40k updates and protocol-v1 observers.

The unchanged `select_hero.py` rule (strict, then relaxed) is applied to all configs, and each arm is reported separately.

A config that qualifies becomes a candidate for its own fresh 20-seed confirmation. That confirmation is proposed to the user, not launched automatically.

The runner change is the optional `split = "qr_difference"` in `model.initialize`. The random-split path is bitwise identical to the locked runner (checked before launch).
