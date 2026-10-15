# Campaign H: Theorem 4.1's configuration at p = 101 (plan fixed before any campaign run)

Written by the authors on 2026-10-06, after a development pilot on one seed (below) and before any campaign run.

## Purpose

Theorem 4.1 proves, from an engineered initialization, that in one window of flat training CE the network stays fitted, its
hidden Fourier fraction A_H grows, and held-out accuracy recovers. The exact simulation of that construction at p = 101 shows
the clauses, but A_H stays near 0 for most of the window and its refit recovers after the network. This campaign trains the
network of the theorem in the theorem's configuration from the paper's ordinary iid initialization, with enough seeds for
figures like the paper's Figures 1–3, and asks whether three things happen inside one window of flat training CE:

1. the training CE stays flat;
2. A_H rises;
3. Theorem 4.1's train-only refit gains held-out accuracy, and does so before the network's own head.

## What is taken from Theorem 4.1, and what is not

| | campaign H | Theorem 4.1 |
|---|---|---|
| p | 101 | p ≥ 101 |
| split | Train = {(a, b): b − a a nonzero quadratic residue}, n = 5,050; Test = the other 5,151 pairs, diagonal included | same |
| network | one hidden layer, bias-free ReLU, full-batch mean CE | same |
| optimizer | eq. (paper:optimizer): Polar hidden step with decoupled decay, normalized head step, λ_V = 0, fixed rates | same |
| clock | η_W λ_W = 1 − (9/10)^(1/32), with λ_W = 1 and η_W = 0.0032871 (Appendix A's λ = 1, s = (1 − q_clk)/λ) | same |
| refit | own-RMS ridge, τ = ‖H_Train‖_F² / (32 n) (eq. paper:refit-penalties); the canonical τ = 10⁻³ n is also logged | same |
| A_H | eq. (paper:AH) on the full grid | same |
| width | 512 | M ≥ p(p² + 3)/2 = 515,302 |
| initialization | iid, as in the paper's experiments: W₀ = 24·N(0, 1)/√p, V₀ = N(0, 1)/(24√M) | an open set around an engineered center |
| head step | η_V ∈ {3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻²} | fixed, small (10⁻¹² in the p = 101 simulation) |

The width and the initialization cannot be matched by training: the theorem's width needs about 10⁸ weights per layer, and its
initial region is engineered. With a nearly frozen head (η_V = 10⁻⁶) the pilot stalls (below), so η_V is the one rate chosen
here.

## Runs (locked here)

All runs: p = 101, QR split, width 512, λ_W = 1, η_W = 1 − (9/10)^(1/32), λ_V = 0, iid initialization as above, 40,000
updates, observations every 50 updates (accuracies, CE, margins) and every 250 updates (A_H, both refits, norms).

| arm | η_V | seeds | runs | role |
|---|---|---|---|---|
| `ev1e-3` | 10⁻³ | 1–10 | 10 | primary (the pilot's setting) |
| `ev3e-4` | 3·10⁻⁴ | 1–6 | 6 | bracket |
| `ev3e-3` | 3·10⁻³ | 1–6 | 6 | bracket |
| `ev1e-2` | 10⁻² | 1–6 | 6 | bracket |
| `ev1e-6` | 10⁻⁶ | 1–2 | 2 | control: nearly frozen head |
| `paper` | 10⁻³, with the paper's η_W = 0.03, λ_W = 0.02 | 1–2 | 2 | control: the experiments' slower clock |

32 runs. Seeds: init_seed = 9100 + s (the QR split has no seed). No seed of this range has run before.

**Early stop** (campaign D's rule): once held-out accuracy has stayed ≥ 0.99 since update s, a run stops at the first observed
update ≥ max(1.25 s, s + 4000). Runs that never generalize go to 40,000 updates.

## Per-run endpoints (fixed here)

Let t_a = 1,000 (after the initial transient, during which the iid initialization decays) and T the last update.

- **J1, flat CE:** the raw training CE varies by at most 1% on [t_a, T], v([t_a, T]) ≤ 0.01, judged on every update.
- **J2, representation:** A_H(T) − A_H(t_a) ≥ 0.02 (the paper's F02 threshold).
- **J3, refit gain:** Theorem 4.1's refit has held-out accuracy ≤ 0.1 at t_a and reaches ≥ 0.9 by T.
- **J4, refit first:** the refit reaches 0.5 held-out accuracy at least 1,000 updates before the network does (or the network
  never does).
- **Joint:** J1–J4 all hold.
- **Reported, not required:** whether training accuracy is 1 at every observation from the refit's 0.5 crossing to T (the
  theorem's clause (i) on that stretch); the first observations at which training accuracy reaches 1, the refit 0.5 and 0.9,
  and the network 0.5 and 0.9; ΔA_H; the canonical refit.

## Report

- Per arm: k/n for J1–J4 and the joint endpoint, with Wilson 95% intervals; medians of v, ΔA_H, the crossing times, the final
  accuracies and A_H.
- **Figure:** the primary arm `ev1e-3` (all 10 seeds, whatever they show), in the layout of the paper's Figures 2–3: rows of
  training CE / log p, accuracies (median training, the network per run and median, the median refit), and A_H. The brackets
  and controls go in a supplementary figure.
- **Commitments:** no setting is re-tuned after these runs and no run is dropped or repeated (except a run that stops with an
  error, which the worker reruns with the same seed).

## Development pilot (one seed, 7001; not part of the campaign)

Run here on 2 cores before this plan was fixed (same configuration, 6,000 updates):

| pilot | η_W, λ_W | η_V | after 6,000 updates |
|---|---|---|---|
| frozen head | clock, 1 | 10⁻⁶ | train acc 0.41, held-out 0, A_H 0.003, refit 0.005: stalled |
| primary | clock, 1 | 10⁻³ | CE flat to 0.15%; train acc 0.85; network held-out 0; A_H 0.0025 → 0.011 rising; refit 0 → 0.25 rising |
| paper clock | 0.03, 0.02 | 10⁻³ | (8,000 updates) CE fell 9.4%; train acc 0.79; held-out 0; A_H 0.003; refit 0 |

## Compute

About 32 runs × 40,000 updates × 0.115 s ≈ 41 core-hours if no run stops early; on 8 sessions × 4 cores about 1.5 hours.
