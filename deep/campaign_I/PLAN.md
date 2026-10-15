# Campaign I: hidden-layer biases at the paper's p = 31 settings (plan fixed before any evaluation run)

Written by the authors on 2026-10-06. It follows an earlier guideline ("campaign C again, with biases as the only change"),
restricted to one, two and four hidden layers at p = 31, and run on 16 cloud sessions. Fixed after the development check below and before any evaluation run.

## Question

Every network in the paper is bias-free. Do the paper's p = 31 results hold when every hidden layer has a bias, with nothing
else changed?

## The change (the only one)

- **Where.** Every hidden layer: h^(ℓ) = ReLU(W^(ℓ) h^(ℓ−1) + b^(ℓ)). The head keeps no bias, so its step and its 1/λ_V cap are
  unchanged.
- **Update.** b^(ℓ) is one more column of W^(ℓ), a weight on a constant input of 1. [W | b] gets one supported-Polar step with
  the same √rank normalization and the same decay: the paper's optimizer applied to the enlarged matrix. The bias gradient is
  the sum over training rows of the layer's backpropagated signal.
- **Initialization.** b = 0, and no random numbers are drawn for it. Each bias run therefore starts from exactly the network
  of the bias-free run with the same seeds; any difference comes from training.
- **First layer.** With one-hot inputs a first-layer bias adds no new functions (it can be folded into the input weights); it
  changes only the optimization. At one hidden layer the campaign therefore tests the optimizer, not the function class.
  Biases in deeper layers do enlarge the function class.

## Runs (locked here)

Each run copies the configuration of the bias-free paper run with the same seeds, adds `bias: true`, and names that run
(`paired_with`).

| family | in the paper | setting | seeds (init, split) | runs | bias-free twin |
|---|---|---|---|---|---|
| 1HL | depth figure (`fig:deep-main`), depth 1; the recipe of Fig. 2's p = 31 column | width 512, λ_W 0.015, η_W = η_V = 0.03, λ_V = 0.2, c = 24, 40k updates, campaign D's early stop | i = 1–10: 5000 + i, 6000 + i | 10 | `H_p31_lw0.0150_s{i}` (campaigns D, F) |
| 2HL | `fig:deep-main`, depth 2 | width 256, fixed last-layer rate r = 0.5, 40k updates, no early stop | i = 0–9: 101 + i, 201 + i | 10 | `seedD2_r0.5_s{i}` (development 0–4, campaign F 5–9) |
| 4HL | `fig:deep-main`, depth 4 | campaign C's D4 cohort: width 256, pilot at r₀ = 0.35 stopping at held-out 0.1, recipe x* = 3.85, main run 100k | i = 1–40: 1000 + i, 2000 + i | 40 | `C2_D4_s{i}` (i ≤ 20), `C1_D4_s{i}` (i > 20) |

60 runs. The depth-4 pilots are short for every seed: t₁₀ is the same quantity as in campaign C's full pilots of seeds 1–20.

`twins.json` (written by `make_twins.py` from campaign C's and campaign F's `analysis.json`) holds each twin's endpoint row as
the paper scored it. Bias-free, 4/10 pass at one hidden layer (the six misses end below 0.99 held-out, as the paper reports),
9/10 at two and 39/40 at four.

## Code

`bias_runner.py` (new, this campaign's runner). Without the bias key it performs the campaign runner's computation operation
for operation; its bias-free reruns reproduce archived paper runs at every stored value (CHECKS.md). With `bias_frozen` (biases
present but never updated) a run equals the bias-free run bit for bit (`tests/test_bias.py`).

## Endpoints (copied from campaign C's PREREGISTRATION, section 5)

Per main run (depth-4 main runs; the 1HL and 2HL runs themselves):
- knee: the first heavy observation with ‖V‖ ≥ 0.99/λ_V; peak band W: the longest run of steps around the largest cycle-mean
  CE after the knee on which CE ≥ CE*/1.03, from the training loss alone;
- a run **passes** when the share of the held-out 0.1 → 0.9 rise inside W is ≥ 0.75, the final held-out accuracy is ≥ 0.99,
  A_H of the last hidden layer rises by ≥ 0.02 across W, and every logged minimum training margin in W is > 0.

Computed with `endpoints_i.py`, campaign C's definitions re-implemented for this campaign. It reproduces campaign C's
`analysis.json` on all 40 archived depth-4 main runs and campaign F's on the 13 archived twins at depths 1 and 2, field for
field (CHECKS.md).

**Claim at depth 4:** at least 37 of 40 runs pass (campaign C's rule). At depths 1 and 2 (10 runs each) the result is
reported as k/10 with a Wilson interval, as in the paper.

**Secondary.** If the paper keeps the post-hoc window of Appendix F.5, the bias runs are also scored with it, reported as a
post-hoc count beside the bias-free one.

## Bias-specific report

- **Paired with the bias-free twin** (same seeds): pass/fail flips (exact McNemar test), the shift in t₁₀ and t₉₀, the
  recipe's rate and landing x at depth 4, and the change in the CE level on W.
- **The biases themselves:** ‖b^(ℓ)‖ over training (heavy observer `normb`).
- **In the paper:** one paragraph and a pass table with and without biases in Appendix F; optionally a figure laid out like
  the depth figure (`fig:deep-main`). The main figures stay as they are.

## Development check (before this plan was fixed)

Development seeds outside the evaluation sets (init 9001–9002, split 9101–9102), two per depth, run here; details in CHECKS.md.
The recipe lands both depth-4 main runs at x = 3.97 and 3.91, inside campaign C's bias-free range (3.53–4.10), so it is kept.
Nothing else was a candidate for change, and nothing was changed.

## Compute

From the archived runs' durations: 1HL 0.1 h, 2HL 0.25 h and a depth-4 chain 1.9 h on one core, about 91 core-hours with
campaign C's 1.15 contention factor. On 16 sessions × 4 cores every run starts at once and the campaign takes about as long as
its longest chain, about 2–2.5 hours.

**Commitments:** no setting is re-tuned after the evaluation runs start, and no run is dropped or repeated, except a run that
stops with an error, which the worker reruns with the same seeds.
