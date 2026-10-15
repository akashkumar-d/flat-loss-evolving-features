# Campaign J: where the one-hidden-layer bias effect comes from (plan fixed before any evaluation run)

Written by the authors on 2026-10-06, as the follow-up to campaign I. The two arms K and S below were chosen first; arm KD
was added as the exact control for K (see "What each arm isolates").
Fixed after the development check below and before any evaluation run.

## Question

Campaign I trained biases in every hidden layer with the joint [W | b] Polar step. At one hidden layer (1HL), on the ten seeds
of the paper's p = 31 setting:

- From its post-knee peak to held-out t₉₀, the training CE fell by a median 5.3% with biases and 2.6% without them. The paired
  difference was 2.1–3.1 points (median 2.74), larger with biases in 10/10 pairs.
- The 3% peak band therefore closes before grokking ends: share 0.62 against 1.00, and 0/10 runs pass campaign C's endpoint
  (bias-free 4/10).
- With one-hot inputs a first-layer bias adds no functions (W_a e_a + b = (W_a + b 1ᵀ) e_a), so this is an optimizer effect.

Which part of the joint update makes the CE fall more during grokking?

## Arms (30 runs, locked here)

Each run copies campaign I's configuration for the same seeds (`B_H_p31_lw0.0150_s{i}`: width 512, λ_W 0.015,
η_W = η_V = 0.03, λ_V = 0.2, c = 24, 40k updates, campaign D's early stop, biases in the hidden layer) and adds one key:

| arm | key | update of the hidden layer | runs |
|---|---|---|---|
| K | `bias_after_knee: frozen` | campaign I's joint [W \| b] step until the knee; from the knee on, W gets the bias-free Polar step and b is held at its value at the knee (no step, no decay) | 10 |
| KD | `bias_after_knee: decay` | as K, but b keeps its decay, b ← (1 − ηλ) b | 10 |
| S | `bias_update: separate` | from the start, W gets the bias-free Polar step, Q/√r with Q, r = polar(−G); b gets one Polar direction along its own gradient, b ← b + η(−g/‖g‖/√r − λb), with the same r | 10 |

- **Seeds:** i = 1–10, init 5000 + i, split 6000 + i: campaign I's 1HL seeds.
- **Knee:** campaign C's, the first heavy observation with ‖V‖ ≥ 0.99/λ_V. The runner detects it from its own heavy observations,
  so it is the knee the endpoint code reads from the output file.
- **References, paired by seed** (`comparison.json`, with the SHA-256 of every file):
  - *joint*: campaign I's `B_H_p31_lw0.0150_s{i}`;
  - *twin*: the bias-free paper run `H_p31_lw0.0150_s{i}` (campaign D for i = 1–2, campaign F for i = 3–10).

## What each arm isolates (1HL, one-hot inputs)

Write G = [G_a | G_b] for the gradient of W = [W_a | W_b] and g for the bias gradient. Then G [1_p; −1_p] = 0, so G has rank
2p − 1 = 61, and g = G_a 1 = G_b 1 lies in G's column space.

- **Joint (campaign I).** The Polar step of M = [G | g] whitens with MMᵀ = GGᵀ + ggᵀ:
  - W moves along (GGᵀ + ggᵀ)^(−1/2) G;
  - b moves along (GGᵀ + ggᵀ)^(−1/2) g. This step's norm is always √(p/(p + 2)) = 0.969 for p = 31 (times η/√r). M's null
    space is spanned by [1_p; −1_p; 0] and [1_p; 0; −1], whatever the weights, so the bias coordinate keeps a fixed share of
    M's row space.
  - The development measurement (CHECKS.md) gives this norm at every update, and a cosine of 0.99 between b's step and −g.
- **S** keeps b's step: its size η/√r is within 3% of the joint one, and its direction is within cosine 0.99. S removes the
  bias column from W's whitening: W moves along (GGᵀ)^(−1/2) G, the bias-free rule.
  - S like the twin: the bias's own step does not cause the effect. It comes from W's step sharing the Polar step with the bias
    column.
  - S like the joint runs: the bias's own step causes it.
- **KD** equals the joint run up to the knee. After it, (W_a + b 1ᵀ, W_b, V) follows the bias-free optimizer exactly: the
  decay of b is the decay of the folded columns, and the gradients and Polar factors are the same (`tests/test_j.py`,
  `test_decay_equals_folded_bias_free`). KD therefore splits the effect in two:
  - KD like the twin: the effect comes from the joint step after the knee.
  - KD like the joint runs: the effect comes from the network reached before the knee.
- **K** equals the joint run up to the knee. After it, W gets the bias-free step, but the decay pulls W_a + b 1ᵀ towards
  b_knee 1ᵀ instead of 0, so the knee offset persists. K against KD shows whether keeping that offset matters.

## Measures (fixed)

- **Primary, per run:** fall = (C(t*) − C(t₉₀)) / C(t*).
  - C is the cycle-mean training CE, (L_t + L_{t+1})/2.
  - t* is campaign C's peak: the largest C after the knee.
  - t₉₀ is the first logged step after which held-out accuracy stays ≥ 0.9 (`endpoints_i.py`).
- **Primary, per arm:** ρ = median_i(fall_arm − fall_twin) / median_i(fall_joint − fall_twin), the share of campaign I's effect
  that remains (0 = the bias-free run, 1 = campaign I).
  - **Reading, fixed now:** ρ ≤ 1/3 removes most of the effect; ρ ≥ 2/3 keeps most of it; otherwise partial.
  - **Reported with it:**
    - exact two-sided sign tests of fall_arm against fall_joint and against fall_twin (10 pairs; p ≤ 0.05 needs 9/10);
    - a paired bootstrap 95% interval for ρ (10,000 resamples of the seeds, generator seed 20261006; descriptive).
- **Secondary:**
  - campaign C's per-run endpoint (`endpoints_i.py`, campaign I's file, unchanged): pass k/10 with a Wilson interval, failed
    checks, share, band W, t₁₀, t₉₀, final held-out, ΔA_H, fit;
  - band end − t₉₀, the minimum training margin at t₉₀, the CE level on W and at t₉₀, and ‖b‖ over training;
  - exploratory, as in campaign I's follow-up: the share with a 5% band.
- **Checks:**
  - K and KD: every knee equals campaign I's knee for the same seed, and up to the knee the losses equal campaign I's at
    7 significant digits. Campaign I's files hold 7 significant digits, and the machines differ (CHECKS.md).
  - Every run finishes without error.

Computed by `analyze_J.py report` from the run files, each checked against its SHA-256.

## Code

- `bias_runner.py`: campaign I's runner (sha256 16bb2ebd…) plus the keys `bias_update` and `bias_after_knee`. Without them it
  equals campaign I's runner bit for bit.
- Tests:
  - `tests/test_bias.py`: campaign I's 7 tests, unchanged, run against this runner.
  - `tests/test_j.py`: 7 new tests:
    - equality with campaign I's runner;
    - the separate step against an SVD;
    - the switch at the knee: identical before it, 'frozen' keeps b exactly, 'decay' shrinks it by (1 − ηλ) per update;
    - KD against the folded bias-free run;
    - resume after the knee;
    - unsupported combinations.
- `run_local.py` runs the manifest here, two runs at a time, and writes a `record.json` per run.

## Development check (before this plan was fixed)

Campaign I's development seeds (init 9001–9002, split 9101–9102), one run per arm and seed; details in CHECKS.md. The only
decision it could change is the size of b's step in S, and the measurement above settles it. Nothing else is a candidate for
change: no setting is tuned on these runs.

## Compute

The development runs took 5.5–9.5 minutes each on one core of this container (runs that do not stop early go to 40k
updates). With 30 runs, two at a time, the campaign takes about 2 hours.

**Commitments:**
- No setting is re-tuned after the evaluation runs start.
- No run is dropped or repeated, except one that stops with an error; it is rerun with the same seeds.
- All three arms are reported, whatever they show.
