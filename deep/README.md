# Deeper networks: code for Section 5.2 and Appendix F

This folder holds the runners, configurations, manifests and analyses of the multi-hidden-layer experiments, the
one-hidden-layer settings of the across-prime figures, campaign H (Theorem 4.1's configuration, one hidden layer), and
campaigns I and J (hidden biases). Figure and section numbers refer to version 1 of the paper on arXiv.

| Folder | Experiment | Runs | In the paper |
|---|---|---|---|
| `development/` | exploratory development runs at depths 1–5, from which the rate recipe was derived | 199 (34 task files), plus 6 cross-machine check files | Figure 3 (depth 2, with campaign F), Appendix F.1 |
| `code/` | `natplat_server.py`, a byte-identical copy of `development/natplat_server.py`, at the path that the campaigns' tests (hash-locked before the campaigns) compare against | — | — |
| `campaign_C/` | **pre-registered depth campaign**: p = 31, width 256, depths 3–5 | 164 chains, 224 scored runs | Section 5.2, Figures 3 (depths 3–5), 8 and 9 (p = 31), Table 2, Appendix F.2 |
| `campaign_D/` | development sweep across p = 23, 31, 47, 61, 97 (not pre-registered) | 127 | chose the settings of Figures 2 and 9; Appendices E.4 and F.3 |
| `campaign_F/` | ten seeds at the 17 settings of the across-prime and depth-2 figures (settings fixed in `PLAN.md` before any run) | 137 new; 170 in all | Figures 2, 3 (depth 2) and 9 (p ≤ 61), Appendices E.4 and F.3 |
| `campaign_G/` | the three p = 97 settings of Figures 2 and 9 run unchanged at p = 101, 10 fresh seeds each (settings and seeds fixed in `PLAN.md` before any run) | 30 | Figures 2 and 9 (p = 101), Appendices E.4 and F.3 |
| `campaign_H/` | Theorem 4.1's configuration at p = 101 (one hidden layer, width 512) from an iid initialization: four head steps and two controls (arms, seeds and endpoints fixed in `PLAN.md` before any campaign run) | 32 | Figure 7, Appendix E.6 |
| `campaign_I/` | the p = 31 runs of Figures 2 and 3 at depths 1, 2 and 4, repeated with a bias in every hidden layer (plan fixed in `PLAN.md` before any run) | 60 (depth 4: 40 chains) | dotted medians in Figures 2 and 3, Appendix F.5 |
| `campaign_J/` | one hidden layer, campaign I's 10 seeds: three other ways of updating the bias (plan and measures fixed in `PLAN.md` before any run) | 30 | Figure 10, Appendix F.5 |
| `campaign_E/` | transformer pilot at p = 31 (rule fixed in `PLAN.md`; the fresh seeds ran only if a development setting passed, and none did) | 24 | not in the paper |

Campaign C's chains in detail:
- 40 recipe runs per depth (C1 + C2);
- the paired fixed-rate arm: the 20 C2 pilots per depth, each run to the horizon at r0;
- depth 4 at p = 23 (C3, 20);
- depth 4 at width 512 (C4, 4);
- depth 3 with normalized-gradient hidden updates (C5, 20).

The post-hoc window and its counts are in `../posthoc_window/`. The figures are drawn by `../paper_figures/`.

## Requirements

- Python ≥ 3.10.
- numpy: the saved runs used 2.4.6.
- matplotlib, for the figure scripts only.
- CPU only: full-batch float64, with no GPU.

## The learner

`campaign_C/natplat_campaign.py` is the learner; every campaign uses it or a byte-identical extension of it.
- **Model.** A bias-free ReLU MLP on (a + b) mod p. The first layer is initialized at scale c = 24 and deeper layers at 8× He.
- **Hidden-layer steps.** Support-Polar steps divided by √rank, with decoupled decay.
- **Head step.** A normalized step with decoupled decay λ_V = 0.2.
- **Observers.** A_H per layer, the train-only ridge refit, class-centered AGOP with random-frame controls, and margins.
  Observers read the state and never enter the update.

The campaign files:
- `campaign_D/natplat_campaign.py`, `campaign_F/natplat_campaign.py` and `campaign_G/natplat_campaign.py` are the same file.
  They add one optional key, `stop_generalized`. A job without it reproduces campaign C's runner bit for bit.
- Without any of the campaign keys, the runner reproduces the development runner `development/natplat_server.py` bit for bit.
- The early stop ends a run once held-out accuracy has stayed ≥ 0.99 from step s through max(1.5·s, s + 10,000). It only
  shortens runs that have long finished grokking.

**Campaign H's runner** is `campaign_H/t41_runner.py`, a separate one-hidden-layer learner for Theorem 4.1's configuration:
- **Data.** The quadratic-residue split of the theorem (the pairs with b − a a nonzero quadratic residue train, n = p(p − 1)/2).
- **Steps.** The same Polar hidden step and normalized head step, with fixed rates and λ_V = 0.
- **Observers.** A_H on the full grid, and two train-only ridge refits: the canonical τ = 10⁻³ n and Theorem 4.1's
  τ = ‖H_train‖_F²/(32n).
- **Tests.** `campaign_H/tests/test_t41.py` checks the split, the gradients, one optimizer step against an SVD, the observers
  and resume.

**Campaigns I and J** use `campaign_I/bias_runner.py` and its extension `campaign_J/bias_runner.py`. Without their new keys
they compute what the campaign runner of campaigns D, F and G computes. With `bias: true`, every hidden layer gets a bias
b^(ℓ) that starts at 0 and is one more column of W^(ℓ), so [W^(ℓ) | b^(ℓ)] takes the supported Polar step and the decay; the
head has no bias. Campaign J adds `bias_update` and `bias_after_knee`. Their READMEs give the tests and the commands.

## Quick checks (minutes, no training)

```bash
python3 -B campaign_C/recipe.py                     # "recipe reproduces T23"
python3 -B campaign_C/run_campaign_local.py --list  # 164 chains and their estimated cost
python3 -B run_local.py list D                      # campaign D's jobs (also E, F, G, H)
```

**Runner tests** (tiny runs, about 2 minutes), including the bit-exact match with the development runner
`code/natplat_server.py`:

```bash
for c in C D F G; do python3 -B campaign_$c/tests/test_runner.py; done
```

**Recomputing the campaigns' reports from the saved runs** (`../DATA.md`; set `PAPER_DATA`):

```bash
python3 -B tools/make_data_bundle.py --campaign-tars $PAPER_DATA/deep/campaign_C --out data
python3 -B campaign_C/analyze.py report --from data/campaign_C --out /tmp/C.md --json /tmp/C.json
python3 -B tools/compare_analysis.py /tmp/C.json campaign_C/results/analysis.json      # every number in Section 5.2
mkdir -p runs_D && for t in $PAPER_DATA/deep/campaign_D/*.tar; do tar -xf "$t" -C runs_D; done
python3 -B campaign_D/analyze.py report --from runs_D --out /tmp/D.md --json /tmp/D.json
python3 -B tools/compare_analysis.py /tmp/D.json campaign_D/results/analysis.json
python3 -B campaign_F/analyze.py report --from $PAPER_DATA/deep/campaign_F --prior-d $PAPER_DATA/deep/campaign_D \
  --prior-dev $PAPER_DATA/deep/development --out /tmp/F.md --json /tmp/F.json
python3 -B tools/compare_analysis.py /tmp/F.json campaign_F/results/analysis.json
python3 -B campaign_G/analyze.py report --from $PAPER_DATA/deep/campaign_G --out /tmp/G.md --json /tmp/G.json
cmp /tmp/G.json campaign_G/results/analysis.json     # p = 101, next to campaign F's p = 97 settings
python3 -B campaign_H/analyze.py report --from $PAPER_DATA/deep/campaign_H --out /tmp/H.md --json /tmp/H.json
cmp /tmp/H.json campaign_H/results/analysis.json     # Appendix E.6
python3 -I campaign_J/analyze_J.py report --out /tmp/J --from $PAPER_DATA/deep/campaign_J/campaignJ.tar \
  $PAPER_DATA/deep/campaign_I/campaignI_H.tar $PAPER_DATA/deep/campaign_D/campaignD_H.tar $PAPER_DATA/deep/campaign_F/campaignF_H.tar
cmp /tmp/J/analysis.json campaign_J/results/analysis.json     # Figure 10
```

Checked on 2026-10-05 with numpy 2.2.6. Every count and verdict agrees with the committed reports. Some floats differ in
their last bits, because the saved runs were analyzed with numpy 2.4.6:

| Campaign | Floats differing in the last bits | Largest relative difference |
|---|---|---|
| C | 163 of 8,846 | 8e-16 |
| D | 52 of 4,626 | 5e-15 |
| E | 4 of 1,272 | 2e-16 |

Campaign F's report rebuilds byte for byte.

## Rerunning

**Campaign C** (pilot → recipe → main run):

```bash
cd campaign_C
python3 -B run_campaign_local.py --chain C2_D3_s001 --out runs_C            # one chain
python3 -B run_campaign_local.py --stage C2 --workers 8 --out runs_C         # one stage, eight chains at a time
python3 -B run_campaign_local.py --cohort D4w512 --threads 2 --out runs_C    # the width-512 cohort ran with 2 BLAS threads
python3 -B analyze.py report --from runs_C --out ANALYSIS_rerun.md --json analysis_rerun.json
```

- **Design.**
  - A chain's pilot runs at the cohort's fixed rate r0 and measures t10, the first logged step with held-out accuracy ≥ 0.1.
  - The main run uses the rate from `recipe.py` (r·K1·t10·exp(0.9(r − r0)) = x*), with the same initialization and split.
  - The rate therefore uses pilot held-out accuracy, as the paper states.
  - `PREREGISTRATION.md` and `HASHES.json` were written and hashed before any campaign run.
- **Cost.**
  - Each chain takes 0.4–4 hours on one core; the width-512 chains took 6–7 hours on two.
  - The saved chains record 341 hours of single-chain wall time, about 367 core-hours.

**Campaigns D–H** (one job per manifest entry):

```bash
python3 -B run_local.py run D --id H_p23_lw0.0050_s1 --out runs_D
python3 -B run_local.py run F --setting D3_p47_r0.45 --out runs_F --workers 4
python3 -B run_local.py run G --setting H_p101_lw0.02 --out runs_G --workers 4
python3 -B run_local.py run H --setting ev1e-3 --out runs_H --workers 4     # campaign H: --setting is the arm
```

`run_local.py` writes the layout that the campaigns' `analyze.py report --from` reads. Recorded costs:
- campaign D: about 270 core-hours (the manifest's estimate);
- campaign F: 398 core-hours measured, on eight 4-core cloud machines;
- campaign G: 294 core-hours measured (wall time × threads), on eight 4-core cloud machines;
- campaign H: 49 core-hours measured (one thread per run).

**Campaigns I and J** have their own drivers, `campaign_I/run_local.py` (single runs and depth-4 chains) and
`campaign_J/run_local.py`; `campaign_I/README.md` and `campaign_J/README.md` give the commands. Recorded costs: about 91 and
4.4 core-hours.

**Development runs.**
- Every development run file stores its full configuration (`cfg`).
- `python3 -B development/natplat_server.py run development/tasks/T21_depth5_dial.json OUT --workers 4` reruns every run of
  one task.
- Earlier runs used `development/nat.py`, which has the same numerics:
  `python3 -B development/nat.py '<cfg json>' OUT.json`.

## How exact a rerun is

The eigendecomposition and rank cutoff in the Polar step amplify last-bit differences. A different numpy or LAPACK build can
therefore make a rerun drift from the saved trajectory within tens of updates.

`campaign_C/run_campaign_local.py --replay <chain> --data <data>/campaign_C --steps 400` reruns the start of a saved pilot.
It then counts how many training-loss values match the saved ones to 7 significant digits.
- With numpy 2.4.4 on x86-64, every value matched: 401 of 401 for a depth-3 Polar chain, and 3001 of 3001 for a
  normalized-gradient chain.
- With numpy 2.2.6 the same Polar chain matched through update 28 and then drifted.

Such runs are reruns, not bitwise replays. Bitwise agreement needs numpy 2.4 on x86-64.

## What is not included

- **The distributed scheduler.** It assigned jobs to cloud sessions and collected their results. It changed no computed
  number. `campaign_C/run_campaign_local.py` and `run_local.py` reproduce its job logic.
- **Later exploratory sweeps** that the paper does not report.
