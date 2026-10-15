# Campaign G: the p = 97 settings at p = 101 (plan fixed before any run)

Written by the authors on 2026-10-05, before any run of this campaign.

## Purpose

The paper's theorems now cover the primes p ≥ 101 (Section 4). Its across-p experiments stop at p = 97, just below that range.
This campaign runs the p = 97 settings that the paper plots unchanged at p = 101, with 10 fresh seeds each, so that the figures
can show p = 101 in place of p = 97. There are three such settings: one hidden layer, depth 3 and depth 4.

## The settings (locked here)

They are campaign F's p = 97 settings. Campaign D chose each as the best setting at p = 97 on its development seeds. Nothing is
tuned at p = 101.

| family | p | width | setting | seeds | runs | horizon | early stop | threads | configuration from |
|---|---|---|---|---|---|---|---|---|---|
| H: one hidden layer | 101 | 512 | λ_W 0.02 | 1–10 | 10 | 50k | stop_generalized | 1 | `H_p97_lw0.0200_s1` |
| D3: depth 3 | 101 | 256 | last-layer r 0.45 | 1–10 | 10 | 100k | stop_generalized | 4 | `D3_p97_r0.45_s1` |
| D4: depth 4 | 101 | 256 | last-layer r 0.45 | 1–10 | 10 | 100k | stop_generalized | 4 | `D4_p97_r0.45_s1` |

That is 30 runs. One p = 97 setting is not included: width 2048 (campaign F's HW setting), which would cost about 120
core-hours at p = 101. The restructured appendix no longer plots it; if it is wanted, it goes in a separate, later manifest.

## How each run is made

- **Configuration.**
  - Each run starts from the seed-1 configuration of the p = 97 setting in `campaign_D/manifest.json`, and only `p`,
    `init_seed` and `split_seed` change.
  - Everything else is the p = 97 run's: rates, decays, initialization scales, horizon, observer cadence (light every 20
    updates, heavy every 1000) and the early-stop rule.
- **Seeds.** init 5000 + i and split 6000 + i, for i = 1..10, the rule of campaigns D and F. No seed has run at p = 101 before.
- **Code.**
  - The runner `natplat_campaign.py` and the per-run definitions `endpoints.py` are byte-identical to campaign F's (which are
    campaign D's).
  - `worker.py` is campaign F's worker with the campaign name changed.
- **Run ids.** `H_p101_lw0.0200_s1`…`s10`, `D3_p101_r0.45_s1`…`s10` and `D4_p101_r0.45_s1`…`s10`.

## Per-run endpoints (fixed here)

1. **Campaign C's per-run endpoint** (`endpoints.py`, unchanged), the paper's primary endpoint for these runs. A run passes when
   all four hold:
   - the loss-only band holds at least 75% of the held-out 10% → 90% rise;
   - the final held-out accuracy is at least 0.99;
   - the last hidden layer's A_H rises by at least 0.02 across the band;
   - every logged training margin in the band is positive.
2. **The paper's post-hoc window** (Appendix F, `code/posthoc_window/`), unchanged. A run passes when the held-out or the
   refit accuracy gains at least 0.2 inside the loss-only window. The paper's code computes it after collection.

## Report

- **Per setting.**
  - Passes k/10 under each rule, with a Wilson 95% interval.
  - The medians of the share, the CE level, t10, t90 and the final held-out accuracy.
  - Every run's fields go in `analysis.json`.
- **Comparison.** Each setting is reported next to the same setting at p = 97 (campaign F, all 10 seeds).
- **In the paper.**
  - The p = 101 runs replace the p = 97 runs in Figure 2 and in the appendix's across-p figures, and the counts in the text
    follow.
  - The p = 97 runs stay in the record (the archives of campaigns D and F) as the origin of the settings.

**Commitments:**
- No setting is re-tuned after these runs, and no other setting is run under this plan.
- No run is dropped or repeated. The exception: a run that stops with an error is rerun with the same seed, as the worker does.

## Compute

About 204 core-hours. This is the measured duration of the p = 97 seeds times the measured cost per update at p = 101 relative
to p = 97: 1.135 for one hidden layer and 1.105 for depth 3 (`CHECKS.md`). On 8 cloud sessions × 4 cores this is about
6–7 hours; each session starts with its 4-thread runs. Runs that do not stop early take longer.
