# Code for "Flat Loss, Evolving Features: Grokking in Modular Addition"

This folder holds the code, configurations, seeds, locks and committed analysis outputs of every experiment in version 2 of
the paper on arXiv. It also lists every experiment run for the paper up to version 2, including the development sweeps and
failed arms; three later campaigns, which version 2 does not report, are named in `campaigns/README.md`. The saved runs
themselves are not included in this release; `DATA.md` lists each dataset, its size and hashes, and how to
regenerate it, except the earlier quadratic-activation runs (L4), whose original learner was not preserved
(`onehl_earlier/README.md`).

Version 1 of the paper reported networks with one hidden layer and deeper MLPs. `campaigns/` holds the experiments run after
it (about 1,180 runs): deeper networks across primes and training fractions, one- and two-block transformers, the
runs that complete two of version 1's figures (depth 2 on 40 seeds, and four more seeds of Theorem 4.1's configuration), and
the readout studies, with their code, plans, tests and analysis outputs, and `campaigns/run_local.py`, which reruns any of
their runs on one machine (`campaigns/README.md`). `paper_figures/arxiv_v2/` draws version 2's figures and recomputes the
numbers of its new text.

Figure, table, section and appendix numbers in this README refer to version 2: Figures 1–5 in the main text; Appendix A
(representation diagnostics, windows and endpoints); Appendix B (one hidden layer: Table 1, Figures 6–8); Appendix C (deeper
networks: Figure 9, Tables 2 and 3, Figures 10–13); Appendix D (transformers: Tables 4–8, Figures 14–17); the proofs in
Appendices E–G. The READMEs inside the folders were written along the way and use the numbers of their time, mostly version
1's; `paper_figures/arxiv_v2/README.md` maps them to version 2's.

## Quick start

**Requirements.**
- Python 3.10 or newer, with numpy and matplotlib. The 1HL runner also needs scipy, the transformer campaigns PyTorch (the
  CPU build is enough), the centring of Figures 1–5 pymupdf, and the tests pytest (`requirements.txt`).
- CPU only. Training is full batch, in float64 (some transformer development runs, campaigns T to V, used float32).

**Redraw the paper's figures from the saved runs.** The runs are not included in this release; the last column of
`DATA.md`'s table names the code that regenerates each dataset, and `campaigns/run_local.py` regenerates the campaigns' runs.
With the runs in place, no training is needed: `paper_figures/arxiv_v2/README.md` gives the command for every figure of
version 2, with the expected statistics and file hashes. For example:

```bash
export PAPER_DATA=/path/to/data            # the layout of DATA.md
cd paper_figures/arxiv_v2
python3 -B fig7_v20.py .. figs 3.7          # Figure 7
python3 -I paper_v4_figs.py ../../campaigns SHARED ANALYSIS_Q.json figs    # Figures 4 and 10-12 (SHARED: the campaigns' runs)
```

**Recompute the reported counts and tables.**
- Deep campaigns: `deep/README.md`, "Quick checks".
- Post-hoc window: `posthoc_window/README.md`.
- 1HL endpoints and diagnostics: `onehl/README.md`, steps 6 and 7.
- The campaigns after version 1: each campaign's analysis script (`campaigns/README.md`, "Analyses and figures"); the numbers
  of version 2's new text: `paper_figures/arxiv_v2/README.md`.

**Retrain any run.** Each experiment folder's README gives the commands; `campaigns/run_local.py` runs the campaigns after
version 1. Training writes the same per-run files the analysis reads, so a rerun can replace any dataset in `DATA.md`, except
the earlier quadratic-activation runs (L4), whose original learner was not preserved. The deep figures read the deep runs'
windows from `posthoc_window/results/`, and the post-hoc rescoring reads campaigns C–F's endpoint fields from
`deep/campaign_*/results/analysis.json`; neither recomputes them from a rerun.

## Layout

| Folder | What it holds |
|---|---|
| `onehl/` | **1HL learner and protocol.** The runner, model and observers of the 1HL experiments. Also: the headline recipe search, lock and confirmation; the pre-registered p = 23 tests; the per-prime development sweep and confirmations; the observers recomputed for the fixed-recipe sweep; the replays; the compact extracts; the diagnostics. |
| `onehl_earlier/` | **Earlier 1HL runs.** Training code of the two earlier cohorts that the paper re-analyzes: the fixed-recipe sweep (L2) and the quadratic-activation contrast (L4). |
| `deep/` | **Deep campaigns.** The deep learner, the development runs, the pre-registered campaign C, the across-prime campaigns D and F, the transformer pilot E, the p = 101 campaign G, campaign H (Theorem 4.1's configuration), campaigns I and J (hidden biases), and a local job runner. |
| `posthoc_window/` | **Post-hoc window.** The rule, the rescoring of every deep run, and the committed per-run windows and verdicts. |
| `paper_figures/` | **Figures.** The renderers of the paper's figures, `_paths.py` (where code and data are), and the expected statistics and hashes; `bias/` draws the with-bias medians of Figures 2 and 3, and Figure 13; `arxiv_v2/` holds the drivers of version 2's figures and of the numbers of its new text, with the map of its figures and tables. |
| `data_manifests/` | **Data hashes.** The SHA-256 of every file of every dataset in `DATA.md`. |
| `campaigns/` | **After version 1.** Sixteen training campaigns and four readout studies run after version 1 of the paper, and `run_local.py`, which reruns their runs (`campaigns/README.md`). |
| `tools/` | **Release.** `build_release.py`, which scans this folder for local paths, e-mail and web addresses and credentials, and copies it with its own `SHA256SUMS`. |
| `PROVENANCE.tsv` | **File origins.** Each file's origin, its SHA-256 there and here, and whether it is verbatim, anonymized, adapted or new. |
| `SHA256SUMS` | **File hashes.** Every file in this folder. |

## Experiments run for the paper

Status:
- **locked**: the configuration, seeds and scoring rule were fixed in a hashed record before the runs.
- **pre-registered**: a locked test with a declared success criterion.
- **development**: selection runs, which use held-out accuracy where stated.
- **earlier**: runs from before this protocol that the paper re-analyzes.

### One hidden layer (bias-free ReLU, width 512, polar feature steps, normalized head steps, decoupled decay)

| Experiment | Setting | Runs | Status | Paper | Code |
|---|---|---|---|---|---|
| Headline search, dev1 | p = 17, a grid of recipes | 96 | development | App. B.1 (no qualifying recipe) | `onehl/headline/dev1` |
| Headline search, dev2 | p = 23, 31; training fraction 0.5 and 0.8; λ_W 0.01–0.02 | 48 | development, rule declared before launch | App. B.1: chose the headline recipe and window [4500, 12000] | `onehl/headline/dev2` |
| Headline search, dev3 | p = 31: no head decay, and a quadratic-residue split | 44 | development | App. B.5 (none met the rule) | `onehl/headline/dev3` |
| **Headline (H)** | p = 31, c = 24, η_W = η_V = 0.03, λ_W = 0.015, λ_V = 0.2; 40k updates; window [4500, 12000] | 20 fresh pairs | locked | Sec. 5.1, Fig. 1; App. B.1, B.2; Table 1 | `onehl/headline/confirm` |
| p = 23 cohorts A and B | headline hyperparameters at p = 23; 20k updates; window [6500, 16500] | 20 + 20 | pre-registered (recovery to 90% and to 50%) | Figs. 6, 7(a); App. B.3 | `onehl/p23_tests` |
| Per-prime development (L1 dev) | 18 recipes × 3 pairs at p = 11, 13, 17, 23, 29, 61 | 54 | development | App. B.4 | `onehl/per_prime_development` |
| **Per-prime confirmations (L1)** | p = 17, 23, 29: (c, η_V) = (24, 0.01), (48, 0.03), (96, 0.03); λ_W = 0.005; 50k updates | 3 × 20 fresh pairs | locked | Fig. 7(b–d); App. B.4; Table 1 | `onehl/per_prime` |
| Fixed-recipe sweep (L2) | one recipe at p = 11, 13, 17, 19, 23, 29; 40k updates; window [15000, 20000] | 6 × 20 | earlier; observers recomputed from saved states | App. B.4 (counts in the text), C.5 | `onehl_earlier/fixed_recipe_sweep`, `onehl/fixed_recipe` |
| Replays (L3) | p = 23 (λ_W = 0.01) and p = 31 (λ_W = 0.015); 30k updates; states saved | 2 × 5 | earlier runs replayed; windows defined from held-out accuracy (descriptive) | App. B.5; Table 1 | `onehl/replays` |
| Quadratic contrast (L4) | square activation, p = 13, width 64, row-sphere features | 40 | earlier, locked | App. B.5; Table 1 | `onehl_earlier/quadratic_contrast` |

### Theorem 4.1's configuration (one hidden layer)

| Experiment | Setting | Runs | Status | Paper | Code |
|---|---|---|---|---|---|
| **Campaign H** | p = 101, quadratic-residue split, width 512, iid initialization (c = 24); λ_W = 1, η_W = 1 − (9/10)^(1/32), λ_V = 0; Theorem 4.1's refit τ = R_t/(32n); 40k updates. Head step η_V = 10⁻³ (primary, 10 seeds), 3·10⁻⁴, 3·10⁻³, 10⁻² (6 each), and two 2-seed controls | 32 | arms, seeds and endpoints fixed after a one-seed pilot and before the runs | App. B.6, Fig. 8 | `deep/campaign_H` |
| Campaign QH | the η_V = 3·10⁻⁴ arm on seeds 7–10, so that both rows of Fig. 8 show 10 runs | 4 | campaign H's runner, configuration and endpoints | App. B.6, Fig. 8 | `campaigns/campaign_QH` |

### Deeper networks (bias-free ReLU MLP, support-Polar hidden steps, normalized head step)

| Experiment | Setting | Runs | Status | Paper | Code |
|---|---|---|---|---|---|
| Development runs | depths 1–5, p = 23–47, widths 256 and 512; 34 tasks | 199 | development | App. C.1; the rate recipe | `deep/development` |
| **Campaign C** | p = 31, width 256, depths 3–5. Rate recipe from a pilot's t10. Arms: paired fixed rate; p = 23; width 512; normalized-gradient hidden steps | 164 chains (224 scored runs) | pre-registered (each depth needs a Wilson lower bound ≥ 0.80) | Sec. 5.2, Figs. 3 (depths 3–5), 9; Table 2; App. C.2 | `deep/campaign_C` |
| Campaign QD | depth 2 at p = 31 on the 40 seeds of depths 3–5, without and with a bias in every hidden layer | 40 + 40 | plan and analysis fixed before the runs | Fig. 3 (depth 2); App. C.3, C.6 | `campaigns/campaign_QD` |
| Campaign D | across p = 23, 31, 47, 61, 97. 1HL λ_W dial; widths 1024 and 2048 at p = 97; depth-3 and depth-4 last-layer rate dials | 127 | development | chose the settings of Fig. 2 and App. C.3; App. B.4, C.3 | `deep/campaign_D` |
| Campaign F | the 17 settings of Fig. 2, of depth 2 at p = 31, of depths 3 and 4 across p, and of depth 2 and width 2048 across p, brought to 10 seeds | 137 new (170 in all) | settings fixed before the runs | Fig. 2 (p ≤ 61); App. B.4, C.3, C.5 | `deep/campaign_F` |
| Campaign G | the three p = 97 settings (one hidden layer, λ_W = 0.02; depths 3 and 4, r = 0.45) run unchanged at p = 101 | 30 (10 per setting) | settings and seeds fixed before the runs | Fig. 2 (p = 101); App. B.4, C.3 | `deep/campaign_G` |
| Campaign E | decoder-only transformer, 1 or 2 blocks, p = 31, three optimizers | 24 (60 fresh runs skipped by its rule) | development pilot | not in the paper | `deep/campaign_E` |

### Deeper networks across training fractions

| Experiment | Setting | Runs | Status | Paper | Code |
|---|---|---|---|---|---|
| Campaigns K and L | depths 3 and 4 at p = 23, 47, 61, 101, head decay scaled with p; training fractions 0.8 (K) and 0.5 (L) | 96 + 120 | development, before campaign M | not reported | `campaigns/campaign_K`, `campaigns/campaign_L` |
| Campaign M | depths 3 and 4 at p = 47 and 61, fractions 0.8 to 0.5, the last hidden layer's rate factor r re-tuned | 216 | development | App. C.4: chose r for each setting, by a fixed order of criteria | `campaigns/campaign_M` |
| **Campaign N** | depths 3 and 4 at p = 47 and 61, fractions 0.8, 0.7, 0.6 and 0.5, width 256, λ_V = 0.2·31/p; ten fresh seeds per setting | 160 | criterion, rates and analysis fixed before the runs | Sec. 5.2, Fig. 4; App. C.4, Table 3, Figs. 10–12 | `campaigns/campaign_N` |

### Hidden biases (one hidden layer and depths 2 and 4 at p = 31; a preliminary experiment)

| Experiment | Setting | Runs | Status | Paper | Code |
|---|---|---|---|---|---|
| **Campaign I** | the p = 31 runs of version 1's Figures 2 and 3 at depths 1, 2 and 4, each repeated with a bias in every hidden layer ([W \| b] takes the Polar step and the decay; b starts at 0) | 60 (depth 4: 40 chains) | plan, seeds and code fixed before the runs | dotted medians of Fig. 2 (p = 31) and Fig. 3 (depth 4); App. C.6 | `deep/campaign_I` |
| **Campaign J** | one hidden layer, campaign I's 10 seeds: the bias frozen after the knee (K), only decaying after it (KD), or updated by its own normalized step (S) | 30 | plan, measures and code fixed before the runs | Fig. 13; App. C.6 | `deep/campaign_J` |

Version 2 draws the dotted median of Fig. 3's depth-2 column from campaign QD's 40 runs with biases (above).

### Transformers (one and two blocks; App. D)

| Experiment | Setting | Runs | Status | Paper | Code |
|---|---|---|---|---|---|
| Campaigns T, U, V and W | one and two blocks at p = 113: Polar steps, normalized-gradient steps and AdamW; an RMS division before the unembedding; float32 and float64 | 80 + 5 + 71 + 50 + 42 | development | the earlier runs on which the model and settings of App. D.1 were chosen | `campaigns/campaign_{T,T_rerun,U,V,W}` |
| Campaign X | one block at p = 113, fraction 0.2, without the gain, ten fresh seeds (and a second setting); the Fourier readouts | 20 | scored after the runs (App. D.2) | Table 4 (no gain) | `campaigns/campaign_X` |
| **Campaign Y** | campaign X's setting with a gain on the head input relaxing from 2 or 3 to 1 | 20 | rule fixed before the runs | Sec. 5.3, Figs. 5, 14 and 17 (p = 113); Tables 4, 5, 7 and 8 | `campaigns/campaign_Y` |
| Campaign Z | normalized-gradient steps and AdamW with the gain; the gain's recipe; other primes; two blocks | 143 | development, thresholds fixed before the runs (App. D.2) | Tables 4–7; App. D.3; Figs. 5 and 15 (two blocks, seeds 11–15) | `campaigns/campaign_Z` |
| Campaign ZP | one block at p = 61, 89, 149, the training fraction set by a rule fixed before the runs | 30 | development (selection seeds 11–15) | Tables 7 and 8; Figs. 5, 14, 16 and 17 (seeds 11–15) | `campaigns/campaign_ZP` |
| Campaign ZM | p = 61 with a mirror-free split (both orders of a pair on the same side) | 7 | prediction fixed before the runs | Tables 7 and 8; Figs. 14, 16 and 17 (seeds 11–15) | `campaigns/campaign_ZM` |
| **Campaign Q** | seeds 16–20 of the one-block cells (p = 61 with both splits, 89, 149) and of two blocks at p = 113; one rate changed at update 12k (seeds 11–13); fractions 0.175 and 0.225 | 40 | analysis fixed before the runs; the rate changes exploratory | Sec. 5.3, Figs. 5 and 14–17 (seeds 16–20); Tables 6–8; App. D.3–D.5 | `campaigns/campaign_Q` |

### Readout studies (replays and run records; no new training)

| Study | Paper | Code |
|---|---|---|
| Train-only refits restricted to the top Fourier frequencies | the Fourier-filtered refit of App. D.1 | `campaigns/readouts/fourier_refit` |
| Why the ridge refit trails the transformer and leads the MLP; how many frequencies to keep | Sec. 5.3, App. D.5 | `campaigns/readouts/refit_lag`, `paper_figures/arxiv_v2/readout_freqs.py` |
| A_H of single-frequency quadratic and ReLU neurons | Sec. 2 (A_H = 2/5) | `campaigns/readouts/margin_vs_AH` |
| Why A_H is higher in the transformer than in the MLP | not reported | `campaigns/readouts/ah_gap` |

### Analyses of saved runs (no new training)

| Analysis | Paper | Code |
|---|---|---|
| Post-hoc window and pass counts for campaigns C–F (520 runs), campaign G (30 runs) and the 1HL cohorts | Fig. 1 (dashed line), grey bands of Figs. 2, 3 and 7; Sec. 5; App. A.3, C.5 | `posthoc_window`, `paper_figures/scheme2.py` |
| Window start without a head-norm log (L2–L4) | App. A.3 | `paper_figures/check_starts.py` |
| Headline diagnostics: norms, logit scale, energies, loss-only suffix windows | App. B.2 | `onehl/diagnostics/h_diagnostics.py` |
| Strict first-fit-anchored endpoint | Sec. 5; App. A.3, B.2 | `onehl/src/endpoints.py`, `onehl/figures/hl1_figs.py` |

## Reproducing each figure and table

`paper_figures/arxiv_v2/README.md` gives each command with its arguments and expected outputs.

| Item | Code | Data |
|---|---|---|
| Fig. 1 | `paper_figures/arxiv_v2/hero_row.py`, then `rev12_center_figures.py` | `onehl/H` |
| Figs. 2, 3 | `paper_figures/arxiv_v2/fig23_v4.py` (with `paper_figures/bias/`), then `rev12_center_figures.py` | `deep/campaign_{C,D,F,G,I}`; campaign QD |
| Fig. 4 | `paper_figures/arxiv_v2/paper_v4_figs.py`, then `rev12_center_figures.py` | campaign N |
| Fig. 5 | `paper_figures/arxiv_v2/paper_v8_figs.py`, then `rev12_center_figures.py` | campaigns Y, Z, ZP and Q |
| Fig. 6 | `paper_figures/render_fig2_v2.py` (`paper_figures/README.md`) | `onehl/p23{A,B}_compact.tar.gz` |
| Fig. 7 | `paper_figures/arxiv_v2/fig7_v20.py` (runs `paper_figures/scheme2.py`) | `onehl/p23B_compact.tar.gz`, `onehl/L1` |
| Fig. 8 | `paper_figures/thm41_config_fig.py --runs` | `deep/campaign_H`; campaign QH |
| Fig. 9 | `paper_figures/campaign_summary_core.py` (`paper_figures/README.md`) | `deep/campaign_C/results/analysis.json` |
| Figs. 10–12 | `paper_figures/arxiv_v2/paper_v4_figs.py` | campaign N |
| Fig. 13 | `paper_figures/bias/bias_appendix_fig.py` (`paper_figures/bias/README.md`) | `deep/campaign_J/results/analysis.json` (rebuilt by `deep/campaign_J/analyze_J.py` from `deep/campaign_{D,F,I,J}`) |
| Figs. 14, 15 | `paper_figures/arxiv_v2/paper_v7_figs.py` | campaigns Y, Z, ZP, ZM and Q |
| Figs. 16, 17 | `campaigns/campaign_Q/zm10/build_zm10.py` | campaigns Y, ZP, ZM and Q |
| Table 1 | `onehl/figures/render_hero.py` (first row) and `onehl/figures/render_all.py` (endpoint tables) | `onehl/*` |
| Table 2 | `deep/campaign_C/analyze.py report` | `deep/campaign_C` |
| Table 3 | `campaigns/campaign_N/analyze_N.py` (`analysis/analysis_N.json`) | campaign N |
| Tables 4, 5 | `campaigns/campaign_Z/analyze_Z.py` (`analysis/analysis_Z.json`) | campaigns X, Y and Z |
| Table 6 | `campaigns/campaign_Q/analyze_Q.py`; `paper_figures/arxiv_v2/rev3_stats.py` | campaigns Z and Q |
| Table 7 | `campaigns/campaign_ZP/analyze_ZP.py`, `campaigns/campaign_Q/analyze_Q.py` | campaigns Y, Z, ZP, ZM and Q |
| Table 8 | `campaigns/campaign_Q/zm10/build_zm10.py` (`readouts_ZM10.json`) | campaigns Y, ZP, ZM and Q |
| App. B.2 numbers (norms, energies, suffix windows) | `onehl/diagnostics/h_diagnostics.py <data>/onehl/H out.json` | `onehl/H` |
| App. B.3 numbers | `onehl/p23_tests/make_p23_fragment.py` | `onehl/p23{A,B}` |
| App. B.6 numbers | `deep/campaign_H/analyze.py report`; `campaigns/campaign_QH/analyze_QH.py`; `paper_figures/thm41_config_fig.py` (statistics file) | `deep/campaign_H`; campaign QH |
| p = 101 numbers (Fig. 2, App. C.3) | `deep/campaign_G/analyze.py report`; `posthoc_window/make_all2.py G` | `deep/campaign_G` |
| Depth 2 on 40 seeds (App. C.3, C.6) | `campaigns/campaign_QD/analyze_QD.py` | campaign QD |
| Post-hoc counts (Sec. 5, App. C.5) | `posthoc_window/make_all2.py`, `tables2.py`; `paper_figures/scheme2.py` (statistics file) | `deep/*`, `onehl/*` |
| Transformer numbers (Sec. 5.3, App. D) | `paper_figures/arxiv_v2/rev3_stats.py`, `rev7_refit_lags.py`, `readout_freqs.py`; `campaigns/readouts/refit_lag` | campaigns Y, Z, ZP, ZM and Q; replays |
| A_H = 2/5 (Sec. 2) | `campaigns/readouts/margin_vs_AH/fourier_notions_check.py` | none |

## How exact a reproduction is

**Version 2.** Checked for this release with Python 3.11.15, numpy 2.4.4, matplotlib 3.10.9, pymupdf 1.28.2, pytest 9.1.1
and torch 2.5.1:
- Every figure of version 2 was redrawn from the saved runs by the commands of `paper_figures/arxiv_v2/README.md`, byte for
  byte except Figure 6 (identical apart from the PDF creation date) and Figure 9 (drawn with matplotlib 3.7.1; its plotted
  values are identical, and a few text labels move by less than 0.4 pt). The figures' statistics and the statistics behind
  version 2's new text rebuilt byte for byte (the centring step's measurements identically, `paper_figures/arxiv_v2/README.md`).
  So did the committed analyses of campaigns N, Q, QD and QH; those of campaigns Z, ZP and ZM rebuilt identically apart from
  the run paths they record.
- The tests of every campaign and of `deep/` pass (`campaigns/README.md`, `deep/README.md`).
- Rerun for 200 updates, a campaign Q run and a campaign QH run matched their records exactly; two campaign QD runs matched
  theirs exactly with numpy 2.5.3, the campaign's build, and drifted after 37 and 52 updates with numpy 2.4.4 (see Training
  below).

**Version 1.** The checks below were made for version 1 and use its figure numbers. Its Figures 1, 2, 3, 4, 5, 7, 8 and 10
are version 2's 1, 2, 3, 6, 7, 8, 9 and 13, and its Figures 6 and 9 are not in version 2. Version 2's Figures 1, 2, 3 and 7
are redrawn by the drivers of `paper_figures/arxiv_v2/`, and its Figure 8 by the same `thm41_config_fig.py` with campaign
QH's four runs added.

Checked with Python 3.10.12, numpy 2.2.6 and matplotlib 3.10.9.

**Figures.** In that check, the figures drawn by `scheme2.py` and their statistics rebuilt byte for byte, and Figures 4 and 8
identically except for the PDF creation date. Figure 1's centered panel titles and its shading of only the locked evaluation
window [4500, 12000], Figure 5's prime-only titles and Figure 8's three-panel layout were drawn after that check and leave
curves, axes and outcomes unchanged; `paper_figures/expected/` holds the hashes of these files. Figure 1's statistics name
the shaded window (`shaded_window`, `shaded_window_kind`); `band_median` and `windows` are post-hoc diagnostics. Byte
equality of rendered files depends on the environment.

**Analyses.**
- The post-hoc window files `posthoc_window/results/*/settings.jsonl` rebuild byte for byte.
- Campaign F's report rebuilds byte for byte.
- Campaigns C, D and E agree in every count and verdict. A few floats differ in their last bits: up to 5·10⁻¹⁵ relative,
  from numpy 2.2 versus the 2.4.6 that analyzed the saved runs.

**Campaigns G and H.** Checked with Python 3.11.15, numpy 2.4.4 and matplotlib 3.10.9.
- `paper_figures/thm41_config_fig.py` draws Figure 7 and its statistics from campaign H's archive. They match
  `paper_figures/expected/` byte for byte, and the PDF is the paper's. `deep/campaign_H/analyze.py report` rebuilds
  `deep/campaign_H/results/analysis.json` byte for byte.
- Figures 2 and 9 end at p = 101. `paper_figures/expected/` holds their statistics and the hashes of the files that
  `scheme2.py` draws for them.
- Campaign G's committed windows (`posthoc_window/results/G/settings.jsonl`) and report (`deep/campaign_G/results/`) were
  computed from its archive with the rules of `posthoc_window/` and `deep/campaign_G/analyze.py`.

**Training.** The Polar step's eigendecomposition and rank cutoff amplify last-bit differences between numpy or LAPACK builds.
A rerun on another build is therefore a rerun, not a bitwise replay of the saved trajectory. With numpy 2.4 on x86-64, saved
deep-campaign trajectories replay exactly; see `deep/README.md`. The campaigns after version 1 ran in cloud sessions with
Python 3.13 and numpy 2.5.3 (2.5.4 in 14 of campaign Z's runs), except campaign ZM (Python 3.11, numpy 2.4.4); each run's
`chain.json`, or campaign ZM's README, records its build.

## Compute

| Experiment | Recorded cost |
|---|---|
| Campaign C | 341 hours of single-chain wall time (about 367 core-hours) on 4-core cloud machines |
| Campaign D | about 270 core-hours (manifest estimate) |
| Campaign F | 398 core-hours |
| Campaign G | 294 core-hours (wall time × threads, from the runs' records), on eight 4-core cloud machines |
| Campaign H | 49 core-hours (one thread per run), on eight 4-core cloud machines |
| Campaign I | about 91 core-hours (one thread per run), on sixteen 4-core cloud machines |
| Campaign J | 4.4 core-hours (one thread per run), on a 2-core cloud machine |
| 1HL per-prime confirmations | 8 shards on one 64-core machine |
| Campaigns after version 1 | planning estimates from the manifests: about 2,100 core-hours for the deep campaigns K to N, 840 for the transformers T to ZM, and 185, 23 and 5 for Q, QD and QH; each run records its own wall time |

The other 1HL experiments take minutes to a few hours per run on one core. Each saved deep run records its own wall time.

## Protocol for adding an experiment

This is the protocol the experiments above followed. New experiments enter the folder the same way.

1. **Fix it first.** Before any run, write the manifest: configurations, seeds or a seeded generator, horizon, logging
   cadence, the window rule and the endpoint. Hash it, and keep the lock file next to the manifest (examples:
   `onehl/headline/confirm/CONFIRM_LOCK.json`, `deep/campaign_C/PREREGISTRATION.md` with `HASHES.json`,
   `deep/campaign_F/PLAN.md`).
2. **Run the committed runner unchanged.** Keep every run, including failures. Do not replace seeds after seeing outcomes.
   Write per-run files that store the full configuration.
3. **Keep the data out of `code/`.** Add the dataset to `DATA.md` (contents, runs, size), its file hashes to
   `data_manifests/`, and its folder to `paper_figures/_paths.py` if a figure reads it.
4. **Add the code.**
   - Copy it here verbatim, under the experiment's folder.
   - Record its origin and hash in `PROVENANCE.tsv`.
   - When a script must read the paths of this folder, change only its path set-up lines and mark it "adapted".
5. **Document it.** Add a row to the experiment tables above, and to the figure and table map if the paper uses it.
6. **Verify.** Rebuild the affected figures and reports, and compare them with `paper_figures/expected/` and the committed
   reports. Then run `python3 -B tools/build_release.py --check`, which scans for local paths, e-mail and web addresses and
   credentials.

## Release

`python3 -B tools/build_release.py --out <folder> --zip` scans every text file, copies this folder and writes the copy's
own `SHA256SUMS`. A hit stops the build. `python3 -B tools/build_release.py --verify` checks every file against
`SHA256SUMS`. A copy delivered through the Claude apps carries Content Credentials (C2PA metadata in a `caBX` chunk) in each
PNG figure that lacked them, so `sha256sum -c SHA256SUMS` reports those PNGs as changed; `--verify` hashes them without that
chunk, which is the only difference. The data archives listed in `DATA.md` and the run records of `campaigns/` are not
included in this release. The scan also covers `campaigns/`; `build_release.py` lists the words those folders use in their
ordinary meaning and the files that keep the cloud-workspace paths they ran with (`campaigns/README.md`).

This release is version 1's code release with campaigns Q, QD and QH, the readout studies `refit_lag` and `margin_vs_AH`,
and `paper_figures/arxiv_v2/` added. Comments and docstrings are removed from every code file (Python, shell,
`.gitignore`, `requirements.txt`), except the module docstrings of the 20 scripts that write their docstring into their
outputs; the code is otherwise unchanged (the same syntax trees), and the checks above give the same results with and
without the removed text. `PROVENANCE.tsv` gives each file's SHA-256 before the removal (`origin_sha256`), which is the hash
that run records, locks and launch records cite (for example `deep/campaign_*/HASHES.json`, `SOURCE.sha256`,
`RUN_SOURCE_LOCK.sha256`); the campaigns' `campaigns/campaign_*/HASHES.json` are recomputed for the files as they are here,
so that the workers' self-tests pass. Of version 1's other files, only `README.md`, `DATA.md`, `campaigns/README.md`,
`tools/build_release.py`, `PROVENANCE.tsv` and `SHA256SUMS` changed.

## Notes

- **Headline selection and evaluation.** Headline selection also used the auxiliary criteria in
  `onehl/headline/dev2/SELECTION_RULE.md`, and evaluation included the endpoint in `onehl/headline/confirm/CONFIRM_LOCK.json`.
  The paper reports a subset of these recorded diagnostics; it does not define a new complete rule. The auxiliary headline
  endpoint, which the paper does not report, passed 16/20, not 20/20. The selection code (without its comments and
  docstrings), rules and recorded selection outputs are included unchanged.
- **Presentation.** The paper uses descriptive study names; the code keeps its internal dataset IDs. Figure 7 titles its
  columns by prime only. Figure 9 shows three panels, without the auxiliary AGOP panel; `paper_figures/campaign_summary_core.py`
  draws it and checks its plotted values against the original renderer. Figures 1–5 are centred on the page by
  `paper_figures/arxiv_v2/rev12_center_figures.py`, which moves bands of each figure and redraws nothing.
- **p = 101.** The last column of Figure 2 is p = 101: campaign G ran the three p = 97 settings unchanged at p = 101, on 10
  fresh seeds each; the p = 97 runs (campaigns D and F) are unchanged. Appendix C.3 reports the depth-3 and depth-4 settings,
  which version 1 drew in its Figure 9. `paper_figures/median_figs.py` (`specs`) and `scheme2.py` read campaign G from the
  data folder's `deep/campaign_G/`, with its windows in `posthoc_window/results/G/settings.jsonl`.
- **Theorem 4.1's configuration (Figure 8, Appendix B.6).** Campaign H trained it at p = 101 from an iid initialization.
  Its runner is `deep/campaign_H/t41_runner.py`, its report `deep/campaign_H/analyze.py`, and its figure
  `paper_figures/thm41_config_fig.py`. Campaign QH ran the η_V = 3·10⁻⁴ arm on four more seeds with the same runner, so that
  both rows of Figure 8 show 10 runs; `campaigns/campaign_QH/analyze_QH.py` scores the ten with campaign H's endpoints.
- **Hidden biases (Figures 2, 3 and 13; Appendix C.6).** The dotted medians are those of runs with a bias in every hidden
  layer: campaign I's in the p = 31 column of Figure 2 and the depth-4 column of Figure 3, and campaign QD's in the depth-2
  column of Figure 3. `paper_figures/arxiv_v2/fig23_v4.py` draws both figures with `paper_figures/bias/bias_figs.py`, which
  draws every bias-free curve, window and label as `scheme2.py` does. Figure 13 shows campaign J: at one hidden layer, three
  other ways of updating the bias; `deep/campaign_J/analyze_J.py` computes its data and
  `paper_figures/bias/bias_appendix_fig.py` draws it. With Python 3.11.15, numpy 2.4.4 and matplotlib 3.10.9, the PDFs rebuild
  byte for byte from the archives, and so does `deep/campaign_J/results/analysis.json`; the runners' tests pass
  (`deep/campaign_I/README.md`, `deep/campaign_J/README.md`).
