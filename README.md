# Flat Loss, Evolving Features: Grokking in Modular Addition

Code for the paper by Akash Kumar and Karthik Sankaranarayanan,
[arXiv:2610.22593](https://arxiv.org/abs/2610.22593) (cs.LG, stat.ML).

Can a network keep learning while its training cross-entropy stays nearly flat at a positive level? The paper shows that it
can. On modular addition, ReLU networks trained with supported-Polar hidden updates, normalized output-layer updates and
weight decay keep 100% training accuracy while their hidden features change and held-out accuracy rises from zero to above
90%, at relatively flat training cross-entropy. Two finite-window theorems prove this coexistence for one-hidden-layer
networks from constructed initializations. Experiments from random initialization show it on a window fixed in advance with
one hidden layer, on post-hoc windows in deeper networks, and in transformers: one block with a slowly decaying output gain
keeps every training pair correct while its training cross-entropy rises and its held-out accuracy approaches one.

This repository holds the code, configurations, seeds, locks, tests and committed analysis outputs of every experiment the
paper reports, and the scripts that draw its figures and tables and compute the numbers in its text. The saved runs are not
included ([Data](#data)). Section, figure, table and appendix numbers are those of the arXiv paper. The theorems
(Section 4, Appendices E to G) need no code; Theorem 4.1's configuration is trained from iid initialization in
`one_hidden_layer/theorem41_iid_init_p101/` (Appendix B.6). The transformers with two blocks (Figs. 5, 15, Table 6) use
the same code as those with one block; [Two blocks](#two-blocks) lists their runs and commands.

## Repository layout

```
one_hidden_layer/                     One hidden layer: Section 5.1, Appendix B
  learner/                            runner (run_campaign.py), model, observers and endpoints of the 1HL experiments
  headline_cohort/                    recipe search (search_dev*, development, 188 runs) and the locked headline cohort
                                      (locked_confirmation, 20 runs at p = 31): Fig. 1, Table 1, App. B.1, B.2, B.5
  p23_preregistered_tests/            pre-registered cohorts A, B at p = 23 (2 x 20 runs), locks: Figs. 6, 7(a), App. B.3
  per_prime_development/              development, 54 runs at p = 11 to 61: App. B.4
  per_prime_confirmations/            locked, 3 x 20 runs at p = 17, 23, 29: Fig. 7(b-d), Table 1, App. B.4
  fixed_recipe_sweep/                 earlier fixed-recipe sweep (120 runs); observers from saved states: App. B.4, C.5
  replays_p23_p31/                    earlier runs replayed with saved states (2 x 5): Table 1, App. B.5
  quadratic_contrast_p13/             earlier locked cohort with square activation (40 runs): Table 1, App. B.5
  theorem41_iid_init_p101/            Theorem 4.1's configuration from iid initialization (32 runs): Fig. 8, App. B.6
  theorem41_iid_init_p101_seeds7to10/ four more seeds of its η_V = 3·10⁻⁴ arm: Fig. 8, App. B.6
  make_extracts/                      writes the per-run compact extracts that the figures read; observers on saved states
deep_mlp/                             Deeper ReLU networks: Section 5.2, Appendix C
  development_depths1to5/             development runs at depths 1 to 5 (199 runs, 34 task files) and their runners: App. C.1
  depths3to5_preregistered_p31/       pre-registered, p = 31, depths 3 to 5 (164 chains): Figs. 3, 9, Table 2, App. C.2
  across_primes_development/          development across p = 23 to 97 (127 runs): App. B.4, C.3
  across_primes_ten_seeds/            17 chosen settings brought to 10 seeds (137 new runs): Fig. 2, App. B.4, C.3, C.5
  across_primes_p101/                 the three p = 97 settings run at p = 101 (30 runs): Fig. 2, App. B.4, C.3
  depth2_p31_forty_seeds/             depth 2 on 40 seeds, without and with hidden biases (80 runs): Fig. 3, App. C.3, C.6
  training_fractions_rate_tuning/     development across training fractions (216 runs): App. C.4
  training_fractions_p47_p61/         depths 3, 4 at p = 47, 61 across fractions (160 runs): Figs. 4, 10-12, Table 3, App. C.4
  hidden_bias_depths1_2_4/            runs repeated with a bias in every hidden layer (60 runs): Figs. 2, 3 (dotted), App. C.6
  hidden_bias_update_rules/           three other bias updates, one hidden layer (30 runs): Fig. 13, App. C.6
  restore_and_compare/                unpacks the depths3to5 archives; compares rebuilt reports with the committed ones
  run_local.py                        reruns across_primes_* and theorem41_iid_init_p101 on one machine
transformer/                          Transformers with one and two blocks: Section 5.3, Appendix D
  one_block_no_gain_p113/             one block at p = 113 without the gain (20 runs, scored after the runs): Table 4
  one_block_gain_p113/                the same with a relaxing output gain (20 runs): Figs. 5, 14, 17, Tables 4, 5, 7, 8
  gain_recipe_contrasts_two_blocks/   other update rules, the gain's recipe, other primes, and the four planned two-block
                                      settings (143 runs, 20 with two blocks): Figs. 5, 15, Tables 4 to 7, App. D.3
  one_block_other_primes/             one block at p = 61, 89, 149 (30 runs): Figs. 5, 14, 16, 17, Tables 7, 8
  one_block_p61_mirror_free/          p = 61 with a mirror-free split (7 runs): Figs. 14, 16, 17, Tables 7, 8
  seeds16to20_two_block_exploration/  seeds 16 to 20 across primes and with two blocks, and two-block runs with one rate
                                      changed at update 12k or another training fraction (40 runs, 20 with two blocks):
                                      Figs. 5, 14 to 17, Tables 6 to 8, App. D.3 to D.5
  readout_fourier_refit/              train-only refits on the top Fourier frequencies, and the replay code: App. D.1
  readout_refit_lag/                  why the ridge refit trails the transformer and leads the MLP: Sec. 5.3, App. D.5
posthoc_window/                       post-hoc window rule, rescoring of every deep run, committed windows (results/):
                                      grey bands of Figs. 2, 3, 7; Section 5; App. A.3, C.5
figures/                              one script per figure, table or group of numbers (fig*, tab*, numbers_*);
                                      lib/: shared helpers; expected/: expected statistics and figure hashes
run_local.py                          reruns runs of folders with a campaign.json (one_block_p61_mirror_free: --configs)
requirements*.txt                     package requirements: minimum, pinned, and for Intel Macs
SHA256SUMS                            SHA-256 of the files in this tree
```

Status: *locked*, configuration, seeds and scoring rule fixed in a hashed record before the runs; *pre-registered*, a locked
test with a declared success criterion; *development*, selection runs; *earlier*, runs from before this protocol that the
paper re-analyzes. Each experiment folder keeps its runner, manifests, tests and analysis as they ran, so runner versions
differ between folders; manifests, run ids and run-record folders keep internal campaign letters (`campaign_Q`).

## Two blocks

There is no separate two-block model: every `tf_runner.py` under `transformer/` builds a transformer with `n_layers`
blocks (1 by default), and the two-block runs set `n_layers: 2` in their manifest entries. The 40 two-block runs are at
p = 113, and their run ids contain `_B_`:

| Folder | Two-block runs | Paper |
|---|---|---|
| `transformer/gain_recipe_contrasts_two_blocks` | the four planned settings, seeds 11 to 15 (20 runs): training fraction 0.2 (λ_V = 0.275, τ = 10,000), 0.25 (λ_V = 0.4, τ = 10,000) and 0.3 (λ_V = 0.275 or 0.5, τ = 5,000) | Figs. 5, 15, App. D.3 |
| `transformer/seeds16to20_two_block_exploration` | seeds 16 to 20 at fraction 0.2 (5 runs); one rate changed at update 12k, seeds 11 to 13 (9 runs); fractions 0.175 and 0.225, seeds 11 to 13 (6 runs) | Figs. 5, 15, Table 6, App. D.3 |

To list or rerun only these runs (SHARED as in [Data](#data); about 330 core-hours in all, by the manifests' estimates):

```bash
python3 run_local.py transformer/gain_recipe_contrasts_two_blocks --match _B_ --list
python3 run_local.py transformer/gain_recipe_contrasts_two_blocks --match _B_ --jobs 4 --out "$SHARED/campaign_Z/collected"
python3 run_local.py transformer/seeds16to20_two_block_exploration --match _B_ --jobs 4 --out "$SHARED/campaign_Q/collected"
```

Figure 5 (`figures/fig05_transformer_one_two_blocks.py`) and Figure 15 (`figures/fig14_fig15_transformer_primes_blocks.py`)
draw these runs, and Table 6 is the `B_*` cells of campaign Q's analysis (`figures/numbers_sec5_3_tab06.py` writes its
rows under `blocks`); [Reproducing the paper](#reproducing-the-paper) has the commands.

## Requirements

CPU only; training is full batch in float64. numpy and matplotlib are used throughout, scipy by the one-hidden-layer
learner, PyTorch by the transformer experiments, the readout studies and the quadratic contrast, pymupdf by the centring
of Figures 1 to 5, and pytest by the tests. Three requirement files:

- `requirements-pinned.txt`: the versions every check in this README was run with, on Python 3.11 (numpy 2.4.4, scipy
  1.17.1, matplotlib 3.10.9, torch 2.5.1, pymupdf 1.28.2, pytest 9.1.1). Use it to rebuild figures byte for byte.
- `requirements-intel-mac.txt`: for Intel Macs, where PyTorch stops at 2.2.2, a build that needs NumPy 1.x (numpy 1.24.3,
  scipy 1.10.1, matplotlib 3.7.1, torch 2.2.2, pymupdf 1.28.2, pytest 7.4.0, on Python 3.11). Every test passes with it;
  rendered figures can differ in bytes from the expected hashes.
- `requirements.txt`: minimum versions, with numpy below 2 on Intel Macs. The tests also pass with newer releases (checked
  with numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2 and torch 2.14.1 on Python 3.12).

On Linux, install the CPU build of PyTorch first (PyPI's Linux build adds about 3 GB of CUDA libraries):

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-pinned.txt
```

On an Intel Mac, where torch 2.5.1 does not exist:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-intel-mac.txt
```

Elsewhere, `pip install -r requirements-pinned.txt` in a fresh environment. Then check that PyTorch and NumPy work together:

```bash
python3 -c "import numpy, torch; torch.from_numpy(numpy.zeros(2)); print('numpy', numpy.__version__, 'torch', torch.__version__, 'ok')"
```

A PyTorch built for NumPy 1.x (as on Intel Macs) next to NumPy 2 fails this check with "Numpy is not available".

## Data

The saved runs are not in this repository. The analysis and figure scripts read them from two folders:

- **PAPER_DATA** (environment variable; default `data/` at the repository root): the one-hidden-layer compact extracts
  `onehl/{H,L1,L1_dev,L2,L3,L4}/` (`<run id>.npz` and `<run id>.json` per run) and `onehl/p23{A,B}_compact.tar.gz` (folders
  `compact/`, `compactB/`); the deep-network archives `deep/campaign_{C,D,F,G,H,I,J}/` (tar files with one folder per run:
  `run.json.gz`, or `pilot.json.gz` and `main.json.gz` for chains, and metadata) with the names that
  `figures/fig02_fig03_prepare_inputs.py` and the commands below use ([Starting from fresh runs](#starting-from-fresh-runs)
  lists them and how to make them); and the development runs `deep/development/`, `deep/development_extra/`.
- **SHARED** (an absolute path, given on the command line): the run records of the transformer experiments and of the later MLP experiments,
  `campaign_<X>/collected/results/<run id>/run.json.gz` with its `chain.json`, and `campaign_ZM/runs/<run id>/run.json.gz`.
  Do not export it: `figures/fig16_fig17_tab08_mirror_readouts.py`, `figures/lib/plot_ZM10.py` and
  `figures/numbers_sec5_3_tab06.py` read an environment variable `SHARED` as the transformer code folder (default
  `transformer/`).

The folder names keep the campaign letters, because the committed analyses record run paths under them:

| Experiment folder | Run records | Regenerate with |
|---|---|---|
| `one_hidden_layer/headline_cohort` | `onehl/H` | `run_campaign.py` on `locked_confirmation/manifests/` (search: `search_dev*/manifests/`; `search_dev3_p31` ran with its `model_qrsplit.py` in place of the learner's `model.py`), then `compact_line1.py` |
| `one_hidden_layer/p23_preregistered_tests` | `onehl/p23A_compact.tar.gz`, `onehl/p23B_compact.tar.gz` | `run_campaign.py` on `manifests/p23_confirm.json`, `manifests/p23_confirmB.json`, then `compact_line1.py` |
| `one_hidden_layer/per_prime_development` | `onehl/L1_dev` | `run_campaign.py` on `anchor_manifest.json`, `candidate_manifest.json`, then `compact_line1.py` |
| `one_hidden_layer/per_prime_confirmations` | `onehl/L1` | `run_campaign.py` on `shards/shard_0.json` to `shard_7.json`, then `compact_line1.py` |
| `one_hidden_layer/fixed_recipe_sweep` | `onehl/L2` | `training/` (the original code; its drivers stop at a check of source hashes, so only its tests run as shipped), then `observers/run_reobserve_c24.sh` and `compact_other.py line2` |
| `one_hidden_layer/replays_p23_p31` | `onehl/L3` | `natplat_states.py jobs_1hl_seed.json OUT`, then `reobserve_states.py` and `compact_other.py line3` |
| `one_hidden_layer/quadratic_contrast_p13` | `onehl/L4` | not exactly: the learner file that trained these runs was revised later, and `src/` holds the revision |
| `one_hidden_layer/theorem41_iid_init_p101` | `deep/campaign_H` | `deep_mlp/run_local.py run H` |
| `one_hidden_layer/theorem41_iid_init_p101_seeds7to10` | `campaign_QH` | `run_local.py` |
| `deep_mlp/development_depths1to5` | `deep/development`, `deep/development_extra` | `natplat_server.py run tasks/<task>.json OUT` |
| `deep_mlp/depths3to5_preregistered_p31` | `deep/campaign_C` | `run_campaign_local.py` (pilot, recipe, main run per chain) |
| `deep_mlp/across_primes_development` | `deep/campaign_D` | `deep_mlp/run_local.py run D` |
| `deep_mlp/across_primes_ten_seeds` | `deep/campaign_F` | `deep_mlp/run_local.py run F` |
| `deep_mlp/across_primes_p101` | `deep/campaign_G` | `deep_mlp/run_local.py run G` |
| `deep_mlp/depth2_p31_forty_seeds` | `campaign_QD` | `run_local.py` |
| `deep_mlp/training_fractions_rate_tuning` | `campaign_M` | `run_local.py` |
| `deep_mlp/training_fractions_p47_p61` | `campaign_N` | `run_local.py` |
| `deep_mlp/hidden_bias_depths1_2_4` | `deep/campaign_I` | its `run_local.py` |
| `deep_mlp/hidden_bias_update_rules` | `deep/campaign_J` | its `run_local.py` |
| `transformer/one_block_no_gain_p113` | `campaign_X` | `run_local.py` |
| `transformer/one_block_gain_p113` | `campaign_Y` | `run_local.py` |
| `transformer/gain_recipe_contrasts_two_blocks` | `campaign_Z` | `run_local.py` |
| `transformer/one_block_other_primes` | `campaign_ZP` | `run_local.py` |
| `transformer/one_block_p61_mirror_free` | `campaign_ZM` (`runs/`, not `collected/results/`) | `run_local.py --configs` |
| `transformer/seeds16to20_two_block_exploration` | `campaign_Q` | `run_local.py` |
| `transformer/readout_fourier_refit`, `readout_refit_lag` | replay snapshots (the paper uses the committed `kcurve/` and `lag_*.json`) | `replay.py KIND CFG OUT T_END EVERY`, `replay2.py RUNNER_DIR CFG OUT T_END EVERY` |

`onehl/...` and `deep/...` are under PAPER_DATA, `campaign_<X>` under SHARED. `run_local.py` is the one at the root;
`run_campaign.py` is in `one_hidden_layer/learner/`, and `compact_line1.py`, `compact_other.py` and `reobserve_states.py` are
in `one_hidden_layer/make_extracts/`. For example, from the repository root:

```bash
python3 run_local.py transformer/one_block_gain_p113 --list        # runs, settings, estimated core-hours
python3 run_local.py transformer/one_block_gain_p113 --all --jobs 8 --out "$SHARED/campaign_Y/collected"
python3 run_local.py deep_mlp/training_fractions_p47_p61 --match D3_p61_f0.8 --jobs 4 --out "$SHARED/campaign_N/collected"
python3 run_local.py transformer/one_block_p61_mirror_free --out "$SHARED/campaign_ZM/collected" \
    --configs transformer/one_block_p61_mirror_free/configs/*.json && ln -s collected/results "$SHARED/campaign_ZM/runs"
python3 deep_mlp/run_local.py run F --setting D3_p47_r0.45 --workers 4 --out runs_F
python3 deep_mlp/run_local.py run H --setting ev1e-3 --workers 4 --out runs_H      # for H, --setting is the arm
python3 deep_mlp/depths3to5_preregistered_p31/run_campaign_local.py --stage C2 --workers 8 --out runs_C
python3 deep_mlp/hidden_bias_update_rules/run_local.py --jobs 2 --out "$PWD/runs_J"   # a relative --out is inside its folder
python3 one_hidden_layer/learner/run_campaign.py --output runs_h/shard_00 \
    --manifest one_hidden_layer/headline_cohort/locked_confirmation/manifests/shard_00.json
python3 one_hidden_layer/make_extracts/compact_line1.py data/onehl/H runs_h/shard_*/h31_*
```

`run_local.py` writes to `<folder>/collected/` and `<folder>/_work/` unless given `--out` and `--work`; a stopped run resumes
from its last checkpoint (every 2,000 updates), finished runs are skipped unless `--force`, and threads per run come from the
manifest or `--threads`. The deep-network drivers write one folder per run: the reports read these folders, while figure
preparation, window rescoring and the data bundle read archives ([Starting from fresh runs](#starting-from-fresh-runs)).
Without any data, the committed outputs
still let every number be checked: each experiment's `results/analysis.json` or `analysis/*.json`,
`posthoc_window/results/*/settings.jsonl`, `figures/expected/`, the p = 23 scores and the recipe selection files.

**Compute.** Recorded: `depths3to5_preregistered_p31` 341 hours of single-chain wall time (about 367 core-hours);
`across_primes_development` about 270 core-hours (estimate), `across_primes_ten_seeds` 398, `across_primes_p101` 294;
`theorem41_iid_init_p101` 49; `hidden_bias_depths1_2_4` about 91, `hidden_bias_update_rules` 4.4; the per-prime
confirmations, 8 shards on one 64-core machine; the other 1HL runs, minutes to a few hours each on one core. Planning
estimates in core-hours from the manifests (`run_local.py --list`): `training_fractions_rate_tuning` 816,
`training_fractions_p47_p61` 253, `one_block_no_gain_p113` 19, `one_block_gain_p113` 15, `gain_recipe_contrasts_two_blocks` 274,
`one_block_other_primes` 25, `seeds16to20_two_block_exploration` 185, `depth2_p31_forty_seeds` 23,
`theorem41_iid_init_p101_seeds7to10` 5. Each run records its own wall time.

## Reproducing the paper

Every command runs from the repository root. First the shared inputs:

```bash
export PAPER_DATA=/abs/path/to/paper_data
SHARED=/abs/path/to/run_records      # not exported (see Data)
mkdir -p out
# campaign Q's analysis with your run paths; the transformer scripts open the runs it names
python3 transformer/seeds16to20_two_block_exploration/analyze_Q.py out/analysis_Q.json "$SHARED"
python3 figures/fig02_fig03_prepare_inputs.py --out out/bi                                     # Figs. 2, 3
mkdir -p out/p23 && for c in A B; do tar -xzf "$PAPER_DATA/onehl/p23${c}_compact.tar.gz" -C out/p23; done    # Fig. 6
mkdir -p out/runs8 && ln -s "$PAPER_DATA"/deep/campaign_H/*.tar out/runs8/ &&                  # Fig. 8
  for d in "$SHARED"/campaign_QH/collected/results/QH_T41_*; do ln -s "$d" "out/runs8/${d##*/QH_}"; done
mkdir -p out/t1 &&                                                                             # Table 1
  for x in l1_confirm:L1 l1_dev:L1_dev l2:L2 l3:L3 l4:L4; do ln -s "$PAPER_DATA/onehl/${x#*:}" "out/t1/${x%%:*}"; done
# Table 2 and App. C.3:
python3 deep_mlp/restore_and_compare/make_data_bundle.py --campaign-tars "$PAPER_DATA/deep/campaign_C" --out out/bundle
mkdir -p out/runs_D && for t in "$PAPER_DATA"/deep/campaign_D/*.tar; do tar -xf "$t" -C out/runs_D; done
```

In the last column a bare file name is in `figures/expected/`, and `a = b` means the command writes `a`, to compare with `b`.

| Item | Command | Inputs | Output = expected |
|---|---|---|---|
| Fig. 1 | `python3 figures/fig01_headline_1hl.py "$PAPER_DATA/onehl/H" out/fig01`, then the centring row | `onehl/H` | `hero_row_stats.json` = `fig01_stats.json` |
| Figs. 2, 3 | `python3 figures/fig02_fig03_primes_and_depths.py figures/lib figures/lib --data out/bi/data --winv2 posthoc_window/results --bias out/bi/bias --dev2 out/bi/dev2 --qd "$SHARED/campaign_QD/collected/results" --out out/fig02_fig03 --height 3.4`, then the centring row | `deep/campaign_{C,D,F,G,I}`, `deep/development`; `campaign_QD` | `bias_figs_stats.json` = `fig02_fig03_stats.json`, `SOURCES.json` = `fig02_fig03_sources.json` |
| Figs. 4, 10-12 | `python3 figures/fig04_fig10_to_fig12_fractions.py . "$SHARED" out/analysis_Q.json out/fig04`; Fig. 4 then the centring row | `campaign_N` (for two figures not in the paper, also `campaign_{Y,Z,ZP,ZM,Q}`) | `paper_v4_figs_stats.json` = `fig04_fig10_to_fig12_stats.json` |
| Fig. 5 | `python3 figures/fig05_transformer_one_two_blocks.py . "$SHARED" out/analysis_Q.json out/fig05`, then the centring row | `campaign_{Y,Z,ZP,Q}` | `paper_v8_figs_stats.json` = `fig05_stats.json` |
| Figs. 1-5, centring | `mkdir -p out/centred && python3 figures/fig01_to_fig05_center_on_page.py out/centred out/fig01/H_hero_1x4.pdf out/fig02_fig03/fig2_acrossp_1hl_bias.pdf out/fig02_fig03/fig3_depths_2to5_bias.pdf out/fig04/N_D3_p47_main.pdf out/fig05/TF_main.pdf > out/centred/centring.json` | the five PDFs | the centred PDFs (`FIGURES.sha256`); the measurements in `centring.json` = `fig01_to_fig05_centring.json`, which adds the print scale |
| Fig. 6 | `python3 figures/fig06_p23_preregistered.py p23 out/p23/compact out/fig06 --stem p23_confirmA` and `python3 figures/fig06_p23_preregistered.py p23 out/p23/compactB out/fig06 --stem p23_confirmB --dot 0.5` (the first `p23` is the figure type) | `onehl/p23{A,B}_compact.tar.gz` | `p23_confirmA_stats.json` = `fig06_cohort_a_stats.json`, `p23_confirmB_stats.json` = `fig06_cohort_b_stats.json` |
| Fig. 7 | `python3 figures/fig07_locked_1hl_cohorts.py figures/lib out/fig07 3.7` | `onehl/L1`, `onehl/p23B_compact.tar.gz` | `scheme2_stats.json` = `fig07_stats.json` |
| Fig. 8 | `python3 figures/fig08_theorem41_iid_p101.py --out out/fig08 --runs out/runs8` | `deep/campaign_H`; `campaign_QH` | `thm41_config_stats.json` = `fig08_stats.json` |
| Fig. 9 | `python3 figures/fig09_deep_campaign_summary.py --out out/fig09` | committed `deep_mlp/depths3to5_preregistered_p31/results/analysis.json` | `receipt.json` checks the plotted values against `figures/lib/fig09_original_renderer.py` |
| Fig. 13 | `python3 figures/fig13_hidden_bias_updates.py --analysis deep_mlp/hidden_bias_update_rules/results/analysis.json --out out/fig13` | committed analysis (rebuilt in the App. C.6 row) | `bias_ablation_1hl_stats.json` = `fig13_stats.json` |
| Figs. 14, 15 | `python3 figures/fig14_fig15_transformer_primes_blocks.py . "$SHARED" out/analysis_Q.json out/fig14_fig15` | `campaign_{Y,Z,ZP,ZM,Q}` | `paper_v7_figs_stats.json` = `fig14_fig15_stats.json` |
| Figs. 16, 17, Table 8 | `python3 figures/fig16_fig17_tab08_mirror_readouts.py out/analysis_Q.json out/fig16_fig17` | `campaign_{Y,ZP,ZM,Q}` | `analysis_ZM10.json` = `fig16_fig17_tab08_analysis.json`, `readouts_ZM10.json` = `tab08_readouts.json`, `ZM_rise_times.json` = `fig16_fig17_rise_times.json` |
| Table 1 | `python3 figures/tab01_headline_endpoints.py "$PAPER_DATA/onehl/H" out/tab01` (first row) and `python3 figures/tab01_cohort_endpoints.py out/t1 out/tab01` | `onehl/{H,L1,L1_dev,L2,L3,L4}` | `H_endpoints.json`, `endpoint_tables.json` (no committed copy) |
| Table 2 | `python3 deep_mlp/depths3to5_preregistered_p31/analyze.py report --from out/bundle/campaign_C --out out/C.md --json out/C.json`, then `python3 deep_mlp/restore_and_compare/compare_analysis.py out/C.json deep_mlp/depths3to5_preregistered_p31/results/analysis.json` | `deep/campaign_C` | `deep_mlp/depths3to5_preregistered_p31/results/analysis.json` |
| Table 3 | `python3 deep_mlp/training_fractions_p47_p61/analyze_N.py "$SHARED/campaign_N/collected/results" none out/analysis_N.json deep_mlp/training_fractions_rate_tuning/analysis/analysis_M.json` | `campaign_N` | `deep_mlp/training_fractions_p47_p61/analysis/analysis_N.json` |
| Tables 4, 5 | `python3 transformer/gain_recipe_contrasts_two_blocks/analyze_Z.py out/analysis_Z.json "$SHARED/campaign_Z/collected/results" "$SHARED/campaign_Y/collected/results" "$SHARED/campaign_X/collected/results"` | `campaign_{X,Y,Z}` | `transformer/gain_recipe_contrasts_two_blocks/analysis/analysis_Z.json` |
| Table 6 | `out/analysis_Q.json` (cells `B_*`); the Sec. 5.3 row prints the rows | `campaign_{Z,Q}` | `transformer/seeds16to20_two_block_exploration/analysis/analysis_Q.json` |
| Table 7 | `python3 transformer/one_block_other_primes/analyze_ZP.py out/analysis_ZP.json "$SHARED/campaign_ZP/collected/results" "$SHARED/campaign_Z/collected/results" "$SHARED/campaign_Y/collected/results"`; rows with 10 runs: `out/analysis_Q.json` | `campaign_{Y,Z,ZP,ZM,Q}` | `transformer/one_block_other_primes/analysis/analysis_ZP.json` |
| Sec. 2 (A_H = 2/5) | `python3 figures/numbers_sec2_ah_two_fifths.py` | none | prints A_H 0.400 (quadratic neuron) and 0.114 (ReLU neuron) |
| Sec. 5.3, App. D.3-D.5 (counts, crossings, flips, mirror pairs), C.3 (depth 2), B.6 | `python3 figures/numbers_sec5_3_tab06.py out/analysis_Q.json out/fig16_fig17/analysis_ZM10.json out/fig16_fig17/readouts_ZM10.json deep_mlp/depth2_p31_forty_seeds/analysis/analysis_QD.json out/fig08/thm41_config_stats.json out/numbers_sec5_3_tab06.json` | outputs of the rows above | `numbers_sec5_3_tab06.json` (its keys use a draft's numbering: `table6` holds Table 7's 10-run rows, `table7` Table 8, and `blocks` Table 6) |
| Sec. 5.3, App. D.5 (refit lags) | `python3 figures/numbers_sec5_3_refit_lags.py transformer/gain_recipe_contrasts_two_blocks/analysis/analysis_Z.json transformer/seeds16to20_two_block_exploration/analysis/analysis_Q.json transformer/one_block_other_primes/analysis/analysis_ZP.json transformer/one_block_p61_mirror_free/analysis/analysis_ZM.json > out/refit_lags.json` | committed analyses | `numbers_sec5_3_refit_lags.json` |
| App. D.5 (head-input frequencies) | `python3 figures/numbers_appd5_head_frequencies.py out/analysis_Q.json out/head_frequencies.json` | `campaign_{Y,Z,ZP,ZM,Q}` | `numbers_appd5_head_frequencies.json` |
| App. D.5 (replays) | `(cd transformer/readout_refit_lag && python3 k_summary.py ../../out/kcurve_summary.json kcurve/{F_H_p31_s10,H_p61_s1,H_p101_s7,Y_G2_s11,Z_B_s11,R5_s1,R4_s3,W_P1_s8,V_E2_s1}.json && python3 plot_cases.py ../../out/cases)` | committed `kcurve/`, `lag_*.json` | `kcurve_summary.json` = `transformer/readout_refit_lag/kcurve/summary.json`, `cases/plot_cases_stats.json` = `transformer/readout_refit_lag/figs/plot_cases_stats.json` |
| Sec. 5, App. A.3, C.5 (post-hoc windows and counts) | `python3 posthoc_window/rescore_deep_runs.py --out out/rescored --no-figs C D F G`, `python3 posthoc_window/settings_tables.py out/rescored C D F G`; the 1HL cohorts: `python3 figures/lib/scheme2.py --out out/scheme2` | `deep/campaign_{C,D,F,G}`; for `scheme2.py` all of PAPER_DATA | `out/rescored/<X>/settings.jsonl` = `posthoc_window/results/<X>/settings.jsonl`; `scheme2_stats.json` = `scheme2_stats.json` |
| App. A.3 (window starts) | `python3 figures/numbers_appa3_window_starts.py --out out/appa3 H L1` and `python3 figures/numbers_appa3_window_starts.py --out out/appa3 L2 L3 L4` | `onehl/{H,L1,L2,L3,L4}` | `check_starts_H_L1.json` = `numbers_appa3_starts_H_L1.json`, `check_starts_L2_L3_L4.json` = `numbers_appa3_starts_L2_L3_L4.json` |
| App. B.1 (recipe selection and lock) | `python3 one_hidden_layer/headline_cohort/select_hero.py DEV2 out/selection_dev2.json` (DEV2: extracts of the 48 runs of `search_dev2_p23_p31`); `python3 one_hidden_layer/headline_cohort/make_confirm.py one_hidden_layer/headline_cohort/search_dev2_p23_p31 out/confirm` | extracts of the dev2 runs | `one_hidden_layer/headline_cohort/search_dev2_p23_p31/out/selection.json`; `out/confirm/manifests/` = `one_hidden_layer/headline_cohort/locked_confirmation/manifests/` |
| App. B.2 (norms, energies, suffix windows) | `python3 figures/numbers_appb2_headline_diagnostics.py "$PAPER_DATA/onehl/H" out/appb2.json` | `onehl/H` | no committed copy |
| App. B.3 (pre-registered p = 23 tests) | `python3 one_hidden_layer/p23_preregistered_tests/score_p23.py out/p23/compact one_hidden_layer/p23_preregistered_tests/CONFIRM_LOCK_p23.json out/score_A.json`, the same with `score_p23b.py`, `out/p23/compactB`, `CONFIRM_LOCK_p23b.json`, `out/score_B.json`, then `python3 one_hidden_layer/p23_preregistered_tests/make_p23_fragment.py --a out/p23/compact out/score_A.json --b out/p23/compactB out/score_B.json --out out/p23.tex --numbers out/p23_numbers.json` | `onehl/p23{A,B}_compact.tar.gz` | `one_hidden_layer/p23_preregistered_tests/results/score_{A,B}_locked.json` |
| App. B.6 (Theorem 4.1's configuration) | `python3 one_hidden_layer/theorem41_iid_init_p101/analyze.py report --from "$PAPER_DATA/deep/campaign_H" --out out/H.md --json out/H.json`; `python3 one_hidden_layer/theorem41_iid_init_p101_seeds7to10/analyze_QH.py one_hidden_layer/theorem41_iid_init_p101 "$PAPER_DATA/deep/campaign_H" "$SHARED/campaign_QH/collected/results" out/analysis_QH.json`; Fig. 8's statistics | `deep/campaign_H`; `campaign_QH` | `one_hidden_layer/theorem41_iid_init_p101/results/analysis.json`, `one_hidden_layer/theorem41_iid_init_p101_seeds7to10/analysis/analysis_QH.json` |
| App. C.1, C.2 (rate recipe, depths 3 to 5) | `python3 deep_mlp/depths3to5_preregistered_p31/recipe.py`; Table 2 | none | prints `recipe reproduces T23` |
| App. C.3 (across primes, depth 2) | `python3 deep_mlp/across_primes_development/analyze.py report --from out/runs_D --out out/D.md --json out/D.json`; `python3 deep_mlp/across_primes_ten_seeds/analyze.py report --from "$PAPER_DATA/deep/campaign_F" --prior-d "$PAPER_DATA/deep/campaign_D" --prior-dev "$PAPER_DATA/deep/development" --out out/F.md --json out/F.json`; `python3 deep_mlp/across_primes_p101/analyze.py report --from "$PAPER_DATA/deep/campaign_G" --out out/G.md --json out/G.json`; `python3 deep_mlp/depth2_p31_forty_seeds/analyze_QD.py "$SHARED/campaign_QD/collected/results" out/analysis_QD.json` | `deep/campaign_{D,F,G}`, `deep/development`; `campaign_QD` | each folder's `results/analysis.json` (D and F with `compare_analysis.py`); `deep_mlp/depth2_p31_forty_seeds/analysis/analysis_QD.json` |
| App. C.4 (training fractions) | `python3 deep_mlp/training_fractions_rate_tuning/analyze_M.py "$SHARED/campaign_M/collected/results" none none none out/analysis_M.json`; Table 3 | `campaign_M` | the `runs` section of `deep_mlp/training_fractions_rate_tuning/analysis/analysis_M.json` |
| App. C.6 (hidden biases) | `python3 deep_mlp/hidden_bias_update_rules/analyze_J.py report --out out/J --from "$PAPER_DATA/deep/campaign_J/campaignJ.tar" "$PAPER_DATA/deep/campaign_I/campaignI_H.tar" "$PAPER_DATA/deep/campaign_D/campaignD_H.tar" "$PAPER_DATA/deep/campaign_F/campaignF_H.tar"`; depth 2: the App. C.3 row | `deep/campaign_{D,F,I,J}` | `out/J/analysis.json` = `deep_mlp/hidden_bias_update_rules/results/analysis.json` |

`figures/expected/FIGURES.sha256` holds the SHA-256 of the 18 figure files as the paper prints them (Figures 1 to 5 after
centring, in `out/centred`), `FIGURES_before_centring.sha256` those of Figures 1 to 5 as their scripts write them; the scripts keep the file names
they had when the paper was built. Compare
statistics files with `cmp`, and figures with `sha256sum -c --ignore-missing` inside an output folder, for example
`(cd out/fig08 && sha256sum -c --ignore-missing ../../figures/expected/FIGURES.sha256)`. Give each script its own output
folder: the Figure 4 and Figure 14 scripts both write a `TF_acrossp_app.pdf`, and only the second is Figure 14.

## Starting from fresh runs

The commands of [Reproducing the paper](#reproducing-the-paper) read the saved runs in the layout of the original data, and
some of them read analyses, post-hoc windows or selections committed to this repository. After retraining, take four steps in
this order: train, pack, rebuild the analyses and windows, then draw the figures and tables. Work in a copy of the repository,
because step 3 overwrites committed files, and set PAPER_DATA and SHARED as in Reproducing the paper. Below, `runs_C` to
`runs_J` are the training output folders at the repository root, as in the examples of [Data](#data); keep them outside
`$PAPER_DATA/deep/`. The commands are for bash.

**1. Train** with the commands of the Data table. The runs under SHARED (`run_local.py --out "$SHARED/campaign_<X>/collected"`)
and the one-hidden-layer extracts (`compact_line1.py` into `$PAPER_DATA/onehl/<X>`) are written in their final layout. The
deep-network drivers write one folder per run, named by its run id: `chain.json` (`record.json` for the two `hidden_bias_*`
folders), the run files (`run.json.gz`, or `pilot.json.gz` and `main.json.gz` for a chain), and `raw/` or a log.

**2. Pack.** The reports (`analyze.py report --from`), `analyze_J.py`, `analyze_QH.py` and Figure 8 (`--runs`) read run
folders, but `figures/fig02_fig03_prepare_inputs.py` and `make_data_bundle.py` look for the archives by name,
`posthoc_window/rescore_deep_runs.py` reads only archives, and the reports of campaigns F and G record the archive member each
run was read from, so they match the committed reports only when built from the archives. An archive holds
`<run id>/<file>` members, without `raw/` and logs. On macOS, keep `COPYFILE_DISABLE=1` set for every `tar` command in this
section: otherwise macOS tar adds `._*` metadata members, which the readers reject.

```bash
export COPYFILE_DISABLE=1      # macOS: no ._* metadata members (no effect elsewhere)
# pack ARCHIVE RUNS PATTERN...: the finished runs of RUNS whose folder names match a PATTERN
pack() {
  a=$1 d=$2; shift 2
  ids=$(cd "$d" && for g in "$@"; do for f in $g/chain.json $g/record.json; do [ -f "$f" ] && echo "${f%/*}"; done; done | sort -u)
  tar -cf "$a" -C "$d" --exclude=raw --exclude='*.log' --exclude='*.tmp' --exclude='*.ckpt.npz' $ids
}
D=$PAPER_DATA/deep; for x in C D F G H I J; do mkdir -p "$D/campaign_$x"; done
for c in C1_D3 C1_D4 C1_D5 C2_D3 C2_D4 C2_D5 C3_D4p23 C4_D4w512 C5_D3ngd; do
  pack "$D/campaign_C/campaignC_$c.tar" runs_C "${c}_s*"
done
(cd "$D/campaign_C" && sha256sum campaignC_*.tar > SHA256SUMS)     # make_data_bundle.py checks the tars against it
pack "$D/campaign_D/campaignD_H.tar" runs_D 'H_*' 'HW_*'
pack "$D/campaign_D/campaignD_D3.tar" runs_D 'D3_*'
pack "$D/campaign_D/campaignD_D4.tar" runs_D 'D4_*'
pack "$D/campaign_F/campaignF_H.tar" runs_F 'H_*' 'HW_*'
pack "$D/campaign_F/campaignF_D2.tar" runs_F 'T16_*' 'T3_*' 'seedD2_*'
pack "$D/campaign_F/campaignF_D3.tar" runs_F 'D3_*'
pack "$D/campaign_F/campaignF_D4.tar" runs_F 'D4_*'
for s in H D3 D4; do pack "$D/campaign_G/campaignG_$s.tar" runs_G "${s}_*"; done
pack "$D/campaign_H/campaignH.tar" runs_H 'T41_*'
for s in H D2 D4; do pack "$D/campaign_I/campaignI_$s.tar" runs_I "B_${s}_*"; done
pack "$D/campaign_J/campaignJ.tar" runs_J 'J*'
```

When no finished run matches, `tar` stops with an error instead of writing an empty archive. The development runs of
`deep/development/` are archives of the files that `natplat_server.py run tasks/<task>.json OUT` writes, for example
`tar -czf "$D/development/runs.tar.gz" -C OUT --exclude='*.ckpt.npz' --exclude=queue.log .`. Five of them, `seedD2_r0.5_s0` to
`s4` (prior seeds of campaign F), have no task file here: each is `seedD2_r0.5_s5` of
`deep_mlp/across_primes_ten_seeds/manifest.json` with `init_seed` 101 to 105 and `split_seed` 201 to 205.
`fig02_fig03_prepare_inputs.py` stops without them (as uncompressed `seedD2_r0.5_s<i>.json`), although Figures 2 and 3 do not
draw them; in that case fill `out/bi` from the run folders instead:

```bash
mkdir -p out/bi/data/C out/bi/data/C5 out/bi/data/D out/bi/data/F out/bi/data/F_D2 out/bi/data/G out/bi/bias out/bi/dev2
ln -s "$PWD"/runs_C/C[12]_D[34]_s* out/bi/data/C/ && ln -s "$PWD"/runs_C/C[12]_D5_s* out/bi/data/C5/
ln -s "$PWD"/runs_D/* out/bi/data/D/ && ln -s "$PWD"/runs_G/* out/bi/data/G/ && ln -s "$PWD"/runs_I/* out/bi/bias/
ln -s "$PWD"/runs_F/H_* "$PWD"/runs_F/HW_* "$PWD"/runs_F/D3_* "$PWD"/runs_F/D4_* out/bi/data/F/
ln -s "$PWD"/runs_F/T16_* "$PWD"/runs_F/T3_* "$PWD"/runs_F/seedD2_* out/bi/data/F_D2/
```

The pre-registered p = 23 cohorts are read from archives with folders `compact/` (cohort A) and `compactB/` (cohort B);
`runs_p23A` and `runs_p23B` are the `--output` folders of `run_campaign.py` on `manifests/p23_confirm.json` and
`manifests/p23_confirmB.json`:

```bash
python3 one_hidden_layer/make_extracts/compact_line1.py out/p23/compact runs_p23A/h23_*
python3 one_hidden_layer/make_extracts/compact_line1.py out/p23/compactB runs_p23B/h23_*
tar -czf "$PAPER_DATA/onehl/p23A_compact.tar.gz" -C out/p23 compact
tar -czf "$PAPER_DATA/onehl/p23B_compact.tar.gz" -C out/p23 compactB
```

For Figure 8, `out/runs8` can hold the run folders (`ln -s "$PWD"/runs_H/T41_* out/runs8/`) in place of the archive.

**3. Rebuild the analyses and windows** that later commands read, from the root of the copy. The window rescoring needs every
chain of campaign C and the development runs. It takes each run's window-search start (the knee) and campaign D's list of runs
from the reports of campaigns C, D and F, so these replace the committed ones first:

```bash
mkdir -p out
python3 deep_mlp/restore_and_compare/make_data_bundle.py --campaign-tars "$PAPER_DATA/deep/campaign_C" --out out/bundle
python3 deep_mlp/depths3to5_preregistered_p31/analyze.py report --from out/bundle/campaign_C --out out/C.md --json out/C.json
python3 deep_mlp/across_primes_development/analyze.py report --from runs_D --out out/D.md --json out/D.json
python3 deep_mlp/across_primes_ten_seeds/analyze.py report --from "$PAPER_DATA/deep/campaign_F" \
    --prior-d "$PAPER_DATA/deep/campaign_D" --prior-dev "$PAPER_DATA/deep/development" --out out/F.md --json out/F.json
cp out/C.json deep_mlp/depths3to5_preregistered_p31/results/analysis.json
cp out/D.json deep_mlp/across_primes_development/results/analysis.json
cp out/F.json deep_mlp/across_primes_ten_seeds/results/analysis.json
python3 posthoc_window/rescore_deep_runs.py --out out/rescored --no-figs C D F G
for X in C D F G; do cp out/rescored/$X/settings.jsonl posthoc_window/results/$X/settings.jsonl; done
python3 deep_mlp/training_fractions_rate_tuning/analyze_M.py "$SHARED/campaign_M/collected/results" none none none \
    out/analysis_M.json
python3 deep_mlp/training_fractions_p47_p61/analyze_N.py "$SHARED/campaign_N/collected/results" none out/analysis_N.json \
    out/analysis_M.json
cp out/analysis_N.json deep_mlp/training_fractions_p47_p61/analysis/analysis_N.json
python3 transformer/gain_recipe_contrasts_two_blocks/analyze_Z.py out/analysis_Z.json "$SHARED/campaign_Z/collected/results" \
    "$SHARED/campaign_Y/collected/results" "$SHARED/campaign_X/collected/results"
cp out/analysis_Z.json transformer/gain_recipe_contrasts_two_blocks/analysis/analysis_Z.json
for d in runs_I/B_H_p31_*; do cp "$d/record.json" "$d/chain.json"; done     # make_comparison.py reads chain.json
python3 deep_mlp/hidden_bias_update_rules/make_comparison.py runs_I runs_D runs_F       # rewrites comparison.json
```

**4. Draw the figures and tables** with the commands of [Reproducing the paper](#reproducing-the-paper). The items below read
committed files; without step 3 or the argument in the last column they use the paper's analyses, windows and verdicts, even
next to your run records:

| Item | Committed input it reads | With fresh runs |
|---|---|---|
| Figs. 2, 3 | windows `posthoc_window/results/*/settings.jsonl` (`--winv2`) | replaced in step 3, or pass `--winv2 out/rescored` |
| Figs. 4, 10-12 | windows and verdicts in `deep_mlp/training_fractions_p47_p61/analysis/analysis_N.json` (fixed path) | replaced in step 3 |
| Figs. 4, 14, 15 | `transformer/gain_recipe_contrasts_two_blocks/analysis/analysis_Z.json` (fixed path), for the panel `TF_contrast_main`, not in the paper but in the statistics files | replaced in step 3 |
| Fig. 9 | `deep_mlp/depths3to5_preregistered_p31/results/analysis.json` (default of `--analysis`) | replaced in step 3, or pass `--analysis out/C.json` |
| Fig. 13 | `deep_mlp/hidden_bias_update_rules/results/analysis.json` (in the command) | `--analysis out/J/analysis.json`, from the App. C.6 command run after step 3 |
| Table 3 | `deep_mlp/training_fractions_rate_tuning/analysis/analysis_M.json` (in the command) | `out/analysis_M.json` (step 3) |
| Sec. 5.3, App. D.3-D.5 (counts), C.3, B.6 | `deep_mlp/depth2_p31_forty_seeds/analysis/analysis_QD.json` (in the command) | `out/analysis_QD.json` (App. C.3 command) |
| Sec. 5.3, App. D.5 (refit lags) | `analysis_Z.json`, `analysis_Q.json`, `analysis_ZP.json`, `analysis_ZM.json` of `transformer/` (in the command) | `out/analysis_Z.json out/analysis_Q.json out/analysis_ZP.json out/analysis_ZM.json`; the last from `python3 transformer/one_block_p61_mirror_free/analyze_ZM.py out/analysis_ZM.json "$SHARED/campaign_ZM/runs" "$SHARED/campaign_ZP/collected/results"` |
| App. D.5 (replays) | `transformer/readout_refit_lag/kcurve/*.json`, `lag_*.json`: replays of the configurations in `cfgs/`, not of the saved runs | rerun `replay2.py` or `replay.py`, then `k_curve.py RUN_DIR OUT.json --runner DIR` and `lag_tau.py RUN_DIR OUT.json --runner DIR`; give the new curves to `k_summary.py`, and run `plot_cases.py` in a folder holding the new `lag_*.json` (it opens them by name) |
| Sec. 5, App. A.3, C.5 | the rescoring: the reports of C, D and F (fixed paths); `figures/lib/scheme2.py`: `posthoc_window/results/*/settings.jsonl` (fixed path) | both replaced in step 3 |
| App. B.1 | `search_dev2_p23_p31/out/selection.json`, which `make_confirm.py` reads as `DEV/out/selection.json` | `mkdir -p out/dev2/out && cp -r one_hidden_layer/headline_cohort/search_dev2_p23_p31/manifests out/dev2/ && cp out/selection_dev2.json out/dev2/out/selection.json`, then `make_confirm.py out/dev2 out/confirm` |
| App. C.3 | the report of campaign G: `deep_mlp/across_primes_ten_seeds/results/analysis.json` (default of `--compare`) | replaced in step 3, or pass `--compare out/F.json` |
| App. C.6 | `deep_mlp/hidden_bias_update_rules/comparison.json`: the reference runs and their SHA-256, which fresh reference runs do not match | rewritten in step 3 |

The other rows read only run records and the outputs of earlier rows. Figure 1 and App. B.2 (window 4500 to 12000), Figure 6
(6500 to 16500) and App. B.3 (`CONFIRM_LOCK_p23*.json`) use windows and scoring rules fixed before the runs: keep them.
Comparisons with `figures/expected/` and with the analyses of the unmodified repository (step 3 replaces some in the copy)
then show how your runs differ from the paper's.

## Tests

From the repository root:

```bash
for d in deep_mlp/training_fractions_rate_tuning deep_mlp/training_fractions_p47_p61 deep_mlp/depth2_p31_forty_seeds \
         one_hidden_layer/theorem41_iid_init_p101_seeds7to10 transformer/one_block_no_gain_p113 \
         transformer/one_block_gain_p113 transformer/gain_recipe_contrasts_two_blocks transformer/one_block_other_primes \
         transformer/seeds16to20_two_block_exploration; do
  (cd "$d" && python3 -m pytest -q tests/)
done
(cd transformer/one_block_p61_mirror_free && python3 -m pytest -q tests/ -k "not manifest")
for d in depths3to5_preregistered_p31 across_primes_development across_primes_ten_seeds across_primes_p101; do
  python3 deep_mlp/$d/tests/test_runner.py
done
python3 one_hidden_layer/theorem41_iid_init_p101/tests/test_t41.py
python3 deep_mlp/hidden_bias_depths1_2_4/tests/test_bias.py
python3 deep_mlp/hidden_bias_update_rules/tests/test_bias.py
python3 deep_mlp/hidden_bias_update_rules/tests/test_j.py
(cd transformer/readout_fourier_refit && python3 test_refits.py)
(cd one_hidden_layer/quadratic_contrast_p13/standard_modarith_1hl_muon_v1 && PYTHONPATH=src python3 -m pytest -q tests/)
(cd one_hidden_layer/fixed_recipe_sweep/training/multip_seed_signed_relu_20_v1 && python3 -m pytest -q test_*.py)
python3 run_local.py transformer/seeds16to20_two_block_exploration --selftest
```

With the versions above all pass, in a few minutes. pytest counts, in loop order: 5, 5, 9, 7, 22, 28, 31, 31 and 44, then
36 for `one_block_p61_mirror_free` (its copy of the `one_block_other_primes` suite includes that folder's manifest test,
deselected). The scripts print 5, 8, 9 and 9 passing runner tests, among them bit-exact matches with the development
runner, then 6, 7, 7 and 7; `test_refits.py` runs 4 checks on synthetic data (6 as `test_refits.py RUN_DIR_TF RUN_DIR_MLP`,
with replays made by `replay.py`). The quadratic contrast has 73 tests, the fixed-recipe sweep 17.
`run_local.py FOLDER --selftest` trains the tiny configuration in the folder's `campaign.json` in a temporary folder and
prints `selftest ok`.

Nine tests in eight files compare a run with its resumed or rearranged twin (`test_resume*`,
`test_canonical_observers_never_change_a_depth2_run`, `test_default_equals_campaign_I`,
`test_cached_design_identical_update_trajectory`). They allow a relative and absolute difference of 1e-12, which absorbs
last-bit differences between NumPy and BLAS builds; with `EXACT_TESTS=1` they require bit-for-bit equality, which holds
with the pinned versions. The fixed-recipe sweep's tests start only with 8 GiB free in the temporary folder (`TMPDIR`).

## How exact a reproduction is

- With Python 3.11.15, numpy 2.4.4, matplotlib 3.10.9, pymupdf 1.28.2 and torch 2.5.1, the commands above rebuild from the
  run records every figure file of `FIGURES.sha256` and `FIGURES_before_centring.sha256` byte for byte, except Figure 6,
  identical apart from the PDF creation date, and Figure 9, drawn with matplotlib 3.7.1 for the paper: with 3.10.9 its
  plotted values are identical (its `receipt.json` checks them) and a few labels move by less than 0.4 pt.
- The statistics files in `figures/expected/` rebuild byte for byte, except that `fig16_fig17_tab08_analysis.json` matches
  apart from the run paths it records and `fig01_to_fig05_centring.json` adds the print scale to the measurements. The
  committed analyses that record no run paths also rebuild byte for byte (for example `analysis_N.json`, `analysis_QD.json`,
  `analysis_QH.json`, the reports of `theorem41_iid_init_p101` and `across_primes_p101`, the post-hoc windows), and
  `analysis_Q.json`, `analysis_Z.json`, `analysis_ZP.json` and `analysis_ZM.json` match apart from the run paths they record.
  Byte equality of rendered files depends on the plotting environment. With numpy 2.2, the reports of
  `depths3to5_preregistered_p31` and `across_primes_development` agree in every count and verdict; a few floats differ in
  their last bits (up to 5·10⁻¹⁵ relative).
- Training: the Polar step's eigendecomposition and rank cutoff amplify last-bit differences between numpy or LAPACK builds,
  so a rerun on another build is a rerun, not a bitwise replay. `run_campaign_local.py --replay CHAIN --data DIR` reruns the
  start of a saved depths3to5 pilot: with numpy 2.4.4 on x86-64 every logged loss matched (401 of 401 for a Polar chain),
  and with numpy 2.2.6 it drifted after update 28. The runs recorded under SHARED ran with Python 3.13 and numpy
  2.5.3 (2.5.4 in 14 runs of `gain_recipe_contrasts_two_blocks`), except `one_block_p61_mirror_free` (Python 3.11, numpy 2.4.4);
  `chain.json` records each run's build. Rerun for 200 updates with numpy 2.4.4, a run of `seeds16to20_two_block_exploration` and
  one of `theorem41_iid_init_p101_seeds7to10` matched their records exactly; two runs of `depth2_p31_forty_seeds` drifted
  after 37 and 52 updates and matched exactly with numpy 2.5.3.

## Citation

```bibtex
@misc{kumar2026flatloss,
  title         = {Flat Loss, Evolving Features: Grokking in Modular Addition},
  author        = {Kumar, Akash and Sankaranarayanan, Karthik},
  year          = {2026},
  eprint        = {2610.22593},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.22593}
}
```
