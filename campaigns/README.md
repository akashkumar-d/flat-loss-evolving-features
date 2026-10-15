# Experiments after version 1 of the paper (campaigns K to ZM, Q, QD and QH)

This folder holds the experiments run after version 1 of the paper, from 7 to 14 October 2026: sixteen training campaigns
with about 1,180 runs, and four readout studies. Version 1 does not report them. Version 2 reports the campaigns and studies
that `../README.md` maps to its figures, tables and text; the others are the development runs behind them. Four campaigns
train deep MLPs across primes and training fractions (K, L, M, N); ten train one- and two-block transformers (T to ZM, and
Q); two complete experiments of version 1: QD runs depth 2 on the 40 seeds of depths 3-5 (Figure 3 of version 2), and QH adds
four seeds to an arm of campaign H (Figure 8). The letters continue the paper's campaigns C to J (`../deep/`).

Each campaign folder is a copy of the campaign's folder as it ended, with comments and docstrings removed from its code
(`../README.md`, "Release"): README, PLAN (the design, and where stated the
rules fixed before the runs), manifest and configurations, runner, tests, worker, and `analysis/` with the summary
(`SUMMARY.md` in most folders, `RESULTS.md` in Q, QD and QH), JSON outputs, logs and figures. The readout folders are
copied the same way, with the exceptions listed under "Files kept as they ran". `../PROVENANCE.tsv` lists each file's origin
and hash; for the PNG figures, see the note on Content Credentials in `../README.md`, "Release". The new files are this
README, `requirements.txt`, `run_local.py` and `readouts/margin_vs_AH/README.md`.

**Not included: three later campaigns.** Campaign QE (ten fresh seeds for the two-block change of the feature rate at
update 12k, which Table 6 of version 2 reports on three seeds), campaign ZG (Figure 5's one-block contrast with the gain
starting at 1.5) and campaign MW (Morwani et al.'s recipe for a quadratic network at p = 71, with held-out pairs and the
paper's measures) were set up on 13 to 15 October 2026. Version 2 does not report them, and they are not in this release.

**Not included: the run records.** Each run wrote `run.json.gz` (every update's loss and the logged observers), and the
analyses read those files. This release leaves them out, together with the cloud sessions' status files, the replays of
`readouts/fourier_refit` and `readouts/refit_lag` (parameter snapshots and logs), campaign X's MLP-side runs and caches. Every
run can be regenerated with the code here.

## Requirements

Python 3.10 or newer with numpy, matplotlib and pytest; PyTorch (the CPU build is enough) for the transformer campaigns and
`readouts/` (`requirements.txt`). Checked with Python 3.11.15, numpy 2.4.4, matplotlib 3.10.9, pytest 9.1.1 and torch 2.5.1.

## Tests

From each campaign folder: `python3 -m pytest -q tests/`. With the versions above all pass: K 4, L 5, M 5, N 5, T 7,
T rerun 8, U 12, V 12, W 16, X 22, Y 28, Z 31, ZP 31, Q 44, QD 9 and QH 7 tests. Q's are its own 8 with campaign ZP's suite
for runner 1.6 (without ZP's manifest test) and campaign ZM's suite, run against Q's runner 1.8; QD's and QH's are the tests
of the runners they reuse (campaigns I and H) and their own checks.

- `campaign_ZM`: `python3 -m pytest -q tests/ -k "not manifest"` (36 tests). `tests/test_tf_16suite.py` is campaign ZP's suite,
  run against ZM's runner; its manifest test needs ZP's manifest, so it is deselected. The suite uses `../campaign_ZP`.
- `readouts/fourier_refit`: `python3 test_refits.py` runs four checks on synthetic data;
  `python3 test_refits.py RUN_DIR_TF RUN_DIR_MLP` adds the two checks against replays made with `replay.py`.

## Running the training runs

The campaigns were run by cloud sessions with each folder's `worker.py` (instructions in `WORKER.md`), which took runs from
the manifest and pushed each finished run to a branch of a private repository; a script outside this release
(`collect_results.sh`, named in the READMEs) copied them into `collected/`. `run_local.py` runs the same runner on the same
configurations on one machine, without git, and writes each finished run where the analyses read it:

    <campaign>/collected/results/<run id>/run.json.gz    floats rounded to 7 significant digits, as worker.py did
    <campaign>/collected/results/<run id>/chain.json     configuration, code hashes, host and timing

From this folder:

    python3 run_local.py campaign_N --list                           # the runs, their settings and estimated cost
    python3 run_local.py campaign_N --match D3_p61_f0.8 --jobs 4     # the runs whose id contains the text, four at a time
    python3 run_local.py campaign_ZP --ids ZP_P_p89_f0.25_lam_v0.33_s11
    python3 run_local.py campaign_Y --all --jobs 8
    python3 run_local.py campaign_T --selftest                       # campaign.json's tiny configuration, in a temporary folder

- A run that stops resumes from the runner's last checkpoint (every 2,000 updates) when started again. Finished runs are
  skipped unless `--force` is given. Threads per run: the manifest's value, or `--threads`.
- Campaign ZM has configuration files (`campaign_ZM/configs/`) instead of a manifest:
  `python3 run_local.py campaign_ZM --configs campaign_ZM/configs/ZM_f0.55_s1*.json`, then give `collected/results` to
  `analyze_ZM.py` as ZM_RUNS_DIR. Its README's `queue.sh` writes the same files to `runs/` instead.
- Campaign X's MLP side: `campaign_X/mlp_side/run_all.sh` (eight runs, about 6 minutes each) writes `mlp_side/runs/`, the
  MLP_RUNS_DIR of `analyze_X.py` and `plot_X.py`.
- `readouts/fourier_refit`: `replay.py` reruns a configuration from `cfgs/` and saves parameter snapshots; its README gives the
  commands. `readouts/refit_lag/replay2.py` does the same for runs with the gain or late rates (`cfgs/`); its `NOTE.md`
  names the replays it read.
- Campaign Q's analysis reads campaign ZM's runs from `campaign_ZM/runs/`, where `queue.sh` wrote them; with runs made by
  `run_local.py`, link `campaign_ZM/collected/results` there. Campaign QH's four runs complete campaign H's arm: its analysis
  and Figure 8 read them with campaign H's archive (`../paper_figures/arxiv_v2/README.md`).
- **Cost.** `--list` prints each run's estimate from the manifest. The manifests add up to about 2,100 core-hours for the deep
  campaigns K to N, 840 for the transformers T to ZM, and 185, 23 and 5 for Q, QD and QH (planning estimates, not measured).

**How exact a rerun is.** With the same numerical build a rerun repeats a run exactly; on another build it is a rerun, not a
bitwise replay. For this release, a campaign N run (numpy 2.4.4 here, 2.5.3 in the campaign) matched its record for the first
28 updates and then drifted: the Polar step's eigendecomposition and rank cutoff amplify last-bit differences, as for the
paper's deep campaigns (`../README.md`). A campaign ZP run matched its record exactly over the 200 updates checked. The replays
of `readouts/fourier_refit` matched their records to the stored precision for about 3,000 updates and then drifted; their
held-out crossings came within 40 to 140 updates of the records'. For version 2's release, one campaign Q run (one block,
p = 61) and one campaign QH run matched their records exactly over 200 updates; two QD runs (depth 2, with and without biases)
drifted after 37 and 52 updates with numpy 2.4.4 and matched exactly over the 200 updates with numpy 2.5.3 under Python 3.13,
the build the campaign used.

## The campaigns

| Folder | Question | Runs | Outcome (the folder's analysis) |
|---|---|---|---|
| `campaign_K` | Deep MLPs (depths 3 and 4) at p = 23, 47, 61, 101, with the head decay scaled with p; training fraction 0.8 | 96 | The paper's joint endpoint passes only at depth 4, p = 47; p = 23 rises too slowly, p ≥ 61 loses training fit |
| `campaign_L` | Campaign K at training fraction 0.5 | 120 | The held-out rise comes 4 to 6 times later and never at p = 23; the refit leads, but the training CE falls 5 to 21% meanwhile |
| `campaign_M` | Training fractions 0.8 to 0.5, with the last hidden layer's rate r re-tuned | 216 | 135/216 pass; r must shrink with the fraction; chose campaign N's r per cell |
| `campaign_N` | Depths 3 and 4 at p = 47 and 61, fractions 0.8 to 0.5, ten fresh seeds per cell; analysis fixed before the runs | 160 | 155/160 pass; at 0.8 the training CE stays within −3% and +10% while held-out accuracy rises |
| `campaign_T` | One- and two-block transformers at p = 113: a grid of the paper's Polar steps, and AdamW | 80 | No setting meets the planned rule; between the fit and the rise the CE falls by orders of magnitude |
| `campaign_T_rerun` | Campaign T's five runs that crashed in the float32 SVD, with runner 1.1's safeguards | 5 | Read by campaign T's `analyze_T.py` with T's other 75 runs |
| `campaign_U` | RMS division before the unembedding, larger head decay, AdamW; averaging windows of the flatness test | 71 | Flat stretches only in the Polar steps' capped regime, at the CE level the caps set; AdamW passes 0/24 |
| `campaign_V` | Fresh seeds for U's two settings (rule fixed before the runs); RMS head at training fraction 0.2 | 50 | Setting C1 10/10, but its plateau is a float32 floor (campaign W); C2 2/10 |
| `campaign_W` | Transformers under the paper's definitions, in float64 | 42 | No setting passes (best 2/3); with up to 10 misclassified pairs 6/6, but the CE falls 6.5 to 9% across the rise |
| `campaign_X` | Fourier readouts on W's two best settings, ten fresh seeds each | 20 | The top-8 Fourier refit leads the network in 20/20, the paper's refit trails in 20/20; the CE still falls 7 to 12% |
| `campaign_Y` | X's setting with a gain on the RMS head input relaxing from 2 or 3 to 1; rule fixed before the runs | 20 | 10/10 in both arms; the training CE rises 33% and 111% across the held-out rise |
| `campaign_Z` | NGD and AdamW with the same gain; the recipe's range; other p; two blocks | 143 | Only the Polar steps pass (NGD 0/10, AdamW 0/10); recipe 18/18; other p at most 1/5; two blocks 0/20 |
| `campaign_ZP` | One block at p = 61, 89, 149, with the training fraction set per p | 30 | p = 89 5/5; p = 149 3/5; p = 61 3/5 on all held-out pairs and 5/5 on the pairs whose mirror is also held out |
| `campaign_ZM` | p = 61 with a mirror-free split (both orders of a pair on the same side); prediction fixed before the runs | 7 | 5/5 on all held-out pairs at fraction 0.55 |
| `campaign_Q` | Seeds 16-20 of the one-block cells of ZP and ZM (p = 61 with both splits, 89, 149), so that every prime has ten; two blocks at p = 113: seeds 16-20 of Z's cell, one rate changed at update 12k (seeds 11-13), fractions 0.175 and 0.225; analysis fixed before the runs | 40 | Ten seeds per prime: 9/10 (p = 61, mirror-free), 5/10 (p = 61, random split; 10/10 on the no-mirror pairs), 10/10 (p = 89), 7/10 (p = 149; loose 10/10). Two blocks 0/10: training pairs flip at 17.8k-22.0k; a smaller feature rate from 12k keeps the fit in 3/3 (exploratory) |
| `campaign_QD` | Depth 2 at p = 31 on the 40 seeds of depths 3-5, without and with hidden biases | 80 | Primary endpoint 34/40 (Wilson 0.71-0.93) without biases, 18/40 with them |
| `campaign_QH` | Campaign H's eta_V = 3e-4 arm on seeds 7-10 | 4 | With H's seeds 1-6, every endpoint holds in 10/10 |
| `readouts/fourier_refit` | Train-only refits restricted to the top Fourier frequencies, on replays of two transformer runs and one MLP run | 3 replays | In the transformers the top-K refit leads and the paper's refit trails; in the MLP the paper's refit already leads |
| `readouts/ah_gap` | Why A_H is higher in the transformer than in the MLP | none (reads the replays) | The transformer's A_H is measured one linear map after its neurons |
| `readouts/refit_lag` | Why the ridge refit trails the transformer's held-out accuracy and leads the MLP's, and how many frequencies the Fourier refit should keep | none (the records of 352 transformer and 870 MLP runs; nine replays) | At the network's held-out rise the transformer's head input carries 6 to 13 frequencies and the ridge fits the others from memorized features; the MLP's features carry every frequency before the rise |
| `readouts/margin_vs_AH` | A_H of single-frequency neurons, quadratic and ReLU (one script of the study) | none | 2/5 for each neuron of the maximum-margin quadratic network, 0.114 for a bias-free ReLU neuron |

## Analyses and figures

Each folder's README gives the order of its scripts and their outputs. Run them
from the campaign's folder, with the folders of the campaigns it reads next to it, as here. A STATUS_DIR argument (the
sessions' status files) only labels unfinished runs; any folder works, even a missing one.

| Folder | Scripts, in order | Reads besides its own runs |
|---|---|---|
| `campaign_K` | `analyze_K.py`, `plot_K.py`, `plot_K_refit.py` | the paper's deep runs (campaigns D, F and G of `../DATA.md`, unpacked), for `plot_K_refit.py` |
| `campaign_L` | `analyze_L.py`, `summarize_L.py`, `extra_L.py`, `plot_L.py` | campaign K's runs; the paper's deep runs |
| `campaign_M` | `analyze_M.py`, `plot_M.py` | campaigns L's and K's runs |
| `campaign_N` | `analyze_N.py`, `plot_N.py`, `plot_N_cut.py`, `paper_N.py` | `../campaign_M/analysis/analysis_M.json` (included) |
| `campaign_T` | `analyze_T.py`, `plot_T.py` | campaign T rerun's runs |
| `campaign_U` | `flat_windows.py`, `plot_U.py`, `plot_flat_scales.py` | four campaign T runs, for `plot_flat_scales.py` |
| `campaign_V` | `confirm_V.py`, `analyze_V.py`, `plot_V.py` | |
| `campaign_W` | `analyze_W.py`, `tf_figs.py`; `score_existing.py` | campaigns T, U and V's runs, for `score_existing.py` |
| `campaign_X` | `analyze_X.py`, `plot_X.py` | the MLP side (`mlp_side/run_all.sh`) |
| `campaign_Y` | `analyze_Y.py`, `plot_Y.py` | campaign X's runs |
| `campaign_Z` | `analyze_Z.py`, `plot_Z.py`, `paper_contrast.py` | campaigns Y's and X's runs |
| `campaign_ZP` | `analyze_ZP.py`, `plot_ZP.py`, `paper_ZP.py` | campaigns Z's and Y's runs |
| `campaign_ZM` | `analyze_ZM.py`, `plot_ZM.py`, `readouts_ZM.py`, `plot_readouts_ZM.py`, `freqs_ZM.py` | campaign ZP's runs and analysis; campaign Z's code |
| `readouts/fourier_refit` | `replay.py`, `score.py`, `analyze.py`, `plot_refits.py` | |
| `readouts/ah_gap` | `ah_decomp.py` | the replays of `readouts/fourier_refit` |
| `campaign_Q` | `analyze_Q.py`; `zm10/build_zm10.py` (it imports `zm10/plot_ZM10.py`) | the runs of campaigns ZP, ZM, Z and Y, and their analysis code; for `build_zm10.py`, campaign ZM's code (SHARED: this folder) |
| `campaign_QD` | `analyze_QD.py` | |
| `campaign_QH` | `analyze_QH.py` | campaign H's archive (`../DATA.md`) and code (`../deep/campaign_H`) |
| `readouts/refit_lag` | `scan_runs.py`, `summarize.py`; `replay2.py`, `lag_tau.py`, `analyze_lag.py`, `plot_lag.py`, `plot_cases.py`; `k_curve.py`, `k_summary.py`, `k_val.py`, `k_val_summary.py`, `k_rules.py`, `mlp/scan_mlp.py`, `plot_k.py` | the runs of campaigns V to Q; the MLP runs (the paper's deep runs and campaigns L, M, N and QD: the 870 of `mlp/scan_mlp.json`; `mlp/scan_mlp.py` also reads campaign K's, which were not in its folder when it ran); `readouts/fourier_refit` (`replay.py`, the runners) |
| `readouts/margin_vs_AH` | `fourier_notions_check.py` | |

## Files kept as they ran

- `worker.py`, `WORKER.md`, `campaign.json`: the cloud-session worker, its instructions and its settings (runner, selftest
  configuration). `run_local.py` uses only `campaign.json`.
- `HASHES.json`: written before the runs with the SHA-256 of the files as the runs used them, and recomputed for this
  release's files without comments, so that `worker.py selftest` passes (`../PROVENANCE.tsv` keeps the earlier hashes). `manifest_test.json`: the miniature
  manifest of the worker's end-to-end check (`make_manifest.py --test`).
- `campaign_X/testdata/`: two records in campaign format, made by `make_testdata.py` from the two transformer replays of
  `readouts/fourier_refit`; the fixtures of campaign X's tests, not campaign X runs.
- Exploratory scripts and records that name the cloud workspace folders where they ran: `campaign_L/extra_L.py`,
  `campaign_M/design/`, `campaign_T/analysis/ce64_check/`, `campaign_W/` (`draft_existing.py`, `draft_pilots.py`,
  `score_existing.py`, `explore/`, `analysis/zoom/`), `campaign_X/testdata/make_testdata.py`,
  `campaign_X/analysis/rise_paperwindow_X.json`, `campaign_Z/pilots/peek.py`, `campaign_Z/prep/heldout_defs.py`,
  `readouts/ah_gap/ah_decomp.py`, the log `campaign_L/analysis/plot_L.log`, `campaign_Q/zm10/build_zm10.py` and `plot_ZM10.py`
  (the default of SHARED), and in `readouts/refit_lag` `scan_runs.py`, `k_rules.py`, `k_curve.py`, `k_val.py`, `lag_tau.py`,
  `mlp/scan_mlp.py` and the log `analyze_lag.log`. To rerun one of the scripts, change the paths in its first lines (or set
  SHARED). Three notes name the folder where the authors kept the runs (`campaign_W/CAMPAIGN_W_PLAN_20261010.md`,
  `campaign_X/analysis/SUMMARY.md`, `campaign_Y/analysis/SUMMARY.md`). `../tools/build_release.py` lists all of these files.
- `campaign_Q/analysis/analysis_Q.json` and `campaign_Q/zm10/analysis_ZM10.json` record each run's file by its path in the
  cloud workspace where it was analyzed, and the scripts that read them open the runs there. To use your own runs, rebuild
  `analysis_Q.json` with `analyze_Q.py OUT.json SHARED`, which writes your paths (`../paper_figures/arxiv_v2/README.md`).
- `readouts/fourier_refit`, `readouts/ah_gap`, `readouts/refit_lag` and `readouts/margin_vs_AH` were `fourier_refit/`,
  `ah_gap/`, `refit_lag/` and `margin_vs_AH/` next to the campaign folders; where a script or note cites `../fourier_refit` or
  `kt/fourier_refit`, it is `readouts/fourier_refit` here. Of `margin_vs_AH/`, this release includes only
  `fourier_notions_check.py`, which checks the values of A_H that version 2 states in Section 2; the folder's other files
  concern a comparison that the paper does not report. `refit_lag/`'s own `SHA256SUMS`, which lists its replays, is left out;
  so is `figs/refit_k_choice-1.png`, a second copy of `figs/refit_k_choice.png`. Its four PNG figures are included without
  the Content Credentials chunk that its folder's copies gained when they were written to the authors' computer.
- The READMEs and plans keep their working references: the branch name and dates of the work, the requests that
  started each campaign, and the project notes they cite (for example `claude/FIGURE_COLOUR_KEY.md`), which are not part of
  this release.
