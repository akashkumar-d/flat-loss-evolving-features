# Data: the saved runs behind every reported number

The saved runs are about 680 MB and are not included in this release; the last column of the table below names the code
that regenerates each dataset, except the earlier quadratic-activation runs (`onehl/L4/`), whose original learner was not
preserved (`onehl_earlier/README.md`). The scripts read them from one data folder: `code/data/` by default, or the folder named by
the environment variable `PAPER_DATA`. That folder has the layout below.
`data_manifests/<name>.sha256` lists the SHA-256 of every file of each dataset; check one with, for example,
`(cd $PAPER_DATA/onehl/H && sha256sum -c /path/to/code/data_manifests/onehl_H.sha256)`.

| Path in the data folder | Contents | Files | Size | Manifest | Produced by |
|---|---|---|---|---|---|
| `onehl/H/` | headline confirmation: per-run compact extracts (`.npz` arrays + `.json` metadata and scores) | 20 runs, 40 files | 29.8 MB | `onehl_H.sha256` | `onehl/` steps 2 and 6 |
| `onehl/p23A_compact.tar.gz`, `onehl/p23B_compact.tar.gz` | pre-registered p = 23 cohorts A and B (folders `compact/`, `compactB/`, with their locks) | 2 × 20 runs | 30.7 MB | `onehl_p23.sha256` | `onehl/` step 7 |
| `onehl/L1/` | per-prime confirmations | 60 runs, 120 files | 115.4 MB | `onehl_L1.sha256` | `onehl/` step 3 |
| `onehl/L1_dev/` | per-prime development sweep | 54 runs, 108 files | 103.4 MB | `onehl_L1_dev.sha256` | `onehl/` step 3 |
| `onehl/L2/` | fixed-recipe sweep, with observers recomputed from saved states | 120 runs, 240 files | 69.2 MB | `onehl_L2.sha256` | `onehl_earlier/`, `onehl/` step 5 |
| `onehl/L3/` | replays | 10 runs, 20 files | 2.4 MB | `onehl_L3.sha256` | `onehl/` step 4 |
| `onehl/L4/` | quadratic contrast (its own observers, 500-update logs) | 40 runs, 80 files | 0.4 MB | `onehl_L4.sha256` | `onehl_earlier/` (original learner not preserved), `onehl/tools/compact_other.py line4` |
| `deep/campaign_C/` | campaign C: nine tars of chain folders `<id>/{chain.json, pilot.json.gz, main.json.gz}` | 164 chains | 108.9 MB | `deep_campaign_C.sha256` | `deep/campaign_C/run_campaign_local.py` |
| `deep/campaign_D/` | campaign D: tars of `<id>/{chain.json, run.json.gz}` | 127 runs | 28.0 MB | `deep_campaign_D.sha256` | `deep/run_local.py run D` |
| `deep/campaign_E/` | campaign E: the development runs, and the records of the 60 fresh runs its rule skipped | 24 runs | 6.2 MB | `deep_campaign_E.sha256` | `deep/run_local.py run E` |
| `deep/campaign_F/` | campaign F: the new seeds | 137 runs | 30.9 MB | `deep_campaign_F.sha256` | `deep/run_local.py run F` |
| `deep/campaign_G/` | campaign G (the p = 97 settings at p = 101): tars of `<id>/{chain.json, run.json.gz}`, one per setting | 30 runs | 5.9 MB | `deep_campaign_G.sha256` | `deep/run_local.py run G` |
| `deep/campaign_H/` | campaign H (Theorem 4.1's configuration at p = 101): one tar of `<id>/{chain.json, run.json.gz}` | 32 runs | 4.4 MB | `deep_campaign_H.sha256` | `deep/run_local.py run H` |
| `deep/campaign_I/` | campaign I (hidden biases at p = 31): tars of `<id>/{chain.json, run.json.gz}` (depths 1 and 2) and `<id>/{chain.json, pilot.json.gz, main.json.gz}` (depth 4) | 60 runs | 26.9 MB | `deep_campaign_I.sha256` | `deep/campaign_I/run_local.py` |
| `deep/campaign_J/` | campaign J (three bias updates at one hidden layer): one tar of `<id>/{cfg.json, record.json, run.json.gz}`, and `SHA256SUMS_runs` | 30 runs | 11.2 MB | `deep_campaign_J.sha256` | `deep/campaign_J/run_local.py` |
| `deep/development/` | development runs: archives of run files | 11 archives | 67.9 MB | `deep_development.sha256` | `deep/development/` |
| `deep/development_extra/` | development runs kept as loose `.json.gz` files, one folder per task | 60 files | 59.6 MB | `deep_development_extra.sha256` | `deep/development/` |

## The per-run files

**1HL compact extracts.** Each run has `<id>.npz` and `<id>.json`.
- The `.npz` holds training CE, held-out CE and both accuracies at every logged update (every update, except L4 at every
  500).
- It also holds the observer rows (A_H, the refit, AGOP purity and its random-frame nulls), at their own cadence, and, where
  logged, margins and norms.
- The `.json` holds the configuration, the seeds, the window and the protocol scores.
- Values are copied from the runs, never smoothed.
- `onehl/tools/compact_line1.py` and `compact_other.py` write them.

**Deep run files.** Each run file (`run.json.gz`, `main.json.gz` or `pilot.json.gz`) stores:
- the full configuration (`cfg`);
- the training CE at every update (`loss`);
- the observations (`obs`): accuracies and margins every 20 updates, and the heavier observers every 500 or 1000.

Floats are rounded to 7 significant digits. `chain.json` records the seeds, the code hashes (of the files before their comments were removed, `PROVENANCE.tsv`'s
`origin_sha256`), the host and a run summary.
Campaign H's run files (`deep/campaign_H/t41_runner.py`) also hold the training accuracy at every update (`train_acc`); their
observations come every 50 updates, and the heavy ones (A_H, the two refits, norms) every 250.
`deep/development/RUN_MANIFEST.tsv` lists every development run file with the SHA-256 of its uncompressed JSON.
`deep/tools/make_data_bundle.py` restores and checks them.

## Results committed with the code

These let a reader check every number without the data:
- **Deep campaigns.** `deep/campaign_*/results/analysis.json` (every per-run field and endpoint; campaigns C–H and J), and
  `ANALYSIS.md` for campaigns C–H.
- **Post-hoc window.** `posthoc_window/results/*/settings.jsonl` (every deep run's window and verdict).
- **Figures.** `paper_figures/expected/` and `paper_figures/bias/expected/` (the figures' statistics).
- **1HL.** The scores of the p = 23 tests in `onehl/p23_tests/results/`, and the selection outputs of the recipe search in
  `onehl/headline/`.

## Not in the data folder

**Full trajectories and saved weights.** The 1HL runs' full-precision trajectories, the fixed-recipe sweep's saved weights,
the replays' saved states and the L1 confirmation's raw results (830 MB) are kept by the authors. Every reported number is
computed from the compact extracts above. The weights and states are needed only to recompute observers.

## The campaigns after version 1

The run records of `campaigns/` (each run's `run.json.gz` and `chain.json`) are not included either, and this folder has no
data manifests for them. `campaigns/run_local.py` regenerates any run in the layout its campaign's analysis reads
(`campaigns/README.md`); each campaign's `analysis/` holds the outputs computed from the original runs.

Version 2's figures and numbers read the records of campaigns N, X, Y, Z, ZP, ZM, Q, QD and QH, and the readout scan of
`campaigns/readouts/refit_lag` also those of V and W: 596 runs, about 190 MB. They read them from one folder, SHARED in
`paper_figures/arxiv_v2/README.md`, with the layout that `run_local.py` writes, `campaign_<X>/collected/results/<run id>/`
(campaign ZM: `campaign_ZM/runs/<run id>/`). Figure 8 reads campaign QH's four runs together with campaign H's archive
(`deep/campaign_H/` above). `paper_figures/arxiv_v2/expected/rev4_fig23_SOURCES.json` lists the SHA-256 of the 300 run files
that Figures 2 and 3 read, campaign QD's 80 among them. The replays of `campaigns/readouts/refit_lag` (parameter snapshots,
70 to 190 MB per run) are not included either; its `replay2.py` and `campaigns/readouts/fourier_refit/replay.py` regenerate them.
