# Figures 2, 3 and 10: the runs with hidden biases

| Figure | Stem | Script | Data |
|---|---|---|---|
| 2 | `fig2_acrossp_1hl_bias` | `bias_figs.py` | as `fig2_acrossp_1hl` (`../README.md`), plus `deep/campaign_I` |
| 3 | `fig3_depths_2to5_bias` | `bias_figs.py` | as `fig3_depths_2to5`, plus `deep/campaign_I` |
| 10 | `bias_ablation_1hl` | `bias_appendix_fig.py` | `../../deep/campaign_J/results/analysis.json` |

- **Figures 2 and 3** are `scheme2.py`'s `fig2_acrossp_1hl` and `fig3_depths_2to5` with the medians of campaign I's with-bias
  runs drawn dotted: the p = 31 column of Figure 2, and the depth-2 and depth-4 columns of Figure 3
  (`../../deep/campaign_I/`). Everything else is drawn as in the bias-free figures. The shaded windows are still those of
  the bias-free runs.
- **Figure 10** shows campaign J's three ways of updating the bias, next to the joint step and to no bias
  (`../../deep/campaign_J/`).

## Commands

From this folder:

```bash
export PAPER_DATA=<data folder>                       # layout: ../../DATA.md
python3 -I prepare_data.py --out bias_inputs          # copies the run files bias_figs.py reads out of the archives
python3 -I bias_figs.py --data bias_inputs/data --winv2 ../../posthoc_window/results --bias bias_inputs/bias \
  --dev2 bias_inputs/dev2 --out figs                  # Figures 2 and 3, their statistics and SOURCES.json
python3 -I bias_appendix_fig.py --analysis ../../deep/campaign_J/results/analysis.json --out figs   # Figure 10
(cd figs && sha256sum -c ../expected/FIGURES.sha256)
cmp figs/bias_figs_stats.json expected/bias_figs_stats.json; cmp figs/SOURCES.json expected/SOURCES.json
cmp figs/bias_ablation_1hl_stats.json expected/bias_ablation_1hl_stats.json
```

With `--no-bias`, `bias_figs.py` draws the bias-free figures `fig2_acrossp_1hl` and `fig3_depths_2to5` instead.

## The scripts

- `bias_figs.py` uses `acrossp_figs.py`, a transcription of the steps of `../scheme2.py` and `../median_figs.py` that draw these
  figures (run selection, medians, window bands and layout). `endpoints_i.py` is campaign C's per-run endpoint, as in
  `../../deep/campaign_I/`.
- `SOURCES.json` records the SHA-256 of all 240 run files that `bias_figs.py` reads.
- `prepare_data.py` copies run files byte for byte; it is needed because `bias_figs.py` reads folders, not archives.

## Checks

Python 3.11.15, numpy 2.4.4, matplotlib 3.10.9:
- **Figures.** The three PDFs that these commands draw are byte-identical to the paper's. `expected/FIGURES.sha256` holds
  the hashes of the PDFs and of the PNGs drawn with them. Rendered images can differ in other plotting environments.
- **Statistics.** `expected/bias_figs_stats.json` and `expected/bias_ablation_1hl_stats.json` are what these commands
  write.
- **The bias-free figures.** With `--no-bias`, `bias_figs.py` draws `fig3_depths_2to5` (PDF and PNG) and the PNG of
  `fig2_acrossp_1hl` exactly as pinned in `../expected/FIGURES.sha256`. The PDF of `fig2_acrossp_1hl` differs from its pin
  only in the PDF Creator field, which names `acrossp_figs.py` instead of `scheme2.py`.
