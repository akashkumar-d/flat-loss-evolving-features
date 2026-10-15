# The paper's figures

Every trajectory figure is drawn from the saved runs; nothing here trains. Figure numbers follow version 1 of the paper on arXiv.
Figures 2 and 3 of the paper add the medians of runs with hidden biases, drawn dotted; those versions, and Figure 10, are
drawn by `bias/` (`bias/README.md`). The figures of the draft before the
restructure of 2026-10-05 are still drawn; they are listed after the map.

- `_paths.py` says where the code and the data are. By default the data are in `../data/`; set `PAPER_DATA` to read them
  elsewhere.
- `../DATA.md` lists the datasets.

## Commands

```bash
export PAPER_DATA=<data folder>          # layout: ../DATA.md
python3 -B scheme2.py --out figs                       # all 22 figures below that scheme2.py draws (about 3 minutes)
python3 -B scheme2.py --out figs H_hero_2x2 fig2_acrossp_1hl  # or any subset, by stem
python3 -B check_starts.py --out figs H L1             # the start-rule check; then L2 L3 L4

# Figure 7: Theorem 4.1's configuration at p = 101 (campaign H), and its statistics
python3 -B thm41_config_fig.py --out figs

# Figure 4: the two pre-registered p = 23 cohorts, from their compact archives
mkdir -p p23 && tar -xzf $PAPER_DATA/onehl/p23A_compact.tar.gz -C p23 && tar -xzf $PAPER_DATA/onehl/p23B_compact.tar.gz -C p23
python3 -B render_fig2_v2.py p23 p23/compact  figs --stem p23_confirmA
python3 -B render_fig2_v2.py p23 p23/compactB figs --stem p23_confirmB --dot 0.5

# Figure 8: three primary-metric panels; uses committed analysis, no trajectories
python3 -B campaign_summary_core.py --out figs
```

## Figure map

| Figure | Stem | Script | Data |
|---|---|---|---|
| 1 | `H_hero_2x2` | `scheme2.py` | `onehl/H` |
| 2 | `fig2_acrossp_1hl_bias` (bias-free: `fig2_acrossp_1hl`) | `bias/bias_figs.py` (bias-free: `scheme2.py`) | `deep/campaign_D`, `deep/campaign_F`, `deep/campaign_G` (p = 101), `deep/campaign_I` |
| 3 | `fig3_depths_2to5_bias` (bias-free: `fig3_depths_2to5`) | `bias/bias_figs.py` (bias-free: `scheme2.py`) | `deep/campaign_C`, `deep/campaign_F`, `deep/development`, `deep/campaign_I` |
| 4 | `p23_confirmA`, `p23_confirmB` | `render_fig2_v2.py` | `onehl/p23A_compact.tar.gz`, `onehl/p23B_compact.tar.gz` |
| 5 | `locked_cohorts_1hl` | `scheme2.py` | `onehl/p23B_compact.tar.gz`, `onehl/L1` |
| 6 | `L2_cohort_all` | `scheme2.py` | `onehl/L2` |
| 7 | `thm41_config_p101` | `thm41_config_fig.py` | `deep/campaign_H` |
| 8 | `deep_campaign_summary` | `campaign_summary_core.py` | `deep/campaign_C/results/analysis.json` |
| 9 | `acrossp_depth3_3row` (top), `acrossp_depth4_3row` (bottom) | `scheme2.py` | `deep/campaign_C`, `deep/campaign_D`, `deep/campaign_F`, `deep/campaign_G` (p = 101) |
| 10 | `bias_ablation_1hl` | `bias/bias_appendix_fig.py` | `../deep/campaign_J/results/analysis.json` |

**The draft before the restructure** (its figure numbers), all drawn by `scheme2.py` and checked the same way: `fig2_grid` (2),
`fig3_depths_p31` (3), `H_cohort` (4), `l1_phase_grid` and `L1_cohort_fullCE` (6, 7), `L2_cohort_p11_p13_p17` and
`L2_cohort_p19_p23_p29` (8, 9), `L3_cohort` and `L4_cohort` (10, 11), `deep_development_overview` (13, with `scheme_deepdev.py`),
`campaign_medians_D345` (14), and `acrossp_1hl`, `acrossp_depth2`, `acrossp_depth3`, `acrossp_depth4` (15–18). Its Figure 1(b),
`H_summary_2x2`, came from `legacy_fig1b/` (below).

**Deep windows and verdicts.** The deep figures take their windows and verdicts from `../posthoc_window/results/`. Those are
rebuilt by `../posthoc_window/make_all2.py`.

**1HL windows and verdicts.** These are computed in `scheme2.py`, with the same function.

**The rest of each figure.**
- The figure layout, colours and run selection come from `median_figs.py`.
- `scheme_deepdev.py` reads the development runs.
- `lib/` holds the campaign run reader and the endpoint definitions.

## Current presentation updates, 2026-10-06

Internal dataset IDs remain stable in the code; the paper uses descriptive
names. Figure 5 now labels its columns only by prime, with settings identified
in its caption. Its curves and statistics are unchanged. Figure 8 retains the
first three metrics and all 164 runs of the original summary, omitting the
auxiliary AGOP panel. The new renderer checks scatter positions, medians,
counts and metric values against the preserved original function.
The receipts of both figures (commands, runtime and checks) are kept with the
paper's figure files. The older renderer remains
available for reproducing the historical panel.

## Historical redraw check

Checked on 2026-10-05 with Python 3.10.12, numpy 2.2.6 and matplotlib 3.10.9 (DejaVu Sans).
- **The 22 figures of `scheme2.py`.** Every PDF and PNG is byte-identical to the paper's, and so is `scheme2_stats.json`
  (`expected/`).
- **`check_starts.py`.** Both JSON files are byte-identical to those in `expected/`.
- **Figures 5 and 7 at that checkpoint (now Figures 4 and 8).** The PNGs and statistics are byte-identical. The PDFs differ only in their creation-date field.
- **Hashes.** `expected/FIGURES.sha256` holds the hashes of the paper's figure files.

## p = 101 and Figure 7, 2026-10-06

- **Figures 2 and 9.** The last column of `fig2_acrossp_1hl`, `acrossp_depth3_3row` and `acrossp_depth4_3row` is now p = 101
  (campaign G; `median_figs.specs`). `scheme2.py` reads those runs from `deep/campaign_G` and their windows from
  `../posthoc_window/results/G/settings.jsonl`. `expected/scheme2_stats.json` and `expected/FIGURES.sha256` hold the new
  statistics and hashes. The other columns are unchanged.
- **Figure 7.** `thm41_config_fig.py` draws it from campaign H's archive. Its PDF, PNG and statistics match
  `expected/FIGURES.sha256` and `expected/thm41_config_stats.json` byte for byte (Python 3.11.15, numpy 2.4.4,
  matplotlib 3.10.9), and the PDF is the paper's. The statistics hold the numbers that Appendix E.6 quotes for the two
  plotted arms.

## Not yet portable

`legacy_fig1b/` holds the renderer of the earlier draft's Figure 1(b), the endpoint summary of the headline cohort, which the
restructured draft no longer shows. It still reads the authors' working tree: the earlier figure folders, a 1HL bundle path
given by `ONEHL_BUNDLE`, and a receipt of its inputs. It is included for the record only.
