# Version 2 of the paper: figures, tables and numbers

This folder draws the figures of version 2 of the paper on arXiv that version 1 did not have or that version 2 redrew, and
recomputes the numbers of its new text. The scripts are the drafts' own (without comments and docstrings, and with the path set-up lines of three adapted,
`../../PROVENANCE.tsv`), plus `readout_freqs.py`, written for this release. Nothing is rescored: windows, passes and counts
come from the campaigns' committed analysis files, except the windows of Figure 3's depth-2 column (campaign QD), which
`fig23_v4.py` computes by the rule of the development runs' windows (`../bias/bias_figs.py`, `scored`).

## Inputs

- `DATA`: the paper's data folder (`../../DATA.md`), for Figures 1, 2, 3, 6, 7 and 8.
- `SHARED`: a folder holding `campaign_<X>/collected/results/<run>/run.json.gz` for campaigns N, X, Y, Z, ZP, Q, QD and QH
  (and V and W for the readout scan), and `campaign_ZM/runs/<run>/run.json.gz`. `../../campaigns/run_local.py` writes this
  layout inside `../../campaigns/`, which can then serve as SHARED (`../../campaigns/README.md`).
- `ANALYSIS_Q.json`: campaign Q's analysis. The committed one (`../../campaigns/campaign_Q/analysis/analysis_Q.json`) records
  each run by its path in the workspace where it was analyzed, and the scripts open the runs there. For your runs, rebuild it:
  `(cd ../../campaigns/campaign_Q && python3 -I analyze_Q.py ANALYSIS_Q.json SHARED)`.
- Python with numpy, matplotlib and pymupdf (for the centring step); `../../requirements.txt`.

## Figures

Commands from this folder. Give each driver its own new folder `OUT`: `paper_v4_figs.py` and `paper_v7_figs.py` both write a
`TF_acrossp_app.pdf`, and only `paper_v7_figs.py`'s is version 2's Figure 14. `expected/` holds the statistics each driver
writes (under the name given here) and the SHA-256 of the figure files as the paper prints them (`FIGURES.sha256`; Figures
1-5 after centring) and of Figures 1-5 before centring (`FIGURES_before_centring.sha256`).

| Figure | Command | Data | Writes, and its expected file |
|---|---|---|---|
| 1 | `python3 -I hero_row.py DATA/onehl/H OUT`, then the centring step | `onehl/H` | `H_hero_1x4.pdf`; `hero_row_stats.json` |
| 2, 3 | `python3 -I fig23_v4.py ../bias . --data BI/data --winv2 ../../posthoc_window/results --bias BI/bias --dev2 BI/dev2 --qd SHARED/campaign_QD/collected/results --out OUT --height 3.4`, then the centring step; BI is the folder that `PAPER_DATA=DATA python3 -I ../bias/prepare_data.py --out BI` fills | deep campaigns C, D, F, G and I; campaign QD | `fig2_acrossp_1hl_bias.pdf`, `fig3_depths_2to5_bias.pdf`; `bias_figs_stats.json` = `rev4_fig23_stats.json`, `SOURCES.json` (SHA-256 of the 300 run files read) = `rev4_fig23_SOURCES.json` |
| 4, 10-12 | `python3 -I paper_v4_figs.py ../../campaigns SHARED ANALYSIS_Q.json OUT`; Figure 4 then goes through the centring step | campaign N; for the two figures it also draws that version 2 does not show, campaign Z's and Q's cells | `N_D3_p47_main.pdf`, `N_D3_p61_app.pdf`, `N_D4_p47_app.pdf`, `N_D4_p61_app.pdf`; `paper_v4_figs_stats.json` = `rev4_paper_v4_figs_stats.json` |
| 5 | `python3 -I paper_v8_figs.py ../../campaigns SHARED ANALYSIS_Q.json OUT`, then the centring step | campaign Q's cells `P_p89`, `P_p113`, `B_base` (runs of ZP, Y, Z and Q) | `TF_main.pdf`; `paper_v8_figs_stats.json` = `rev8_paper_v8_figs_stats.json` |
| 1-5, centring | `mkdir -p OUTC && python3 -I rev12_center_figures.py OUTC OUT1/H_hero_1x4.pdf OUT2/fig2_acrossp_1hl_bias.pdf OUT2/fig3_depths_2to5_bias.pdf OUT4/N_D3_p47_main.pdf OUT5/TF_main.pdf > OUTC/centring.json` (OUT1 to OUT5: the folders of the commands above) | the five files | the five centred files, as printed; it prints its measurements, which `rev12_center_figures.json` holds with the paper's print scale (0.8826) added |
| 6 | `../render_fig2_v2.py` (version 1's Figure 4; `../README.md`) | `onehl/p23A_compact.tar.gz`, `onehl/p23B_compact.tar.gz` | `p23_confirmA.pdf`, `p23_confirmB.pdf`; `../expected/` |
| 7 | `PAPER_DATA=DATA python3 -B fig7_v20.py .. OUT 3.7` | `onehl/L1`, `onehl/p23B_compact.tar.gz` | `locked_cohorts_1hl.pdf`; `scheme2_stats.json` = `rev20_fig7_stats.json` |
| 8 | `python3 -B ../thm41_config_fig.py --out OUT --runs RUNS`, RUNS as below | campaign H's archive; campaign QH | `thm41_config_p101.pdf`; `thm41_config_stats.json` = `fig8_thm41_config_stats.json` |
| 9 | `python3 -B ../campaign_summary_core.py --out OUT` (version 1's Figure 8) | `../../deep/campaign_C/results/analysis.json` | `deep_campaign_summary.pdf`; `../expected/` |
| 13 | `python3 -I ../bias/bias_appendix_fig.py --analysis ../../deep/campaign_J/results/analysis.json --out OUT` (version 1's Figure 10) | the committed analysis | `bias_ablation_1hl.pdf`; `../bias/expected/` |
| 14, 15 | `python3 -I paper_v7_figs.py ../../campaigns SHARED ANALYSIS_Q.json OUT` | campaign Q's cells `P_p61_mf`, `P_p89`, `P_p113`, `P_p149`, `B_base`; for the contrast it also draws, campaign Z's | `TF_acrossp_app.pdf`, `TF_two_blocks_app.pdf`; `paper_v7_figs_stats.json` = `rev7_paper_v7_figs_stats.json` |
| 16, 17 | from `../../campaigns/campaign_Q/zm10`: `SHARED=<this release>/campaigns python3 -I build_zm10.py ANALYSIS_Q.json OUT` (SHARED here: where campaign ZM's code is) | campaign Q's cells `P_p61_mf`, `P_p61`, `P_p89`, `P_p113`, `P_p149` | `ZM_mirrorfree_paper.pdf`, `ZM_readouts.pdf`; `analysis_ZM10.json`, `readouts_ZM10.json` and `ZM_rise_times.json`, as committed in `zm10/` |

RUNS for Figure 8: campaign H's archive and campaign QH's four runs (seeds 7-10 of the eta_V = 3e-4 row), under the names the
script reads:

    mkdir RUNS && ln -s DATA/deep/campaign_H/*.tar RUNS/
    for d in SHARED/campaign_QH/collected/results/QH_T41_*; do ln -s "$d" "RUNS/$(basename "$d" | sed 's/^QH_//')"; done

`paper_v4_figs.py` and `paper_v7_figs.py` also draw figures that version 2 no longer shows (the one-block contrast of three
update rules, `TF_contrast_main`, and revision 4's `TF_acrossp_app`).

## Tables

| Table | Source |
|---|---|
| 1 | version 1's Table 1: `../../onehl/figures/render_hero.py` (first row) and `render_all.py` (endpoint tables) |
| 2 | version 1's Table 2: `../../deep/campaign_C/analyze.py report` |
| 3 | campaign N: the cells of `../../campaigns/campaign_N/analysis/analysis_N.json` (`analyze_N.py`) |
| 4 | campaign Z's analysis (`analysis_Z.json`), cells `Y_G2` (gain 2 to 1), `X_R5` (no gain), `R_G1.5_fixed`, `R_G2_fixed`, `R_G3_fixed` |
| 5 | campaign Z's cells `R_G1.5_*`, `R_G2_*` and `R_G3_*` (tau from 2,500 to 20,000, and fixed); the g0 = 2, tau = 5,000 entry: the first three runs of `Y_G2`. The text's one-at-a-time changes: cells `R_lam_v*`, `R_eta_w*`, `R_lam_w*`, `R_frac0.3` |
| 6 | campaign Q's cells `B_base`, `B_F12`, `B_V12`, `B_E12` (`analysis_Q.json`); `rev3_stats.py` prints the rows |
| 7 | campaign ZP's analysis (`analysis_ZP.json`), which scores both sets of runs: campaign Z's (`P_p61_f0.3_*`, `P_p89_f0.2_*`, `P_p149_f0.2_*`) and ZP's; the rows with 10 runs: campaign Q's cells `P_p61`, `P_p61_mf`, `P_p89`, `P_p149`, and `Y_G2` (p = 113) |
| 8 | `readouts_ZM10.json` (Figures 16 and 17's command), every row: readouts `paper refit`, `Fourier refit top 8`, `Fourier refit top 16`, `net logits top 8` and `logistic probe` |

## Numbers in the text

- `rev3_stats.py ANALYSIS_Q.json ANALYSIS_ZM10.json READOUTS_ZM10.json ../../campaigns/campaign_QD/analysis/analysis_QD.json
  FIG8_STATS.json OUT.json` (FIG8_STATS: Figure 8's statistics): the counts, crossing times, flips and frequencies of
  Section 5.3 and Appendices D.3-D.5 for campaign Q's cells and the two mirror-pair settings, depth 2 (campaign QD, Appendix
  C.3) and Figure 8 (Appendix B.6). Expected: `rev3_stats.json`. Its section headings use the numbers of the draft of
  13 October: its "Table 6" rows are Table 7's rows with 10 runs, its "Figure 14", "Figure 15" and "Table 7" are Figures 16
  and 17 and Table 8.
- `rev7_refit_lags.py ../../campaigns/campaign_Z/analysis/analysis_Z.json ANALYSIS_Q.json
  ../../campaigns/campaign_ZP/analysis/analysis_ZP.json ../../campaigns/campaign_ZM/analysis/analysis_ZM.json > OUT.json`:
  how far the ridge refit trails the transformer (Section 5.3, Appendix D.5). Expected: `rev7_refit_lags.json`. Its key
  `figure5` is the earlier one-block contrast of three update rules; the lags of version 2's Figure 5 are under
  `across_primes` and `two_blocks`.
- `readout_freqs.py ANALYSIS_Q.json OUT.json`: the frequencies the head input carries at the network's held-out 0.5 crossing
  (6 to 13, medians 7 to 9) and the held-out R^2 of the frequencies outside the top 8 (medians -0.27 to -0.79), per setting
  of Table 8 (Appendix D.5). Expected: `readout_freqs.json`.
- `../../campaigns/readouts/refit_lag`: Appendix D.5's one-hidden-layer replays (`kcurve/F_H_p31_s10.json`,
  `kcurve/H_p61_s1.json`, `kcurve/H_p101_s7.json`, summarized in `kcurve/summary.json`) and transformer replays
  (`figs/plot_cases_stats.json`, which `plot_cases.py` writes from the replays' `lag_*.json`: the training and held-out R^2
  of the frequencies outside the top 8, 0.38 and 0.53, -0.42 and -0.49); its `NOTE.md`.
- `../../campaigns/readouts/margin_vs_AH`: Section 2's A_H = 2/5.
- The campaigns' analysis files: Appendix C.4 (`campaign_N`), C.3 (`campaign_QD`), B.6 (`campaign_QH`) and D (`campaign_Z`,
  `campaign_ZP`, `campaign_ZM`, `campaign_Q`).

## How exact

Checked for this release with Python 3.11.15, numpy 2.4.4, matplotlib 3.10.9, pymupdf 1.28.2 and torch 2.5.1, from the
run records:
- Every figure file of `expected/FIGURES.sha256` and `expected/FIGURES_before_centring.sha256` rebuilt byte for byte, except
  Figure 6, identical apart from the PDF creation date (as in version 1), and Figure 9, drawn with matplotlib 3.7.1: with
  3.10.9 its plotted values are identical (its own receipt compares them) and a few text labels move by less than 0.4 pt.
- Every statistics file in `expected/` rebuilt byte for byte, except `rev12_center_figures.json`, whose measurements rebuilt
  identically (the script prints them with the paths it was given; the file adds the print scale).
- The committed analyses rebuilt from the run records: byte for byte those of campaigns N, Q, QD and QH (`analysis_N.json`,
  `analysis_Q.json`, `analysis_QD.json`, `analysis_QH.json`, and `zm10/analysis_ZM10.json`, `readouts_ZM10.json`,
  `ZM_rise_times.json`); those of campaigns Z, ZP and ZM identically apart from the run paths they record, which name the
  folders they were given.
- In `../../campaigns/readouts/refit_lag`, `analysis_lag.json`, `kcurve/summary.json`, `kcurve/val_summary.json` and
  `figs/plot_cases_stats.json` rebuilt byte for byte from the included outputs, and `scan_runs.json` and `k_rules.json` from
  the run records.
Byte equality of rendered files depends on the plotting environment.

## Figure numbers of version 1

Version 1's numbers, which the other READMEs of this release use, map to
version 2's as follows: Figures 1, 2 and 3 keep their numbers, Figure 4 is Figure 6, 5 is 7, 7 is 8, 8 is 9 and 10 is 13;
version 1's Figures 6 and 9 are not in version 2 (their counts are in the text of Appendices B.4 and C.3); Tables 1 and 2 keep
their numbers. Appendices D, E and F of version 1 are A, B and C of version 2 (F.4 is C.5, F.5 is C.6), and its proofs,
Appendices A, B and C, are E, F and G.
