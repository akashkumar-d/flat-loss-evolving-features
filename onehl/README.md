# One-hidden-layer experiments

This folder holds the code behind the paper's one-hidden-layer (1HL) results:
- **The headline confirmation and the search that chose it.** Section 5.1, Figure 1, Appendices E.1 and E.2, and Table 1.
  The recipe search is in Appendix E.1 (dev1, dev2) and under "Other settings" in Appendix E.5 (dev3).
- **The pre-registered p = 23 cohorts.** Figures 4 and 5(a), Appendix E.3.
- **The per-prime confirmations and their development sweep (L1).** Figure 5(b–d), Appendix E.4, Table 1.
- **The observers recomputed on the fixed-recipe sweep (L2).** Figure 6, Appendix E.4. The training code of those earlier
  runs is in `../onehl_earlier/fixed_recipe_sweep/`.
- **The replays with held-out-defined windows (L3).** Appendix E.5, Table 1.
- **The diagnostics** quoted in Appendix E.2 ("Norms and energies" and the suffix windows).

The figures in the paper are drawn by `../paper_figures/` from the same per-run extracts. The renderers in `figures/` and
`figure2/` draw earlier versions of those figures. They also compute the endpoint tables that the text and Table 1 quote.

**Requirements.**
- Python 3.10 or newer, with numpy, scipy and matplotlib. Nothing else is needed.
- CPU only. All training is full-batch float64.
- The runner trains at most 8 runs of a manifest in parallel, with one BLAS thread per run.

## Layout

| Folder | Contents |
|---|---|
| `src/` | **Runner and model.** `run_campaign.py` trains the runs listed in a JSON manifest. It logs CE and accuracies every update, and observers every 100 or 500 updates. `model.py` defines the task, the split, the bias-free 1HL ReLU network and the exact update: a polar feature step (relative singular-value cutoff 1e-10), a Frobenius-normalized head step, and decoupled decays. `observers.py` computes A_H and the energies, class-centered AGOP purity against eight fixed random frames, and the train-only ridge refit. `endpoints.py` computes the window endpoints and the strict joint endpoint. `postprocess.py`, `plotting.py` and `legacy_core.py` are helpers. |
| `headline/` | **Recipe search and lock.** `dev1`–`dev3` hold the grids (`SWEEP.json`), the run manifests and the selection rules, each declared before its launch: 96, 48 and 44 runs. `select_hero.py` implements the rule. `make_confirm.py` locks the selected recipe and window, then draws 20 fresh initialization/split pairs with `numpy.random.default_rng(20260930)`. `confirm/` holds the executed lock file and manifests. `dev3` used `dev3/model_qrsplit.py` in place of `src/model.py`; the only addition is the quadratic-residue split. |
| `per_prime_development/` | **The L1 development sweep.** 54 runs: 18 recipes × 3 pairs at p = 11, 13, 17, 23, 29 and 61. `anchor_manifest.json` (18 runs) and `candidate_manifest.json` (36 runs) were written before launch, with the protocol `PROTOCOL_SOURCE_v1.md` and `prepare_campaign.py`. `candidate_manifest.json` records the source hashes at preparation. The learner files (`model.py`, `observers.py`, `endpoints.py`, `legacy_core.py`, `run_campaign.py`) equal those in `src/`. The plotting and post-processing helpers were earlier versions. |
| `per_prime/` | Locked manifest of the per-prime confirmations (`focus_manifest.json`: three recipes, 20 fresh pairs each, 50k updates) and its 8-shard split. |
| `fixed_recipe/` | Observers recomputed from the saved states of the 120 fixed-recipe runs. |
| `replays/` | Learner of the replayed runs (`natplat_server.py`), the replay with state saving (`natplat_states.py`), the job list, and the check against the original runs. |
| `tools/` | Per-run compact extracts (`compact_line1.py`, `compact_other.py`) and observers on replay states (`reobserve_states.py`). |
| `figures/` | Plotting and endpoint module (`hl1_figs.py`). `render_hero.py` draws the earlier headline figures and writes `H_endpoints.json`. `render_all.py` draws the earlier per-prime, fixed-recipe, replay and quadratic figure set and writes the endpoint tables. `dev_fig.py` draws the search overviews. |
| `diagnostics/` | `h_diagnostics.py`: norms, logit scale, energy ratios and loss-only suffix windows of the 20 headline runs. The "Norms and energies" and "Retrospective suffix windows" paragraphs of Appendix E.2 come from it. |
| `figure2/` | `render_fig2.py`: the earlier draft's Figures 2 (phase grid at p = 23, 29, 31), 6 (the three L1 recipes) and 5 (the p = 23 cohorts). The statistics quoted in the captions are in the included `*_stats.json`. The paper's Figure 4 is drawn by `../paper_figures/render_fig2_v2.py`, which changes only the refit's colour and line style. |
| `p23_tests/` | **Pre-registered p = 23 tests (Appendix E.3).** `scan_dev_recipes.py` screens the development recipes. `select_window.py` is the window rule used; its first version, `select_window_first_version.py`, admitted no window. `CONFIRM_LOCK_p23.json` and `CONFIRM_LOCK_p23b.json` are the two locks, each written before its launch, with the manifests in `manifests/`. `score_p23.py` and `score_p23b.py` score them exactly as declared, and `results/` holds the scores. `make_p23_fragment.py` recomputes every number quoted in Appendix E.3. |

## Reproduce

Run everything from this folder. `<data>` is the data folder of `../DATA.md`.

**1. Headline recipe search.** This step uses held-out accuracy by design; see Appendix E.1.

```bash
for k in 00 01 02 03 04 05; do
  python src/run_campaign.py --manifest headline/dev2/manifests/shard_$k.json --output runs/dev2/shard_$k
done
python tools/compact_line1.py compact/dev2 runs/dev2/shard_*/h*_*
python headline/select_hero.py compact/dev2 selection_dev2.json   # compare with headline/dev2/out/selection.json
```

**2. Lock and confirmation (20 fresh pairs).** `make_confirm.py` regenerates `headline/confirm/manifests` exactly; only the
lock time differs.

```bash
python headline/make_confirm.py headline/dev2 confirm_rerun
for k in 00 01 02; do
  python src/run_campaign.py --manifest headline/confirm/manifests/shard_$k.json --output runs/h/shard_$k
done
python tools/compact_line1.py compact/h runs/h/shard_*/h*_*
```

**3. Per-prime development sweep and confirmations.**

```bash
python src/run_campaign.py --manifest per_prime_development/anchor_manifest.json --output runs/l1_dev/anchor
python src/run_campaign.py --manifest per_prime_development/candidate_manifest.json --output runs/l1_dev/candidate
for k in 0 1 2 3 4 5 6 7; do
  python src/run_campaign.py --manifest per_prime/shards/shard_$k.json --output runs/l1/shard_$k
done
python tools/compact_line1.py compact/l1 runs/l1/shard_*/p*_*
```

**4. Replays with held-out-defined windows.**

```bash
python replays/natplat_states.py replays/jobs_1hl_seed.json runs/l3 --workers 10 --every 500
mkdir -p obs/l3
# for each replay <name>:
python tools/reobserve_states.py runs/l3/<name>_states.npz runs/l3/<name>.json.gz obs/l3/<name>.jsonl
python tools/compact_other.py line3 runs/l3/<name>.json.gz obs/l3/<name>.jsonl compact/l3
```

**5. Fixed-recipe runs.** Train them with `../onehl_earlier/fixed_recipe_sweep/`. The observers are then recomputed on the
runs' saved states:

```bash
bash fixed_recipe/run_reobserve_c24.sh <folder of the saved runs> obs/l2
python tools/compact_other.py line2 <run_dir> obs/l2/<run>.jsonl compact/l2
```

**6. Diagnostics and endpoint tables.** No training is needed.

```bash
python diagnostics/h_diagnostics.py <data>/onehl/H h_diagnostics.json        # Appendix E.2: "Norms and energies", suffix windows
python figures/render_hero.py <data>/onehl/H figs                             # H_endpoints.json (Table 1, first row)
python figures/render_all.py <data_root> figs   # data_root/{l1_confirm,l1_dev,l2,l3,l4} -> the compact folders; endpoint tables
```

**7. Pre-registered p = 23 tests (Appendix E.3).** The development runs are those of step 1 (recipe
`h23_tf0p8_lw0p015_ev0p03_lv0p2`). Each confirmation trains 20 pairs for 20k updates.

```bash
SCAN_OUT=scan_rows.json python p23_tests/scan_dev_recipes.py compact/dev2
python p23_tests/select_window.py compact/dev2 h23_tf0p8_lw0p015_ev0p03_lv0p2 window.json   # [6500, 16500]
cd p23_tests
python ../src/run_campaign.py --manifest manifests/p23_confirm.json --output ../runs/p23A
python ../src/run_campaign.py --manifest manifests/p23_confirmB.json --output ../runs/p23B
python ../tools/compact_line1.py ../compact/p23A ../runs/p23A/h23_*
python ../tools/compact_line1.py ../compact/p23B ../runs/p23B/h23_*
python score_p23.py ../compact/p23A CONFIRM_LOCK_p23.json score_A.json      # compare with results/score_A_locked.json
python score_p23b.py ../compact/p23B CONFIRM_LOCK_p23b.json score_B.json     # compare with results/score_B_locked.json
python make_p23_fragment.py --a ../compact/p23A score_A.json --b ../compact/p23B score_B.json \
  --out e4.tex --numbers e4_numbers.json
```

Every reported 1HL number and figure is computed from per-run compact extracts. `../DATA.md` lists them. With them, steps 6
and 7's scoring and `../paper_figures/` need no training.
