# Earlier one-hidden-layer runs that the paper re-analyzes

Two cohorts in Appendices E.4 and E.5 were trained before the paper's protocol existed. The paper re-analyzes their saved
logs, and for the first cohort also its saved states. This folder holds their original training code, with each folder's
own relative layout kept so that the original commands still work.

## Fixed-recipe sweep (L2): `fixed_recipe_sweep/`

**The runs.** 120 runs: p = 11, 13, 17, 19, 23 and 29, with 20 fresh initialization/split pairs each.
- Width 512, a random 80% split, c = 24.
- η_W = η_V = 0.03, λ_W = 0.005, λ_V = 0.2.
- 40,000 updates each.

`SCOPE.md` states the learner, the roster and the logged quantities. It was fixed before launch.

**The code.**
- `multip_seed_signed_relu_20_v1/batch_config.py` holds the roster. For the zero-based roster index i, init seed = 2026091301 + i
  and split seed = 2026191301 + i.
- `train_stage.py` trains the roster with the learner of `multip_seed_signed_relu_v1/core.py`, a frozen earlier copy.
- `audit_stage.py` checks the outputs independently. `postprocess.py` draws its figures.
- `pipeline.py` runs the three stages in order, with no automatic restart.
- Before their comments and docstrings were removed, the files here had the SHA-256 recorded in the sweep's launch record
  (`../PROVENANCE.tsv`, `origin_sha256`).

```bash
cd fixed_recipe_sweep/multip_seed_signed_relu_20_v1
python3 pipeline.py          # training -> independent audit -> figures; writes p<p>_seed<k>/ run folders here
```

**How the paper reads them.**
- Training CE and accuracies are logged every update.
- Weights are saved every 1000 updates. The paper's observers (A_H, the train-only refit, AGOP) are recomputed from those
  states by `../onehl/fixed_recipe/run_reobserve_c24.sh`.
- `../onehl/tools/compact_other.py line2` writes the compact extracts.
- The declared window [15000, 20000] comes from this sweep.

## Quadratic-activation contrast (L4): `quadratic_contrast/`

**The runs.** 40 locked initialization/split pairs at p = 13.
- Width 64, 135 training pairs.
- Square activation, row-sphere feature geometry.
- Feature step 5·10⁻⁴, head step 0.5, head decay 10⁻³.
- Logged every 500 updates, with their own observers.

**The code.**
- `standard_modarith_1hl_muon_v1/src/run_observer_manifest.py` trains the four 10-pair stages of
  `outputs/manyseed_direction_confirmation_v1/locked_manifest.json`.
- `REPRODUCTION.md` gives the commands.
- `RUN_SOURCE_LOCK.sha256` records the hashes of the training sources at launch.

**One gap.** Before the removal of comments, the runner and the manifest matched their recorded hashes. The learner file `src/modarith_1hl.py` (and its test)
does not: the version that trained these runs (sha256 `6abcbf4a…`) was later edited and is not preserved. The file here is
that later revision. The paper uses only these runs' logged trajectories and their own observers. It does not retrain them,
and their saved logs are the evidence.

**How the paper reads them.** `../onehl/tools/compact_other.py line4` writes the compact extracts from each run folder.
