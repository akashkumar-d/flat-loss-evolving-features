# Campaign I: hidden biases at the p = 31 settings (dotted curves of Figures 2 and 3)

The paper's networks are bias-free. Campaign I reruns the p = 31 runs of Figure 2 (one hidden layer) and of Figure 3
(depths 2 and 4) with a bias in every hidden layer, as a first, preliminary look at biases. The dotted curves of Figures 2
and 3 are the medians of these runs, and Appendix F.5 describes them. `PLAN.md` fixed the design before any evaluation run.

## The change

- **Model.** h^(ℓ) = ReLU(W^(ℓ) h^(ℓ−1) + b^(ℓ)) in every hidden layer; the head has no bias.
- **Update.** b^(ℓ) is one more column of W^(ℓ): [W^(ℓ) | b^(ℓ)] gets the supported Polar step and the decay of the paper's
  optimizer.
- **Start.** b = 0 and no random numbers are drawn for it, so each run starts from exactly the network of its bias-free
  counterpart, with the same split, rates, rate rule and stopping rule.

## Runs (60)

| family | runs | seeds (init, split) | bias-free counterparts |
|---|---|---|---|
| one hidden layer, width 512, λ_W = 0.015 | 10 | 5000 + i, 6000 + i, i = 1–10 | `H_p31_lw0.0150_s{i}` (campaigns D and F) |
| depth 2, width 256, r = 0.5 | 10 | 101 + i, 201 + i, i = 0–9 | `seedD2_r0.5_s{i}` (development runs and campaign F) |
| depth 4, width 256: pilot, rate recipe, main run | 40 chains | 1000 + i, 2000 + i, i = 1–40 | `C2_D4_s{i}`, `C1_D4_s{i}` (campaign C) |

`twins.json` lists every counterpart with its endpoint fields as scored in the bias-free campaigns (`make_twins.py`).

## Files

| file | what |
|---|---|
| `PLAN.md` | design, fixed before any evaluation run |
| `make_manifest.py` → `manifest.json` | the 60 jobs; each copies its counterpart's configuration and adds `bias: true` |
| `bias_runner.py` | the runner: the campaign runner of campaigns D, F and G plus the `bias` key; without it, the same computation |
| `tests/test_bias.py` | bias gradients against finite differences, the [W \| b] Polar step against an SVD, frozen biases equal to the bias-free run bit for bit, resume, the recipe, a chain |
| `endpoints_i.py` | campaign C's per-run endpoint, re-implemented for the analysis; it reproduces campaign C's `analysis.json` fields |
| `run_local.py` | runs the manifest on one machine (new for this release; the distributed scheduler is not included) |
| `HASHES.json` | SHA-256 of the files as the runs used them, written before the evaluation runs. The code files here differ from those files only in author fields (the copies marked anonymized in `../../PROVENANCE.tsv`) and in their removed comments and docstrings; that file lists both hashes. |
| `results/ARCHIVES.sha256` | SHA-256 of the three run archives (`../../DATA.md`, `deep/campaign_I/`) |

## Commands

```bash
python3 -I tests/test_bias.py                                # the runner's tests
python3 -I run_local.py --list                               # the 60 jobs and their estimated cost (about 91 core-hours)
python3 -I run_local.py --family H --jobs 4 --out runs       # the ten one-hidden-layer runs (about 4 minutes each)
python3 -I run_local.py --family D2 --jobs 4 --out runs
python3 -I run_local.py --family D4 --jobs 4 --out runs      # forty chains, about 2 core-hours each
```

The figures read the archived runs; `../../paper_figures/bias/README.md` gives the commands. Reruns on another numpy or LAPACK
build drift from the archived trajectories after many updates (`../README.md`, "How exact a rerun is").
