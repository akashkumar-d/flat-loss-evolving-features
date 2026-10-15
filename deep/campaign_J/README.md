# Campaign J: three other ways of updating the hidden bias (Figure 10)

A follow-up to campaign I at one hidden layer and p = 31, on campaign I's ten seeds. Each run copies campaign I's
configuration for the same seeds and changes only how the bias is updated. Appendix F.5 describes the runs, Figure 10 shows
them, and the paper treats them as a preliminary experiment. `PLAN.md` fixed the design and the measures before any
evaluation run.

| arm | key | the bias is | runs |
|---|---|---|---|
| K | `bias_after_knee: frozen` | updated with the joint [W \| b] step until the knee, then held fixed (no step, no decay) | 10 |
| KD | `bias_after_knee: decay` | updated with the joint step until the knee, then only decayed | 10 |
| S | `bias_update: separate` | updated from the start by its own normalized step, b ← (1 − ηλ)b + η g/(‖g‖√r), g = −∇_b L, r the rank of W's step | 10 |

From the knee on, W takes the bias-free step of the paper's optimizer in K and KD, and in S from the start. The two
references of Figure 10 are campaign I's runs (the joint step) and the bias-free runs with the same seeds (campaigns D and F);
`comparison.json` lists them with the SHA-256 of their run files.

## Files

| file | what |
|---|---|
| `PLAN.md` | design and measures, fixed before any evaluation run |
| `make_manifest.py` → `manifest.json` | the 30 jobs |
| `bias_runner.py` | campaign I's runner plus the keys `bias_update` and `bias_after_knee`; without them, campaign I's computation |
| `tests/test_bias.py`, `tests/test_j.py` | campaign I's tests, and seven new ones: equality with campaign I's runner bit for bit, the separate step against an SVD, the switch at the knee, KD against the bias-free run from the folded network, resume, and invalid key combinations |
| `make_comparison.py` → `comparison.json` | the paired references |
| `run_local.py` | runs the manifest on one machine, a few runs at a time |
| `analyze_J.py` | the analysis: campaign C's per-run endpoint (`endpoints_i.py`) and the fall of the cycle-mean training CE from its post-knee peak to held-out t₉₀ |
| `lock.py` → `HASHES.json` | SHA-256 of the files above, written before the evaluation runs. The code files here differ from those files only in author fields (the copies marked anonymized in `../../PROVENANCE.tsv`) and in their removed comments and docstrings; that file lists both hashes. |
| `results/analysis.json` | the analysis of the 30 runs and their references; Figure 10 is drawn from it |
| `results/ARCHIVES.sha256` | SHA-256 of the run archive (`../../DATA.md`, `deep/campaign_J/`) |

## Commands

```bash
python3 -I tests/test_bias.py && python3 -I tests/test_j.py     # test_j.py also reads ../campaign_I/bias_runner.py
python3 -I run_local.py --jobs 2 --out runs                     # all 30 runs, about 4.4 core-hours; --arm K|KD|S for one arm

# Rebuild results/analysis.json from the archives (byte for byte):
python3 -I analyze_J.py report --out /tmp/J --from $PAPER_DATA/deep/campaign_J/campaignJ.tar \
  $PAPER_DATA/deep/campaign_I/campaignI_H.tar $PAPER_DATA/deep/campaign_D/campaignD_H.tar $PAPER_DATA/deep/campaign_F/campaignF_H.tar
cmp /tmp/J/analysis.json results/analysis.json
```

`analyze_J.py` checks every run file against its SHA-256 (`comparison.json`, and `SHA256SUMS_runs` next to the campaign J
archive). It also writes `ANALYSIS.md`, a table of the preregistered measures. Figure 10:
`../../paper_figures/bias/bias_appendix_fig.py`.
