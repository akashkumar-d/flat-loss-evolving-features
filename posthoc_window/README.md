# The post-hoc window: every deep run rescored

The paper reports one rule that was fixed after all runs had been seen (Section 5.2; Appendices D.3 and F.4). The figures' grey
bands show its windows. This folder holds:
- the rule;
- the code that scores every run of the deep campaigns C–G with it (campaign G, the p = 101 runs, added on 2026-10-06, after
  the rule was fixed);
- the committed per-run results.

The 1HL cohorts are scored with the same function in `../paper_figures/scheme2.py`. Campaign C's pre-registered endpoint
(`../deep/campaign_C/`) is unchanged and remains the confirmatory result.

## The rule

**Window (loss only).**
- Start at the knee, the first logged state with head norm ‖V‖_F ≥ 0.99/λ_V. In the deep campaigns that state is on the
  500-update grid. In the 1HL runs it is at update resolution.
- After the knee, a stretch is non-falling if the cycle-mean training CE never drops below its running peak divided by 1.03.
  It may rise by any amount.
- The window is the non-falling stretch in which the CE climbs most (the largest maximum minus its value at the start). Ties
  go to the longer stretch.

**Start for runs without a head-norm log (1HL L2–L4).** These start instead at the CE minimum before their largest relative
CE climb (`dip_start` in `../paper_figures/scheme2.py`). On the 80 headline and L1 runs, which log both, that start is within
230 updates of the knee and gives the same window end and verdict (`../paper_figures/check_starts.py`).

**Pass.** Held-out accuracy or the train-only refit's held-out accuracy rises by at least 0.2 inside the window: the last
logged value inside minus the first. Final accuracy, training fit and ΔA_H are reported but not required.

**Refit.**
- The canonical train-only ridge readout, wherever a run logs it.
- Otherwise the runner's legacy refit: the depth-2 development runs and campaign E.

## Files

| File | What it does |
|---|---|
| `win.py` | O(n) non-falling stretches |
| `ovw.py` | run loading from the campaign archives (read in place), the per-run endpoint fields, helpers |
| `ovw2.py` | the window and pass rule, per-run records, and the seed-grid and raw-sheet figures |
| `make_all2.py` | rescores campaigns C, D, E, F and G, and writes `<out>/<campaign>/settings.jsonl` (one JSON line per setting or grid column) and the figures |
| `tables2.py` | the per-setting tables, from the `settings.jsonl` files |
| `windows_survey.py` | why the window is the stretch with the largest rise rather than the literal longest one: the verdicts under three choices |
| `results/{C,D,E,F,G}/settings.jsonl` | the committed per-run windows and verdicts. The paper's deep figures read these. |
| `results/tables.md`, `results/windows_survey.json` | the committed tables and survey |

## Reproduce

The commands read the campaign archives from the data folder (`../DATA.md`; set `PAPER_DATA`) and the campaigns' committed
`analysis.json` and manifests from `../deep/`.

```bash
python3 -B make_all2.py --out rescored --no-figs C D E F G   # settings.jsonl only; drop --no-figs for the figures
python3 -B tables2.py rescored C D E F G > tables.md
python3 -B windows_survey.py --out rescored                  # campaigns C-F
for c in C D E F G; do cmp rescored/$c/settings.jsonl results/$c/settings.jsonl; done
```

The four `settings.jsonl` files of campaigns C–F rebuild byte for byte (checked on 2026-10-05). Campaign F takes the longest,
a few minutes on one core. Campaign G's file was added on 2026-10-06; its runs' endpoint fields are computed with campaign C's
`endpoints.py`, as for the development runs.

**Totals.**

| Campaign | Runs passing |
|---|---|
| C | 204/224 |
| D | 115/127 |
| E | 0/24 |
| F | 170/170 |
| G | 30/30 |

The 20 failures in C are the normalized-gradient arm. The rule is not a plateau test. Inside the window the training CE
rises after the knee, and with p ≥ 47 some training pairs are misclassified. Appendix F.4 gives the numbers.
