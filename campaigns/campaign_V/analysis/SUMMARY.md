# Campaign V: results (claude/1hl-figures, 2026-10-09)

50 of 50 runs finished on 11 sessions (no failures, no push errors). The results were read from the session branches
(through Akash's browser; collect_results.sh had not written `campaign_V/collected/` in this folder) and laid out as
collect_results.sh does (`collected/results`, `collected/status`, `collected/INDEX.json`); every run file matches the
SHA-256 in its `chain.json`. Rule and analysis as fixed in PLAN.md before the runs.

The key message (Akash, 2026-10-08): in a flat or rising training loss, the refit gets better, and either within the same
window or later the held-out accuracy goes up.

## 1. Part A: confirmation on fresh seeds (pre-registered rule, `confirm_V.py`)

| setting (frac 0.3, seeds 6-15) | rule (ridge refit) | probe | without the training-fit condition | raw training acc. >= 0.97 | verdict |
|---|---|---|---|---|---|
| C1: 1 layer, lam_w 0.02, eta_w 0.1, lam_v 0.2 | **10/10** | 10/10 | 10/10 | 10/10 | **confirmed** |
| C2: 2 layers, lam_w 0.01, eta_w 0.1, lam_v 1 | **2/10** | 2/10 | 2/10 | 0/10 | **not confirmed** |

**C1.** The training set is fit by update 540-560; the CE then sits at its cap level, about 3e-7 (5e-8 to 9e-8 of log p),
while the held-out accuracy rises (0.5 at updates 3400-6500, 0.99 at 5000-9100) and the ridge refit with it (gains
0.33-0.95 inside the passing stretches, at windows of 500 or 2000 updates). Training accuracy is 1.000 throughout.
Sensitivity (not deciding): 10/10 at delta 0.2, 5/10 at delta 0.05 (the CE is flat to within 10% but not 5%).

**C2.** All 8 failures fail on the final held-out accuracy: on these seeds the held-out rise came later than on the
development seeds (held-out 0.5 at updates 13700-38900, or not by 40000) and did not finish within the 40000-update horizon.
Final held-out (median over the last 2000 updates): 0.999 and 0.992 (the two passes, seeds 6 and 13), 0.97, 0.96, 0.84,
0.48, 0.32, 0.14, 0.10, 0.05. The other conditions (CE flat to 10% at w = 2000, training accuracy averaged over 500 updates
>= 0.97, refit +0.2 inside the stretch) held in 7/10 seeds. When the rise comes, the refit and the held-out accuracy move
together (within 0.7k updates), and the 500-update CE drifts up slightly (about 0.003 -> 0.005 log p).

## 2. Part B: the final RMS division at training fraction 0.2 (exploration, seeds 1-3)

Question (PLAN.md): does the training fit come first, with the CE then flat at 0.5-0.8 log p while the refit and the
held-out accuracy rise? "Fit first" below: the held-out accuracy is <= 0.25 when the 500-update average of the training
accuracy first reaches 0.97.

| setting | rule | CE / log p in the stretch | fit first | final held-out | what happens |
|---|---|---|---|---|---|
| 1 layer, lam_v 0.5, eta_w 0.03 | **3/3** | 0.58-0.59 | 2/3 | 0.998-0.999 | fit, then held-out 0.2 -> 1.0 at a flat CE |
| 1 layer, lam_v 0.5, eta_w 0.1 | **3/3** | 0.59 | 2/3 | 0.993-0.999 | the same, faster |
| 1 layer, lam_v 0.75 (eta_w 0.03, 0.1) | 0/6 | 0.72 | 1/6 | 0.87-0.99 | held-out rises before the training fit |
| 1 layer, lam_v 1 (eta_w 0.03, 0.1) | 0/6 | 0.79 | 0/6 | 0.90-0.98 | the same; training acc. reaches 0.99 late or never |
| 2 layers, lam_v 0.5 | 0/3 | 0.57 | **3/3** | 0.92-0.95 | fit first, refit leads; unstable after about 24k |
| 2 layers, lam_v 1 | 0/3 | 0.78 | - | 0.25-0.37 | no generalization in 40000 updates |
| AdamW, 1 layer | 0/3 | - | - | 0.01-0.16 | fits by update 350; no generalization in 60000 |
| AdamW, 1 layer, RMS | 0/3 | (no flat stretch) | - | 1.00 | generalizes at 4-6k while its CE falls 1.4-3.4x |

**The high-loss plateau after the training fit exists: one layer, lam_v 0.5.** In all six runs the CE stays between 0.57 and 0.65
log p from the training fit on (stretch averages 0.58-0.59, about 2.8 nats) while the ridge refit gains 0.55-0.83 and the held-out accuracy reaches 0.99; the strict rule passes
6/6, also with the logistic probe, at delta 0.05, 0.1 and 0.2, and campaign U's key message holds at every averaging window
from the raw CE to 5000 updates (with training accuracy >= 0.97 at every logged step). In 4 of the 6 the order is clean:
the training accuracy (500-update average) reaches 0.97 at updates 920-4000 while the held-out accuracy is still 0.12-0.18,
and the held-out accuracy reaches 0.5 1200-4600 updates later; between the fit and the held-out 0.9 crossing the CE falls
by 2-7%. Figure: `V_highloss` (first two columns).

Two caveats. (i) Here the ridge refit trails the held-out accuracy (by 800-3400 updates at the 0.5 crossing); the logistic
probe tracks the held-out accuracy. (ii) The level is set by lam_v (0.5 -> 0.58-0.62, 0.75 -> 0.72, 1 -> 0.79 log p), and
above lam_v 0.5 the order reverses: the held-out accuracy rises first and the training set is fit late or not at all.

**Two layers, lam_v 0.5** has the cleanest order (training fit while the held-out accuracy is 0.04-0.15, the refit leading
the held-out accuracy by 650-1300 updates at 0.5 and 1000-2200 at 0.9, CE 0.56-0.62 log p), but after about 24000 updates
the held-out accuracy becomes unstable (drops to 0.2-0.9 at single updates), so the final median is 0.92-0.95 and the
smoothed training accuracy dips to 0.94-0.97: the rule fails.

**AdamW at fraction 0.2**: without the RMS division it memorizes and does not generalize in 60000 updates; with it, it
generalizes at updates 4000-6000 while the 500-update CE falls from about 0.007 to 0.003 log p, so there is no flat stretch.

## 3. What it shows

1. The one-layer low-CE example is confirmed (10/10 fresh seeds): a fit training set, a CE pinned at its cap (about 3e-7),
   and a refit and held-out rise inside the flat stretch.
2. The two-layer example (CE about 2e-2) is not confirmed: on fresh seeds the held-out rise often starts after 20000-40000
   updates and was unfinished at the pre-registered horizon. The flat, fit, refit-rising stretch itself was present in 7/10.
3. At a high CE the key message holds with the training fit first: RMS head, one layer, lam_v 0.5, training fraction 0.2,
   CE about 0.6 log p, 6/6 runs (4/6 with a clean fit-then-rise order). This is an exploratory result on 3 seeds per setting.
4. AdamW shows no such stretch at fraction 0.2 either.

## 4. Files

- `confirm_V.json`: the rule's output per run (both readouts, both windows, the best stretch and its numbers);
  `confirm_V_delta_sensitivity.json`: counts at delta 0.05, 0.1, 0.2.
- `runs_V.json` (`analyze_V.py`): per run, the order of events (training accuracy 0.97 and 0.99, held-out 0.1/0.5/0.9/0.99,
  refit 0.5/0.9), the CE / log p at those updates, the best stretch, and the first failing condition.
- `key_message_by_window_V_d0.1.json` (`flat_windows.py`): campaign U's key message at windows 1-10000, both readouts.
- Figures: `V_scoreboard` (rule and window counts per setting), `V_events` (order of events, every run), `V_highloss`
  (four runs at fraction 0.2), `settings/V_dyn_<setting>` (every seed: CE with averages, accuracies, best stretch).
- Code: `confirm_V.py` (unchanged since the plan), `analyze_V.py`, `plot_V.py`, `flat_windows.py` (campaign U's).

## 5. Possible next steps

- Confirm the high-loss example (RMS head, one layer, lam_v 0.5, frac 0.2, eta_w 0.03 or 0.1) on 10 fresh seeds with this
  rule fixed beforehand, adding "held-out <= 0.25 when the training accuracy reaches 0.97" as the fit-first condition.
- C2 at a longer horizon (for example 100000 updates) on the same fresh seeds, to see whether the late rises complete.
- Two layers, lam_v 0.5 at fraction 0.2 with a smaller hidden step after the rise, to remove the late instability.
