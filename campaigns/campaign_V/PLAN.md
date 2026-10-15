# Campaign V: confirmation of the two cleanest flat-loss settings, and the RMS head at training fraction 0.2

claude/1hl-figures, 2026-10-09, at Akash's request after campaign U ("Let's try both"): (A) confirm the two cleanest
settings on fresh seeds, with the strict rule judged on smoothed training accuracy so loss spikes do not decide it;
(B) run the RMS head at a smaller training fraction (0.2), aiming for a high-loss plateau that comes after the training fit.

The key message (Akash, 2026-10-08): in a flat or rising training loss, the refit gets better, and either within the same
window or later the held-out accuracy goes up.

## 1. Background (campaign U, analysis/SUMMARY.md there)

- One layer, lam_w 0.02, eta_w 0.1, lam_v 0.2: the key message held in 3/3 seeds with the training set fit throughout (CE
  about 3e-7).
- Two layers, lam_w 0.01, eta_w 0.1, lam_v 1: held in 3/3 seeds at CE about 2e-2 (4 of 5 with campaign T), but training
  accuracy dips to 0.73-0.91 at single loss spikes, so the strict (3% training error at every logged step) version failed.
- With the final RMS division (lam_v 0.5-1, frac 0.3) the CE sits at 0.6-0.8 log p, but the training fit happens inside the
  flat stretch: held-out and training accuracy rise together, so the strict version held in 1 of 27 runs.

## 2. Part A: confirmation (fixed before the runs)

Settings, ten fresh seeds each (i = 6-15; seeds 1-5 were development):
- C1: one layer, lam_w 0.02, eta_w 0.1, lam_v 0.2, eta_v 0.03, frac 0.3, campaign-T layout; at most 30000 updates.
- C2: two layers, lam_w 0.01, eta_w 0.1, lam_v 1, eta_v 0.03, frac 0.3, campaign-T layout; at most 40000 updates.
Both stop once the held-out accuracy has stayed >= 0.99 for max(0.5 t, 5000) updates after first reaching it at t.

Rule (confirm_V.py; its docstring is the definition). A run passes if, for w = 500 or w = 2000 updates, there is a stretch
[a, b] with:
1. flat or rising loss: the centred geometric moving average of the training CE over w updates never falls more than 10%
   below its running maximum since a, and b - a >= 2w;
2. the training set stays fit: the training accuracy averaged over 500 updates is >= 0.97 at every logged step of [a, b];
3. the ridge refit's held-out accuracy rises by >= 0.2 inside [a, b];
4. the held-out accuracy rises by >= 0.2 from a to the end of the run, and its final value (median over the last 2000
   updates) is >= 0.99.

A setting is confirmed if at least 8 of its 10 fresh seeds pass. Secondary, reported but not deciding: the same rule with
the logistic probe instead of the ridge refit; campaign U's two earlier versions at the same windows.

On the development seeds (campaign T seeds 1-2 and campaign U seeds 3-5) this rule gives: C1 5/5; C2 4/5 (campaign T's seed
2 never generalized within 60000 updates). The smoothed training-accuracy minimum in C2's passing stretches was 0.975-0.980,
close to the 0.97 threshold.

## 3. Part B: exploration, the RMS head at training fraction 0.2 (seeds 1-3)

- E1: one layer, final RMS division, lam_w 0.01, frac 0.2: lam_v 0.5, 0.75, 1, each with eta_w 0.03 and 0.1; at most
  60000 updates.
- E2: two layers, same, eta_w 0.1, lam_v 0.5 and 1; at most 40000 updates.
- A1 (control): AdamW (lr 1e-3, betas 0.9/0.98, wd 1) at frac 0.2, one layer, with and without the RMS division; at most
  60000 updates, continuing 20000 updates past generalization.

Question: at frac 0.2, does the training fit come first, with the CE then flat at 0.5-0.8 log p while the refit and the
held-out accuracy rise? Analysis: confirm_V.py's rule (and its pass3 / loose versions), flat_windows.py at all windows, and
for each run the order of events (updates at which training accuracy reaches 0.99 and held-out accuracy reaches 0.5 and 0.9).
Pilot (this container, seed 1, one layer, cut at 4000 updates; not part of the results). lam_v 0.5, eta_w 0.1: the training
set is fit first (training accuracy 0.95 at update 500, 0.99 at 1500) while the held-out accuracy stays at 0.16-0.22; then,
from update 1500 to 3000, held-out 0.22 -> 0.99, ridge refit 0.22 -> 0.84, probe 0.25 -> 1.0, while the CE goes 0.634 ->
0.601 log p (-5%). On the partial run the confirmation rule finds the stretch [1000, 3999] (CE 2.9 = 0.61 log p, refit +0.74,
held-out +0.81) and fails only on the final held-out median (0.989, the run was cut during the rise). lam_v 1, eta_w 0.03:
training accuracy 0.89-0.91 and held-out 0.14 -> 0.30 by update 3500 at CE 0.80 log p (slower).

## 4. Records and code

tf_runner.py 1.2 (campaign U's, unchanged; SHA-256 fa77aa19...): training CE (float64) at every update; accuracies,
margins and held-out loss every 20 updates; A_H, norms and the ridge refit every 250; the logistic probe every 500.
worker.py, tests and the worker protocol are campaign U's.

## 5. Cost

50 runs, about 119 core-hours; 12 sessions (V1-V12) of 4 cores, about 4-5 hours each (the longest runs, two layers at 40000
updates, take about 4 hours).
