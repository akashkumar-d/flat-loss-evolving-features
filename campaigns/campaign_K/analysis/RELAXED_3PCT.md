# Campaign K with a 3% training-error tolerance (post hoc, 2026-10-08)

At Akash's request ("about 3% training accuracy failure is fine"), the fit check of campaign C's per-run endpoint is relaxed
from "every logged training margin in the band W is positive" to "training accuracy is at least 0.97 at every logged step in
W". The other three checks are unchanged (share >= 0.75, final held-out accuracy >= 0.99, A_H of the last hidden layer rises
>= 0.02 across W). This rule was chosen after seeing the results, so these counts are exploratory; a confirmation run must fix
the rule in its plan first.

Passes (relaxed rule; strict counts in `SUMMARY.md`). Paper setting = lam_v 0.2, lam_w 0.015, the 10 seeds per cell of
campaigns D and F (p = 23, 47, 61) and G (p = 101), rescored with the same code.

| cell | paper setting | sqrt rule, lam_w 0.015 | sqrt rule, lam_w 0.025 | linear rule, lam_w 0.015 | linear rule, lam_w 0.025 |
|---|---|---|---|---|---|
| depth 3, p = 23 | 0/10 | 0/3 | 0/3 | 0/3 | 0/3 |
| depth 3, p = 47 | 7/10 | 0/3 | 0/3 | 0/3 | 0/3 |
| depth 3, p = 61 | 7/10 | 3/3 | 0/3 | 3/3 | 0/3 |
| depth 3, p = 101 | 0/10 | 0/3 | 0/3 | 0/3 | 0/3 |
| depth 4, p = 23 | 0/10 | 0/3 | 0/3 | 0/3 | 0/3 |
| depth 4, p = 47 | 10/10 | 3/3 | 3/3 | 3/3 | 3/3 |
| depth 4, p = 61 | 0/10 | 3/3 | 1/3 | 3/3 | 3/3 |
| depth 4, p = 101 | 0/10 | 0/3 | 0/3 | 0/3 | 0/3 |

(sqrt rule: lam_v = 0.2 (31/p)^(1/2); linear rule: lam_v = 0.2 (31/p).)

Still failing:
- p = 23: share (the held-out rise is slow and the flat-loss band holds less than 75% of it); training accuracy stays 1.0.
- p = 101: fit; training accuracy falls to 0.51-0.58 (paper setting) or 0.59-0.83 (most K runs) inside the band; the few K runs
  that keep it above 0.97 (depth 3, lam_w 0.025) fail on share instead.
- depth 3, p = 47 at the K settings: share (the held-out rise starts before the band); the paper setting passes 7/10.
- depth 4, p = 61 at the paper setting: training accuracy falls to 0.94-0.97 in the band (median 0.956).

Minimum training accuracy inside the band, K runs at p = 61: depth 3 0.990-1.000; depth 4 0.941-0.998.
