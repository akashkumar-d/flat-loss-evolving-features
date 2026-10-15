# Twenty-pair, six-modulus locked replication

User request: more values of p, at least 20 seeds, and an explanation of how
the existing useful configuration was found. Interpret 20 seeds as 20 fresh
initialization/split pairs PER modulus, not 20 runs pooled across moduli.

New roster: p = 11, 13, 17, 19, 23, 29, twenty pairs each (120 new runs).
Each complete run has 40,000 updates. Maximum new updates: 4,800,000.
Roster interleaves these six moduli in each of twenty replicate rounds.
For zero-based roster index i, init_seed = 2026091301+i and
split_seed = 2026191301+i. These ranges exclude the earlier nine-run batch.
No historical run counts toward the twenty; no rejection or replacement.

The unchanged learner is imported from the frozen prior nine-run core.py:
biasless dense 1HL ReLU, width 512; input [e_a;e_b], label (a+b) mod p.
Full-batch mean training CE, float64 stable CE/residual calculation.
W_ij ~ N(0,24^2/p), V_ij ~ N(0,1/(24^2*512)), independent NumPy RNG.
Uniform fixed-size entry subset without replacement, floor(.8*p^2) train:
11:96/25, 13:135/34, 17:231/58, 19:288/73, 23:423/106, 29:672/169.
Initialization/split pairs are independent draws; sample entries within a
fixed-size split are not iid draws with replacement.

Simultaneous departing-state updates, unchanged across all moduli:

    W_next = W + .03*(supportPolar(-grad_W CE)/sqrt(retained rank) - .005*W)
    V_next = V + .03*(unitF(-grad_V CE) - .2*V)

No momentum Muon, row-sphere projection, adaptive balancing/gating,
learning-rate schedule, head substitution, or Fourier training feedback.
ReLU gates change naturally; "no gating" means no intervention that gates
the optimizer based on a loss target. This is replication/transfer of a
previously selected configuration, NOT a new hyperparameter search.

Keep every-step raw train/test CE, strict counts, and worst margins.
Record hidden/logit addition-Fourier energy and signed actual-head bounds
every 100 steps, including an observed terminal partial state. Save weights
and fixed split every 1,000 steps plus every terminal observed state.
At 10k/20k evaluate all four saved output combinations A(rule time)+Z(other
time), purely as observers; these are not trained intervention networks.
These metrics are NOT AGOP purity or ridge refit accuracy.

Reuse the prior fixed 10k–20k and 15k–20k window definitions: every training
row strictly correct, CE (max-min)/mean <=10%, hidden energy gain >=.01,
test start <=50%, test end >=90%, signed lower-bound fraction at end >=90%.
No favorable window is selected after outcomes. Also report all component
criteria, raw CE levels, finite-horizon test recovery, and accuracy at the
common 20k endpoint separately. A failed strict flag does not rule out a
different type/timing of recovery. Persistence is through an observed
finite horizon only. Do not label a censored run a full-horizon failure.

All 120 declared runs remain in result tables. Report completion, partial,
failed, and unstarted counts explicitly. Wilson intervals for binary flags
are small-sample empirical uncertainty summaries, not a theorem. With
censoring, report observed counts/denominators and unresolved trials; never
silently treat unresolved outcomes as measured negative observations.

Resource envelope: three single-thread CPU workers; cooperative per-arm
limit 3,600 seconds and global training work limit 43,200 seconds (12h).
No outcome-based stopping, seed replacement, retuning, or automatic extension.
These cooperative checks cannot interrupt a stalled numerical kernel.
A pipeline supervisor uses a separate process group for the training stage
and a 43,800-second stage timeout; on a true timeout it stops that owned
process group and reports failure, without fabricating complete outcomes.
Independent saved-output audit stage timeout 1,200 seconds; plotting/report
timeout 900 seconds. No retries of a launched stage. Disk must have at least
8 GiB free at launch; each worker resource-stops if this reserve is crossed.
No remote compute, paid service, package installation, or machine-wide
setting change. A lifetime-scoped caffeinate -i wrapper may keep this local
job awake; it exits when the pipeline exits.

The long-running local pipeline performs training, read-only independent
FFT/direct-column saved-checkpoint auditing, then saved-array plots/report
only if the audit passes. It writes progress and stage status on disk.
New sources and all imported scientific sources are hashed before launch;
existing sources/results/figures remain unchanged. Fabricated tests and
independent source review precede scientific launch. Scientific values are
not changed for nicer plots. No initialization or causal optimizer theorem
is inferred from this batch.
