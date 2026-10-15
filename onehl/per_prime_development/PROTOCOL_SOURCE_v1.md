# Modular arithmetic: corroborating experiments and uniform plotting protocol

Version **1.0-proposed**, 27 September 2026. Owner: the authors.

This is the cross-thread specification for the **next experiments**, not a report that they have been performed. The numerical thresholds below are proposed operational definitions, not consequences of a theorem or universal definitions of grokking. Freeze the version, configurations, seeds, windows, and endpoints before confirmation. Do not retrospectively change the accepted criteria of archived campaigns.

**The target:** on the **same run and same stated interval**, fitted training, nearly flat original training loss, non-negligible Fourier-representation change, and delayed improvement of the **actual network's held-out predictions**. A stronger target additionally shows that a specified train-only refit becomes useful before the actual head recovers.

Navigation: the machine-readable plotting style is `../src/plot_style_v1.json`. The plot collection, experiment index and theory notes that this protocol linked to are not included.

### At a glance

- **Run first:** reproduce 1HL ReLU/quadratic anchors, then lock ungated CE recipes and fresh seed/split confirmation; extend width/p before deeper claims.
- **Always plot:** original train/test loss, a linear plateau zoom, actual and train-only-refit accuracy, hidden Fourier fractions, explicitly defined AGOP alignment, and held-out refit error. Add energy, signed-margin, and norm diagnostics.
- **Proposed joint screen:** fitted training, raw loss `max/min <= 1.10`, absolute `Delta A_H >= 0.02`, and delayed actual test accuracy reaching 90% **inside the same declared interval**. Report stricter 3%/1% flatness and 0.05/0.10/0.15 Fourier-gain tiers separately.
- **Do not conflate:** high training accuracy with low CE; cycle-mean with raw flatness; Fourier energy with AGOP; oracle projection with train-only refit; selected success with replication.
- **Colors:** train blue, actual test green, refit amber, hidden Fourier purple, availability sky blue, AGOP magenta, residual/difference vermilion.
- **Status:** a reviewed specification for future runs, not an additional experiment or a new theorem.

## 1. What the figures can and cannot corroborate

Keep these endpoints separate; report all their pass/fail fields, not just a composite success label.

| Claim | Required evidence | Insufficient substitute |
|---|---|---|
| Raw-loss plateau | Small **within-window range** of the original training loss | Small endpoint drift, a logarithmic axis, smoothed loss, or flat median across runs |
| Plateau plus Fourier change | Plateau and a declared absolute increase in centered hidden Fourier fraction on that interval | AGOP gain substituted for hidden energy; a larger fraction with no decomposition of its numerator/denominator |
| Delayed test recovery | Training fitted earlier, a nontrivial bad-test delay, then high actual-head test accuracy | Refit accuracy alone, high train accuracy, or alignment alone |
| Joint plateau–alignment–recovery | All preceding conditions on the same trajectory, with the recovery time inside the declared plateau window | Combining different seeds, layers selected after inspection, or nonoverlapping intervals |
| Useful representation improvement | A **specified train-only refit** improves held-out predictions substantially from an initially imperfect refit | Full-grid oracle projection error; an initially perfect refit |
| Representation-first chronology | Useful-refit improvement demonstrably precedes actual-head recovery | Two threshold crossings indistinguishable at the logging resolution |
| AGOP–Fourier improvement | A fully specified AGOP/mapping/rank/row-set statistic improves | Treating AGOP, hidden Fourier energy, and hidden-span availability as the same metric |
| A reproducible regime | Fresh, predeclared initialization/split pairs; unconditional success counts and uncertainty | A selected good run, outcome-selected windows, or counting continuation branches as independent seeds |

High training CE is **allowed**. Perfect training accuracy with CE near `log(p)` is a high-loss/low-confidence phenotype, not automatically an invalid run. It must be labeled and accompanied by margins, logit scale, and test accuracy. Low-CE and high-CE cases should not be silently pooled.

Observed flat loss does not mean that CE has reached its minimum, that the gradient vanishes, or that the parameters have stopped moving. Training accuracy equal to one establishes none of those statements.

Experiments can corroborate a mechanism or find a counterexample. They do not prove an initialization theorem, establish all primes from a finite sweep, or show that scalar Fourier energy alone causes generalization.

### Theorem-matched versus ordinary dense-network experiments

The consolidated theory notes (not included) contain **separate** constructions. In particular, its R91 half-grid result has engineered initialization and a net `A_H` gain greater than `0.0998` for primes at least 97; its R62 random-split/low-CE/refit-first result has different width, initialization, penalty, and sufficiently-large-prime assumptions. Do not combine their conclusions into one theorem.

- **Theorem-matched track:** reproduce the exact split, initialization, optimizer, diagnostic, and constants of the selected theorem. Check its hypotheses before comparing its conclusions. Label structured examples and supplied arithmetic features. Certificate replay is not a new training trajectory.
- **Dense empirical track:** independent random-entry splits and declared random weights, without supplied Fourier features. This tests extension of the mechanism beyond the proved setting; it is not automatically theorem verification.
- **Controlled diagnostic track:** adaptive balancing, a head rescale, freezing, or a prescribed switch is allowed only as a separately labeled intervention. It cannot be counted as autonomous fixed-optimizer plateau evidence.

## 2. Experiments to perform, in order

This is a staged plan, **not a request to launch its full Cartesian product**. No jobs are authorized or launched by this document. Check runtime/storage first; obtain approval before paid GPU use.

### E0. Measurement and archive-reproduction checks

Before broad searches, reproduce saved anchor configurations exactly, then compute the new observers from new checkpoints where needed. Preserve original metrics alongside versioned new ones.

| Anchor | Why it is needed | Known limitation to preserve |
|---|---|---|
| 1HL quadratic 40-pair report (`onehl_earlier/quadratic_contrast/`) | Accuracy-grokking/reference alignment implementation | Accuracy success is not automatically joint CE flatness |
| 1HL ReLU multi-seed source (`onehl_earlier/fixed_recipe_sweep/multip_seed_signed_relu_v1/core.py`) and a selected p17 plot (not included) | Ungated CE plateau, Fourier change, actual recovery | Selected p17 interval is not fresh confirmation; its refit may already be good |
| Deep quadratic-MSE setup (not included) | Period-two/raw-loss distinction and deeper observers | Many measured phases train only the input matrix; low accuracy is not grokking |
| ReLU depth-6 controlled protocol (not included) | Positive intervention/control comparison | Deliberate head rescale/freeze, not autonomous stalling |
| Newer natural-plateau ReLU report (not included) | Candidate all-layer, fixed-parameter depth/width experiments | Separate workstream; later than the atlas inventory. Recheck raw loss and selection rules; do not import its success counts uncritically |

Unit-check observers on synthetic arrays, not only trained networks: constant features, a single nonconstant addition mode, a complete addition basis, pure difference features, rank-deficient features, zero AGOP, and sign-reversed output heads. Check projector/FFT equivalence, CE stability, centering, rank cutoffs, and seed independence. These are code tests, not learning experiments.

### E1. Main ungated CE experiment

Task: `x=[e_a;e_b]`, `y=(a+b) mod p`, all `p^2` **ordered** pairs. For random-entry experiments use a uniform subset of `n=floor(rho*p^2)` distinct training pairs and its complement as held-out data. This is sampling without replacement, **not literally iid examples**. Store separate initialization and split seeds and the actual pair lists. The default `rho=0.8` reproduces common archive conditions; `rho=0.5` is an important transfer arm, not an invitation to optimize ever-smaller fractions.

| Stage | Initial cells | Purpose |
|---|---|---|
| 1HL ReLU CE | `p in {13,17,31}`, width 512, `rho=0.8` | Primary plateau–Fourier–recovery target |
| 1HL quadratic CE | Same task cells; exact historical width-64 recipe first, then a separately labeled width-512 comparison | Activation/optimizer mechanism comparison, not a ReLU replication |
| Deep ReLU CE | Hidden depths 2, 3, 4; `p=31`; widths 256 and 512 | All hidden matrices and head train throughout; identify where depth helps or hurts |
| Deep quadratic MSE | Exact archived D6/D8 cells at `p in {17,31}`, widths 64/128 | Reproduce flat-cycle/feature-learning evidence separately from accuracy grokking |

Within a locked CE recipe use fixed learning rates and fixed decay, with no CE-target controller, adaptive gating, head rescale/freeze, temperature change, loss offset, or loss clipping. A fixed decay is allowed and must be visible in the configuration. A preplanned warmup/switch is a different **scheduled** arm, not the no-switch primary arm.

For a concrete ReLU 1HL development starting point, use the archived large-init family:

```text
W_ij ~ N(0, 24^2/p), independently
V_ki ~ N(0, 1/(24^2*m)), independently
f(x) = V ReLU(Wx), no bias
full batch, float64; eta_W = eta_V = 0.03
feature support-Polar / sqrt(retained rank); head Frobenius-NGD
fixed lambda_V = 0.2; develop lambda_W in {0.005, 0.01, 0.015}
initial budget B = 50,000 simultaneous updates
```

These are **starting configurations, not predicted successes at every p**. For quadratic and deep anchors import the complete saved configuration; do not silently transplant this ReLU law. Record deeper-layer initialization, output scaling, biases, residual connections, and every trainable/frozen matrix. “Depth” always counts hidden activation layers; an input activation plus seven residual blocks is eight, not seven.

The canonical new CE optimizer is specified, rather than merely called “Muon”:

\[
W_{\ell,t+1}=(1-\eta_\ell\lambda_\ell)W_{\ell,t}
-\eta_\ell\,\frac{U_rZ_r^\top}{\sqrt r},\qquad
V_{t+1}=(1-\eta_V\lambda_V)V_t
-\eta_V\frac{\nabla_VL_t}{\|\nabla_VL_t\|_F}.
\]

All gradients are evaluated at the same pre-update state. `U_r Sigma_r Z_r^T` is the SVD of the CE gradient of that **whole matrix**, not a rowwise or blockwise normalization. Zero gradient gives zero gradient increment. Proposed numerical support: singular values greater than `1e-10 * sigma_max`, absolute cutoff zero; log the retained rank and threshold. Use the exact historical threshold when reproducing an archive. Decay is outside the normalization. No momentum or row-sphere constraint is implicit. This is **memoryless normalized support-Polar + NGD**, not a claim about standard momentum Muon.

### E2. Transfer and ablations after locking a working recipe

Change one factor at a time before considering interactions:

- **Modulus:** extend to `p in {7,11,19,23,29,47,61,97}` with the locked scaling rule. Report all failures. Finite tests at these values do not imply all-prime coverage.
- **Width:** `{128,256,512,1024}` at fixed p/depth; keep explicit weight/output scaling and actual update norms. Width changes numerical ranks and effective Polar step sizes if normalization differs.
- **Split:** `rho in {0.5,0.8}` and fresh random-entry splits. Run a structured theorem split only in its separately labeled arm. Do not assume arbitrary balanced splits satisfy a theorem for quadratic-residue difference classes.
- **Optimizer:** feature Polar versus feature NGD and ordinary GD; optionally AdamW or an explicitly implemented momentum-Muon arm. Use paired seeds and a predeclared development budget for each optimizer. Equal nominal learning rate is not equal update size: plot realized matrix and prediction changes.
- **Decay:** no decay, hidden-only, head-only, and both, using fixed values. Do not tune decay from held-out losses during a confirmation run.
- **Feature-freezing control:** clone the full optimizer state at a **training-only selected or fixed** checkpoint, freeze hidden matrices, and continue the original head optimizer for the same remaining budget. Compare with the joint-training continuation. A head-freezing control is separate. Neither intervention is counted in the autonomous arm.
- **Scale versus direction:** log normalized features and norms; a counterfactual that imports the original run's future norms is an oracle diagnostic, not an independently implementable learner.
- **Signed-rule ablation:** at fixed checkpoints remove addition-projected logits, compare with predeclared rank/norm-matched non-rule removals, and report perturbation sizes. This is an observer-only, full-grid intervention on the current predictor, not a deployable train-only refit and not proof of the learning dynamics.

### E3. Development, confirmation, and seed accounting

Use development pairs to choose recipes and window rules. Keep the chosen set small; a suggested budget is at most 24 configurations per family using the same 3 development pairs. Record every tried setting. This is a budget proposal, not a guarantee of adequate optimization.

For each selected recipe, predeclare **40 fresh independent initialization/split pairs**, disjoint from development, possibly executed as two scheduled batches of 20. Do not change the recipe after seeing the first batch and call the second a continuation of the same confirmation. A revised recipe gets a new version and new confirmation seeds; earlier failures stay reported. A budget-limited first batch is an interim result, not grounds for optional stopping on a favorable interval.

Use the same fresh pair across optimizer/control arms for paired comparisons. Runs within one arm use independent pairs. If using a crossed initialization-by-split design instead, record the dependence and use cluster-aware uncertainty; do not treat every grid cell as an independent Bernoulli trial. Continuations, branches, duplicate plots, and two README copies are not additional seeds.

## 3. Exactly what to log

Store the state convention: metrics at `t` refer to parameters **after t updates**, with `t=0` before training. Warmup and main-phase counters must also map to this global counter. Do not splice away the earlier poor-feature or poor-test phase.

| Cadence: proposed CE default | Required quantities |
|---|---|
| Every update | Original training CE, actual train accuracy and minimum margin, matrix/gradient/update norms, retained Polar ranks, numerical-health flags; accumulated min/max CE even if plotting is thinned |
| Every 20 updates | Actual held-out CE, accuracy, margin quantiles, logit RMS; include t=0 and terminal state |
| Every 100 updates | Full-grid per-layer `A_H`, addition/difference/total/DC energies, gate-change fraction for ReLU, fixed-grid representation displacement |
| Every 500 updates | Mapped AGOP observers, fixed/random-axis controls, hidden-span availability, specified train-only refit, actual-head tracking discrepancy, signed Fourier/residual margins |
| Explicit event/window boundaries | All above observers at the exact endpoints; save reusable states and optimizer state |

These cadences are upper bounds on spacing, not automatic runtime commitments. Profile observers and lock any coarser schedule before confirmation. Period-two MSE phases are short: log raw losses every update and observers at every complete cycle where feasible. Save enough information to audit all updates within any claimed plateau; sparse checkpoints cannot certify unseen raw-loss flatness.

For an offline training-only window detector, retain its candidate-boundary states or a tested deterministic replay path. Fixed-axis observers and checkpoint-freezing interventions cannot be reconstructed from scalar metric logs alone. Choose a storage/replay plan before collecting runs, rather than silently omitting these controls afterward.

**Minimum run record:** protocol version, run/parent/branch IDs, full config hash, code version/hash, p, n, split lists and seeds, depth convention, activation, widths, output scaling, exact optimizer equations/implementation, schedule and all interventions, dtype/device, stopping reason, declared horizon, initialization law or checkpoint hash, observer row-set hashes, centering/rank thresholds, refit rule, and raw metric arrays. A stopped/incomplete run is not silently omitted.

## 4. Metric definitions: do not use “alignment” without a suffix

### 4.1 Original loss, accuracy, and margins

Use raw mean CE on the stated row set:

\[
L_E(t)=|E|^{-1}\sum_{x\in E}\left[\log\sum_c e^{f_{t,c}(x)}-f_{t,y(x)}(x)\right].
\]

Use a stable implementation; for tiny losses avoid subtractive cancellation with a target-relative `log(1+sum_{c != y} exp(f_c-f_y))` evaluation in a stable log domain. Do not add an epsilon/floor to measured CE to create a plateau. Zero/underflow-censored CE makes relative flatness **unresolved**, not an automatic pass. A display floor, if unavoidable, is separately marked and never used for scoring.

Report `L_train`, `L_test`, `L_train/log(p)`, `log(p)-L_train`, and logit RMS. CE is not classification error. The example margin is `f_y - max_{c != y} f_c`. Report argmax accuracy, strict-positive-margin accuracy, near-tie fraction, and minimum/5th/50th/95th margin percentiles. Record the tie rule. Signs below numerical resolution require higher precision or an “unresolved” label, not confidence from arbitrary tie breaking.

For MSE, specify whether it is per-coordinate or per-example. Canonical secondary MSE is `mean_x ||f(x)-onehot(y)||_2^2`, equal to `p * elementwise_MSE`; zero logits have loss 1. CE and MSE never share an unlabeled loss scale.

### 4.2 Hidden Fourier-energy fraction: `AH_add_centered_fullgrid`

For each hidden layer, let `H` have all `N=p^2` grid rows and all physical hidden columns. Let `H_c=C_N H` subtract the full-grid column means. Let `Q_+` be an orthonormal real basis of the nonconstant functions of `(a+b) mod p`, and `P_+=Q_+ Q_+^T`, of rank `p-1`.

\[
A_H=\frac{\|P_+H_c\|_F^2}{\|H_c\|_F^2},\quad
E_+=\|P_+H_c\|_F^2/N,\quad E_{\rm total}=\|H_c\|_F^2/N.
\]

Log `A_H`, `E_+`, `E_total`, `E_residual=E_total-E_+`, and the squared DC-mean norm. Compute the analogous difference fraction for `(a-b) mod p`. FFT implementations use the nonzero `(k,k)` modes for addition with a documented sign convention; verify against conditional class means. Zero/negligible denominator is flagged undefined; do not infer a gain from a legacy zero-return convention.

**This is the theory's centered hidden fraction.** An increase of 0.02 means two percentage points of centered energy, not a 2% relative increase. Report gains at 0.02, 0.05, 0.10, and 0.15 as different effect sizes. A larger fraction can arise from shrinking non-rule energy without growing rule energy. The two energy curves are mandatory if claiming to explain that mechanism. Net gain is not monotone gain.

For depth greater than one, predeclare the primary layer, by default the last hidden layer. Show first and last layers in the main report and all layers in a companion sheet. Do not select the best layer after seeing the test curves. For skip/multiple-head models, layer metrics and the complete readout-input representation are separate records.

### 4.3 Hidden-span availability and oracle error

Let `U_H` span the numerical column space of `H_c`. Define

\[
A_+=\|U_H^TQ_+\|_F^2/(p-1),\qquad R_{\rm oracle}=1-A_+.
\]

Record the SVD cutoff and numerical rank. `R_oracle` is **full-grid oracle projection error**, not test CE, not held-out ridge MSE, and not a training-only head-fit risk. Do not label it merely “head-fit risk.” Availability can already be one in an unhelpfully conditioned or overcomplete representation. It is not interchangeable with energy concentration.

### 4.4 AGOP–Fourier alignment: coordinate, centering, mapping, and rank

New canonical **CE** observer: the Jacobian of **class-centered logits** with respect to hidden representation `h_l`. Form its Gram using training inputs S, then assess its mapped Fourier overlap on held-out observer rows `E_obs`. These are two separately recorded row sets, not one implicit “held-out AGOP” convention:

\[
J_l(x)=\partial(C_p f(x))/\partial h_l(x),\qquad
G_l=|S|^{-1}\sum_{x\in S}J_l(x)^TJ_l(x).
\]

At the last layer of `f=Vh`, this is `V^T C_p V`; include `1/m` if logits are divided by `sqrt(m)`. At earlier layers propagate through the downstream network. This is **hidden-coordinate AGOP**, not input-gradient AGOP. An input AGOP in dimension `2p` cannot be identified directly with the addition-function subspace in `p^2` sample coordinates.

Select top-q positive numerical eigenmodes `U_q`, then map them to sample functions `Z=C_E H_E U_q`. Re-orthonormalize the columns of `Z` and the restricted, centered Fourier basis on the **same rows**, obtaining `Q_Z,Q_F`. Let their effective ranks be `r_Z,r_F`, and `O=||Q_Z^T Q_F||_F^2`. Store separately:

- `agop_effective_purity = O/r_Z`;
- `agop_requested_rank_score = O/q`;
- `agop_coverage = O/r_F`;
- `agop_mean_squared_CCA = O/min(r_Z,r_F)`.

These may coincide in nondegenerate small-q cases, but they are not generally identical. Undefined denominators are NA. Requested q, actual selected rank, mapped rank, target rank, numerical tolerance, eigenvalues, and the boundary eigengap are mandatory. Do not fill unsupported q with arbitrary null eigenvectors. A primary fixed-q comparison is unresolved when that positive rank is unavailable; report reduced-rank diagnostics separately.

The primary fixed-q gain also requires mapped rank `r_Z=q` at both compared endpoints and unchanged target rank at least q. Then effective purity equals the requested-rank score. Otherwise report rank-changing diagnostics separately: dividing by a shrinking effective rank can create an apparent purity gain without a larger fixed-rank overlap.

For a primary fixed-q **gain**, also require a resolved gap between the selected and omitted eigenspaces at the compared checkpoints: proposed float64 criterion `(lambda_q-lambda_{q+1})/lambda_max > 1e-8`, with `lambda_max > 0` and `lambda_{m+1}=0` if selecting the whole hidden space. Record numerical uncertainty. A tie inside the selected subspace is harmless; a tie across its boundary makes the top-q subspace unidentifiable. Mark that endpoint unresolved rather than allowing an eigensolver's arbitrary basis to manufacture a gain. A whole-degenerate-cluster observer is a separately declared diagnostic with its own rank, not a post-hoc replacement for q=4.

Use **q=4** as the new cross-depth primary rank when feasible, with `q=min(p-1,width_l)` as a separate comprehensive diagnostic. Primary mapping rows are all held-out rows for the small-p core; any computational subsample of the Gram or mapping row sets must be fixed before training and identified separately. Full-grid AGOP/mapping is a separate observer. No observer gradients or labels enter optimization.

For **MSE**, keep the raw-logit Jacobian Gram as the objective-relevant primary convention: a class-common output error contributes to MSE. A class-centered companion is allowed but does not replace it. For CE, also retain a raw-logit compatibility observer when reproducing an archive. Record `output_convention`, `gram_rows`, and `mapping_rows` in every metric ID.

Add eight fixed-seed random orthonormal q-frame controls mapped through the same representation/rows, with the same rank pipeline. Plot their median and range. Also keep the axes selected at the window start fixed and remap them as features change; distinguish **current-axis** from **fixed-axis** alignment. High-dimensional finite-row overlap and moving-axis selection can increase scores without the intended mechanism.

Report the late score minus the random-frame median and the difference in gains. An increasing AGOP score that remains below the matched null is not evidence of preferential Fourier selection by its leading eigenspace. Eight frames are a diagnostic control, not a calibrated hypothesis test.

**Legacy warning:** archived implementations differ in raw versus class-centered logits, requested versus effective rank normalization, and held-out versus full-grid rows. Keep their metric IDs intact when reproducing them. Compute the canonical observer additionally; do not rename a legacy curve and call it a uniform comparison.

### 4.5 Train-only readout and the actual head

New canonical full-hidden diagnostic: raw hidden features, no intercept, centered one-hot targets `Y_c`, and a **fixed numerical** penalty `alpha=1e-3` in the mean-loss convention:

\[
\widehat B_t=\arg\min_B \left\{n^{-1}\|H_{S,t}B-Y_{S,c}\|_F^2+\alpha\|B\|_F^2\right\}.
\]

Freeze alpha for the entire run. This is a proposed diagnostic, not an asserted optimal choice. Predeclared sensitivity arms `alpha in {1e-4,1e-2}` are separate; do not choose one using test outcomes. If tuning is necessary, use only an inner training split during development and lock the rule for confirmation. No test/full-grid normalization, standardization, label use, or adaptive penalty is hidden in this primary fit.

Historical standardized/Fourier-code, training-RMS-normalized, or AGOP-selected refits remain separate diagnostics. A fixed RMS-normalization **rule** is not a fixed numerical regularization penalty. State whether an objective is a sum or a mean; its ridge constants differ by n.

Report held-out refit accuracy, held-out squared prediction error to `Y_c`, and margin quantiles. Label that error **“train-only ridge refit: held-out MSE”**. It is not original-network CE. Do not replace the learner's head with this diagnostic.

Where the actual readout is linear in the chosen H, also log

\[
d_t(x)=\|C_p(f_t(x)-\widehat B_t^T h_t(x))\|_\infty,\qquad
m_{f_t}(x)\ge m_{\rm refit,t}(x)-2d_t(x).
\]

Row/column orientations in code must be consistent. For skip/multiple-head networks H must contain **all actual linear readout inputs**, not just one branch. A probe at an earlier hidden layer is not the actual nonlinear downstream head. Compute this bound pointwise before aggregating quantiles; report certified fraction and actual accuracy separately. A good actual head proves the features admit a good predictor, but does not guarantee this particular train-only ridge fit will select it.

### 4.6 Signed rule margins and residuals, with the actual head

Class-center the full-grid actual logit table `F`; decompose sample rows as `F^+=P_+F` and `R=F-F^+`. For each example x and incorrect class c define

\[
u_c(x)=F^+_y(x)-F^+_c(x),\quad b_c(x)=R_c(x)-R_y(x),\quad
\gamma(x)=\min_{c\ne y}u_c(x),\quad\beta(x)=\max_{c\ne y}b_c(x).
\]

Then `m_actual(x)=min_c(u_c-b_c) >= gamma(x)-beta(x)`. Retain the **sign** of beta; a negative residual contrast can help. Optionally also log `max(beta,0)` as a conservative nonnegative harmful budget, under a different name. Include the input-constant class bias in R.

Plot gamma, beta, the pointwise lower bound, and actual margin quantiles on held-out rows; report `fraction(gamma-beta > 0)`. Do not subtract independently aggregated quantiles and call it a margin certificate. These full-grid projections are explanatory observers, not train-only predictors or evidence that the optimizer had access to test labels.

## 5. What counts as a valid behavior

### 5.1 Window selection must be declared before confirmation

For each recipe choose **one primary window rule** using development data:

1. **Fixed:** predeclare absolute endpoints `[t0,t1]` per recipe. No seed-specific test-based adjustments.
2. **Training-only:** predeclare a length W and select the **earliest** window after training becomes fitted that satisfies the locked loss-range test. Use only training loss/accuracy to choose it; do not skip an earlier qualifying window because its features/test scores are disappointing.

For new long CE runs propose `W >= max(1000,ceil(0.05*B))`, with endpoints on the declared observer grid; for `B=50,000`, W is at least 2,500 updates. Short historical/theorem/period-two protocols use their separately declared horizons, not this default. A long update count obtained solely by reducing a step size is not evidence of meaningful learning: require an effect-size gain and report elapsed time and parameter/representation path length.

The training-only detector may use a completed window's future **training** losses for offline analysis; it does not alter the optimizer. A test-defined transition window is a useful **separate** analysis, even if its rule was predeclared. A plot selected after inspecting all outcomes is explicitly exploratory. Keep all failed/missing windows in the outcome table.

### 5.2 Per-run numerical criteria

For `I=[t0,t1]`, computed from all original training-loss observations,

\[
v_{\rm min}(I)=\frac{\max_I L-\min_I L}{\min_I L},\qquad
v_{\rm mean}(I)=\frac{\max_I L-\min_I L}{\operatorname{mean}_I L}.
\]

| Field | Proposed operational criterion |
|---|---|
| `P10_raw` | Positive, numerically resolved original training loss; `v_min <= 0.10`, equivalently `max/min <= 1.10` |
| `P03_raw`, `P01_raw` | Same **primary window**, with `v_min <= 0.03` or `<= 0.01`; do not search a new favorable window for each tier |
| `fit_strict` | Every training example has a resolved positive margin at every scored update in I; also report the weaker accuracy >=99% endpoint separately |
| `F02` | Primary layer has `A_H(t1)-A_H(t0) >= 0.02`, resolved nonzero denominators, with both endpoint energies logged |
| `F05`, `F10`, `F15` | Absolute gains >=0.05, >=0.10, >=0.15 on that same layer/window; distinguish this from reaching an absolute A_H level |
| `AGOP02` | Canonical fixed-q effective purity gain >=0.02, valid ranks and resolved boundary eigengaps at both endpoints; report requested-rank score, coverage, null and fixed-axis changes too |
| `useful_refit` | Refit accuracy at **t0** <=0.70; reaches >=0.90 at some `T_refit in (t0,t1]`, with gain >=0.20 relative to t0, using the same fitting rule |
| `finite_recovery90` | Actual held-out accuracy reaches >=0.90 at some logged finite time; also report 95%, 99%, and 100% times if attained |

`F02` is an **evidence floor**, not a claim that two percentage points is universally a large or useful gain. The theorem-matched track must check its theorem's own quantitative threshold (for example >0.0998), not replace it by F02. None of the Fourier/AGOP criteria alone certifies predictive usefulness.

Report `v_mean`, signed endpoint drift, maximum drawdown, fitted slope, and CE level in addition to `v_min`. Historical “3% of the mean” or “±1.5%” thresholds are not automatically the same as a 3% max/min condition. Do not remove transients inside I or average over seeds before testing flatness.

**Period-two exception:** define nonoverlapping pairs with a fixed phase origin, `Lbar_j=(L_{t_origin+2j}+L_{t_origin+2j+1})/2`. Apply the same ratios to complete pairs only; log both raw branches and `|L_even-L_odd|`. Proposed short-phase minimum is 50 complete cycles. A pass is `P10_cycle`/`P01_cycle`, **not** a raw-loss plateau. Show alignment/accuracy for both raw branches. Compare Fourier, AGOP, and refit endpoints at matched parity; a phase-robust cycle F02 requires at least 0.02 gain on **both** branches, and a phase-robust useful-refit flag requires its improvement criteria on both branches. Do not manufacture a gain by comparing an even initial state with an odd final state. Single-phase findings are named as such. A full period-two claim also needs state/update diagnostics, not only an oscillating loss.

### 5.3 Delayed recovery, overlap, and feature-first chronology

New CE default: `T_fit` is the first logged resolved strictly positive training margin on every training example; fittedness must persist throughout the subsequent scored delay and window. Starting at T_fit, require a bad-test prefix of at least `D=max(1000,ceil(0.02*B))` updates on which **every logged held-out accuracy is <=0.20**. If that prefix does not exist, this strict delayed-acquisition endpoint fails; do not move T_fit or hunt across later dips. Then define `T_grok` as the first subsequent logged actual accuracy >=0.90, and require that **no earlier logged checkpoint, including initialization, had reached 90%**. Report the delay in updates and wall time. If 90% was achieved earlier and then lost, use the separate label `recovery_after_collapse`, not first-acquisition grokking. This bad-test cutoff is intended for the stated p>=7 grid; use a separately locked chance-adjusted rule if expanding to other tasks/moduli.

This establishes a **finite-time empirical recovery event**. Later dips do not erase it; show them. Record a secondary short-persistence check (three consecutive evaluation checkpoints >=0.90) and terminal accuracy. Neither amounts to infinite-time permanence. Sparse held-out evaluations establish observed chronology, not a claim about every unobserved update.

- **`PA10_F02`:** P10_raw + F02 on I. Training fit and test recovery are separate fields.
- **`joint_PAG10_F02`:** PA10_F02 + fit_strict + the bad-test delay + `t0 <= T_grok <= t1`, with actual accuracy at t0 <=0.20. This is the main joint endpoint.
- **`AGOP_supported_joint`:** joint endpoint plus AGOP02; do not require AGOP to claim a theorem whose metric is A_H alone.
- **`full_transition_covered`:** additionally the first 20% crossing after the bad-test delay and T_grok both lie inside I. Show this as a stronger overlap descriptor, not a different selected interval.
- **`late_overlap`:** only part of recovery lies in I. Label it partial overlap, not full plateau-before-and-through-grokking.
- **`representation_first`:** joint_PAG10_F02 and useful_refit both hold, with `t0 < T_refit < T_grok <= t1`; the refit improves relative to **t0**, reaches 90% while the actual network is still <=20%, and its upper crossing-time bracket precedes the actual-head lower crossing-time bracket. Plot both brackets. If refit is already excellent before I, the run may pass joint_PAG but not demonstrate discovery of usefulness during I. Any chronology outside I is reported as `anytime_refit_lead`, not this common-window endpoint.

Always report transition duration from the first post-delay 20% crossing to 90%, and its ratio to the fitted/bad-test delay. Call a transition “sharp” only under an additional predeclared bound on that ratio. Do not make an abruptness claim solely because a long horizontal axis compresses the recovery.

Missing AGOP/refit observations mean **not measured**, not “no gain” and not a pass. If logging resolution cannot order two events, chronology is unresolved. If the selected interval is missing, numeric failures occur, or required data are censored, the run cannot count as a primary joint success.

## 6. The plots to produce

### Sheet 1: mandatory per-run trajectory figure (2 rows x 3 columns)

All time panels use the **same global update axis**. Show the complete run plus the common-window zoom; use the same shaded interval I in every relevant panel.

| Panel | Contents |
|---|---|
| A. Original objective | Training and actual held-out CE (or explicitly labeled MSE), full trajectory. CE logarithmic overview with a linear-scale companion; show the CE level relative to log(p) |
| B. Is it genuinely flat? | Raw training loss in I on a linear axis, and percent change from L(t0); print min/mean/max, v_min, v_mean, endpoints, train accuracy, and window-selection rule |
| C. Actual and refitted predictions | Train accuracy, actual held-out accuracy, train-only refit held-out accuracy, and chance 1/p; show fit/refit/recovery events and any later dips |
| D. Hidden Fourier fractions | Addition A_H and difference fraction at the primary layer, plus endpoint gains; first/all other layers on a labeled companion plot, not a best-layer envelope |
| E. AGOP–Fourier | Canonical current-axis purity, fixed-axis counterpart, matched random-frame band, with q/ranks/row set in caption; coverage in a separate inset/companion, not silently substituted |
| F. Head-fit risk, explicitly named | **Train-only ridge refit held-out MSE**, with optional train MSE. Do not substitute the oracle error 1-A_+ or the actual network's CE |

If a required observer is unavailable, retain the panel labeled “not measured” and its reason; do not fabricate a curve or substitute a different alignment. Legacy-only figures keep their original names and metric IDs.

### Sheet 2: mechanism and interpretation (2 rows x 3 columns)

1. Addition energy E_+, non-rule energy, total centered energy, and feature mean/DC energy: fraction growth versus numerator growth.
2. Hidden-span availability A_+ and explicitly labeled oracle error 1-A_+; numerical ranks/conditioning. This is separate from panel F above.
3. Actual-head signed rule margin gamma, signed residual beta, pointwise bound, actual margin quantiles, and certified correct fraction.
4. Refit/actual margin distributions and prediction discrepancy d_t; proportion certified by `m_refit-2d > 0`.
5. Per-layer/head norms, centered logit RMS, gate-change fractions, and normalized representation displacement. Use separate axes/panels for unlike units, not a misleading shared scale.
6. Actual CE increments versus `sum_l <grad_Wl L, Delta W_l> + <grad_V L, Delta V>` and their difference; show gradient and decay contributions separately if the implementation permits. This is an observed first-order decomposition, **not** a certified Taylor bound across ReLU gate changes. It helps distinguish small motion from cancellation.

Add selected frequency spectra and layer heatmaps at fixed pre-fit, plateau-start, pre-recovery, recovery, and final checkpoints. Lock the selected modes on training/development information or display the full spectrum; do not choose only the best-looking frequencies.

### Sheet 3: reproducibility across runs

- Full-time trajectories: thin individual lines; thick median; shaded IQR. IQR is not a confidence interval. State N at every time if horizons differ; do not carry a truncated trajectory forward.
- Per-run chronology: T_fit, useful-refit crossing, T_grok, and plateau interval. Failed/nonattained events remain as censored marks, not omitted rows.
- Success rates: retain `k/N_declared` for every endpoint and composite, but call it the **operational confirmed-success fraction** if infrastructure censoring exists. Keep scientific fail and scientifically unknown distinct. With u unresolved runs also show the identification range `[k/N,(k+u)/N]`; completion-only rates never replace it. For fully observed independent seed pairs show 95% Wilson intervals. For censored scientific outcomes a conservative interval can use the Wilson lower endpoint at k successes and upper endpoint at k+u, clearly labeled censoring-aware; do not call unknown outcomes observed scientific failures.
- Scatter/Pareto: raw CE range versus Delta A_H, actual test improvement, and refit improvement, on the **same window**, including failures.
- Transfer summary: hidden layers first, then p; separate CE/MSE, activation, optimizer, intervention, and split family. Within cells show width and sample count. Do not pool unlike protocols into a single success rate.
- Paired-control differences with pair-level bootstrap intervals. If doing many formal significance tests, predeclare the primary comparisons and multiplicity treatment.

For a strong “reliable regime” claim, predeclare the desired success probability (suggested 0.8) and require its **95% lower confidence bound**, not just the observed fraction, to exceed that target. Predeclare the single primary recipe/endpoint for this claim, or use simultaneous/multiplicity-adjusted bounds when selecting among many. Unadjusted marginal Wilson intervals for a large grid are descriptive, not a license to select any passing winner. With censoring use the conservative lower bound above and disclose the operational/scientific distinction. Smaller cohorts remain descriptive. A median trajectory cannot establish that any individual run passes the joint criterion.

## 7. Uniform color and visual grammar across threads

The semantic palette is versioned in `../src/plot_style_v1.json`. Reuse it; do not assign a different meaning to the same color in another thread's core panels. Color is supplemented by line style, labels, and markers.

| Quantity | Hex | Default style |
|---|---|---|
| Actual training loss/accuracy | `#0072B2` blue | Dashed |
| Actual held-out loss/accuracy/margin | `#009E73` green | Solid |
| Train-only refit, evaluated held-out | `#E69F00` amber | Dotted |
| Train-only refit, evaluated on train | `#0072B2` blue | Dotted; explicitly label “refit train” |
| Addition hidden fraction A_H / addition energy / signed rule margin | `#7B4AB5` purple | Solid; separate axes for different units |
| Hidden-span availability A_+ | `#56B4E9` sky blue | Dash-dot |
| Canonical AGOP–Fourier current-axis score | `#CC79A7` magenta | Solid |
| Same AGOP statistic, fixed axes | `#CC79A7` magenta | Dashed |
| Difference-mode / non-rule / signed residual | `#D55E00` vermilion | Dashed |
| Full-grid oracle projection error 1-A_+ | `#666666` gray | Dotted |
| Total energy / neutral baseline | `#333333` charcoal | Solid |
| Random-frame baseline / chance / threshold | `#999999` gray | Thin dotted; label which |
| Common candidate window | `#DDE3E8` | Background span, alpha 0.30 |
| Uncertainty band | Same semantic curve color | IQR alpha 0.18; individual runs alpha 0.12 |

Rules:

- Keep the training/held-out/refit mapping in every objective and accuracy plot. For a scalar Fourier panel use the Fourier mapping, not “test green” simply because its observer rows are held out.
- Different depths, p, widths, optimizers, and layers normally get **facets**, not a competing rainbow. For a two-arm overlay retain the metric color, use labeled solid/dashed arms in a separately keyed control panel; do not mix that encoding with the train/test style legend in the same panel.
- Event lines: T_fit gray dashed; useful-refit crossing amber dotted; T_grok green dash-dot. Use direct labels. Window shading is not a success certificate.
- Main trajectories width 2.2 points; raw individual lines 0.6 points; controls 1.3 points. White background, light `#E5E7EB` grid, no 3D effects. Default 15 x 8.5 inch 2x3 sheet, at least 9-point axis/legend text. Split crowded figures instead of shrinking labels.
- Accuracy/fraction axes use [0,1] on the overview. A zoom may use a narrower range only with explicit limits. Signed margins include a zero reference line. Small CE changes need linear and relative-change views, not only log scale.
- Primary plots are unsmoothed. Optional moving averages are secondary and show their window; period-two cycle means are a distinct mathematical observable, not cosmetic smoothing.
- Per-run figures shade that run's declared interval. Do not shade a single median “plateau” on a cohort figure when runs have different windows; show the interval distribution or individual panels.
- Export vector PDF/SVG and a 180–220 dpi PNG, without recomputing/rounding the raw arrays. Keep readable per-depth/per-p bundles; do not start with another hundreds-of-megabytes monolith.

## 8. Pass/fail table and release checklist

One row per declared run, including non-completions. Required summary columns:

```text
run_id, config_hash, init_seed, split_seed, depth, p, n, widths,
activation, loss_id, optimizer_id, intervention_id, status,
window_rule, t0, t1, primary_layer, min_train_acc, min_train_margin,
CE_min, CE_mean, CE_max, CE_over_logp, v_min, v_mean,
AH_start, AH_end, delta_AH, Eadd_start, Eadd_end, Etotal_start, Etotal_end,
AGOP_metric_id, q_requested, ranks_start_end, delta_AGOP, delta_null,
refit_rule_id, refit_acc_start, refit_acc_end, refit_MSE_start, refit_MSE_end,
T_fit, bad_test_delay, T_refit_bracket, T_grok_bracket, test_acc_start_end,
P10_raw, P03_raw, P01_raw, P10_cycle, F02, F05, F10, F15,
fit_strict, useful_refit, finite_recovery90, joint_PAG10_F02,
AGOP_supported_joint, representation_first, missing_observers, failure_reason
```

Before calling a run “joint plateau–alignment–grokking,” check:

- [ ] It is the actual learner's unmodified original objective and predictions.
- [ ] The same run, same predeclared layer, and same interval meet all named requirements.
- [ ] Every-update loss extrema were checked; the plateau is not a smoothed/cycle-only/aggregate artifact.
- [ ] Training was fitted before the observed delayed test recovery; ties and high-CE confidence are disclosed.
- [ ] The Fourier gain is absolute, numerically resolved, and decomposed into rule and non-rule energy.
- [ ] Any useful-refit or AGOP claim uses its actual defined metric; missing curves are labeled.
- [ ] No test-driven optimizer adjustment, window retuning, or unreported outcome-based selection occurred.
- [ ] All declared seeds, failures, censoring, configs, sources, and plotting versions are included.
- [ ] A direct source link accompanies every plot; local links are relative/percent-encoded and verified.
- [ ] Another reviewer checks metric semantics and endpoint logic before results are described as confirmation.

Each campaign writes its own immutable config/seed manifest, trajectory arrays, observer tables, outcome table, figures, and short README citing **this protocol version** and listing any deviations. If a definition changes, increment the version and recompute comparisons explicitly. Do not overwrite legacy results, edit another thread's runner without coordination, or present this protocol as already adopted by all threads.

## 9. Immediate execution priority

1. Validate observer code and reproduce the 1HL ReLU/quadratic anchors with full loss, A_H, AGOP, refit, and actual-head trajectories.
2. Lock one ungated ReLU recipe and its window rule; perform fresh paired confirmation and the hidden-freezing control.
3. Extend that recipe to width and p, then to all-layer ReLU depth 2–4. Keep failures and partial overlap visible.
4. Reproduce the deep-MSE cycle-plateau branch separately. Treat its lack of high actual accuracy as a scientific distinction to explain, not a plotting defect to hide.
5. Only then choose the clearest defensible figures for the paper. Do not spend the first pass optimizing visual smoothness or favorable windows.

This README specifies the work; **no new training or numerical revalidation was performed to create it**.
