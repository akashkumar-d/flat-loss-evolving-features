# Campaign QD: depth 2 on the 40 evaluation seeds of depths 3-5 (claude/1hl-figures, 2026-10-13)

Fixed before any campaign-QD run.

## 1. Why

Akash, on the arXiv draft (v2, revision 2): every panel of a figure should have the same number of runs. The paper's depth
figure (p = 31) shows 10 runs at depth 2 (5 development and 5 fresh seeds) and 40 at each of depths 3, 4 and 5 (campaign C's
primary cohorts, seeds init 1000 + i, split 2000 + i, i = 1-40). QD runs depth 2 on those 40 seeds, so that all four columns
show the same 40 initialization and split seeds, and the dotted with-bias median of the depth-2 column stays paired with them.

## 2. Runs (80)

| family | runs | configuration |
|---|---|---|
| D2 | i = 1-40 | the paper's depth-2 setting: width 256, init scales 24 and 8, eta_w = eta_v = 0.03, last-layer rate 0.03 x 0.015 / 0.07 x 0.5 (r = 0.5), decays 0.015 / 0.07 / head 0.2, 80% split, full-batch CE, 40k updates, no early stop; bias-free |
| B_D2 | i = 1-40 | the same with a trained hidden bias (campaign I's D2 family) |

Both log the canonical observers (`observers_v1`: the paper's train-only ridge refit, AGOP and Fourier observers, as at depths
3-5); the old depth-2 runs logged only the legacy refit. Observers never change the training (tests/test_qd.py). Seeds init
1000 + i, split 2000 + i. Everything else is campaign I's D2 configuration, unchanged.

## 3. Code

`bias_runner.py` and `tests/test_bias.py`: campaign I's, byte for byte (also used unchanged by campaigns K, L, M and N);
`worker.py`: worker 2.0 (campaign ZP's), unchanged; `tests/test_qd.py` (9 tests): campaign I's 7, that observers_v1 never
changes a depth-2 run with or without the bias, and the manifest (checked against campaign I's manifest). Numerical note:
this training is chaotic at the level of rounding. On numpy 2.4.4 the with-bias run B_D2_r0.5_s0 matches its archived copy
(numpy 2.5.3) to 3e-7 relative for 61 updates and then drifts (0.6% by update 100); runs from different numpy builds are
equivalent statistically, not bit for bit.

## 4. Analysis (fixed before the runs)

- The per-run primary endpoint of the paper's deep runs (campaign C's, as computed by campaign I's `endpoints_i.py`: knee,
  peak band, share >= 0.75, final held-out accuracy >= 0.99, dA_H >= 0.02, fit), and the post-hoc window of Appendix D.3,
  on all 40 D2 runs; the same on the 40 B_D2 runs (secondary).
- The paper's depth figure: depth 2 shows the 40 D2 runs (thin) and their median, with the canonical refit (as at depths
  3-5); the dotted with-bias median is that of the 40 B_D2 runs. The old 10-run depth-2 cohort is reported in the appendix.

## 5. Sessions

QD1-QD4, 20 runs each, four at once on one thread (about 15 minutes per run): about 1.5 hours, 23 core-hours.

## 6. Checks before launch

The tests above; the worker's selftest; an end-to-end run of `manifest_test.json` through the worker to a local bare
repository.
