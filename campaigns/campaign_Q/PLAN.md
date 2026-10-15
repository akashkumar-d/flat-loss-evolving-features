# Campaign Q: ten seeds per prime for one block, and why two blocks fail late (claude/1hl-figures, 2026-10-13)

Fixed before any campaign-Q run.

## 1. Why

Akash, on the arXiv draft (v2, revision 2): every panel of a figure should have the same number of runs, ideally 10. The
one-block figure across primes has 5, 5, 10 and 5 runs at p = 61, 89, 113 and 149, and the mirror-pair figures have 5 per
split. He chose the mirror-free split for p = 61 in the across-primes figure. For two blocks (0/20 so far): "try more passes
plus different fraction as well to see what is it that is not working or if something works."

## 2. What the two-block runs already show (campaign Z, cell B_f0.2_lam_v0.275_tau10k, seeds 11-15)

The fit fails late and abruptly. From update 12k until shortly before the first flip (17.8k-22.0k) the smallest training
margin stays at +1.7 to +3.5 (one seed dips to +0.1) while the gain falls from 1.30 to 1.11-1.17; within 2k updates of the
first flip it reaches -4.7 to -5.9, and 45-438 training pairs are misclassified at the worst log. The MLP matrices grow only
after the flip (W_out1 of the second block: 18-21 at the flip, 33-42 two thousand updates later, at most 50-64; all hidden
matrices from 55-58 at 12k to 80-93), and the head norm stays at 3.6. So it is not a slow loss of margin as the gain decays,
and the growth is a response to the flip, not its cause. Three candidates, each changed alone at update 12k (after the
held-out rise, before any flip): the gain's continued decay (freeze it); the fixed size of the Polar steps on the hidden
matrices, which no longer shrinks as the loss does (eta_w 0.1 -> 0.03); the fixed size of the head's normalized step (eta_v
0.03 -> 0.01; at the head norm 3.6 this step is about 0.8% of it per update).

## 3. Runs (40; seeds: init 7100 + seed, split 8100 + seed, as in campaigns X, Y, Z, ZP and ZM)

| part | cell | seeds | what |
|---|---|---|---|
| P | P_p61_f0.33_lam_v0.45 | 16-20 | campaign ZP's cell (random split) |
| P | M_p61_f0.55_lam_v0.45 | 16-20 | campaign ZM's cell (mirror-free split) |
| P | P_p89_f0.25_lam_v0.33 | 16-20 | campaign ZP's cell |
| P | P_p149_f0.16_lam_v0.23 | 16-20 | campaign ZP's cell |
| B | B_f0.2_lam_v0.275_tau10k | 16-20 | campaign Z's two-block cell, unchanged |
| B | ..._F12 | 11-13 | from update 12000: eta_g = 0 (the gain stays at its value, about 1.30) |
| B | ..._E12 | 11-13 | from update 12000: eta_w = 0.03 |
| B | ..._V12 | 11-13 | from update 12000: eta_v = 0.01 |
| B | B_f0.175_lam_v0.275_tau10k | 11-13 | training fraction 0.175 |
| B | B_f0.225_lam_v0.275_tau10k | 11-13 | training fraction 0.225 |

Every other setting is campaign Y's G2 (one block, d_model 128, 4 heads, MLP 512, RMS head input with the gain 2 -> 1,
float64, Polar feature steps eta_w 0.1, lam_w 0.01, head eta_v 0.03, Fourier readouts, heavy logs every 100, probe every 200,
early stop once held-out accuracy >= 0.99 has held for max(1.5 t, t + 5000), at most 30k updates); two blocks use Z's time
constant 10k (eta_g 1e-4). F12, E12 and V12 equal Z's base runs of seeds 11-13 update for update until update 12000, so each
is a paired comparison. Seeds 16-20 complete seeds 11-20 for every one-block prime (p = 113 is campaign Y's G2, seeds 11-20)
and for the two-block base cell.

## 4. Code

- `tf_runner.py` 1.8 = campaign ZM's 1.7 plus optional late rates (`late_at`, `late`: from update late_at on, the given
  eta_w, lam_w, eta_v, lam_v, eta_g or lam_g replace the configured ones). Without them every update and record is 1.7's.
- `tests/test_q.py` (44 tests): ZP's suite for 1.6 without its manifest test, ZM's suite for 1.7, and 8 new: 1.8 without
  late rates equals 1.7 (`tests/ref_tf_runner_17.py`, ZM's runner) on seven configurations; the switch happens exactly after
  the update at late_at for each of the six rates; after the switch an update equals that of a learner configured with the
  late rates; eta_g = 0 freezes the gain; resume is exact before and after the switch; invalid late settings raise; records
  carry the version and the late settings; the manifest is the table above, checked against ZP's and Z's manifests. Five
  deliberate bugs in the late rates (switch one update early, hidden or gain or head ignoring them, no switch in `run`) each
  fail at least two tests.
- On the real configurations (Z's two-block s11 and ZP's p = 149 s11, 60 updates), 1.8 equals 1.7 bit for bit and the archived
  worker runs to 7 significant digits.
- `worker.py`: campaign ZP's worker 2.0, unchanged.

## 5. Analysis (fixed before the runs)

- Scoring: campaign Z's per-run analysis, unchanged (analyze_Z.py: share of the held-out 10% -> 90% rise inside the post-hoc
  window >= 0.75, Fourier top-8 refit gain >= 0.2, dA_H >= 0.02, misclassified training pairs inside the window 0 (strict)
  or at most 10 (loose), final held-out accuracy on all pairs >= 0.99; a run passes if it passes on both held-out
  definitions, all pairs and the no-mirror pairs).
- Part P: strict and loose passes out of 10 per cell (seeds 11-15 from ZP or ZM, 16-20 from Q), for the across-primes figure
  (p = 61 mirror-free, 89, 113 (Y's G2), 149) and the mirror-pair figures (p = 61 random and mirror-free).
- Part B: the same counts per cell, and per run: the first update after 12k with a misclassified training pair (or none),
  the smallest training margin after 12k, the growth of the second block's MLP norms after 12k (largest value over the value
  at 12k), the CE change across the no-mirror rise, the window share and the final held-out accuracy; F12, E12 and V12 next to
  Z's base run of the same seed.
- Reading: a variant prevents the late failure if none of its 3 runs has a misclassified training pair after 12k and all end
  at held-out accuracy >= 0.99; it works if it also passes strictly in 3 of 3. The same for the two fractions. Anything that
  works is exploratory: the paper would need it confirmed on 10 fresh seeds first.

## 6. Sessions

24 sessions of 4 cores. Q1-Q4: the one-block runs, one thread each, four at once (about 1-1.7 h; the mirror-free p = 61 and
p = 149 runs are the long ones). Q5-Q24: one two-block run each on 4 threads (Z's two-block runs took 0.32 s per update on 2
threads; about 1.8-2.1 h expected for 30k updates on 4). About 185 core-hours. Fewer sessions also work; after 15 minutes the
others take over the runs of IDs that never started.

## 7. Checks before launch

The tests above; the worker's selftest; an end-to-end run of `manifest_test.json` (4 miniature runs, one per kind, on two
sessions) through the worker to a local bare repository.
