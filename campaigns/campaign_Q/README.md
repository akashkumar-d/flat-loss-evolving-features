# Campaign Q: ten seeds per prime for one block, and why two blocks fail late (2026-10-13)

claude/1hl-figures, at Akash's request on the arXiv draft (v2, revision 2): every panel of a figure should have the same
number of runs, ideally 10; p = 61 in the across-primes figure uses the mirror-free split; for two blocks, more seeds,
other fractions, and a look at what fails. `PLAN.md` has the design and the analysis (fixed before the runs).

- Part P (20 runs): seeds 16-20 of campaign ZP's cells at p = 61 (random split), 89 and 149 and of campaign ZM's mirror-free
  p = 61 cell, so that every prime has seeds 11-20 (p = 113: campaign Y's G2).
- Part B (20 runs, two blocks at p = 113): seeds 16-20 of campaign Z's cell B_f0.2_lam_v0.275_tau10k; three paired tests that
  change one thing from update 12000 on (gain frozen; eta_w 0.1 -> 0.03; eta_v 0.03 -> 0.01), seeds 11-13; fractions 0.175
  and 0.225, seeds 11-13.

Files for the workers (registered in `HASHES.json`; `push_Q.sh` in `natural_plateau_shared/` checks and pushes them):
`tf_runner.py` 1.8 (campaign ZM's 1.7 plus optional late rates; without them identical to 1.7), `worker.py` (campaign ZP's,
unchanged), `manifest.json` (written by `make_manifest.py`), `campaign.json`, `WORKER.md`, `tests/test_q.py` (44 tests; it
runs `tests/test_tf_16suite.py` and `tests/test_zm_suite.py`; `tests/ref_tf_runner_17.py` is campaign ZM's runner, for the
identity test). 40 runs; sessions Q1-Q24.

Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/`.
