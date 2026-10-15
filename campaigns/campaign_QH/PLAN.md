# Campaign QH: Theorem 4.1's configuration, eta_V = 3e-4, seeds 7-10 (claude/1hl-figures, 2026-10-13)

Fixed before any campaign-QH run.

Akash, on the arXiv draft (v2, revision 2): every panel of a figure should have the same number of runs. The paper's figure
of Theorem 4.1's configuration from an iid initialization (campaign H, p = 101, width 512, the quadratic-residue split, the
theorem's clock, lam_V = 0, 40k updates) shows 6 runs at head step eta_V = 3e-4 and 10 at the primary eta_V = 1e-3. QH runs
eta_V = 3e-4 on seeds 7-10 (init 9107-9110), so that both rows show the same 10 initializations; every run is campaign H's run
of that arm with only the seed changed (`tests/test_qh.py` checks it against campaign H's manifest).

Code: `t41_runner.py` and `tests/test_t41.py` are campaign H's, byte for byte; `worker.py` is worker 2.0 (campaign ZP's),
unchanged. Here the runner reproduces campaign H's archived run T41_ev3e-4_s1 to 7 significant digits on its first 300
updates (0.14 s per update).

Analysis (fixed): campaign H's analysis (`analyze.py`, endpoints J1-J4 and the joint endpoint of its PLAN.md) on the 10 runs of
the arm (seeds 1-6 from campaign H, 7-10 from QH); the paper's figure redrawn with 10 runs per row.

Session: QH1, four runs at once on one thread each, about 1.6 hours (campaign H's runs took 1.6 hours on a worker).
