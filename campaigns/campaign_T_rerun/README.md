# Campaign T rerun: the five campaign-T runs that crashed (2026-10-08)

claude/1hl-figures, at Akash's request. Five campaign-T runs crashed on every session that tried them, always in the float32 SVD
of the Polar step (`torch.linalg.svd`: Intel oneMKL "Parameter 4 was incorrect on entry to SLASCL", then LinAlgError "failed to
converge ... (error code: 1)"; traceback from session T11, run T_P1_f0.3_lam_w0.03_eta_w0.1_lam_v3_s1). The same MKL message
and error appear when the matrix holds an infinite entry; an infinite gradient or an SVD failure on a finite matrix are both
possible causes.

`tf_runner.py` here is version 1.1. It differs from campaign T's 1.0 only in two safeguards:
- when the float32 SVD raises or returns non-finite factors, the same matrix is decomposed again in float64; such updates are
  listed in `svd_fallback_steps`;
- when the loss or any gradient is not finite, the run stops there and is written out with stop_reason 'nonfinite'.

Every update on which the float32 SVD works is computed exactly as in 1.0: on the four test configurations of campaign T
(200 updates, checkpoint and resume every 50), 1.1 gives byte-identical losses and observations; `tests/test_tf.py` adds a test
of both safeguards.

Runs (same ids, configurations, seeds and one thread each as in campaign T's manifest):
- R1: T_P2_f0.3_lam_w0.03_eta_w0.1_lam_v0.2_s2, T_P1_f0.3_lam_w0.03_eta_w0.1_lam_v3_s1, T_P1_f0.3_lam_w0.03_eta_w0.1_lam_v3_s2
- R2: T_P1_f0.5_lam_w0.03_eta_w0.1_lam_v3_s1, T_P1_f0.5_lam_w0.03_eta_w0.1_lam_v3_s2

Results go to `results/<run>/` on the session branches; `../collect_results.sh` copies them to `collected/` here. Campaign T's
`analyze_T.py` reads them together with campaign T's 75 runs.
