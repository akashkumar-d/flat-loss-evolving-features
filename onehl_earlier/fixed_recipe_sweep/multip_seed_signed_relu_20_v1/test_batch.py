import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import batch_config as config
import train_stage as stage
import pipeline


def toy_config():
    return dict(name="fabricated_toy", p=3, width=8, replicate=1,
                init_seed=715, split_seed=716, steps=5, train_fraction=.8,
                initial_scale=24., eta_w=.03, eta_v=.03, decay_w=.005, decay_v=.2)


class Tests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {key: "1" for key in config.THREAD_VARS})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_roster_and_fixed_params(self):
        self.assertTrue(config.validate_roster())
        self.assertEqual([r["p"] for r in config.ROSTER[:6]], config.MODULI)
        self.assertEqual([int(.8 * p * p) for p in config.MODULI], [96, 135, 231, 288, 423, 672])
        for r in config.ROSTER:
            self.assertEqual((r["width"], r["initial_scale"], r["eta_w"], r["eta_v"], r["decay_w"], r["decay_v"]),
                             (512, 24., .03, .03, .005, .2))

    def test_cached_design_identical_update_trajectory(self):
        cfg = toy_config()
        x, y, tr, te, w, v = stage.core.initialize(3, 8, cfg["init_seed"], cfg["split_seed"])
        expected = []
        for t in range(6):
            rec, pre, h, _, residual = stage.core.forward(x, y, tr, te, w, v)
            expected.append(rec)
            if t != 5:
                c, d = stage.core.forces(x[tr], pre[tr], h[tr], residual, v)
                w, v, _ = stage.core.update(w, v, c, d)
        with tempfile.TemporaryDirectory(prefix="relu20-toy-") as folder:
            root = Path(folder)
            with mock.patch.object(stage, "HERE", root), contextlib.redirect_stdout(io.StringIO()):
                result = stage.run_one(cfg, time.monotonic() + 60)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["updates_completed"], 5)
            self.assertEqual(result["observed_states"], 6)
            self.assertFalse(result["pending_unobserved_update"])
            self.assertEqual([c["step"] for c in result["checkpoints"]], [0, 5])
            with np.load(root / cfg["name"] / "trajectory.npz") as trace:
                for key in expected[0]:
                    np.testing.assert_array_equal(trace[key], [r[key] for r in expected])
            with np.load(root / cfg["name"] / "checkpoint_00005.npz") as cp:
                np.testing.assert_array_equal(cp["W"], w)
                np.testing.assert_array_equal(cp["V"], v)
            for path, digest in result["output_sha256"].items():
                self.assertEqual(stage.sha(path), digest)
            with mock.patch.object(stage, "HERE", root), self.assertRaises(FileExistsError):
                stage.run_one(cfg, time.monotonic() + 60)

    def test_expired_before_start_no_initialization(self):
        with tempfile.TemporaryDirectory(prefix="relu20-expired-") as folder:
            with mock.patch.object(stage, "HERE", Path(folder)), mock.patch.object(stage.core, "initialize") as init, contextlib.redirect_stdout(io.StringIO()):
                result = stage.run_one(toy_config(), time.monotonic() - 1)
            init.assert_not_called()
            self.assertEqual(result["status"], "unrun")
            self.assertEqual(result["observed_states"], 0)
            self.assertEqual(result["updates_completed"], 0)
            self.assertEqual(result["windows"], [])

    def test_expired_after_one_observation_retained_as_partial(self):
        with tempfile.TemporaryDirectory(prefix="relu20-partial-") as folder:
            clock = [0., .1, 2., 2.1]
            with mock.patch.object(stage, "HERE", Path(folder)), mock.patch.object(stage.time, "monotonic", side_effect=clock), contextlib.redirect_stdout(io.StringIO()):
                result = stage.run_one(toy_config(), 1.)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["resource_stop_reason"], "global_time")
            self.assertEqual(result["observed_states"], 1)
            self.assertEqual(result["updates_completed"], 0)
            self.assertFalse(result["final"]["delayed_recovery"])
            self.assertTrue(all(w["joint_window_pass"] is None for w in result["windows"]))

    def test_persistent_suffix_and_zero_ce_window(self):
        self.assertEqual(stage.first_suffix(np.array([False, True, False, True, True])), 3)
        self.assertIsNone(stage.first_suffix(np.array([True, False])))
        trace = {"step": np.arange(3), "train_ce": np.zeros(3),
                 "train_strict_count": np.full(3, 7), "test_strict_count": np.array([0, 2, 2])}
        diags = [dict(step=0, fourier_energy=0., separated_certified=0),
                 dict(step=2, fourier_energy=.2, separated_certified=2)]
        result = stage.window_summary(trace, diags, 0, 2, 7, 2)
        self.assertIsNone(result["relative_ce_range"])
        self.assertFalse(result["joint_window_pass"])

    def test_progress_prefers_terminal_parent_record(self):
        cfg = toy_config()
        with tempfile.TemporaryDirectory(prefix="relu20-progress-") as folder:
            root = Path(folder)
            with mock.patch.object(stage, "HERE", root), mock.patch.object(stage, "ROSTER", [cfg]), mock.patch.object(stage, "MODULI", [3]):
                counts = stage.progress_snapshot(time.monotonic(), {cfg["name"]: dict(status="failed", completed_steps=0)})
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(counts["queued"], 0)

    def test_supervisor_owns_timed_out_process_group(self):
        child = mock.Mock(pid=12345)
        child.wait.side_effect = [subprocess.TimeoutExpired("toy", 1), 0, 0]
        with tempfile.TemporaryDirectory(prefix="relu20-supervisor-") as folder:
            with mock.patch.object(pipeline.subprocess, "Popen", return_value=child) as spawn, mock.patch.object(pipeline.os, "killpg") as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    pipeline.supervise_stage(Path("fabricated.py"), 1, Path(folder) / "log")
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            self.assertEqual(kill.call_args_list, [mock.call(12345, pipeline.signal.SIGTERM),
                                                   mock.call(12345, pipeline.signal.SIGKILL)])

    def test_failed_observed_toy_passes_independent_checker(self):
        import audit_stage
        cfg = toy_config()
        with tempfile.TemporaryDirectory(prefix="relu20-failed-") as folder:
            root = Path(folder)
            with mock.patch.object(stage, "HERE", root), mock.patch.object(stage.core, "update", side_effect=ValueError("fabricated failure")), contextlib.redirect_stdout(io.StringIO()):
                result = stage.run_one(cfg, time.monotonic() + 60)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["observed_states"], 1)
            with mock.patch.object(audit_stage.checker, "HERE", root):
                audit = audit_stage.checker.Audit(time.monotonic() + 60)
                checked = audit_stage.checker.check_case(audit, cfg, result)
            self.assertEqual(checked["checkpoints_checked"], 1)
            self.assertEqual(checked["status"], "failed")


if __name__ == "__main__":
    unittest.main()
