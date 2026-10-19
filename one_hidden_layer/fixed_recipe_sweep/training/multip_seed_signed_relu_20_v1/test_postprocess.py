import time
import unittest

import matplotlib.pyplot as plt
import numpy as np

import postprocess as p


def fake_run(name="fixture", status="complete", modulus=11):
    cfg = next(c for c in p.ROSTER if c["p"] == modulus).copy()
    cfg["name"] = name
    n_test = modulus * modulus - int(.8 * modulus * modulus)
    observed_window = dict(start=15000, end=20000, status="available", joint_window_pass=True)
    record = dict(config=cfg, status=status, completed_steps=40000 if status == "complete" else 20000,
                  observed_states=40001 if status == "complete" else 20001,
                  final=dict(test_correct=n_test, train_ce=.01, delayed_recovery=True),
                  windows=[observed_window], error={"message":"fabricated"} if status == "failed" else None,
                  resource_stop_reason="fabricated resource cutoff" if status == "partial" else None)
    return dict(name=name, p=modulus, status=status, config=cfg, record=record,
                n_test=n_test, n_train=modulus * modulus-n_test,
                trajectory=dict(step=np.array([0, 20000]), test_strict_count=np.array([0, n_test])),
                diagnostics=None, cross=None)


class PostprocessTests(unittest.TestCase):
    def test_wilson_all_twenty_known(self):
        out = p.binary_summary([True] * 10 + [False] * 10)
        self.assertEqual((out["declared"], out["known_positive"], out["known_negative"], out["unresolved"]), (20, 10, 10, 0))
        self.assertAlmostEqual(out["interval_low"], .2992980081982123)
        self.assertAlmostEqual(out["interval_high"], .7007019918017877)
        self.assertEqual(out["interval_kind"], "ordinary nominal Wilson")

    def test_censor_envelope_covers_all_possible_completions(self):
        out = p.binary_summary([True] * 7 + [False] * 10 + [None] * 3)
        self.assertEqual(out["declared"], 20)
        self.assertEqual(out["unresolved"], 3)
        for k in range(7, 11):
            lo, hi = p.wilson(k, 20)
            self.assertLessEqual(out["interval_low"], lo)
            self.assertGreaterEqual(out["interval_high"], hi)
        self.assertIn("not a measured-rate CI", out["interval_kind"])
        all_missing = p.binary_summary([None] * 20)
        self.assertAlmostEqual(all_missing["interval_low"], 0.)
        self.assertAlmostEqual(all_missing["interval_high"], 1.)

    def test_partial_window_known_but_40k_unresolved(self):
        events = p.run_events(fake_run(status="partial"))
        self.assertTrue(events["joint_15000_20000"])
        self.assertTrue(events["test100_at20k"])
        self.assertIsNone(events["test100_at40k"])
        self.assertIsNone(events["delayed_recovery"])

    def test_unrun_is_not_a_measured_negative(self):
        run = fake_run(status="unrun")
        run["trajectory"] = None
        run["record"]["windows"] = []
        self.assertTrue(all(value is None for value in p.run_events(run).values()))

    def test_regular_statistics_unchanged_without_offgrid_artifact(self):
        series = [(np.array([0, 100, 200]), np.array([1., 2., 3.])),
                  (np.array([0, 100, 153]), np.array([2., 4., 999.]))]
        original = p.exact_summary(series)
        regular = p.exact_summary(series, 100)
        take = original["step"] % 100 == 0
        for key in original:
            np.testing.assert_array_equal(original[key][take], regular[key])
        np.testing.assert_array_equal(regular["n"], [2, 2, 1])
        self.assertEqual(series[1][0][-1], 153)

    def test_derived_window_observer_regular_grid_and_terminal_retention(self):
        a, b = fake_run("a"), fake_run("b", "partial")
        a["window_observer"] = dict(step=np.array([15000, 15100, 15200]), hidden_change=np.array([0., .1, .2]))
        b["window_observer"] = dict(step=np.array([15000, 15100, 15153]), hidden_change=np.array([0., .2, 999.]))
        view = p.Presenter([a, b], time.monotonic() + 60)
        fig, ax = plt.subplots()
        out = view.plot(ax, [a, b], "window_observer", "hidden_change", "blue", "fixture")
        self.assertNotIn(15153, out["step"])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), [0., .2, 999.])
        self.assertEqual(ax.lines[2].get_xdata()[0], 15.153)
        self.assertEqual(ax.lines[2].get_ydata()[0], 999.)
        plt.close(fig)

    def test_summary_retains_all_statuses_errors_and_resource_reasons(self):
        runs = []
        for modulus in p.MODULI:
            runs += [fake_run(f"p{modulus}_{j}", status, modulus) for j, status in
                     enumerate(["complete", "partial", "failed", "unrun"])]
        summary = p.make_summary(runs)
        self.assertEqual(summary["declared_runs"], 24)
        self.assertEqual(summary["by_p"][0]["statuses"], dict(complete=1, partial=1, failed=1, unrun=1))
        self.assertEqual(summary["runs"][1]["resource_stop_reason"], "fabricated resource cutoff")
        self.assertEqual(summary["runs"][2]["error"], {"message":"fabricated"})

    def test_zero_ce_retained_with_symlog_not_a_floor(self):
        values = np.array([1., 0.])
        mode, threshold = p.ce_scale([(np.array([0, 1]), values)])
        self.assertEqual(mode, "symlog")
        self.assertGreater(threshold, 0.)
        np.testing.assert_array_equal(values, [1., 0.])
        self.assertEqual(p.ce_scale([(np.array([0]), np.array([.01]))])[0], "log")
        tiny_values = np.array([np.nextafter(0., 1.), 0.])
        mode, threshold = p.ce_scale([(np.array([0, 1]), tiny_values)])
        self.assertEqual(mode, "symlog")
        self.assertGreater(threshold, 0.)
        self.assertEqual(tiny_values[0], np.nextafter(0., 1.))

    def test_every_individual_is_plotted(self):
        runs = [fake_run(str(i)) for i in range(20)]
        for i, run in enumerate(runs):
            run["trajectory"]["train_ce"] = np.array([1., .01 + i * .0001])
        view = p.Presenter(runs, time.monotonic() + 60)
        fig, ax = plt.subplots()
        result = view.plot(ax, runs, "trajectory", "train_ce", "blue", "fixture")
        self.assertEqual(len(ax.lines), 21)
        np.testing.assert_array_equal(result["n"], [20, 20])
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
