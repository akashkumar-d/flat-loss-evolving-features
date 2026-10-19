from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import time
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
ATOL = 2e-9
RTOL = 2e-10
THREADS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {name: f[name] for name in f.files}


def reconstruct(w, v, p):
    first = np.repeat(np.arange(p), p)
    second = np.tile(np.arange(p), p)
    pre = w[:, first].T + w[:, p + second].T
    hidden = np.maximum(pre, 0.)
    return hidden, hidden @ v.T, (first + second) % p


def metrics(logits, labels, train, test):
    relative = logits - logits[np.arange(len(labels)), labels, None]
    ce = np.logaddexp.reduce(relative, axis=1)
    rivals = relative.copy()
    rivals[np.arange(len(labels)), labels] = -np.inf
    margin = -rivals.max(axis=1)
    return dict(train_ce=float(ce[train].mean()), test_ce=float(ce[test].mean()),
                train_strict_count=int(np.sum(margin[train] > 0)),
                test_strict_count=int(np.sum(margin[test] > 0)),
                train_margin=float(margin[train].min()),
                test_margin=float(margin[test].min())), margin


def fft_diagnostics(hidden, logits, labels, test, p):
    centered_logits = logits - logits.mean(axis=1, keepdims=True)
    coeff = np.fft.fft2(centered_logits.reshape(p, p, p), axes=(0, 1), norm="ortho")
    keep = np.zeros_like(coeff)
    k = np.arange(1, p)
    keep[k, k] = coeff[k, k]
    restored = np.fft.ifft2(keep, axes=(0, 1), norm="ortho").reshape(p * p, p)
    imaginary_error = float(np.max(np.abs(restored.imag)))
    addition = restored.real
    remainder = centered_logits - addition

    hcoeff = np.fft.fft2(hidden.reshape(p, p, hidden.shape[1]), axes=(0, 1), norm="ortho")
    hcoeff[0, 0] = 0.
    hdenom = float(np.sum(np.abs(hcoeff) ** 2))
    hnum = float(np.sum(np.abs(hcoeff[k, k]) ** 2))
    fdenom = float(np.sum(centered_logits ** 2))

    rule = addition[:p]
    g = np.diag(rule)[:, None] - rule
    g[np.arange(p), np.arange(p)] = np.inf
    gamma = g.min(axis=1)
    harmful = remainder - remainder[np.arange(p * p), labels, None]
    harmful[np.arange(p * p), labels] = -np.inf
    beta = harmful.max(axis=1)
    lower = gamma[labels] - beta
    actual = centered_logits - centered_logits[np.arange(p * p), labels, None]
    actual[np.arange(p * p), labels] = -np.inf
    actual_margin = -actual.max(axis=1)
    ratio = None
    if np.all(gamma > 0):
        ratio = float(np.max(np.maximum(harmful[test], 0.) / g[labels[test]]))
    return dict(fourier_energy=hnum / hdenom if hdenom else 0.,
                logit_fourier_energy=float(np.sum(np.abs(keep) ** 2)) / fdenom if fdenom else 0.,
                rule_min=float(gamma.min()), harmful_max=float(beta[test].max()),
                global_lower=float(gamma.min() - beta[test].max()),
                row_lower_min=float(lower[test].min()),
                separated_certified=int(np.sum(lower[test] > 0)),
                test_min_margin=float(actual_margin[test].min()), ratio_max=ratio), addition, remainder, dict(
                    max_imaginary_fft=imaginary_error,
                    max_lower_bound_excess=float(np.max(lower - actual_margin)),
                    strict_count=int(np.sum(actual_margin[test] > 0)))


def suffix_start(mask):
    mask = np.asarray(mask, dtype=bool)
    if not len(mask) or not mask[-1]:
        return None
    bad = np.flatnonzero(~mask)
    return int(bad[-1] + 1) if len(bad) else 0


class Audit:
    def __init__(self, deadline):
        self.deadline = deadline
        self.maximum_discrepancy = {}
        self.scalar_comparisons = 0
        self.hash_bindings = 0
        self.verified_hashes = {}

    def tick(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError("Cooperative saved-output audit work limit reached")

    def same(self, name, actual, expected, exact=False):
        self.scalar_comparisons += 1
        if actual is None:
            assert expected is None or np.isnan(expected), (name, actual, expected)
            return
        if expected is None:
            raise AssertionError((name, actual, expected))
        actual = float(actual)
        expected = float(expected)
        assert np.isfinite(actual) and np.isfinite(expected), (name, actual, expected)
        delta = abs(actual - expected)
        self.maximum_discrepancy[name] = max(self.maximum_discrepancy.get(name, 0.), delta)
        if exact:
            assert actual == expected, (name, actual, expected)
        else:
            assert delta <= ATOL + RTOL * abs(expected), (name, actual, expected, delta)

    def bindings(self, mapping, base):
        for name, expected in mapping.items():
            self.tick()
            path = Path(name)
            if not path.is_absolute():
                path = base / path
            assert path.exists(), f"Missing hash-bound input: {path}"
            value = sha(path)
            assert value == expected, f"Hash mismatch: {path}"
            self.hash_bindings += 1
            self.verified_hashes[str(path)] = value


def check_windows(audit, record, trace, diag):
    ntr, nte = record["n_train"], record["n_test"]
    for window in record["windows"]:
        left, right = window["start"], window["end"]
        if trace["step"][-1] < right:
            assert window["status"] == "unavailable" and window["joint_window_pass"] is None
            continue
        assert window["status"] == "available"
        ce = trace["train_ce"][left:right + 1]
        relative = float((ce.max() - ce.min()) / ce.mean()) if ce.mean() > 0 else None
        positions = {int(s): i for i, s in enumerate(diag["step"])}
        gain = float(diag["fourier_energy"][positions[right]] - diag["fourier_energy"][positions[left]])
        fitted = bool(np.all(trace["train_strict_count"][left:right + 1] == ntr))
        start, end = float(trace["test_strict_count"][left] / nte), float(trace["test_strict_count"][right] / nte)
        certified = float(diag["separated_certified"][positions[right]] / nte)
        flag = bool(fitted and relative is not None and relative <= .10 and gain >= .01
                    and start <= .50 and end >= .90 and certified >= .90)
        for name, value in dict(relative_ce_range=relative, ce_min=ce.min(), ce_max=ce.max(),
                                hidden_energy_gain=gain, test_start_fraction=start,
                                test_end_fraction=end, certified_end_fraction=certified).items():
            audit.same("window_" + name, value, window[name])
        assert window["all_train_fit"] == fitted and window["joint_window_pass"] == flag


def check_case(audit, cfg, parent_record):
    case = HERE / cfg["name"]
    report = dict(name=cfg["name"], status=parent_record["status"], checkpoints_checked=0,
                  checkpoint_steps_without_signed_observation=[], crossed_tables_checked=0,
                  missing_artifacts=[])
    assert parent_record["config"] == cfg
    path = case / "RESULTS.json"
    if path.exists():
        record = json.loads(path.read_text())
        assert record == parent_record, f"Parent/case result disagreement: {cfg['name']}"
        closepath = case / "CLOSED.json"
        assert closepath.exists()
        close = json.loads(closepath.read_text())
        assert close["results_sha256"] == sha(path) and close["status"] == record["status"]
        audit.hash_bindings += 1
        launch = json.loads((case / "LAUNCH.json").read_text())
        assert launch["config"] == cfg
    else:
        assert parent_record["status"] in ["failed", "unrun"]
        report["missing_artifacts"].append("RESULTS.json / CLOSED.json")
        record = parent_record
    audit.bindings(record["output_sha256"], case)
    existing_checkpoints = {str(p.resolve()) for p in case.glob("checkpoint_*.npz")}
    declared_checkpoints = {str(Path(cp["path"]).resolve()) for cp in record.get("checkpoints", [])}
    assert existing_checkpoints == declared_checkpoints, "Unbound or missing saved checkpoint"
    p = cfg["p"]
    ntr = int(cfg["train_fraction"] * p * p)
    nte = p * p - ntr
    assert record["n_train"] == ntr and record["n_test"] == nte
    last_observed = record["observed_states"] - 1
    updates = record["updates_completed"]
    if updates is None:
        assert record["status"] == "failed" and record["observed_states"] == 0
        expected_pending = False
    else:
        assert 0 <= updates <= cfg["steps"]
        assert updates in [max(0, last_observed), max(0, last_observed + 1)]
        expected_pending = bool(last_observed >= 0 and updates > last_observed)
    assert record.get("pending_unobserved_update", False) == expected_pending
    if not (case / "trajectory.npz").exists():
        assert record["observed_states"] == 0 and record["status"] in ["failed", "unrun"]
        report["missing_artifacts"].append("trajectory: no observed state")
        return report
    trace = load_npz(case / "trajectory.npz")
    count = record["observed_states"]
    np.testing.assert_array_equal(trace["step"], np.arange(count))
    assert record["completed_steps"] == count - 1
    for key, value in trace.items():
        assert value.shape == (count,) and np.isfinite(value).all(), key
    for key, maximum in [("train_strict_count", ntr), ("test_strict_count", nte)]:
        assert np.all(trace[key] == trace[key].astype(int))
        assert np.all((trace[key] >= 0) & (trace[key] <= maximum))
    diag = load_npz(case / "diagnostics.npz") if (case / "diagnostics.npz").exists() else None
    diag_positions = {}
    if diag is not None:
        dstep = diag["step"].astype(int)
        assert np.all(diag["step"] == dstep) and np.all(np.diff(dstep) > 0)
        assert dstep[0] == 0 and dstep[-1] <= trace["step"][-1]
        diag_positions = {int(s): i for i, s in enumerate(dstep)}
        assert np.all(diag["separated_certified"] <= trace["test_strict_count"][dstep])
        assert np.all(diag["separated_certified"] >= 0)
    if record["status"] == "complete":
        assert count == cfg["steps"] + 1 and record["updates_completed"] == cfg["steps"]
        assert diag is not None
        np.testing.assert_array_equal(diag["step"], np.arange(0, cfg["steps"] + 1, 100))
        assert {cp["step"] for cp in record["checkpoints"]} == set(range(0, cfg["steps"] + 1, 1000))
    expected_perm = np.random.default_rng(cfg["split_seed"]).permutation(p * p)
    expected_train = np.sort(expected_perm[:ntr])
    expected_test = np.sort(expected_perm[ntr:])
    components = {}
    for cp in record.get("checkpoints", []):
        audit.tick()
        step = cp["step"]
        assert 0 <= step < count
        z = load_npz(Path(cp["path"]))
        w, v, tr, te = z["W"], z["V"], z["train_idx"], z["test_idx"]
        assert w.shape == (cfg["width"], 2 * p) and v.shape == (p, cfg["width"])
        assert np.isfinite(w).all() and np.isfinite(v).all()
        np.testing.assert_array_equal(tr, expected_train)
        np.testing.assert_array_equal(te, expected_test)
        np.testing.assert_array_equal(np.sort(np.r_[tr, te]), np.arange(p * p))
        if step == 0:
            rng = np.random.default_rng(cfg["init_seed"])
            ew = cfg["initial_scale"] * rng.normal(size=w.shape) / np.sqrt(p)
            ev = rng.normal(size=v.shape) / (cfg["initial_scale"] * np.sqrt(cfg["width"]))
            np.testing.assert_array_equal(w, ew)
            np.testing.assert_array_equal(v, ev)
        h, f, y = reconstruct(w, v, p)
        computed, margins = metrics(f, y, tr, te)
        for name, value in computed.items():
            audit.same(name, value, trace[name][step], exact="count" in name)
        signed, addition, remainder, extra = fft_diagnostics(h, f, y, te, p)
        assert extra["max_imaginary_fft"] <= ATOL
        assert extra["max_lower_bound_excess"] <= ATOL
        audit.maximum_discrepancy["max_imaginary_fft"] = max(
            audit.maximum_discrepancy.get("max_imaginary_fft", 0.), extra["max_imaginary_fft"])
        audit.maximum_discrepancy["max_lower_bound_excess"] = max(
            audit.maximum_discrepancy.get("max_lower_bound_excess", 0.), extra["max_lower_bound_excess"])
        assert signed["separated_certified"] <= computed["test_strict_count"]
        if step in diag_positions:
            i = diag_positions[step]
            for name, value in signed.items():
                audit.same(name, value, diag[name][i], exact=name == "separated_certified")
        else:
            report["checkpoint_steps_without_signed_observation"].append(step)
            assert record["status"] != "complete"
        if step in [10000, 20000]:
            components[step] = addition, remainder, y, tr, te
        report["checkpoints_checked"] += 1
    if (case / "crossed_components.npz").exists():
        crossed = load_npz(case / "crossed_components.npz")
        np.testing.assert_array_equal(crossed["step"], [10000, 20000])
        assert set(components) == {10000, 20000}
        for i, left in enumerate([10000, 20000]):
            for j, right in enumerate([10000, 20000]):
                a, _, y, tr, te = components[left]
                _, z, _, _, _ = components[right]
                values, _ = metrics(a + z, y, tr, te)
                for target, source in [("train_ce", "train_ce"), ("test_ce", "test_ce"),
                                        ("train_correct", "train_strict_count"),
                                        ("test_correct", "test_strict_count")]:
                    audit.same("cross_" + target, values[source], crossed[target][i, j], exact="correct" in target)
                report["crossed_tables_checked"] += 1
    else:
        assert record["status"] != "complete"
        report["missing_artifacts"].append("crossed components: no complete saved pair")
    if record["final"]:
        final = record["final"]
        for name, source in [("train_ce", "train_ce"), ("test_ce", "test_ce"),
                              ("train_correct", "train_strict_count"),
                              ("test_correct", "test_strict_count")]:
            audit.same("final_" + name, final[name], trace[source][-1], exact="correct" in name)
        fitted = suffix_start(trace["train_strict_count"] == ntr)
        recovered = suffix_start(trace["test_strict_count"] / nte >= .90)
        assert final["first_train_full_suffix"] == fitted
        assert final["first_test90_suffix"] == recovered
        assert final["first_test100_suffix"] == suffix_start(trace["test_strict_count"] == nte)
        bad = (fitted is not None and fitted + 1000 <= count - 1
               and bool(np.all(trace["test_strict_count"][fitted:fitted + 1001] / nte <= .10)))
        delayed = bool(record["status"] == "complete" and bad and recovered is not None and recovered >= fitted + 1000)
        assert final["initial_fitted_bad_test_1000"] == bool(bad)
        assert final["delayed_recovery"] == delayed
        if diag is None or not len(diag["step"]):
            assert final["hidden_energy"] is None and final["separated_certified"] is None
            report["final_diagnostic_step"] = None
        else:
            audit.same("final_hidden_energy", final["hidden_energy"], diag["fourier_energy"][-1])
            audit.same("final_separated_certified", final["separated_certified"],
                       diag["separated_certified"][-1], exact=True)
            report["final_diagnostic_step"] = int(diag["step"][-1])
    if record["windows"]:
        assert diag is not None
        check_windows(audit, record, trace, diag)
    report["observed_states"] = count
    return report


def main():
    assert all(os.environ.get(name) == "1" for name in THREADS)
    start = time.monotonic()
    result_hash = sha(HERE / "RESULTS.json")
    checker_hash = sha(__file__)
    scope_hash = sha(HERE / "POSTCHECK_SCOPE.md")
    dump(HERE / "CHECK_LAUNCH.json", dict(source_sha256=checker_hash,
         scope_sha256=scope_hash, results_sha256=result_hash,
         fabricated_test_source_sha256=sha(HERE / "test_check_outputs.py"),
         work_seconds=120, total_seconds=150, new_training_updates=0))
    audit = Audit(start + 120)
    reports = []
    status = "failed"
    error = None
    scientific_status = None
    try:
        result = json.loads((HERE / "RESULTS.json").read_text())
        scientific_status = result["status"]
        launch = json.loads((HERE / "LAUNCH.json").read_text())
        closed = json.loads((HERE / "CLOSED.json").read_text())
        assert closed["results_sha256"] == result_hash and closed["status"] == result["status"]
        audit.hash_bindings += 1
        audit.verified_hashes[str(HERE / "CLOSED.json")] = sha(HERE / "CLOSED.json")
        assert launch["roster"] == result["roster"]
        assert launch["sources"] == result["source_sha256"]
        assert len(result["roster"]) == result["declared_runs"] == 9
        assert len(result["runs"]) == 9
        assert result["total_new_updates"] == sum(r["updates_completed"] or 0 for r in result["runs"])
        assert not result["source_binding_errors"]
        audit.bindings(result["source_sha256"], HERE)
        audit.bindings(result["output_sha256"], HERE)
        for cfg, record in zip(result["roster"], result["runs"]):
            audit.tick()
            reports.append(check_case(audit, cfg, record))
            print(json.dumps(reports[-1]), flush=True)
        assert sum(r["status"] == "complete" for r in reports) == result["completed_runs"]
        assert sha(HERE / "RESULTS.json") == result_hash
        assert sha(__file__) == checker_hash and sha(HERE / "POSTCHECK_SCOPE.md") == scope_hash
        status = "complete"
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    elapsed = time.monotonic() - start
    if elapsed > 150:
        status = "resource_overrun"
    output = dict(status=status, scientific_status=scientific_status,
                  results_sha256=result_hash, source_sha256=sha(__file__),
                  runs=reports, declared_roster_retained=9,
                  checkpoints_checked=sum(r["checkpoints_checked"] for r in reports),
                  crossed_tables_checked=sum(r["crossed_tables_checked"] for r in reports),
                  scalar_comparisons=audit.scalar_comparisons, hash_bindings=audit.hash_bindings,
                  maximum_absolute_discrepancy=audit.maximum_discrepancy,
                  verified_hashes=audit.verified_hashes, elapsed_seconds=elapsed, error=error,
                  new_training_updates=0, interval_arithmetic=False)
    dump(HERE / "CHECKS.json", output)
    print(json.dumps({k: output[k] for k in ["status", "checkpoints_checked", "crossed_tables_checked",
                                             "scalar_comparisons", "hash_bindings", "elapsed_seconds", "error"]}), flush=True)
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
