import concurrent.futures as futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from batch_config import (HERE, PRIOR, MODULI, ROSTER, WORKERS, ARM_SECONDS,
                          TRAIN_SECONDS, DISK_RESERVE, THREAD_VARS, validate_roster)
sys.path.insert(0, str(PRIOR))
import core
from run_sweep import first_suffix, window_summary
assert Path(core.__file__).resolve().parent == PRIOR


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value, exclusive=True):
    path = Path(path)
    if exclusive:
        with path.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
    else:
        temp = path.with_name(path.name + ".tmp")
        with temp.open("w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
        os.replace(temp, path)


def run_one(cfg, global_deadline):
    assert all(os.environ.get(k) == "1" for k in THREAD_VARS)
    start = time.monotonic()
    deadline = min(global_deadline, start + ARM_SECONDS)
    folder = HERE / cfg["name"]
    folder.mkdir()
    outputs, diags, checkpoints, components = {}, [], [], {}
    error = None
    observed, updates, status, trace = -1, 0, "unrun", {}
    ntr = int(cfg["train_fraction"] * cfg["p"]**2)
    nte = cfg["p"]**2 - ntr
    stop_reason = None
    dump(folder / "LAUNCH.json", dict(config=cfg, work_seconds=ARM_SECONDS,
                                      global_seconds_remaining=max(0, global_deadline - start)))

    def save(path, **arrays):
        with Path(path).open("xb") as f:
            np.savez_compressed(f, **arrays)
        outputs[str(path)] = sha(path)

    try:
        if time.monotonic() < deadline and shutil.disk_usage(HERE).free >= DISK_RESERVE:
            p = cfg["p"]
            x, y, tr, te, w, v = core.initialize(p, cfg["width"], cfg["init_seed"], cfg["split_seed"])
            assert len(tr) == ntr and len(te) == nte
            x_train = x[tr]
            trace = {k: np.empty(cfg["steps"] + 1, dtype=int if "count" in k else float) for k in
                     ["train_ce", "test_ce", "train_strict_count", "test_strict_count", "train_margin", "test_margin"]}
            for step in range(cfg["steps"] + 1):
                rec, pre, h, f, residual = core.forward(x, y, tr, te, w, v)
                assert all(np.isfinite(val) for val in rec.values()), "Nonfinite observed scalar"
                for key, value in rec.items():
                    trace[key][step] = value
                observed = step
                clock = time.monotonic()
                disk_stop = step % 1000 == 0 and shutil.disk_usage(HERE).free < DISK_RESERVE
                terminal = step == cfg["steps"] or clock >= deadline or disk_stop
                if step % 100 == 0 or terminal:
                    diag, a, z = core.diagnostics(h, f, y, te, p)
                    assert diag["separated_certified"] <= rec["test_strict_count"]
                    diags.append(dict(step=step, **diag))
                    if step in [10000, 20000]:
                        components[step] = (a.copy(), z.copy())
                if step % 1000 == 0 or terminal:
                    cp = folder / f"checkpoint_{step:05d}.npz"
                    save(cp, W=w, V=v, train_idx=tr, test_idx=te)
                    checkpoints.append(dict(step=step, path=str(cp)))
                if step % 1000 == 0 or terminal:
                    progress = dict(run=cfg["name"], status="running", step=step,
                                    train_ce=rec["train_ce"], test_ce=rec["test_ce"],
                                    test_correct=rec["test_strict_count"], n_test=nte,
                                    elapsed_seconds=round(clock - start, 2))
                    dump(folder / "PROGRESS.json", progress, exclusive=False)
                    if step % 5000 == 0 or terminal:
                        print(json.dumps(progress), flush=True)
                if terminal:
                    status = "complete" if step == cfg["steps"] else "partial"
                    stop_reason = ("horizon_complete" if status == "complete" else
                                   "disk_reserve" if disk_stop else
                                   "global_time" if clock >= global_deadline else "arm_time")
                    break
                c, d = core.forces(x_train, pre[tr], h[tr], residual, v)
                wn, vn, _ = core.update(w, v, c, d)
                assert np.isfinite(wn).all() and np.isfinite(vn).all()
                w, v = wn, vn
                updates += 1
        else:
            stop_reason = "resource_limit_before_initialization"
    except Exception as exc:
        status, stop_reason = "failed", "exception"
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    windows, summary = [], {}
    try:
        if observed >= 0:
            trace = {k: value[:observed + 1] for k, value in trace.items()}
            trace["step"] = np.arange(observed + 1)
            save(folder / "trajectory.npz", **trace)
            if diags:
                save(folder / "diagnostics.npz", **{
                    k: np.array([np.nan if r[k] is None else r[k] for r in diags]) for k in diags[0]})
            if updates == observed and not any(cp["step"] == observed for cp in checkpoints):
                cp = folder / f"checkpoint_{observed:05d}.npz"
                save(cp, W=w, V=v, train_idx=tr, test_idx=te)
                checkpoints.append(dict(step=observed, path=str(cp)))
            if all(s in components for s in [10000, 20000]):
                crosses = {k: np.empty((2, 2), dtype=int if "correct" in k else float) for k in
                           ["train_ce", "test_ce", "train_correct", "test_correct"]}
                for i, a_time in enumerate([10000, 20000]):
                    for j, z_time in enumerate([10000, 20000]):
                        values = core.table_metrics(components[a_time][0] + components[z_time][1], y, tr, te)
                        for key, value in values.items():
                            crosses[key][i, j] = value
                        if i == j:
                            assert values["train_correct"] == trace["train_strict_count"][a_time]
                            assert values["test_correct"] == trace["test_strict_count"][a_time]
                            assert abs(values["train_ce"] - trace["train_ce"][a_time]) <= 2e-10
                            assert abs(values["test_ce"] - trace["test_ce"][a_time]) <= 2e-10
                save(folder / "crossed_components.npz", step=np.array([10000, 20000]), **crosses)
            windows = [window_summary(trace, diags, l, r, ntr, nte) for l, r in [(10000, 20000), (15000, 20000)]]
            fitted = first_suffix(trace["train_strict_count"] == ntr)
            recovered = first_suffix(trace["test_strict_count"] / nte >= .90)
            bad = bool(fitted is not None and fitted + 1000 <= observed and
                       np.all(trace["test_strict_count"][fitted:fitted + 1001] / nte <= .10))
            summary = dict(train_ce=float(trace["train_ce"][-1]), test_ce=float(trace["test_ce"][-1]),
                           train_correct=int(trace["train_strict_count"][-1]), test_correct=int(trace["test_strict_count"][-1]),
                           first_train_full_suffix=fitted, first_test90_suffix=recovered,
                           first_test100_suffix=first_suffix(trace["test_strict_count"] == nte),
                           initial_fitted_bad_test_1000=bad,
                           delayed_recovery=bool(status == "complete" and bad and recovered is not None and recovered >= fitted + 1000),
                           hidden_energy=diags[-1]["fourier_energy"] if diags else None,
                           separated_certified=diags[-1]["separated_certified"] if diags else None)
    except Exception as exc:
        status, stop_reason = "failed", "finalization_exception"
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(), previous_error=error)
    result = dict(status=status, config=cfg, p=cfg["p"], n_train=ntr, n_test=nte,
                  completed_steps=max(0, observed), observed_states=observed + 1, updates_completed=updates,
                  pending_unobserved_update=updates > observed if observed >= 0 else False,
                  checkpoints=checkpoints, windows=windows, final=summary, error=error,
                  resource_stop_reason=stop_reason, output_sha256=outputs,
                  elapsed_seconds=time.monotonic() - start,
                  initialization_theorem_proved=False, causal_optimizer_theorem_proved=False)
    dump(folder / "RESULTS.json", result)
    dump(folder / "CLOSED.json", dict(status=status, results_sha256=sha(folder / "RESULTS.json")))
    dump(folder / "PROGRESS.json", dict(run=cfg["name"], status=status, step=max(0, observed),
                                       resource_stop_reason=stop_reason, final=summary), exclusive=False)
    print(json.dumps(dict(run=cfg["name"], status=status, completed_steps=max(0, observed),
                          resource_stop_reason=stop_reason)), flush=True)
    return result


def progress_snapshot(start, collected=None):
    collected = {} if collected is None else collected
    rows = []
    for cfg in ROSTER:
        path = HERE / cfg["name"] / "PROGRESS.json"
        if cfg["name"] in collected:
            result = collected[cfg["name"]]
            row = dict(run=cfg["name"], status=result["status"], step=result["completed_steps"])
        elif path.exists():
            row = json.loads(path.read_text())
        else:
            row = dict(run=cfg["name"], status="queued", step=0)
        rows.append(dict(p=cfg["p"], **row))
    counts = {s: sum(r["status"] == s for r in rows) for s in ["complete", "running", "partial", "failed", "queued", "unrun"]}
    out = dict(stage="training", elapsed_seconds=time.monotonic() - start,
               declared_runs=len(ROSTER), counts=counts, runs=rows)
    dump(HERE / "PROGRESS.json", out, exclusive=False)
    lines = ["# Twenty-pair sweep: live training progress", "",
             "120 declared runs: 20 fresh initialization/split pairs per modulus. Target40,000 updates each.", "",
             "This file is generated during training; completed plots are produced only after the independent audit passes.", "",
             "| p | Complete /20 | Running | Queued | Partial | Failed/unrun |", "|---:|---:|---:|---:|---:|---:|"]
    for p in MODULI:
        group = [r for r in rows if r["p"] == p]
        n = lambda s: sum(r["status"] == s for r in group)
        lines.append(f"| {p} | {n('complete')}/20 | {n('running')} | {n('queued')} | {n('partial')} | {n('failed') + n('unrun')} |")
    lines += ["", f"Elapsed training wall time: {out['elapsed_seconds']/3600:.2f} hours.", "",
              "Active checkpoints:"]
    lines += [f"- {r['run']}: last reported step{r['step']:,}." for r in rows if r["status"] == "running"]
    lines += ["", "Resource limits: 3 single-thread workers; 1h/arm, 12h total training work. "
              "No outcome-based stopping, retuning or automatic replacement. See PIPELINE_STATUS.json for later stage status.", ""]
    temp = HERE / "RUN_STATUS.md.tmp"
    temp.write_text("\n".join(lines))
    os.replace(temp, HERE / "RUN_STATUS.md")
    return counts


def main():
    validate_roster()
    assert all(os.environ.get(k) == "1" for k in THREAD_VARS)
    start = time.monotonic()
    deadline = start + TRAIN_SECONDS
    sources = json.loads((HERE / "PIPELINE_LAUNCH.json").read_text())["sources"]
    assert all(sha(path) == expected for path, expected in sources.items())
    dump(HERE / "LAUNCH.json", dict(roster=ROSTER, sources=sources, max_workers=WORKERS,
                                    per_arm_work_seconds=ARM_SECONDS, global_work_seconds=TRAIN_SECONDS,
                                    declared_new_updates=4800000, original_run_included=False))
    collected, failures = {}, []
    with futures.ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {pool.submit(run_one, cfg, deadline): cfg for cfg in ROSTER}
        pending = set(jobs)
        while pending:
            done, pending = futures.wait(pending, timeout=30, return_when=futures.FIRST_COMPLETED)
            for job in done:
                cfg = jobs[job]
                try:
                    collected[cfg["name"]] = job.result()
                except Exception as exc:
                    failure = dict(run=cfg["name"], message=str(exc), traceback=traceback.format_exc())
                    failures.append(failure)
                    ntr = int(.8 * cfg["p"]**2)
                    collected[cfg["name"]] = dict(status="failed", config=cfg, p=cfg["p"], n_train=ntr,
                        n_test=cfg["p"]**2 - ntr, completed_steps=0, observed_states=0,
                        updates_completed=None, output_sha256={}, windows=[], final={}, error=failure)
            progress_snapshot(start, collected)
    runs = [collected[cfg["name"]] for cfg in ROSTER]
    outputs = {}
    for cfg in ROSTER:
        for name in ["RESULTS.json", "CLOSED.json"]:
            path = HERE / cfg["name"] / name
            if path.exists():
                outputs[str(path)] = sha(path)
    binding_errors = [path for path, expected in sources.items() if sha(path) != expected]
    status = "complete" if all(r["status"] == "complete" for r in runs) and not failures and not binding_errors else "partial_or_failed"
    result = dict(status=status, roster=ROSTER, runs=runs, declared_runs=len(ROSTER),
                  completed_runs=sum(r["status"] == "complete" for r in runs),
                  total_new_updates=sum(r["updates_completed"] or 0 for r in runs),
                  output_sha256=outputs, source_sha256=sources, failures=failures,
                  source_binding_errors=binding_errors, elapsed_seconds=time.monotonic() - start,
                  independent_seed_pairs=True, iid_entry_sampling=False,
                  initialization_theorem_proved=False, causal_optimizer_theorem_proved=False)
    dump(HERE / "RESULTS.json", result)
    dump(HERE / "CLOSED.json", dict(status=status, results_sha256=sha(HERE / "RESULTS.json")))
    print(json.dumps(dict(status=status, completed_runs=result["completed_runs"],
                          total_new_updates=result["total_new_updates"])), flush=True)


if __name__ == "__main__":
    main()
