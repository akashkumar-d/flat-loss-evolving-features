from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

import model

VERSION = "1hl-bounded-runner-1.0-20260927"
SAFE_NAME = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False, default=json_safe)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_npz(path, values):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value, name):
    if not isinstance(value, str) or not value or len(value) > 128 or not set(value) <= SAFE_NAME:
        raise ValueError(f"unsafe {name}")
    return value


def validate_manifest(manifest):
    result = dict(manifest)
    _identifier(result["campaign_id"], "campaign_id")
    budget = result.get("updates", 50000)
    if isinstance(budget, bool) or int(budget) != budget or budget < 0:
        raise ValueError("updates must be a nonnegative integer")
    result["updates"] = int(budget)
    cadence = {"cheap_every": 100, "full_every": 500, "checkpoint_every": 500}
    cadence.update(result.get("observation", {}))
    for key in ("cheap_every", "full_every", "checkpoint_every"):
        if isinstance(cadence[key], bool) or int(cadence[key]) != cadence[key] or cadence[key] < 1:
            raise ValueError("cadences must be positive integers")
        cadence[key] = int(cadence[key])
    if cadence["full_every"] % cadence["cheap_every"]:
        raise ValueError("full observation grid must lie on cheap grid")
    result["observation"] = cadence
    limits = {"workers": 8, "global_seconds": 10800, "per_arm_seconds": 7200,
              "cooperative_grace_seconds": 60, "minimum_free_bytes": 3 * 1024 ** 3}
    limits.update(result.get("limits", {}))
    for key in ("workers", "global_seconds", "per_arm_seconds", "cooperative_grace_seconds"):
        if isinstance(limits[key], bool) or not math.isfinite(float(limits[key])) or limits[key] <= 0:
            raise ValueError("positive finite resource caps required")
    if int(limits["workers"]) != limits["workers"] or not 1 <= limits["workers"] <= 8:
        raise ValueError("at most eight workers permitted in this runner")
    if (isinstance(limits["minimum_free_bytes"], bool) or
        int(limits["minimum_free_bytes"]) != limits["minimum_free_bytes"] or limits["minimum_free_bytes"] < 0):
        raise ValueError("minimum_free_bytes must be a nonnegative integer")
    result["limits"] = limits
    configs, seen = [], set()
    for item in result["configs"]:
        config = model.validated_config(item)
        key = _identifier(config["config_id"], "config_id")
        if key in seen:
            raise ValueError("duplicate config_id")
        seen.add(key)
        windows = config.get("windows", [])
        if len(windows) != 1:
            raise ValueError("exactly one primary fixed window is required per recipe")
        for window in windows:
            t0, t1 = window["t0"], window["t1"]
            if any(isinstance(t, bool) or int(t) != t for t in (t0, t1)) or not 0 <= t0 < t1 <= budget:
                raise ValueError("invalid fixed window")
            if t0 % cadence["full_every"] or t1 % cadence["full_every"]:
                raise ValueError("window boundaries must lie on full observer grid")
            _identifier(window.get("window_id", "primary"), "window_id")
        configs.append(config)
    if not configs:
        raise ValueError("at least one configuration required")
    result["configs"] = configs
    ids = set()
    runs = []
    for run in result["runs"]:
        run = dict(run)
        run_id = _identifier(run["run_id"], "run_id")
        if run_id in ids or run["config_id"] not in seen:
            raise ValueError("duplicate run or unknown config")
        ids.add(run_id)
        for key in ("init_seed", "split_seed"):
            if isinstance(run[key], bool) or int(run[key]) != run[key] or run[key] < 0:
                raise ValueError("seeds must be nonnegative integers")
        runs.append(run)
    if not runs:
        raise ValueError("empty roster")
    result["runs"] = runs
    return result


def _trace_arrays(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    if any(row.keys() != rows[0].keys() for row in rows):
        raise ArithmeticError("scalar trace schema changed during run")
    return {key: np.asarray([row[key] for row in rows]) for key in keys}


def _incoming_zero():
    return {key: 0.0 for key in (
        "polar_rank", "polar_threshold", "polar_sigma_max", "polar_relative_threshold",
        "grad_W_norm", "grad_V_norm", "update_W_norm", "update_V_norm",
        "gradient_update_W_norm", "gradient_update_V_norm", "decay_update_W_norm", "decay_update_V_norm",
        "ce_linear_gradient", "ce_linear_decay", "ce_linear_total", "ce_linear_rounding")}


def run_case(config, run, manifest, directory, global_deadline=None, stop_event=None):
    from observers import ObserverContext, VERSION as observer_version, hidden_fourier

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "checkpoints").mkdir()
    (directory / "observer_arrays").mkdir()
    start = time.monotonic()
    wall_start = time.time()
    budget = manifest["updates"]
    deadline = min(start + manifest["limits"]["per_arm_seconds"],
                   float("inf") if global_deadline is None else global_deadline)
    cadence = manifest["observation"]
    window = config["windows"][0]
    boundaries = {window["t0"], window["t1"]}
    rows, observations, checkpoints = [], 0, []
    status, reason, error = "running", None, None
    step, observer_seconds, training_seconds = 0, 0.0, 0.0
    metadata = dict(run, config=config, config_hash=model.config_hash(config),
                    model_version=model.VERSION, runner_version=VERSION,
                    observer_version=observer_version, declared_updates=budget,
                    dtype="float64", device="CPU", activation="ReLU", hidden_depth=1,
                    widths=[config["width"]], loss_id="raw_mean_cross_entropy",
                    optimizer_id="memoryless_support_Polar_W_Frobenius_NGD_V_fixed_decay",
                    intervention_id="none", state_convention="metrics at t are after t simultaneous updates",
                    update_statistics_convention="at t>0, incoming update from t-1; at t=0 zeros with incoming_update_valid=false",
                    linearization_note="Observed first-order CE decomposition, not a certified Taylor bound across ReLU gates.",
                    tie_rule="argmax uses first class; strict accuracy and resolved margins separately",
                    margin_tolerance=config["margin_tolerance"],
                    margin_resolution_rule="max(margin_tolerance,64*float64_eps*max(1,max_abs_class_centered_logit)) per example",
                    start_unix=wall_start, random_frame_seed=int(manifest.get("observer_seed", 20260927)),
                    source_hashes=manifest.get("source_hashes", {}),
                    numpy_version=np.__version__, python_version=platform.python_version(),
                    platform=platform.platform(), no_outcome_stopping=True, no_retries=True)
    atomic_json(directory / "RUN_METADATA.json", metadata)
    try:
        x, y, tr, te, w, v = model.initialize(config, run["init_seed"], run["split_seed"])
        p = config["p"]
        pairs = np.column_stack((np.repeat(np.arange(p), p), np.tile(np.arange(p), p)))
        atomic_npz(directory / "split.npz", dict(train_indices=tr, test_indices=te, labels=y,
                                                  ordered_pairs=pairs, train_pairs=pairs[tr], test_pairs=pairs[te]))
        metadata.update(n=len(tr), n_test=len(te), split_sha256=sha256_file(directory / "split.npz"))
        ctx = ObserverContext(p, config["width"], tr, te, random_seed=metadata["random_frame_seed"])
        frame_arrays = {"Qadd_full_grid": ctx.Qadd, "Qtarget_heldout_centered": ctx.Qtarget}
        for name, frames in ctx.random_frames.items():
            for k, frame in enumerate(frames):
                frame_arrays[f"random_{name}_{k}"] = frame
        atomic_npz(directory / "observer_context.npz", frame_arrays)
        metadata["observer_context_sha256"] = sha256_file(directory / "observer_context.npz")
        atomic_json(directory / "RUN_METADATA.json", metadata)
        incoming = _incoming_zero()
        path_w = path_v = representation_path = 0.0
        previous_ce = None
        first_h_unit = previous_h_unit = previous_gate = None
        fixed_axes = fixed_axes_validity = None
        with (directory / "observations.jsonl").open("x", buffering=1) as observer_stream:
            while True:
                timer = time.monotonic()
                values, pre, h, f, residual = model.forward(x, y, tr, te, w, v, config["margin_tolerance"])
                training_seconds += time.monotonic() - timer
                if step >= budget:
                    status, reason = "complete", "declared_horizon"
                elif stop_event is not None and stop_event.is_set():
                    status, reason = "partial", "campaign_stop"
                elif time.monotonic() >= deadline:
                    status, reason = "partial", "resource_deadline"
                terminal = status != "running"
                ce_increment = 0.0 if previous_ce is None else values["train_ce"] - previous_ce
                path_w += incoming["update_W_norm"]
                path_v += incoming["update_V_norm"]
                values.update(step=step, wall_seconds=time.monotonic() - start, **incoming,
                              incoming_update_valid=bool(step), ce_increment=ce_increment,
                              ce_remainder=ce_increment - incoming["ce_linear_total"],
                              W_parameter_path_length=path_w, V_parameter_path_length=path_v,
                              numerical_health=True)
                rows.append(values)
                cheap = terminal or step in boundaries or step % cadence["cheap_every"] == 0
                full = terminal or step in boundaries or step % cadence["full_every"] == 0
                checkpoint = terminal or step in boundaries or step % cadence["checkpoint_every"] == 0
                if checkpoint:
                    checkpoint_path = directory / "checkpoints" / f"step_{step:06d}.npz"
                    atomic_npz(checkpoint_path, dict(step=np.asarray(step), W=w, V=v, F=f,
                                                   train_indices=tr, test_indices=te, labels=y))
                    checkpoints.append({"step": step, "file": str(checkpoint_path.relative_to(directory)),
                                        "sha256": sha256_file(checkpoint_path)})
                    atomic_npz(directory / "trace.npz", _trace_arrays(rows))
                    atomic_json(directory / "CHECKPOINTS.json", checkpoints)
                if cheap or full:
                    timer = time.monotonic()
                    record = dict(step=step, wall_seconds=values["wall_seconds"], full_observation=full)
                    record.update(hidden_fourier(h, p))
                    norm_h = np.linalg.norm(h)
                    h_unit = h / norm_h if norm_h > 0 else np.zeros_like(h)
                    gates = pre > 0
                    if first_h_unit is None:
                        first_h_unit = h_unit.copy()
                    record.update(gate_change_fraction=0.0 if previous_gate is None else float(np.mean(gates != previous_gate)),
                                  normalized_representation_displacement_previous=0.0 if previous_h_unit is None else float(np.linalg.norm(h_unit - previous_h_unit)),
                                  normalized_representation_displacement_initial=float(np.linalg.norm(h_unit - first_h_unit)),
                                  representation_raw_Frobenius_norm=float(norm_h))
                    representation_path += record["normalized_representation_displacement_previous"]
                    record["normalized_representation_observed_path_length"] = representation_path
                    previous_h_unit, previous_gate = h_unit.copy(), gates.copy()
                    if full:
                        result = ctx.evaluate(h, v, fixed_axes=fixed_axes,
                                              fixed_axes_validity=fixed_axes_validity, actual_logits=f)
                        if step == window["t0"]:
                            fixed_axes = {key: value.copy() for key, value in result["axes"].items()}
                            fixed_axes_validity = dict(result["axes_validity"])
                            ctx.set_fixed_axes(fixed_axes, fixed_axes_validity)
                            atomic_npz(directory / "fixed_axes_at_window_start.npz", fixed_axes)
                            atomic_json(directory / "fixed_axes_at_window_start.json", fixed_axes_validity)
                            for name, q in ctx.qs.items():
                                prefix = "agop_" + name + "_"
                                m = result["metrics"]
                                for suffix in ("purity", "coverage", "mapped_rank", "target_rank", "requested_rank_score"):
                                    m[prefix + "fixed_" + suffix] = m[prefix + suffix]
                                m[prefix + "fixed_valid"] = bool(fixed_axes_validity[name])
                        record.update(result["metrics"])
                        result["metadata"]["fixed_axis_reference_step"] = window["t0"] if fixed_axes is not None else None
                        detail_path = directory / "observer_arrays" / f"step_{step:06d}.npz"
                        detailed = dict(result["arrays"])
                        detailed.update({"axes_" + key: value for key, value in result["axes"].items()})
                        atomic_npz(detail_path, detailed)
                        atomic_json(directory / "observer_arrays" / f"step_{step:06d}.json", result["metadata"])
                        record["observer_arrays_file"] = str(detail_path.relative_to(directory))
                        record["observer_arrays_sha256"] = sha256_file(detail_path)
                    record["observer_seconds"] = time.monotonic() - timer
                    observer_seconds += record["observer_seconds"]
                    observer_stream.write(json.dumps(record, sort_keys=True, allow_nan=False, default=json_safe) + "\n")
                    observations += 1
                if checkpoint or terminal:
                    atomic_json(directory / "STATUS.json", dict(run_id=run["run_id"], status=status,
                        reason=reason, step=step, declared_updates=budget, elapsed_seconds=time.monotonic() - start,
                        observer_seconds=observer_seconds, training_seconds=training_seconds,
                        observations=observations))
                if terminal:
                    break
                timer = time.monotonic()
                fw, fv = model.forces(x[tr], pre[tr], h[tr], residual, v)
                w, v, incoming = model.update(w, v, fw, fv, config)
                previous_ce = values["train_ce"]
                step += 1
                training_seconds += time.monotonic() - timer
    except Exception as exc:
        status, reason = "failed", "execution_error"
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        atomic_json(directory / "ERROR.json", error)
    finally:
        if rows:
            atomic_npz(directory / "trace.npz", _trace_arrays(rows))
        result_status = dict(run_id=run["run_id"], status=status, reason=reason,
                             step=rows[-1]["step"] if rows else None, declared_updates=budget,
                             elapsed_seconds=time.monotonic() - start, observer_seconds=observer_seconds,
                             training_seconds=training_seconds, observations=observations,
                             error=error, end_unix=time.time())
        atomic_json(directory / "STATUS.json", result_status)
        metadata.update(terminal_status=result_status, checkpoints=len(checkpoints))
        if (directory / "trace.npz").exists():
            metadata["trace_sha256"] = sha256_file(directory / "trace.npz")
        if (directory / "observations.jsonl").exists():
            metadata["observations_sha256"] = sha256_file(directory / "observations.jsonl")
        atomic_json(directory / "RUN_METADATA.json", metadata)
    return result_status


def _worker(config, run, manifest, directory, deadline, event):
    result = run_case(config, run, manifest, directory, deadline, event)
    if manifest.get("postprocess_after_each_run", False) and result["status"] in ("complete", "partial"):
        try:
            import postprocess
            receipt = postprocess.process_run(Path(directory), manifest, config, run)
            atomic_json(Path(directory) / "POSTPROCESS_STATUS.json", dict(status="complete", result=receipt))
        except Exception as exc:
            atomic_json(Path(directory) / "POSTPROCESS_ERROR.json", dict(status="failed",
                type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    raise SystemExit(1 if result["status"] == "failed" else 0)


def campaign(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    original = json.loads(manifest_path.read_text())
    manifest = validate_manifest(original)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "MANIFEST_ORIGINAL.json", original)
    source_dir = Path(__file__).resolve().parent
    manifest["source_hashes"] = {path.name: sha256_file(path) for path in sorted(source_dir.glob("*.py"))}
    atomic_json(output / "MANIFEST_EFFECTIVE.json", manifest)
    configs = {config["config_id"]: config for config in manifest["configs"]}
    roster = [dict(run, status="queued", reason=None, step=None) for run in manifest["runs"]]
    start, wall_start = time.monotonic(), time.time()
    deadline = start + manifest["limits"]["global_seconds"]
    context = mp.get_context("spawn")
    stop_event = context.Event()
    active, next_index = {}, 0
    interrupted = {"signal": None}

    def handler(signum, _frame):
        interrupted["signal"] = signum
        stop_event.set()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    atomic_json(output / "LAUNCH.json", dict(start_unix=wall_start, pid=os.getpid(),
        declared_runs=len(roster), workers=manifest["limits"]["workers"],
        manifest_sha256=sha256_file(manifest_path), no_automatic_retry=True,
        global_seconds=manifest["limits"]["global_seconds"], python=sys.executable))

    def save_progress():
        counts = {name: sum(row["status"] == name for row in roster)
                  for name in ("queued", "running", "complete", "partial", "failed", "unrun")}
        atomic_json(output / "PROGRESS.json", dict(elapsed_seconds=time.monotonic() - start,
            declared_runs=len(roster), counts=counts, runs=roster, updated_unix=time.time()))

    save_progress()
    while next_index < len(roster) or active:
        now = time.monotonic()
        import shutil
        free_bytes = shutil.disk_usage(output).free
        resource_stop = now >= deadline or free_bytes < manifest["limits"]["minimum_free_bytes"]
        if resource_stop:
            stop_event.set()
        while (next_index < len(roster) and len(active) < manifest["limits"]["workers"]
               and not stop_event.is_set()):
            index = next_index
            run = manifest["runs"][index]
            process = context.Process(target=_worker,
                args=(configs[run["config_id"]], run, manifest, output / run["run_id"], deadline, stop_event),
                name=run["run_id"])
            process.start()
            active[index] = (process, time.monotonic())
            roster[index].update(status="running", pid=process.pid)
            next_index += 1
        if stop_event.is_set():
            why = ("interrupted" if interrupted["signal"] else
                   "free_disk_guard" if free_bytes < manifest["limits"]["minimum_free_bytes"] else "global_resource_deadline")
            while next_index < len(roster):
                roster[next_index].update(status="unrun", reason=why)
                next_index += 1
        for index, (process, case_start) in list(active.items()):
            case_dir = output / roster[index]["run_id"]
            status_file = case_dir / "STATUS.json"
            if status_file.exists():
                current = json.loads(status_file.read_text())
                roster[index].update({key: current[key] for key in ("status", "reason", "step")})
            hard_deadline = min(deadline, case_start + manifest["limits"]["per_arm_seconds"]) + manifest["limits"]["cooperative_grace_seconds"]
            if process.is_alive() and time.monotonic() > hard_deadline:
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                if roster[index]["status"] in ("complete", "partial"):
                    roster[index]["postprocess_cutoff"] = "owned_worker_hard_resource_cutoff"
                else:
                    roster[index].update(status="partial", reason="owned_worker_hard_resource_cutoff")
            if not process.is_alive():
                process.join()
                roster[index]["exit_code"] = process.exitcode
                if roster[index]["status"] in ("running", "queued"):
                    roster[index].update(status="failed", reason="worker_exit_without_terminal_receipt")
                del active[index]
        save_progress()
        if active:
            time.sleep(.5)
    summary = dict(completed_unix=time.time(), elapsed_seconds=time.monotonic() - start,
                   declared_runs=len(roster), runs=roster,
                   counts={name: sum(row["status"] == name for row in roster)
                           for name in ("complete", "partial", "failed", "unrun")},
                   state="closed", no_automatic_retry=True)
    atomic_json(output / "CLOSED.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = campaign(args.manifest, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, sort_keys=True))
    return 1 if summary["counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
