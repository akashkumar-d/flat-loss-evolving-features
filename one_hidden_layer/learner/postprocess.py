from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from endpoints import score_run
from plotting import plot_run


def aliases(trace):
    trace = dict(trace)
    mapping = {"train_acc": ["train_accuracy", "train_resolved_accuracy"],
               "test_acc": ["test_accuracy", "test_resolved_accuracy"],
               "train_min_margin": ["train_margin"], "test_min_margin": ["test_margin"],
               "linear_gradient_term": ["ce_linear_gradient"],
               "linear_decay_term": ["ce_linear_decay"],
               "linear_total": ["ce_linear_total"],
               "linear_remainder": ["ce_remainder", "ce_linear_remainder"],
               "logit_rms": ["test_logit_centered_rms"]}
    for target, candidates in mapping.items():
        if target not in trace:
            for key in candidates:
                if key in trace:
                    trace[target] = trace[key]
                    break
    return trace


def endpoint_data(trace, observations):
    trace = aliases(trace)
    data = dict(train_step=trace["step"], train_ce=trace["train_ce"],
                train_min_margin=trace["train_min_margin"], train_acc=trace.get("train_acc"),
                eval_step=trace["step"],
                test_acc=trace.get("test_resolved_accuracy", trace.get("test_acc")))
    for dest, source in [("train_margin_resolved", "train_margin_resolved"),
                         ("train_ce_resolved", "train_ce_positive_resolved")]:
        if source in trace:
            data[dest] = trace[source]
    groups = [
        ("obs_step", "hidden_AH_add", {"AH":"hidden_AH_add", "Eadd":"hidden_E_add", "Etotal":"hidden_E_total"}),
        ("agop_step", "agop_q4_purity", {
            "agop_purity":"agop_q4_purity", "agop_selected_rank":"agop_q4_selected_rank",
            "agop_mapped_rank":"agop_q4_mapped_rank", "agop_target_rank":"agop_q4_target_rank",
            "agop_gap_ratio":"agop_q4_relative_eigengap", "agop_lambda_max":"agop_q4_lambda_max",
            "agop_null_median":"agop_q4_null_median", "agop_fixed_purity":"agop_q4_fixed_purity",
            "agop_requested_rank_score":"agop_q4_requested_rank_score", "agop_coverage":"agop_q4_coverage"}),
        ("refit_step", "refit_test_accuracy", {"refit_acc":"refit_test_resolved_accuracy",
                                               "refit_mse":"refit_test_mse"})]
    for step_key, discriminator, mapping in groups:
        rows = [r for r in observations if discriminator in r]
        if not rows:
            continue
        data[step_key] = np.array([r["step"] for r in rows], dtype=int)
        for dest, source in mapping.items():
            if any(source in r for r in rows):
                data[dest] = np.array([np.nan if r.get(source) is None else r[source] for r in rows])
        if step_key == "refit_step" and "refit_acc" not in data:
            data["refit_acc"] = np.array([r["refit_test_accuracy"] for r in rows])
    return data


def process_arrays(trace, observations, config, outdir, *, stem="protocol_v1"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trace = aliases(trace)
    window = config.get("windows", [{}])[0]
    scoring = dict(budget=config.get("budget", config.get("updates", 50000)),
                   p=config["p"], window_rule="fixed", t0=window.get("t0", config.get("t0")),
                   t1=window.get("t1", config.get("t1")), observer_grid=500,
                   q_requested=4, margin_tolerance=1e-10, energy_tolerance=1e-20)
    score = score_run(endpoint_data(trace, observations), scoring)
    score["evidence_scope"] = config.get("evidence_label", "prospective development")
    score["actual_accuracy_scoring"] = "margin >max(1e-10,64eps*centered_logit_scale); unrounded, every logged update"
    path = outdir / f"{stem}_scores.json"
    if path.exists():
        raise FileExistsError(f"Refusing to replace an earlier scored artifact: {path}")
    path.write_text(json.dumps(score, indent=2, allow_nan=False) + "\n")
    plot_manifest = plot_run(trace, observations, score, dict(config, window_rule="fixed"), outdir, stem)
    plot_manifest["render_source_sha256"] = {name: hashlib.sha256(
        (Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("plotting.py", "postprocess.py", "endpoints.py", "plot_style_v1.json")}
    plot_manifest["render_patch_note"] = "Event-label correction v1.1: a window refit event is called useful only if useful_refit passes. Training and observer code unchanged."
    (outdir / f"{stem}_figures.json").write_text(json.dumps(plot_manifest, indent=2) + "\n")
    return score, plot_manifest


def process_run(run_dir, manifest=None, config=None, run=None):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "RUN_METADATA.json").read_text())
    config = dict(metadata["config"] if config is None else config)
    config.update(run_id=metadata["run_id"], updates=metadata["declared_updates"],
                  source_label=f"{run_dir.name}/trace.npz + observations.jsonl; saved matrices every500",
                  evidence_label=(manifest or {}).get("evidence", "fresh prospective DEVELOPMENT; not confirmation"),
                  gate_interval_updates=100, representation_interval_updates=100)
    with np.load(run_dir / "trace.npz", allow_pickle=False) as saved:
        trace = dict(saved)
    observations = [json.loads(line) for line in (run_dir / "observations.jsonl").read_text().splitlines() if line.strip()]
    return process_arrays(trace, observations, config, run_dir / "plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    score, _ = process_run(args.run_dir)
    print(json.dumps({k: score.get(k) for k in ("P10_raw", "F02", "joint_PAG10_F02", "AGOP_supported_joint")}))
