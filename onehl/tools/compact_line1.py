import json, os, sys
import numpy as np

DENSE = ("train_ce", "test_ce", "train_accuracy", "test_accuracy", "train_resolved_accuracy", "train_margin",
         "train_logit_centered_rms", "W_norm", "V_norm")
OBS = ("step", "full_observation", "hidden_AH_add", "hidden_E_add", "hidden_E_total", "availability_A_plus",
       "agop_q4_purity", "agop_q4_valid", "agop_q4_gap_resolved", "agop_q4_null_median", "agop_q4_null_min", "agop_q4_null_max",
       "agop_q4_fixed_purity", "agop_q4_fixed_valid", "agop_qcomprehensive_purity", "agop_qcomprehensive_null_median",
       "refit_test_accuracy", "actual_test_accuracy", "actual_train_accuracy", "gate_change_fraction",
       "signed_test_certified_fraction")


def compact(run_dir, out_dir):
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(run_dir, "RUN_METADATA.json")
    meta = json.load(open(meta_path))
    rid = meta["run_id"]
    if os.path.exists(os.path.join(out_dir, rid + ".json")):
        return rid, "exists", ""
    with np.load(os.path.join(run_dir, "trace.npz"), allow_pickle=False) as z:
        dense = {k: np.asarray(z[k]) for k in DENSE if k in z.files}
        dense["step"] = np.asarray(z["step"])
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "observations.jsonl")) if l.strip()]
    obs = {}
    for k in OBS:
        vals = [r.get(k) for r in rows]
        obs["obs_" + k] = np.asarray([np.nan if v is None else float(v) for v in vals], dtype=float)
    np.savez_compressed(os.path.join(out_dir, rid + ".npz"), **dense, **obs)
    score = None
    for cand in ("score.json", os.path.join("plots", "protocol_v1_scores.json")):
        if os.path.exists(os.path.join(run_dir, cand)):
            score = json.load(open(os.path.join(run_dir, cand)))
            break
    keep = {k: meta.get(k) for k in ("run_id", "config_id", "config", "init_seed", "split_seed", "n", "n_test", "declared_updates",
                                      "terminal_status", "runner_version", "observer_version", "model_version", "numpy_version",
                                      "platform", "confirmation_pair", "development_pair")}
    tmp = os.path.join(out_dir, rid + ".json.tmp")
    json.dump({"meta": keep, "score": score, "source": os.path.abspath(run_dir)}, open(tmp, "w"),
              default=lambda v: v.item() if hasattr(v, "item") else str(v))
    os.replace(tmp, os.path.join(out_dir, rid + ".json"))
    return rid, len(dense["step"]), len(rows)


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for d in sys.argv[2:]:
        try:
            print(*compact(d, out), flush=True)
        except Exception as e:
            print("FAILED", d, repr(e), flush=True)
