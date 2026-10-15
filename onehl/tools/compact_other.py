import gzip, json, os, sys
import numpy as np

OBS_KEYS = ("hidden_AH_add", "agop_q4_purity", "agop_q4_valid", "agop_q4_null_median", "agop_q4_null_min", "agop_q4_null_max",
            "agop_q4_fixed_purity", "refit_test_accuracy", "availability_A_plus", "test_accuracy", "train_accuracy")


def _obs(jsonl):
    lines = open(jsonl).read().splitlines()
    meta = json.loads(lines[0])["meta"]
    rows = [json.loads(l) for l in lines[1:] if l.strip()]
    out = {"obs_step": np.asarray([r["step"] for r in rows], float)}
    for k in OBS_KEYS:
        name = {"test_accuracy": "actual_test_accuracy", "train_accuracy": "actual_train_accuracy"}.get(k, k)
        out["obs_" + name] = np.asarray([np.nan if r.get(k) is None else float(r.get(k)) for r in rows], float)
    return meta, out


def _write(out_dir, rid, arrays, meta):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, rid + ".npz"), **arrays)
    tmp = os.path.join(out_dir, rid + ".json.tmp")
    json.dump(meta, open(tmp, "w"), default=lambda v: v.item() if hasattr(v, "item") else str(v))
    os.replace(tmp, os.path.join(out_dir, rid + ".json"))
    return rid


def line2(run_dir, jsonl, out_dir):
    R = json.load(open(os.path.join(run_dir, "RESULTS.json"))); cfg = R["config"]
    z = np.load(os.path.join(run_dir, "trajectory.npz"))
    n, nt = int(R["n_train"]), int(R["n_test"])
    meta_o, obs = _obs(jsonl)
    arrays = dict(step=np.asarray(z["step"], float), train_ce=np.asarray(z["train_ce"], float), test_ce=np.asarray(z["test_ce"], float),
                  train_accuracy=np.asarray(z["train_strict_count"], float) / n, test_accuracy=np.asarray(z["test_strict_count"], float) / nt,
                  **obs)
    rid = cfg["name"]
    conf = dict(p=int(R["p"]), c=cfg["initial_scale"], eta_V=cfg["eta_v"], eta_W=cfg["eta_w"], decay_W=cfg["decay_w"],
                decay_V=cfg["decay_v"], width=cfg["width"], config_id=f"p{R['p']}_c24_20seed",
                windows=[{"t0": 15000, "t1": 20000, "window_id": "joint15_20k (predeclared in the replication)"}])
    meta = {"meta": {"run_id": rid, "config_id": conf["config_id"], "config": conf, "init_seed": cfg["init_seed"],
                     "split_seed": cfg["split_seed"], "n": n, "n_test": nt, "line": "L2", "evidence": "20 fresh pairs per modulus (historical)",
                     "observer": meta_o.get("observer_version"), "observer_grid": 1000},
            "score": None, "source": os.path.abspath(run_dir)}
    return _write(out_dir, rid, arrays, meta)


def line3(run_gz, jsonl, out_dir):
    run = json.load(gzip.open(run_gz, "rt")); cfg = run["cfg"]
    loss = np.asarray(run["loss"], float); T = len(loss) - 1
    ob = run["obs"]
    st = np.asarray([o["step"] for o in ob], float)
    tr = np.asarray([o.get("train_acc", np.nan) for o in ob], float); te = np.asarray([o.get("test_acc", np.nan) for o in ob], float)
    tl = np.asarray([o.get("test_loss", np.nan) for o in ob], float)
    meta_o, obs = _obs(jsonl)
    arrays = dict(step=np.arange(T + 1, dtype=float), train_ce=loss, acc_step=st, train_accuracy=tr, test_accuracy=te, test_ce_logged=tl, **obs)
    rid = os.path.basename(run_gz)[:-len(".json.gz")]
    conf = dict(p=int(cfg["p"]), c=cfg["init_scale"], eta_V=cfg["eta_v"], eta_W=cfg["eta_w"], decay_W=cfg["lam_w"], decay_V=cfg["lam_v"],
                width=cfg["width"], config_id=f"natplat_p{cfg['p']}_lw{cfg['lam_w']}", windows=[])
    meta = {"meta": {"run_id": rid, "config_id": conf["config_id"], "config": conf, "init_seed": cfg["init_seed"], "split_seed": cfg["split_seed"],
                     "line": "L3", "evidence": "natural-plateau 1HL seed cohort (rerun with saved states)", "observer": meta_o.get("observer_version"),
                     "observer_grid": 500, "rerun_reference_bitwise_equal": run.get("reference_bitwise_equal"),
                     "rerun_reference_max_rel_loss_diff": run.get("reference_max_rel_loss_diff")},
            "score": None, "source": os.path.abspath(run_gz)}
    return _write(out_dir, rid, arrays, meta)


def line4(run_dir, out_dir):
    z = np.load(os.path.join(run_dir, "trajectory.npz"), allow_pickle=True)
    j = json.load(open(os.path.join(run_dir, "run.json"))); cfg = j["config"]
    g = lambda k: np.asarray(z[k], float) if k in z.files else None
    arrays = dict(step=g("step"), train_ce=g("train_ce"), test_ce=g("test_ce"), train_accuracy=g("train_accuracy"), test_accuracy=g("test_accuracy"),
                  obs_step=g("step"), obs_hidden_AH_add=g("fourier_addition_fraction"),
                  obs_refit_test_accuracy=g("full_hidden_train_only_refit_heldout_modular_accuracy"),
                  obs_availability_A_plus=g("fourier_sum_subspace_alignment"))
    for k in ("agop_top_rule_fourier_sum_purity", "agop_top4_fourier_sum_purity", "agop_top_rule_fourier_sum_coverage"):
        if k in z.files:
            arrays["obs_" + k] = g(k)
    arrays = {k: v for k, v in arrays.items() if v is not None}
    rid = os.path.basename(run_dir.rstrip("/"))
    conf = dict(p=int(cfg["p"]), width=int(cfg["width"]), config_id="quadratic_p13_k64_40pair", windows=[], **{k: cfg[k] for k in (
        "lr_feature", "lr_head", "weight_decay_head", "feature_init_scale", "head_init_scale", "activation", "feature_geometry") if k in cfg})
    meta = {"meta": {"run_id": rid, "config_id": conf["config_id"], "config": conf, "line": "L4", "evidence": "40-pair locked confirmation (historical)",
                     "observer_grid": int(cfg.get("log_every", 500)), "summary": j.get("summary")},
            "score": None, "source": os.path.abspath(run_dir)}
    return _write(out_dir, rid, arrays, meta)


if __name__ == "__main__":
    kind = sys.argv[1]
    try:
        print({"line2": lambda: line2(*sys.argv[2:5]), "line3": lambda: line3(*sys.argv[2:5]), "line4": lambda: line4(*sys.argv[2:4])}[kind]())
    except Exception as e:
        print("FAILED", sys.argv[2], repr(e))
