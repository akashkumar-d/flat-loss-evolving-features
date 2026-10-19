import argparse, glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "learner"))
import model
from observers import ObserverContext, hidden_fourier, VERSION as OBS_VERSION

KEEP = ("train_ce", "test_ce", "train_accuracy", "test_accuracy", "train_resolved_accuracy", "test_resolved_accuracy",
        "train_margin_resolved", "train_logit_centered_rms", "test_logit_centered_rms", "W_norm", "V_norm")


def json_safe(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("out"); ap.add_argument("--t0", type=int, default=15000)
    a = ap.parse_args()
    R = json.load(open(os.path.join(a.run_dir, "RESULTS.json")))
    cfg = R["config"]; p = int(R["p"]); width = int(cfg["width"])
    traj = np.load(os.path.join(a.run_dir, "trajectory.npz"))
    ce_logged = dict(zip(traj["step"].astype(int).tolist(), traj["train_ce"].tolist()))
    files = sorted(glob.glob(os.path.join(a.run_dir, "checkpoint_*.npz")))
    first = np.repeat(np.arange(p), p); second = np.tile(np.arange(p), p)
    x = np.concatenate([np.eye(p)[first], np.eye(p)[second]], 1); y = (first + second) % p
    z0 = np.load(files[0]); tr, te = np.sort(z0["train_idx"]), np.sort(z0["test_idx"])
    ctx = ObserverContext(p, width, tr, te, random_seed=20260927)
    fixed = fixed_valid = None
    out = []
    max_rel = 0.0
    for f in files:
        z = np.load(f)
        step = int(os.path.basename(f).split("_")[1].split(".")[0])
        assert np.array_equal(np.sort(z["train_idx"]), tr) and np.array_equal(np.sort(z["test_idx"]), te)
        values, pre, h, logits, residual = model.forward(x, y, tr, te, z["W"], z["V"], 1e-10)
        rel = abs(values["train_ce"] - ce_logged[step]) / ce_logged[step]
        max_rel = max(max_rel, rel)
        if rel > 1e-9:
            raise SystemExit(f"{a.run_dir}: state {step} CE {values['train_ce']!r} != logged {ce_logged[step]!r} (rel {rel:.2e})")
        rec = {"step": step}
        rec.update(hidden_fourier(h, p))
        res = ctx.evaluate(h, z["V"], fixed_axes=fixed, fixed_axes_validity=fixed_valid, actual_logits=logits)
        m = res["metrics"]
        if step == a.t0:
            fixed = {k: v.copy() for k, v in res["axes"].items()}
            fixed_valid = dict(res["axes_validity"])
            ctx.set_fixed_axes(fixed, fixed_valid)
            for name in ctx.qs:
                pre_ = "agop_" + name + "_"
                for suffix in ("purity", "coverage", "mapped_rank", "target_rank", "requested_rank_score"):
                    m[pre_ + "fixed_" + suffix] = m[pre_ + suffix]
                m[pre_ + "fixed_valid"] = bool(fixed_valid[name])
        rec.update(m)
        rec.update({k: values[k] for k in KEEP if k in values})
        out.append(rec)
    meta = {"run": R["config"]["name"], "p": p, "config": cfg, "observer_version": OBS_VERSION, "t0_fixed_axes": a.t0,
            "states": len(out), "max_rel_ce_mismatch": max_rel, "source": os.path.abspath(a.run_dir)}
    with open(a.out, "w") as fh:
        fh.write(json.dumps({"meta": meta}, default=json_safe) + "\n")
        for r in out:
            fh.write(json.dumps(r, default=json_safe, allow_nan=True) + "\n")
    print(json.dumps(meta)[:300])


if __name__ == "__main__":
    main()
