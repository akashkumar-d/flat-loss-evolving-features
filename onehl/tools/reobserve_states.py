import gzip, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import model
from observers import ObserverContext, hidden_fourier, VERSION as OBS_VERSION

KEEP = ("train_ce", "test_ce", "train_accuracy", "test_accuracy", "train_resolved_accuracy", "train_margin_resolved",
        "train_logit_centered_rms", "W_norm", "V_norm")


def json_safe(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def main(states_path, run_path, out_path):
    run = json.load(gzip.open(run_path, "rt")); cfg = run["cfg"]; loss = np.asarray(run["loss"])
    p, width = int(cfg["p"]), int(cfg["width"])
    z = np.load(states_path)
    tr, te = np.sort(z["train_idx"]), np.sort(z["test_idx"])
    first = np.repeat(np.arange(p), p); second = np.tile(np.arange(p), p)
    x = np.concatenate([np.eye(p)[first], np.eye(p)[second]], 1); y = (first + second) % p
    ctx = ObserverContext(p, width, tr, te, random_seed=20260927)
    out, max_rel = [], 0.0
    for k, step in enumerate(z["steps"].astype(int)):
        W, V = z["W"][k], z["V"][k]
        values, pre, h, logits, residual = model.forward(x, y, tr, te, W, V, 1e-10)
        rel = abs(values["train_ce"] - loss[step]) / loss[step]
        max_rel = max(max_rel, rel)
        if rel > 1e-9:
            raise SystemExit(f"state {step}: CE {values['train_ce']!r} != logged {loss[step]!r} (rel {rel:.2e})")
        rec = {"step": int(step)}
        rec.update(hidden_fourier(h, p))
        rec.update(ctx.evaluate(h, V, actual_logits=logits)["metrics"])
        rec.update({kk: values[kk] for kk in KEEP if kk in values})
        out.append(rec)
    meta = {"run": os.path.basename(run_path)[:-len(".json.gz")], "p": p, "cfg": cfg, "observer_version": OBS_VERSION,
            "states": len(out), "max_rel_ce_mismatch": max_rel, "reference_bitwise_equal": run.get("reference_bitwise_equal")}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps({"meta": meta}, default=json_safe) + "\n")
        for r in out:
            fh.write(json.dumps(r, default=json_safe, allow_nan=True) + "\n")
    os.replace(tmp, out_path)
    print(json.dumps(meta)[:200])


if __name__ == "__main__":
    main(*sys.argv[1:4])
