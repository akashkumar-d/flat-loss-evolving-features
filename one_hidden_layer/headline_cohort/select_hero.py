import glob
import json
import os
import sys

import numpy as np

G = 500
RULES = {"strict": dict(vmax=0.08, ce_min=0.12, dah_min=0.07, dag_null=0.02),
         "relaxed": dict(vmax=0.10, ce_min=0.10, dah_min=0.06, dag_null=0.02)}


def load_run(js):
    d = json.load(open(js)); m = d["meta"]; c = m["config"]
    z = np.load(js[:-5] + ".npz")
    step = z["step"].astype(int); T = int(step[-1])
    assert np.array_equal(step, np.arange(T + 1)), "dense trace must hold every update"
    ce, tra, tea = z["train_ce"], z["train_accuracy"], z["test_accuracy"]
    grid = np.arange(0, T + 1, G)
    nb = len(grid) - 1
    bmax = np.array([ce[grid[k]:grid[k + 1] + 1].max() for k in range(nb)])
    bmin = np.array([ce[grid[k]:grid[k + 1] + 1].min() for k in range(nb)])
    bfit = np.array([bool(np.all(tra[grid[k]:grid[k + 1] + 1] >= 1 - 1e-12)) for k in range(nb)])
    os_ = z["obs_step"].astype(int)

    def at_grid(key, need_full=False):
        v = np.full(len(grid), np.nan)
        y = z[key]
        for i, g in enumerate(grid):
            k = np.flatnonzero(os_ == g)
            if k.size:
                v[i] = y[k[-1]]
        return v
    ah = at_grid("obs_hidden_AH_add")
    ag = at_grid("obs_agop_q4_purity"); agv = at_grid("obs_agop_q4_valid"); nu = at_grid("obs_agop_q4_null_median")
    hit = np.flatnonzero(tea >= 0.9 - 1e-12)
    tgrok = int(hit[0]) if hit.size else None
    return dict(id=m["run_id"], cfg=c, config_id=m["config_id"], T=T, grid=grid, bmax=bmax, bmin=bmin, bfit=bfit,
                ah=ah, ag=ag, agv=agv, nu=nu, test_grid=tea[grid], tgrok=tgrok, ce=ce, V_norm=z["V_norm"] if "V_norm" in z.files else None)


def window_table(r):
    grid = r["grid"]; n = len(grid)
    V = np.full((n, n), np.inf); CEmin = np.full((n, n), -np.inf); FIT = np.zeros((n, n), bool)
    for i in range(n):
        mx, mn, fit = -np.inf, np.inf, True
        for j in range(i + 1, n):
            mx = max(mx, r["bmax"][j - 1]); mn = min(mn, r["bmin"][j - 1]); fit = fit and r["bfit"][j - 1]
            V[i, j] = mx / mn - 1 if mn > 0 else np.inf; CEmin[i, j] = mn; FIT[i, j] = fit
    dAH = r["ah"][None, :] - r["ah"][:, None]
    valid = (r["agv"][:, None] == 1) & (r["agv"][None, :] == 1)
    dAGn = (r["ag"][None, :] - r["ag"][:, None]) - (r["nu"][None, :] - r["nu"][:, None])
    dAGn = np.where(valid, dAGn, -np.inf)
    start_ok = (r["test_grid"] <= 0.2)[:, None]
    tg = r["tgrok"]
    grok_ok = np.zeros((n, n), bool)
    if tg is not None:
        grok_ok = (grid[:, None] < tg) & (tg <= grid[None, :] - G)
    geo = (grid[:, None] >= 1000) & (grid[None, :] - grid[:, None] >= 2000)
    return dict(V=V, CEmin=CEmin, FIT=FIT, dAH=dAH, dAGn=dAGn, start_ok=start_ok, grok_ok=grok_ok, geo=geo)


def select(runs_by_cfg, rule):
    cands = []
    for cid, runs in runs_by_cfg.items():
        W = [window_table(r) for r in runs]
        ok = np.ones_like(W[0]["geo"])
        for w in W:
            ok &= (w["geo"] & (w["V"] <= rule["vmax"]) & (w["CEmin"] >= rule["ce_min"]) & (w["dAH"] >= rule["dah_min"]) &
                   (w["dAGn"] >= rule["dag_null"]) & w["start_ok"] & w["grok_ok"] & w["FIT"])
        minAH = np.min([w["dAH"] for w in W], axis=0); minCE = np.min([w["CEmin"] for w in W], axis=0)
        maxV = np.max([w["V"] for w in W], axis=0)
        grid = runs[0]["grid"]; c = runs[0]["cfg"]
        for i, j in zip(*np.nonzero(ok)):
            cands.append(dict(config_id=cid, t0=int(grid[i]), t1=int(grid[j]), min_dAH=float(minAH[i, j]), min_CEmin=float(minCE[i, j]),
                              max_range=float(maxV[i, j]), train_fraction=c["train_fraction"], decay_V=c["decay_V"], n_runs=len(runs)))
    if not cands:
        return None, []
    best = max(c["min_dAH"] for c in cands)
    top = [c for c in cands if c["min_dAH"] >= best - 0.005]
    top.sort(key=lambda c: (-c["min_CEmin"], c["train_fraction"] != 0.5, c["decay_V"] != 0.2, c["t1"] - c["t0"]))
    cands.sort(key=lambda c: -c["min_dAH"])
    return top[0], cands[:25]


def diagnostics(runs_by_cfg):
    out = {}
    for cid, runs in runs_by_cfg.items():
        c = runs[0]["cfg"]
        med = lambda xs: float(np.median(xs)) if len(xs) else None
        out[cid] = dict(train_fraction=c["train_fraction"], decay_W=c["decay_W"], eta_V=c["eta_V"], decay_V=c["decay_V"], n=len(runs),
                        ce_at={str(t): med([float(r["ce"][t]) for r in runs if t <= r["T"]]) for t in (1000, 5000, 10000, 20000, 30000, 50000)},
                        t_grok=[r["tgrok"] for r in runs], final_AH=med([float(r["ah"][-1]) for r in runs]),
                        V_norm_final=med([float(r["V_norm"][-1]) for r in runs if r["V_norm"] is not None]))
        W = [window_table(r) for r in runs]
        base = np.ones_like(W[0]["geo"])
        for w in W:
            base &= w["geo"] & (w["V"] <= 0.10) & w["FIT"] & w["start_ok"] & w["grok_ok"]
        minAH = np.min([w["dAH"] for w in W], axis=0); minCE = np.min([w["CEmin"] for w in W], axis=0)
        if base.any():
            i, j = np.unravel_index(np.argmax(np.where(base, minAH, -np.inf)), base.shape)
            g = runs[0]["grid"]
            out[cid]["best_flat_window"] = dict(t0=int(g[i]), t1=int(g[j]), min_dAH=float(minAH[i, j]), min_CEmin=float(minCE[i, j]))
        else:
            out[cid]["best_flat_window"] = None
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    runs = [load_run(f) for f in sorted(glob.glob(os.path.join(src, "*.json")))]
    by = {}
    for r in runs:
        by.setdefault(r["config_id"], []).append(r)
    result = dict(n_runs=len(runs), n_configs=len(by), runs_per_config={k: len(v) for k, v in by.items()})
    for name in ("strict", "relaxed"):
        choice, top = select(by, RULES[name])
        result[name] = dict(rule=RULES[name], choice=choice, top=top)
        if choice is not None and name == "strict":
            break
    result["diagnostics"] = diagnostics(by)
    json.dump(result, open(dst, "w"), indent=1)
    for name in ("strict", "relaxed"):
        if name in result:
            print(name, "->", result[name]["choice"])
