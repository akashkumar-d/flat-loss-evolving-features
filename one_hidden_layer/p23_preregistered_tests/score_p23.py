import glob, json, math, os, sys
import numpy as np
src, lockp, outp = sys.argv[1:4]
lock = json.load(open(lockp)); t0, t1 = lock["window"]["t0"], lock["window"]["t1"]
def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
rows = []
for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
    z = np.load(f); j = json.load(open(f[:-4] + ".json")); sc = j.get("score") or {}
    ce = z["train_ce"]; lb = ce.copy(); lb[:-1] = 0.5 * (ce[:-1] + ce[1:]); acc = z["test_accuracy"]; tr = z["train_accuracy"]
    os_ = z["obs_step"].astype(int); AH = z["obs_hidden_AH_add"]; full = z["obs_full_observation"] > 0.5
    w = ce[t0:t1 + 1]; v = w.max() / w.min() - 1
    after = np.flatnonzero(acc[t0 + 1:] >= 0.9); T90 = t0 + 1 + int(after[0]) if len(after) else None
    if T90 is not None:
        seg = lb[t0:T90 + 1]; rm = np.maximum.accumulate(seg); dd = float(np.max((rm - seg) / rm)); net = float(seg[-1] / seg[0] - 1)
    else:
        dd = net = None
    dAH = float(AH[os_ == t1][0] - AH[os_ == t0][0])
    q = z["obs_agop_q4_purity"]; qn = z["obs_agop_q4_null_median"]; qv = z["obs_agop_q4_valid"] > 0.5
    i0, i1 = np.flatnonzero(os_ == t0)[0], np.flatnonzero(os_ == t1)[0]
    dcorr = float((q[i1] - qn[i1]) - (q[i0] - qn[i0])) if (qv[i0] and qv[i1]) else None
    rs, rv = os_[full], z["obs_refit_test_accuracy"][full]; r90 = next((int(s) for s, r in zip(rs, rv) if r >= 0.9), None)
    E = dict(P10=bool(v <= 0.10), ND=bool(T90 is not None and T90 <= t1 - 500 and dd <= 0.005 and net >= 0),
             FIT=bool(np.all(tr[t0:t1 + 1] == 1.0)), H20=bool(acc[t0] <= 0.20), F04=bool(dAH >= 0.04))
    E["joint"] = all(E.values())
    rows.append(dict(run=j["meta"]["run_id"], pair=j["meta"].get("confirmation_pair"), **E, v=v, T90=T90, drawdown=dd, net=net, dAH=dAH,
                     heldout_t0=float(acc[t0]), window_min_ce=float(w.min()), dAGOP_minus_null=dcorr,
                     AGOPnull02=(dcorr is not None and dcorr >= 0.02), refit90=r90, heldout_final=float(acc[-1]),
                     strict_joint=sc.get("joint_PAG10_F02"), steps=int(len(ce) - 1)))
n = len(rows)
summ = {k: sum(r[k] for r in rows) for k in ["P10", "ND", "FIT", "H20", "F04", "joint", "AGOPnull02"]}
summ["strict_joint"] = sum(bool(r["strict_joint"]) for r in rows)
lo, hi = wilson(summ["joint"], n) if n else (None, None)
rng = lambda k: [min(r[k] for r in rows if r[k] is not None), float(np.median([r[k] for r in rows if r[k] is not None])), max(r[k] for r in rows if r[k] is not None)]
res = dict(n=n, counts=summ, joint_wilson95=[lo, hi], decision=("REPLACES Figure 2" if summ["joint"] >= 18 else "does not meet the >= 18/20 rule"),
           ranges={k: rng(k) for k in ["v", "T90", "drawdown", "net", "dAH", "heldout_t0", "window_min_ce", "heldout_final", "refit90"]}, runs=rows)
json.dump(res, open(outp, "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["n", "counts", "joint_wilson95", "decision", "ranges"]}, indent=1, default=lambda x: round(float(x), 5)))
