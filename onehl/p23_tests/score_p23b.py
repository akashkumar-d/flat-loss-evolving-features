import glob
import json
import math
import os
import sys

import numpy as np

src, lockp, outp = sys.argv[1:4]
lock = json.load(open(lockp))
t0, t1 = lock["window"]["t0"], lock["window"]["t1"]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def nd(lb, acc, th):
    after = np.flatnonzero(acc[t0 + 1:] >= th)
    T = t0 + 1 + int(after[0]) if len(after) else None
    if T is None:
        return dict(T=None, drawdown=None, net=None, ok=False)
    seg = lb[t0:T + 1]
    rm = np.maximum.accumulate(seg)
    dd = float(np.max((rm - seg) / rm))
    net = float(seg[-1] / seg[0] - 1)
    return dict(T=T, drawdown=dd, net=net, ok=bool(T <= t1 - 500 and dd <= 0.005 and net >= 0))


rows = []
for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
    z = np.load(f)
    j = json.load(open(f[:-4] + ".json"))
    ce = z["train_ce"]
    lb = ce.copy()
    lb[:-1] = 0.5 * (ce[:-1] + ce[1:])
    acc = z["test_accuracy"]
    tr = z["train_accuracy"]
    os_ = z["obs_step"].astype(int)
    AH = z["obs_hidden_AH_add"]
    full = z["obs_full_observation"] > 0.5
    w = ce[t0:t1 + 1]
    v = float(w.max() / w.min() - 1)
    dAH = float(AH[os_ == t1][0] - AH[os_ == t0][0])
    q, qn, qv = z["obs_agop_q4_purity"], z["obs_agop_q4_null_median"], z["obs_agop_q4_valid"] > 0.5
    i0, i1 = np.flatnonzero(os_ == t0)[0], np.flatnonzero(os_ == t1)[0]
    dcorr = float((q[i1] - qn[i1]) - (q[i0] - qn[i0])) if (qv[i0] and qv[i1]) else None
    rs, rv = os_[full], z["obs_refit_test_accuracy"][full]
    r90 = next((int(s) for s, r in zip(rs, rv) if r >= 0.9), None)
    tpk = t0 + int(np.argmax(lb[t0:t1]))
    n50, n70, n90 = nd(lb, acc, 0.5), nd(lb, acc, 0.7), nd(lb, acc, 0.9)
    E = dict(P10=bool(v <= 0.10), ND50=n50["ok"], FIT=bool(np.all(tr[t0:t1 + 1] == 1.0)),
             H20=bool(acc[t0] <= 0.20), F04=bool(dAH >= 0.04))
    E["joint"] = all(E.values())
    rows.append(dict(run=j["meta"]["run_id"], pair=j["meta"].get("confirmation_pair"), **E, v=v, dAH=dAH,
                     T50=n50["T"], drawdown50=n50["drawdown"], net50=n50["net"],
                     ND70=n70["ok"], T70=n70["T"], drawdown70=n70["drawdown"],
                     ND90=n90["ok"], T90=n90["T"], drawdown90=n90["drawdown"], net90=n90["net"],
                     T90_by_horizon=n90["T"] is not None, peak=tpk, heldout_at_peak=float(acc[tpk]),
                     heldout_at_peak_ge50=bool(acc[tpk] >= 0.5), heldout_t0=float(acc[t0]),
                     window_min_ce=float(w.min()), dAGOP_minus_null=dcorr,
                     AGOPnull02=(dcorr is not None and dcorr >= 0.02), refit90=r90, heldout_final=float(acc[-1]),
                     steps=int(len(ce) - 1)))
n = len(rows)
keys = ["P10", "ND50", "FIT", "H20", "F04", "joint", "ND70", "ND90", "T90_by_horizon", "heldout_at_peak_ge50",
        "AGOPnull02"]
summ = {k: sum(bool(r[k]) for r in rows) for k in keys}
lo, hi = wilson(summ["joint"], n) if n else (None, None)


def rng(k):
    x = [r[k] for r in rows if r[k] is not None]
    return [min(x), float(np.median(x)), max(x)] if x else None


res = dict(n=n, counts=summ, joint_wilson95=[lo, hi],
           decision=("CONFIRMED (>= 18/20)" if summ["joint"] >= 18 else "does not meet the >= 18/20 rule"),
           ranges={k: rng(k) for k in ["v", "dAH", "T50", "drawdown50", "net50", "T70", "drawdown70", "T90",
                                       "drawdown90", "net90", "peak", "heldout_at_peak", "heldout_t0",
                                       "window_min_ce", "heldout_final", "refit90"]},
           runs=rows)
json.dump(res, open(outp, "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["n", "counts", "joint_wilson95", "decision", "ranges"]}, indent=1,
                 default=lambda x: round(float(x), 5)))
