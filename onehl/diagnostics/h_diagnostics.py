import glob
import json
import os
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
T0, T1 = 4500, 12000
files = sorted(glob.glob(os.path.join(src, "*.npz")))
assert len(files) == 20, len(files)


def obs(z, key, t):
    s = z["obs_step"].astype(int)
    k = np.flatnonzero(s == t)
    return float(z[key][k[-1]])


rows = []
for f in files:
    z = np.load(f)
    ce, ta, m = z["train_ce"], z["test_accuracy"], z["train_margin"]
    pre = int(np.argmin(ce[:T0]))
    smax = np.maximum.accumulate(ce[::-1])[::-1]
    smin = np.minimum.accumulate(ce[::-1])[::-1]
    v_suffix = smax / smin - 1
    r = dict(run=os.path.basename(f)[:-4],
             ce_pre_window_min=float(ce[pre]), t_pre_window_min=pre,
             ce_t0=float(ce[T0]), ce_t1=float(ce[T1]),
             V_norm_t0=float(z["V_norm"][T0]), V_norm_t1=float(z["V_norm"][T1]),
             W_norm_t0=float(z["W_norm"][T0]), W_norm_t1=float(z["W_norm"][T1]),
             logit_rms_t0=float(z["train_logit_centered_rms"][T0]),
             logit_rms_t1=float(z["train_logit_centered_rms"][T1]),
             min_margin_t0=float(m[T0]), min_margin_t1=float(m[T1]),
             E_total_ratio=obs(z, "obs_hidden_E_total", T1) / obs(z, "obs_hidden_E_total", T0),
             E_add_ratio=obs(z, "obs_hidden_E_add", T1) / obs(z, "obs_hidden_E_add", T0),
             AH_t0=obs(z, "obs_hidden_AH_add", T0), AH_t1=obs(z, "obs_hidden_AH_add", T1))
    so = z["obs_step"].astype(int)
    for thr in (0.10, 0.05, 0.03):
        s = int(np.flatnonzero(v_suffix <= thr)[0])
        k0 = np.flatnonzero(so <= s)[-1]
        r["lossonly_%02d" % round(thr * 100)] = dict(
            start=s, heldout_at_start=float(ta[s]),
            AH_start=float(z["obs_hidden_AH_add"][k0]), AH_end=float(z["obs_hidden_AH_add"][-1]),
            fitted_to_end=bool(np.all(m[s:] > 1e-10)), final_heldout=float(ta[-1]))
    rows.append(r)


def span(key, sub=None):
    vals = [r[key][sub] if sub else r[key] for r in rows]
    return [min(vals), max(vals)]


summary = {k: span(k) for k in rows[0] if k != "run" and not k.startswith("lossonly")}
for k in [k for k in rows[0] if k.startswith("lossonly")]:
    summary[k] = {s: span(k, s) for s in rows[0][k]}
json.dump(dict(window=[T0, T1], n=len(rows), summary=summary, per_run=rows), open(dst, "w"), indent=1)
print(json.dumps(summary, indent=1))
