import gzip, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import tf_runner as T
RUNS = {'W_R5_s1': ('W_R5_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.275_s1', 'X_R5_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.275_s1'),
        'W_R4_s3': ('W_R4_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.25_s3', 'X_R4_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.25_s3')}
SRC = '/home/claude/kt/fourier_refit'
for run, (cfgname, xid) in RUNS.items():
    rep = json.load(gzip.open(f'{SRC}/runs/{run}/replay.json.gz'))
    cfg = dict(json.load(open(f'{SRC}/cfgs/{cfgname}.json')), observers_fourier=True, heavy_every=200, probe_every=400)
    light = {o['step']: o for o in rep['obs']}
    snaps = sorted(f for f in os.listdir(f'{SRC}/runs/{run}/snaps') if f.endswith('.npz') and '.tmp' not in f)
    m = T.TF(cfg)
    for fn in snaps:
        z = np.load(f'{SRC}/runs/{run}/snaps/{fn}'); t = int(z['t'])
        with torch.no_grad():
            for k in m.P: m.P[k].copy_(torch.tensor(z[k], dtype=m.dt))
        o = T.observe(m, True, t % 400 == 0); o['step'] = t; o['loss'] = rep['loss'][t]
        light[t] = o
        print(run, t, flush=True)
    res = dict(cfg=cfg, cfg_full=T.full_cfg(cfg), version=T.VERSION + ' (test data from replay snapshots)', loss=rep['loss'],
               obs=[light[s] for s in sorted(light)], seconds=0.0, resumed_from=0)
    d = os.path.join(HERE, 'results', xid); os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, 'run.json.gz'), 'wt') as f: json.dump(res, f)
