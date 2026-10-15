#!/usr/bin/env python3
import os
os.environ.setdefault('CAMPAIGN_THREADS', '1')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ['CAMPAIGN_THREADS'])
import gzip, json, sys, time
import numpy as np

RUNNER_DIR, cfg_path, out, T_end, every = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, RUNNER_DIR)
import torch, tf_runner as T


def _write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f)
    os.replace(tmp, path)


cfg = json.load(open(cfg_path)); os.makedirs(out, exist_ok=True)
lr = T.Learner(cfg); m = lr.m; c = lr.cfg
snaps = os.path.join(out, 'snaps'); os.makedirs(snaps, exist_ok=True); prog = os.path.join(out, 'progress.json')
t_start, loss, obs = 0, [], []
if os.path.exists(prog):
    st = json.load(open(prog)); t_start, loss, obs = st['t_next'], st['loss'], st['obs']
    z = np.load(os.path.join(snaps, f't{t_start:06d}.npz'))
    with torch.no_grad():
        for k in m.P: m.P[k].copy_(torch.tensor(z[k], dtype=m.dt))
t0 = time.time()
for t in range(t_start, T_end + 1):
    if t % every == 0:
        fn = os.path.join(snaps, f't{t:06d}.npz'); tmp = fn[:-4] + '.tmp.npz'
        np.savez(tmp, t=t, **{k: v.detach().double().numpy() for k, v in m.P.items()}); os.replace(tmp, fn)
        if t > t_start: _write_json(prog, dict(t_next=t, loss=loss, obs=obs))
    L = m.loss_and_grads(); loss.append(L)
    if t % c['obs_every'] == 0:
        o = T.observe(m, False); o['step'] = t; o['loss'] = L; obs.append(o)
    if t == T_end: break
    if hasattr(lr, 'phase'): lr.phase(t)
    lr.step()
res = dict(kind='tf', cfg=cfg, cfg_full=c, runner=T.VERSION, torch=torch.__version__, loss=loss, obs=obs,
           every=every, T_end=T_end, seconds=time.time() - t0, resumed_from=t_start)
with gzip.open(os.path.join(out, 'replay.json.gz'), 'wt') as f: json.dump(res, f)
print('done', out, flush=True)
