import json, math, sys, time
import numpy as np, torch
sys.path.insert(0, '/home/claude/kt/campaign_T_rerun')
import tf_runner as T
cfg = json.load(open(sys.argv[1])); steps = int(sys.argv[2]); out = sys.argv[3]
torch.set_num_threads(1)
lr = T.Learner(cfg); m = lr.m
ce32, ce64, obs = [], [], []
t0 = time.time()
for t in range(steps + 1):
    for v in m.P.values(): v.grad = None
    F = m.forward(m.tr_t); y = m.Y[m.tr_t]
    L = torch.nn.functional.cross_entropy(F, y); L.backward()
    with torch.no_grad():
        ce32.append(float(L)); ce64.append(float(torch.nn.functional.cross_entropy(F.detach().double(), y)))
    if t % 100 == 0:
        o = T.observe(m, heavy=(t % 500 == 0)); o['step'] = t
        obs.append({k: o[k] for k in ('step', 'train_acc', 'test_acc', 'train_margin_min', 'refit_test_acc') if k in o})
    if t == steps: break
    lr.step()
json.dump(dict(cfg=cfg, ce32=ce32, ce64=ce64, obs=obs, seconds=time.time() - t0), open(out, 'w'))
print('done', out, round(time.time() - t0))
