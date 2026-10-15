import sys, time, os
os.environ['CAMPAIGN_THREADS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_runner_v12 as T
for dt in ('float32', 'float64'):
    cfg = dict(p=113, frac=0.2, n_layers=1, final_norm='rms', optimizer='polar', eta_w=0.1, lam_w=0.01, eta_v=0.03, lam_v=0.3,
               init_seed=7101, split_seed=8101, dtype=dt)
    lr = T.Learner(cfg)
    t0 = time.time()
    for i in range(20):
        lr.m.loss_and_grads(); lr.step()
    t1 = time.time()
    o = T.observe(lr.m, True, True); t2 = time.time()
    print(dt, 'ms/update', round(1000 * (t1 - t0) / 20, 1), 'heavy obs+probe s', round(t2 - t1, 2))
