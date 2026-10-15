#!/usr/bin/env python3
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import gzip, json, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, 'mlp_runner'))
import bias_runner as B
import tf_runner as T


def main(cfg_path, out, T_end, every):
    cfg = json.load(open(cfg_path)); R = B.Run(cfg); p = R.p; allr = np.arange(p * p); tau = 1e-3 * len(R.tr)
    loss, obs, t0 = [], [], time.time()
    for t in range(T_end + 1):
        L, gWs, gV = R.loss_and_grads(); loss.append(L)
        if t % cfg['obs_every'] == 0:
            o = R.observe(heavy=False); o['step'] = t; o['loss'] = L
            if t % every == 0:
                pres, hs, F = R.forward(allr); W0 = R.Ws[0]
                embed = np.concatenate([W0[:, :p], W0[:, p:]], 0).T
                o.update(T.fourier_observers(hs[-1], F, R.y, R.tr, R.te, None, p, tau, head=R.V.T, embed=embed, prefix='f'))
                if t % (2 * every) == 0:
                    pp = T.probe_lr(hs[-1][R.tr], R.y[R.tr], hs[-1], p, 100, 1e-4).argmax(1)
                    o['probe_train_acc'] = float((pp[R.tr] == R.y[R.tr]).mean()); o['probe_test_acc'] = float((pp[R.te] == R.y[R.te]).mean())
            obs.append(o)
        if t == T_end: break
        R.step(gWs, gV, t)
    res = dict(kind='mlp', cfg=cfg, runner=B.VERSION, observers=T.VERSION, loss=loss, obs=obs, every=every, T_end=T_end,
               seconds=time.time() - t0)
    tmp = out + '.tmp'
    with gzip.open(tmp, 'wt') as f: json.dump(res, f)
    os.replace(tmp, out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    print('done', sys.argv[2], flush=True)
