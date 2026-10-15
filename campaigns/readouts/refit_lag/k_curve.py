#!/usr/bin/env python3
import json, gzip, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); FR = '/home/claude/kt/fourier_refit'
sys.path.insert(0, HERE); sys.path.insert(1, FR)
import numpy as np
import refits as RF


def tables(kind, cfg, z, runner_dir):
    if kind == 'tf':
        sys.path.insert(0, runner_dir)
        import torch, tf_runner as T
        m = T.TF(cfg)
        with torch.no_grad():
            for k in m.P: m.P[k].copy_(torch.tensor(z[k], dtype=m.dt))
            F, mlps, resid = m.forward(torch.arange(m.p * m.p), want=True)
        return m, F.double().numpy(), resid.double().numpy()
    import bias_runner as B
    R = B.Run(cfg); R.V = z['V']; R.Ws = [z[f'W{l}'] for l in range(int(z['nW']))]
    pres, hs, F = R.forward(np.arange(R.p * R.p))
    return R, F, hs[-1]


def guesses(E):
    Es = np.sort(E)[::-1]; out = {}
    lg = np.log(np.maximum(Es[:31], 1e-300)); drops = lg[:-1] - lg[1:]
    out['gapK'] = int(np.argmax(drops[1:30]) + 2)
    med = float(np.median(E))
    for c in (3, 5, 10): out[f'medK{c}'] = int((E > c * med).sum())
    out['prK'] = float(E.sum() ** 2 / (E ** 2).sum())
    cs = np.cumsum(Es) / Es.sum()
    for q in (0.5, 0.8): out[f'cumK{q:g}'] = int(np.searchsorted(cs, q) + 1)
    return out


def main():
    run_dir, outp = sys.argv[1], sys.argv[2]
    runner_dir = sys.argv[sys.argv.index('--runner') + 1] if '--runner' in sys.argv else FR
    every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else None
    rj = os.path.join(run_dir, 'replay.json.gz')
    res = json.load(gzip.open(rj)) if os.path.exists(rj) else json.load(open(os.path.join(run_dir, 'progress.json')))
    cfg = res['cfg'] if 'cfg' in res else json.load(open(sys.argv[sys.argv.index('--cfg') + 1] if '--cfg' in sys.argv
                                                        else os.path.join(run_dir, 'cfg.json')))
    snaps = sorted(f for f in os.listdir(os.path.join(run_dir, 'snaps')) if f.endswith('.npz') and '.tmp' not in f)
    kind = 'tf' if 'W_U' in np.load(os.path.join(run_dir, 'snaps', snaps[0])).files else 'mlp'
    out = dict(kind=kind, cfg=cfg, rows=[], obs=res.get('obs', []))
    for fn in snaps:
        z = np.load(os.path.join(run_dir, 'snaps', fn)); t = int(z['t'])
        if every and t % every: continue
        M, F, H = tables(kind, cfg, z, runner_dir)
        y, tr, te, p = M.y, M.tr, M.te, M.p
        ev = getattr(M, 'te_nm', None); ev = ev if ev is not None and len(ev) else te
        n = len(tr); Htr, ytr = H[tr], y[tr]
        E, _, _ = RF.train_energy(Htr, ytr, p); order = np.argsort(-E)
        fr = RF.FourierRidge(Htr, ytr, p, 1e-3 * n); Zc, Zs = fr.predict(H[ev])
        Cc, Cs = RF.freq_basis(np.arange(p), p)
        L = np.zeros((len(ev), p)); acc = []
        for k in order:
            L += (2.0 / p) * (np.outer(Zc[:, k], Cc[:, k]) + np.outer(Zs[:, k], Cs[:, k]))
            acc.append(float((L.argmax(1) == y[ev]).mean()))
        cc, ss = RF.freq_basis(y[ev], p)
        r2 = 1 - ((Zc - cc) ** 2 + (Zs - ss) ** 2).mean(0)
        row = dict(step=t, net=float((F.argmax(1)[ev] == y[ev]).mean()), acc_K=acc, carried=int((r2 > 0.05).sum()),
                   carried_in_topK=[int(np.isin(np.where(r2 > 0.05)[0], order[:K]).sum()) for K in range(1, len(order) + 1)],
                   bestK=int(np.argmax(acc) + 1), E_sorted=(np.sort(E)[::-1] / E.sum()).tolist(), E=(E / E.sum()).tolist(),
                   r2=r2.tolist(), order=order.tolist(), **guesses(E / E.sum()))
        out['rows'].append(row)
        print(t, 'net %.3f' % row['net'], 'carried', row['carried'], 'bestK', row['bestK'], 'acc@8 %.2f @16 %.2f @all %.2f best %.2f' %
              (acc[7], acc[min(15, len(acc) - 1)], acc[-1], max(acc)), {k: row[k] for k in ('gapK', 'medK5', 'medK10', 'prK', 'cumK0.5')}, flush=True)
        json.dump(out, open(outp, 'w'))


if __name__ == '__main__':
    main()
