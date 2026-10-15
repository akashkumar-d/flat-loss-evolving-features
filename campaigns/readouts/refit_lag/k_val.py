#!/usr/bin/env python3
import json, gzip, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); FR = '/home/claude/kt/fourier_refit'
sys.path.insert(0, HERE); sys.path.insert(1, FR)
import numpy as np
import refits as RF
from k_curve import tables

FRACS = (0.5, 0.1)


def r2_of(Zc, Zs, y, p):
    cc, ss = RF.freq_basis(y, p)
    return 1 - ((Zc - cc) ** 2 + (Zs - ss) ** 2).mean(0)


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
    out = dict(kind=kind, cfg=cfg, fracs=list(FRACS), rows=[])
    splits = None
    for fn in snaps:
        z = np.load(os.path.join(run_dir, 'snaps', fn)); t = int(z['t'])
        if every and t % every: continue
        M, F, H = tables(kind, cfg, z, runner_dir)
        y, tr, te, p = M.y, M.tr, M.te, M.p
        ev = getattr(M, 'te_nm', None); ev = np.asarray(ev if ev is not None and len(ev) else te)
        if splits is None:
            a, b = np.asarray(M.a)[ev], np.asarray(M.b)[ev]; g = np.minimum(a, b) * p + np.maximum(a, b)
            u = np.unique(g); rng = np.random.default_rng(12345); perm = rng.permutation(len(u))
            splits = {}
            for f in FRACS:
                val_u = set(u[perm[:int(round(f * len(u)))]].tolist())
                isval = np.array([x in val_u for x in g]); splits[f] = (ev[isval], ev[~isval])
        n = len(tr); Htr, ytr = H[tr], y[tr]
        E, _, _ = RF.train_energy(Htr, ytr, p); order = np.argsort(-E)
        fr = RF.FourierRidge(Htr, ytr, p, 1e-3 * n)
        row = dict(step=t)
        for f, (va, ts_) in splits.items():
            Zc_v, Zs_v = fr.predict(H[va]); Zc_t, Zs_t = fr.predict(H[ts_])
            r2v = r2_of(Zc_v, Zs_v, y[va], p); r2t = r2_of(Zc_t, Zs_t, y[ts_], p)
            sel = np.where(r2v > 0.05)[0]
            if len(sel) == 0: sel = np.array([int(np.argmax(r2v))])
            sets = dict(all=None, top8=order[:8], top16=order[:16], val=sel, valK=order[:len(sel)],
                        oracle=np.where(r2t > 0.05)[0] if (r2t > 0.05).any() else np.array([int(np.argmax(r2t))]))
            acc = {k: float((fr.logits(Zc_t, Zs_t, ks).argmax(1) == y[ts_]).mean()) for k, ks in sets.items()}
            acc['net'] = float((F[ts_].argmax(1) == y[ts_]).mean())
            row[str(f)] = dict(acc=acc, n_val=int(len(sel)), n_oracle=int((r2t > 0.05).sum()), n_val_rows=int(len(va)),
                               n_test_rows=int(len(ts_)), val_in_top8=int(np.isin(sel, order[:8]).sum()))
        out['rows'].append(row)
        r = row['0.1']
        print(t, 'net %.3f' % r['acc']['net'], ' '.join(f"{k} {v:.3f}" for k, v in r['acc'].items() if k != 'net'),
              'n_val(0.1)', r['n_val'], 'n_val(0.5)', row['0.5']['n_val'], 'n_oracle', r['n_oracle'], flush=True)
        json.dump(out, open(outp, 'w'))


if __name__ == '__main__':
    main()
