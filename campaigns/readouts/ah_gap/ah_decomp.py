#!/usr/bin/env python3
import os, sys, glob, json, gzip
import numpy as np, torch
HERE = '/home/claude/kt/fourier_refit'; sys.path.insert(0, HERE)
import tf_runner as T, bias_runner as B


def decomp(H, a, b, p):
    C = H - H.mean(0); den = float((C * C).sum()); out = {}
    for name, key in (('sum', (a + b) % p), ('diff', (a - b) % p), ('a', a), ('b', b)):
        M = np.zeros((p, H.shape[1])); np.add.at(M, key, C); M /= np.bincount(key, minlength=p)[:, None]
        out[name] = float((M[key] ** 2).sum() / den)
    out['other'] = 1 - sum(out.values()); return out


def acc_series(d, key):
    return np.array([o['step'] for o in d['obs']]), np.array([o[key] for o in d['obs']])


def run_tf(name):
    d = json.load(gzip.open(f'{HERE}/runs/{name}/replay.json.gz', 'rt'))
    lr = T.Learner(d['cfg']); m = lr.m; p = m.p; allr = torch.arange(p * p)
    X = m.X.numpy(); a, b = X[:, 0], X[:, 1]; assert np.all((a + b) % p == m.y)
    res = []
    for fn in sorted(glob.glob(f'{HERE}/runs/{name}/snaps/t*.npz')):
        z = np.load(fn)
        with torch.no_grad():
            for k in m.P: m.P[k].copy_(torch.tensor(z[k], dtype=m.dt))
            F, mlps, r_hat = m.forward(allr, want=True)
            m.cfg['final_norm'] = 'none'; _, _, r_raw = m.forward(allr, want=True); m.cfg['final_norm'] = 'rms'
        F = F.numpy(); rows = dict(t=int(z['t']))
        for lab, H in (('mlp_neurons', mlps[0].numpy()), ('resid_raw', r_raw.numpy()), ('head_input', r_hat.numpy()),
                       ('logits', F - F.mean(1, keepdims=True))):
            rows[lab] = decomp(H, a, b, p)
        rn = np.linalg.norm(r_raw.numpy(), axis=1); rows['rms_cv'] = float(rn.std() / rn.mean())
        res.append(rows)
    s, v = acc_series(d, 'test_acc_nm'); return dict(kind='tf', p=p, rows=res, acc_step=s.tolist(), acc=v.tolist())


def run_mlp(name):
    d = json.load(gzip.open(f'{HERE}/runs/{name}/replay.json.gz', 'rt'))
    R = B.Run(d['cfg']); p = R.p; allr = np.arange(p * p); a, b = R.a[allr], R.b[allr]; assert np.all((a + b) % p == R.y)
    res = []
    for fn in sorted(glob.glob(f'{HERE}/runs/{name}/snaps/t*.npz')):
        z = np.load(fn); R.Ws[0] = z['W0'].copy(); R.V = z['V'].copy()
        pres, hs, F = R.forward(allr); rows = dict(t=int(z['t']))
        for lab, H in (('hidden', hs[0]), ('preact', pres[0]), ('logits', F - F.mean(1, keepdims=True))):
            rows[lab] = decomp(H, a, b, p)
        res.append(rows)
    s, v = acc_series(d, 'test_acc'); return dict(kind='mlp', p=p, rows=res, acc_step=s.tolist(), acc=v.tolist())


if __name__ == '__main__':
    out = {'W_R5_s1': run_tf('W_R5_s1'), 'W_R4_s3': run_tf('W_R4_s3'), 'F_H_p31_s10': run_mlp('F_H_p31_s10')}
    json.dump(out, open(sys.argv[1], 'w'))
    for k, D in out.items():
        s, v = np.array(D['acc_step']), np.array(D['acc']); ts = [r['t'] for r in D['rows']]
        marks = {'init': 0}
        for q in (0.1, 0.5, 0.9):
            i = np.where(v >= q)[0]
            if len(i): marks[f'ho{q}'] = min(ts, key=lambda t: abs(t - s[i[0]]))
        marks['end'] = ts[-1]
        print(f'== {k} ({D["kind"]}, p={D["p"]})  snapshots at', marks)
        labs = [l for l in D['rows'][0] if isinstance(D['rows'][0][l], dict)]
        for l in labs:
            line = f'  {l:12s}'
            for mk, t in marks.items():
                r = next(r for r in D['rows'] if r['t'] == t)[l]
                line += f' | {mk}: sum {r["sum"]:.2f} diff {r["diff"]:.2f} a {r["a"]:.2f} b {r["b"]:.2f} oth {r["other"]:.2f}'
            print(line)
