#!/usr/bin/env python3
import gzip, json, math, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import refits as RF


def tf_tables(cfg, z):
    import torch, tf_runner as T
    m = T.TF(cfg)
    with torch.no_grad():
        for k in m.P: m.P[k].copy_(torch.tensor(z[k], dtype=m.dt))
        F, mlps, resid = m.forward(torch.arange(m.p * m.p), want=True)
    tabs = {'resid': resid.double().numpy(), 'mlp': mlps[-1].double().numpy()}
    extra = {'head': RF.weight_spectrum(z['W_U'].T, m.p), 'embed': RF.weight_spectrum(z['W_E'][:m.p], m.p)}
    return m, F.double().numpy(), tabs, extra


def mlp_tables(cfg, z):
    import bias_runner as B
    R = B.Run(cfg); R.V = z['V']; R.Ws = [z[f'W{l}'] for l in range(int(z['nW']))]
    pres, hs, F = R.forward(np.arange(R.p * R.p))
    p = R.p; W0 = R.Ws[0]
    extra = {'head': RF.weight_spectrum(R.V, p), 'embed': RF.weight_spectrum(np.concatenate([W0[:, :p], W0[:, p:]], 0).T, p)}
    return R, F, {'last': hs[-1]}, extra


def score_run(run_dir, probe=False, every=None):
    res = json.load(gzip.open(os.path.join(run_dir, 'replay.json.gz'))) if os.path.exists(os.path.join(run_dir, 'replay.json.gz')) \
        else dict(json.load(open(os.path.join(run_dir, 'progress.json'))), kind=None)
    snaps = sorted(f for f in os.listdir(os.path.join(run_dir, 'snaps')) if f.endswith('.npz') and '.tmp' not in f)
    kind = res.get('kind') or ('tf' if 'W_U' in np.load(os.path.join(run_dir, 'snaps', snaps[0])).files else 'mlp')
    cfg = res['cfg'] if 'cfg' in res else json.load(open(os.path.join(run_dir, 'cfg.json')))
    out = dict(kind=kind, cfg=cfg, rows=[])
    probe_fn = None
    if probe:
        import tf_runner as T
        probe_fn = lambda Htr, ytr, H, p=cfg['p']: T.probe_lr(Htr, ytr, H, p, 100, 1e-4)
    for fn in snaps:
        z = np.load(os.path.join(run_dir, 'snaps', fn)); t = int(z['t'])
        if every and t % every: continue
        M, F, tabs, extra = (tf_tables if kind == 'tf' else mlp_tables)(cfg, z)
        y, tr, te = M.y, M.tr, M.te; p = M.p; nm = getattr(M, 'te_nm', None)
        pred_net = F.argmax(1)
        row = dict(step=t, net_te=float((pred_net[te] == y[te]).mean()), net_tr=float((pred_net[tr] == y[tr]).mean()),
                   head_spec=extra['head'].tolist(), embed_spec=extra['embed'].tolist())
        if nm is not None: row['net_te_nm'] = float((pred_net[nm] == y[nm]).mean())
        for name, H in tabs.items():
            grp = np.minimum(M.a[tr], M.b[tr]) * p + np.maximum(M.a[tr], M.b[tr])
            pred, info = RF.all_readouts(H, y, tr, p, probe=probe_fn if name != 'mlp' else None, groups=grp)
            r = dict(info=info, AH=float(RF.grid_spectrum(H, y, p).sum()), AH_spec=RF.grid_spectrum(H, y, p).tolist())
            ev = nm if nm is not None and len(nm) else te
            fr = RF.FourierRidge(H[tr], y[tr], p, 1e-3 * len(tr)); Zc, Zs = fr.predict(H)
            cc, ss = RF.freq_basis(y[ev], p)
            res_ = ((Zc[ev] - cc) ** 2 + (Zs[ev] - ss) ** 2).mean(0); r2 = 1 - res_ / 1.0
            r['ho_r2_spec'] = r2.tolist()
            ks_or = [int(k) for k in np.where(r2 > 0)[0]] or [int(np.argmax(r2))]
            pred['oracle_filt'] = fr.logits(Zc, Zs, ks_or).argmax(1); r['ks_oracle'] = [k + 1 for k in ks_or]
            Cc, Cs = RF.freq_basis(np.arange(p), p); Fc = F - F.mean(1, keepdims=True)
            for tag, ks in (('top4', [k - 1 for k in info['order'][:4]]), ('top8', [k - 1 for k in info['order'][:8]]),
                            ('oracle', ks_or)):
                ks = np.asarray(ks, int); Pi = (2.0 / p) * (Cc[:, ks] @ Cc[:, ks].T + Cs[:, ks] @ Cs[:, ks].T)
                pred['net_filt_' + tag] = (Fc @ Pi).argmax(1)
            for k, pr in pred.items():
                r[k + '_te'] = float((pr[te] == y[te]).mean()); r[k + '_tr'] = float((pr[tr] == y[tr]).mean())
                if nm is not None: r[k + '_te_nm'] = float((pr[nm] == y[nm]).mean())
            row[name] = r
        out['rows'].append(row)
        print(t, 'net %.3f' % row.get('net_te_nm', row['net_te']),
              ' '.join(f"{k}={row[list(tabs)[0]].get(k + ('_te_nm' if nm is not None else '_te'), float('nan')):.2f}"
                       for k in ('ridge', 'filt_top4', 'filt_top8', 'phase_top8', 'proj_top8', 'oracle_filt', 'filt_cv', 'probe', 'net_filt_top8', 'net_filt_oracle')),
              'n_sig', len(row[list(tabs)[0]]['info']['ks_sig']), 'k_cv', row[list(tabs)[0]]['info']['k_cv'], 'ks_cv', row[list(tabs)[0]]['info']['ks_cv'][:10], flush=True)
    out['obs'] = res.get('obs', [])
    return out


if __name__ == '__main__':
    run_dir, outp = sys.argv[1], sys.argv[2]
    every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else None
    o = score_run(run_dir, probe='--probe' in sys.argv, every=every)
    json.dump(o, open(outp, 'w'))
