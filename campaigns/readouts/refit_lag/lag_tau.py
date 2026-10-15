#!/usr/bin/env python3
import json, gzip, math, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
FR = '/home/claude/kt/fourier_refit'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(1, FR)
import numpy as np
import refits as RF

TAUS = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
K_SIG = 8


def tables(kind, cfg, z, runner_dir):
    if kind == 'tf':
        sys.path.insert(0, runner_dir)
        import torch, tf_runner as T
        m = T.TF(cfg)
        with torch.no_grad():
            for k in m.P: m.P[k].copy_(torch.tensor(z[k], dtype=m.dt))
            out = m.forward(torch.arange(m.p * m.p), want=True)
        F, mlps, resid = out[0], out[1], out[2]
        return m, F.double().numpy(), {'resid': resid.double().numpy(), 'mlp': mlps[-1].double().numpy()}
    import bias_runner as B
    R = B.Run(cfg); R.V = z['V']; R.Ws = [z[f'W{l}'] for l in range(int(z['nW']))]
    pres, hs, F = R.forward(np.arange(R.p * R.p))
    return R, F, {'last': hs[-1]}


def ridge_B(Htr, T, tau):
    return np.linalg.solve(Htr.T @ Htr + tau * np.eye(Htr.shape[1]), Htr.T @ T)


def decompose(Fl, ks, p):
    Cc, Cs = RF.freq_basis(np.arange(p), p); ks = np.asarray(ks, int)
    Pi = (2.0 / p) * (Cc[:, ks] @ Cc[:, ks].T + Cs[:, ks] @ Cs[:, ks].T)
    Fc = Fl - Fl.mean(1, keepdims=True); S = Fc @ Pi
    return S, Fc - S


def score_snapshot(kind, cfg, z, runner_dir, probe):
    M, F, tabs = tables(kind, cfg, z, runner_dir)
    y, tr, te, p = M.y, M.tr, M.te, M.p
    ev = getattr(M, 'te_nm', None)
    ev = ev if ev is not None and len(ev) else te
    acc = lambda pred: float((pred[ev] == y[ev]).mean())
    row = dict(net=acc(F.argmax(1)), net_tr=float((F.argmax(1)[tr] == y[tr]).mean()))
    for name, H in tabs.items():
        Htr, ytr = H[tr], y[tr]; n = len(tr); T = np.eye(p)[ytr] - 1.0 / p
        r = {}
        G = Htr.T @ Htr; ev_ = np.linalg.eigvalsh(G)[::-1] / n
        r['eig_top'] = ev_[:40].tolist(); r['eig_sum'] = float(ev_.sum())
        logits = {}
        for x in TAUS:
            B = ridge_B(Htr, T, x * n); Lg = H @ B; logits[f'ridge_t{x:g}'] = Lg; r[f'ridge_t{x:g}'] = acc(Lg.argmax(1))
            r[f'ridge_t{x:g}_tr'] = float((Lg[tr].argmax(1) == ytr).mean())
        Lc = H @ (Htr.T @ T); logits['centroid'] = Lc; r['centroid'] = acc(Lc.argmax(1))
        r['centroid_tr'] = float((Lc[tr].argmax(1) == ytr).mean())
        E, gc, gs = RF.train_energy(Htr, ytr, p); order = np.argsort(-E); ks = order[:K_SIG]
        fr = RF.FourierRidge(Htr, ytr, p, 1e-3 * n); Zc, Zs = fr.predict(H)
        r['filt8'] = acc(fr.logits(Zc, Zs, ks).argmax(1))
        for x in (1.0, 1e1):
            fr2 = RF.FourierRidge(Htr, ytr, p, x * n); Zc2, Zs2 = fr2.predict(H)
            r[f'filt8_t{x:g}'] = acc(fr2.logits(Zc2, Zs2, ks).argmax(1))
        r['order'] = [int(k) + 1 for k in order[:16]]; r['E_top8_share'] = float(E[ks].sum() / E.sum())
        if probe and kind == 'tf' and name == 'resid':
            import tf_runner as Tm
            PL = Tm.probe_lr(Htr, ytr, H, p, 100, 1e-4); r['probe'] = acc(PL.argmax(1))
        cc_tr, ss_tr = RF.freq_basis(ytr, p); cc_ev, ss_ev = RF.freq_basis(y[ev], p)
        amp_tr = np.sqrt(Zc[tr] ** 2 + Zs[tr] ** 2).mean(0); amp_ev = np.sqrt(Zc[ev] ** 2 + Zs[ev] ** 2).mean(0)
        r2_tr = 1 - ((Zc[tr] - cc_tr) ** 2 + (Zs[tr] - ss_tr) ** 2).mean(0)
        r2_ev = 1 - ((Zc[ev] - cc_ev) ** 2 + (Zs[ev] - ss_ev) ** 2).mean(0)
        r['freq'] = dict(amp_tr=amp_tr.tolist(), amp_ev=amp_ev.tolist(), r2_tr=r2_tr.tolist(), r2_ev=r2_ev.tolist(),
                         E=(E / E.sum()).tolist())
        dec = {}
        dec_list = [('ridge', logits['ridge_t0.001']), ('centroid', logits['centroid']), ('net', F)]
        if probe and kind == 'tf' and name == 'resid': dec_list.append(('probe', PL))
        for lab, Lg in dec_list:
            S, Rr = decompose(Lg, ks, p); Se, Re = S[ev], Rr[ev]
            St, Rt = S[tr], Rr[tr]
            dec[lab] = dict(rms_signal=float(np.sqrt((Se ** 2).mean())), rms_rest=float(np.sqrt((Re ** 2).mean())),
                            rms_signal_tr=float(np.sqrt((St ** 2).mean())), rms_rest_tr=float(np.sqrt((Rt ** 2).mean())),
                            acc_signal=float((Se.argmax(1) == y[ev]).mean()), acc_rest=float((Re.argmax(1) == y[ev]).mean()),
                            acc_full=float(((Se + Re).argmax(1) == y[ev]).mean()))
        r['dec'] = dec
        row[name] = r
    return row


def main():
    run_dir, outp = sys.argv[1], sys.argv[2]
    every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else None
    probe = '--probe' in sys.argv
    runner_dir = sys.argv[sys.argv.index('--runner') + 1] if '--runner' in sys.argv else FR
    res = json.load(gzip.open(os.path.join(run_dir, 'replay.json.gz'))) if os.path.exists(os.path.join(run_dir, 'replay.json.gz')) \
        else dict(json.load(open(os.path.join(run_dir, 'progress.json'))))
    snaps = sorted(f for f in os.listdir(os.path.join(run_dir, 'snaps')) if f.endswith('.npz') and '.tmp' not in f)
    kind = 'tf' if 'W_U' in np.load(os.path.join(run_dir, 'snaps', snaps[0])).files else 'mlp'
    cfg = res['cfg'] if 'cfg' in res else json.load(open(os.path.join(run_dir, 'cfg.json')))
    out = dict(kind=kind, cfg=cfg, taus=TAUS, rows=[], obs=res.get('obs', []))
    for fn in snaps:
        z = np.load(os.path.join(run_dir, 'snaps', fn)); t = int(z['t'])
        if every and t % every: continue
        row = score_snapshot(kind, cfg, z, runner_dir, probe); row['step'] = t; out['rows'].append(row)
        tab = 'resid' if kind == 'tf' else 'last'; r = row[tab]
        print(t, 'net %.3f' % row['net'], ' '.join(f"t{x:g}={r[f'ridge_t{x:g}']:.2f}" for x in TAUS),
              'cent=%.2f filt8=%.2f' % (r['centroid'], r['filt8']), 'probe=%.2f' % r.get('probe', float('nan')),
              '| dec ridge s/r %.3f/%.3f net s/r %.3f/%.3f' % (r['dec']['ridge']['rms_signal'], r['dec']['ridge']['rms_rest'],
                                                          r['dec']['net']['rms_signal'], r['dec']['net']['rms_rest']), flush=True)
        json.dump(out, open(outp, 'w'))


if __name__ == '__main__':
    main()
