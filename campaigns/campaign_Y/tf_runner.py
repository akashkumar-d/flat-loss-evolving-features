#!/usr/bin/env python3
import os
_THREADS = int(os.environ.get('CAMPAIGN_THREADS', '1'))
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, str(_THREADS))
import gzip, json, math, sys, time
import numpy as np
import torch

torch.set_num_threads(_THREADS)
VERSION = 'campaign-Y tf_runner 1.5 (2026-10-11)'
SVD_FALLBACKS = []
_STEP = [0]
DEFAULTS = dict(p=113, frac=0.3, n_layers=1, d_model=128, n_heads=4, d_mlp=512, dtype='float32', optimizer='polar',
                eta_w=0.01, lam_w=0.1, eta_v=0.01, lam_v=0.2, rel_cut=1e-7, lr=1e-3, wd=1.0, betas=[0.9, 0.98], eps=1e-8,
                head_init_std=None, steps=30000, obs_every=20, heavy_every=500, refit_tau=1e-3, final_norm='none',
                probe_every=0, probe_kappa=1e-4, probe_iters=100, hidden_mode='polar', observers_fourier=False,
                gain_init=None, eta_g=0.0, lam_g=1.0)
GROUPS = dict(W_E='embed', W_pos='embed', W_Q='attn', W_K='attn', W_V='attn', W_O='attn', W_in='mlp', W_out='mlp')


def full_cfg(cfg):
    c = dict(DEFAULTS); c.update(cfg); return c


class TF:
    def __init__(s, cfg):
        s.cfg = c = full_cfg(cfg); p = c['p']; s.p = p
        s.dt = torch.float64 if c['dtype'] == 'float64' else torch.float32
        a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); y = (a + b) % p
        perm = np.random.default_rng(c['split_seed']).permutation(p * p); n = int(c['frac'] * p * p)
        s.tr = np.sort(perm[:n]); s.te = np.sort(perm[n:])
        intr = np.zeros(p * p, bool); intr[s.tr] = True
        s.te_nm = s.te[~intr[b[s.te] * p + a[s.te]]]
        s.a, s.b, s.y = a, b, y
        s.X = torch.tensor(np.stack([a, b, np.full(p * p, p)], 1), dtype=torch.long)
        s.Y = torch.tensor(y, dtype=torch.long)
        s.tr_t = torch.tensor(s.tr, dtype=torch.long); s.te_t = torch.tensor(s.te, dtype=torch.long)
        d, H, m, L = c['d_model'], c['n_heads'], c['d_mlp'], c['n_layers']
        assert d % H == 0
        rng = np.random.default_rng(c['init_seed'])
        g = lambda shape, std: torch.tensor(rng.normal(size=shape) * std, dtype=s.dt)
        P = {'W_E': g((p + 1, d), 1 / math.sqrt(d)), 'W_pos': g((3, d), 1 / math.sqrt(d))}
        for l in range(L):
            for k in ('W_Q', 'W_K', 'W_V', 'W_O'): P[f'{k}{l}'] = g((d, d), 1 / math.sqrt(d))
            P[f'W_in{l}'] = g((m, d), math.sqrt(2.0 / d)); P[f'W_out{l}'] = g((d, m), 1 / math.sqrt(m))
        hs = c['head_init_std'] if c['head_init_std'] is not None else 1 / math.sqrt(d)
        P['W_U'] = g((d, p), hs)
        if c['gain_init'] is not None:
            if c['final_norm'] != 'rms': raise ValueError('gain_init needs final_norm rms')
            P['g'] = torch.tensor([float(c['gain_init'])], dtype=s.dt)
        s.P = {k: v.requires_grad_(True) for k, v in P.items()}
        s.mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        if c['final_norm'] not in ('none', 'rms'): raise ValueError(c['final_norm'])

    def names(s): return list(s.P)

    @staticmethod
    def group(name):
        base = name.rstrip('0123456789')
        return GROUPS.get(base, 'head')

    def forward(s, rows, want=False):
        c = s.cfg; P = s.P; d, H, L = c['d_model'], c['n_heads'], c['n_layers']; dh = d // H
        X = s.X[rows]; N = X.shape[0]
        x = P['W_E'][X] + P['W_pos'][None]
        mlps = []
        for l in range(L):
            last = (l == L - 1)
            q_in = x[:, 2:3] if last else x
            T = q_in.shape[1]
            q = (q_in @ P[f'W_Q{l}'].T).view(N, T, H, dh).transpose(1, 2)
            k = (x @ P[f'W_K{l}'].T).view(N, 3, H, dh).transpose(1, 2)
            v = (x @ P[f'W_V{l}'].T).view(N, 3, H, dh).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            msk = s.mask[2:3] if last else s.mask
            att = att.masked_fill(~msk[None, None], float('-inf')).softmax(-1)
            z = (att @ v).transpose(1, 2).reshape(N, T, d) @ P[f'W_O{l}'].T
            r = q_in + z
            h = torch.relu(r @ P[f'W_in{l}'].T)
            r = r + h @ P[f'W_out{l}'].T
            mlps.append(h[:, -1])
            x = r if not last else r
        resid = x[:, -1]
        if c['final_norm'] == 'rms': resid = resid / torch.sqrt((resid * resid).mean(-1, keepdim=True) + 1e-12)
        logits = ((resid * P['g']) if 'g' in P else resid) @ P['W_U']
        return (logits, mlps, resid) if want else logits

    def loss_and_grads(s):
        for v in s.P.values(): v.grad = None
        F = s.forward(s.tr_t)
        L = torch.nn.functional.cross_entropy(F, s.Y[s.tr_t])
        L.backward()
        with torch.no_grad():
            L64 = torch.nn.functional.cross_entropy(F.detach().double(), s.Y[s.tr_t])
        return float(L64) if math.isfinite(float(L.detach())) else float(L.detach())


class NonFinite(RuntimeError):
    pass


def _svd(G):
    try:
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        if bool(torch.isfinite(S).all() and torch.isfinite(U).all() and torch.isfinite(Vh).all()): return U, S, Vh
    except RuntimeError:
        pass
    if not bool(torch.isfinite(G).all()): raise NonFinite('non-finite gradient in the Polar step')
    SVD_FALLBACKS.append(_STEP[0])
    U, S, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    return U.to(G.dtype), S.to(G.dtype), Vh.to(G.dtype)


def polar_dir(G, rel_cut):
    U, S, Vh = _svd(G)
    if S.numel() == 0 or float(S[0]) == 0.0: return torch.zeros_like(G), 0
    keep = S > rel_cut * S[0]; k = int(keep.sum())
    return U[:, keep] @ Vh[keep], k


class Learner:
    def __init__(s, cfg):
        s.m = TF(cfg); c = s.m.cfg; s.cfg = c
        if c['optimizer'] == 'adamw':
            s.opt = torch.optim.AdamW(list(s.m.P.values()), lr=c['lr'], betas=tuple(c['betas']), eps=c['eps'], weight_decay=c['wd'])
        elif c['optimizer'] == 'polar':
            s.opt = None
            if c['hidden_mode'] not in ('polar', 'ngd'): raise ValueError(c['hidden_mode'])
        else:
            raise ValueError(c['optimizer'])

    def eta_lam(s, name):
        c = s.cfg; grp = TF.group(name)
        eta = c.get('eta_groups', {}).get(grp, c['eta_w']); lam = c.get('lam_groups', {}).get(grp, c['lam_w'])
        return eta, lam

    @torch.no_grad()
    def step(s):
        c = s.cfg
        if s.opt is not None:
            s.opt.step(); return
        for name, W in s.m.P.items():
            G = W.grad
            if name == 'g':
                W.add_(c['eta_g'] * (-torch.sign(G) - c['lam_g'] * W))
            elif name == 'W_U':
                gn = float(torch.linalg.norm(G))
                W.add_(c['eta_v'] * ((-G / gn if gn > 0 else torch.zeros_like(G)) - c['lam_v'] * W))
            elif c['hidden_mode'] == 'ngd':
                gn = float(torch.linalg.norm(G)); eta, lam = s.eta_lam(name)
                if not math.isfinite(gn): raise NonFinite('non-finite gradient in the NGD step')
                W.add_(eta * ((-G / gn if gn > 0 else torch.zeros_like(G)) - lam * W))
            else:
                Q, k = polar_dir(-G, c['rel_cut']); eta, lam = s.eta_lam(name)
                W.add_(eta * ((Q / math.sqrt(k) if k else torch.zeros_like(W)) - lam * W))


def ah_full(H, y, p):
    C = H - H.mean(0)
    den = float((C * C).sum())
    if den <= 0: return 0.0
    M = np.zeros((p, H.shape[1])); np.add.at(M, y, C); M /= np.bincount(y, minlength=p)[:, None]
    return float((M[y] ** 2).sum() / den)


def ridge(Htr, ytr, tau, p):
    T = np.eye(p)[ytr] - 1.0 / p
    A = Htr.T @ Htr + tau * np.eye(Htr.shape[1])
    return np.linalg.solve(A, Htr.T @ T)


def probe_lr(Htr, ytr, Hall, p, iters, kappa):
    sc = float(np.sqrt((Htr ** 2).mean()))
    sc = sc if sc > 0 and math.isfinite(sc) else 1.0
    X = torch.tensor(Htr / sc, dtype=torch.float64); Y = torch.tensor(ytr, dtype=torch.long)
    B = torch.zeros(X.shape[1], p, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([B], lr=1, max_iter=iters, history_size=20, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-10, tolerance_change=1e-14)
    def closure():
        opt.zero_grad()
        L = torch.nn.functional.cross_entropy(X @ B, Y) + 0.5 * kappa * (B * B).sum(); L.backward(); return L
    with torch.enable_grad(): opt.step(closure)
    with torch.no_grad(): return (torch.tensor(Hall / sc, dtype=torch.float64) @ B).numpy()


FOURIER_KS = (1, 2, 4, 8, 16)
FOURIER_NET_KS = (4, 8, 16)


def fbasis(v, p):
    k = np.arange(1, (p - 1) // 2 + 1); ang = 2 * np.pi * np.outer(v, k) / p
    return np.cos(ang), np.sin(ang)


def weight_spectrum(Wt, p):
    cc, ss = fbasis(np.arange(p), p)
    P = ((cc.T @ Wt) ** 2 + (ss.T @ Wt) ** 2).sum(1); s = float(P.sum())
    return P / s if s > 0 else P


class FourierReadouts:
    def __init__(s, H, tr, ytr, p, tau):
        s.p = p; K = (p - 1) // 2; s.K = K; n = len(tr); M = H.shape[1]
        Htr = H[tr]; C, S = fbasis(ytr, p); Hc = Htr - Htr.mean(0)
        s.E = ((Hc.T @ C) ** 2 + (Hc.T @ S) ** 2).sum(0) / n ** 2
        s.order = np.argsort(-s.E, kind='stable')
        coef = np.linalg.solve(Htr.T @ Htr + tau * np.eye(M), Htr.T @ np.concatenate([C, S], 1))
        s.Zc = H @ coef[:, :K]; s.Zs = H @ coef[:, K:]
        s.Cc, s.Cs = fbasis(np.arange(p), p)

    def logits(s, ks, phase=False):
        ks = np.asarray(ks, int); zc, zs = s.Zc[:, ks], s.Zs[:, ks]
        if phase:
            r = np.sqrt(zc * zc + zs * zs); r[r == 0] = 1.0; zc, zs = zc / r, zs / r
        return zc @ s.Cc[:, ks].T + zs @ s.Cs[:, ks].T

    def restrict(s, F, ks):
        ks = np.asarray(ks, int); Pi = (2.0 / s.p) * (s.Cc[:, ks] @ s.Cc[:, ks].T + s.Cs[:, ks] @ s.Cs[:, ks].T)
        return (F - F.mean(1, keepdims=True)) @ Pi

    def predictions(s, F=None, full=True):
        K = s.K; out = {'ref_all': s.logits(np.arange(K)).argmax(1)}
        for k_ in FOURIER_KS:
            if k_ < K: out[f'ref_top{k_}'] = s.logits(s.order[:k_]).argmax(1)
        if full:
            out['phase_top8'] = s.logits(s.order[:8], phase=True).argmax(1)
            if F is not None:
                for k_ in FOURIER_NET_KS:
                    if k_ < K: out[f'net_top{k_}'] = s.restrict(F, s.order[:k_]).argmax(1)
        return out


def fourier_observers(H, F, y, tr, te, nm, p, tau, head=None, embed=None, prefix='f', full=True):
    o = {}; fr = FourierReadouts(H, tr, y[tr], p, tau); K = fr.K; M = H.shape[1]
    has_nm = nm is not None and len(nm) > 0

    def score(pred, tag):
        o[f'{prefix}{tag}_train_acc'] = float((pred[tr] == y[tr]).mean())
        o[f'{prefix}{tag}_test_acc'] = float((pred[te] == y[te]).mean())
        if has_nm: o[f'{prefix}{tag}_test_acc_nm'] = float((pred[nm] == y[nm]).mean())

    for tag, pred in fr.predictions(F, full).items(): score(pred, tag)
    if not full: return o
    ev = nm if has_nm else te
    ce, se = fbasis(y[ev], p)
    r2 = 1.0 - ((fr.Zc[ev] - ce) ** 2 + (fr.Zs[ev] - se) ** 2).mean(0)
    ks_or = np.where(r2 > 0)[0]
    if len(ks_or) == 0: ks_or = np.array([int(np.argmax(r2))])
    score(fr.logits(ks_or).argmax(1), 'ref_oracle')
    if F is not None: score(fr.restrict(F, ks_or).argmax(1), 'net_oracle')
    Cg = H - H.mean(0); den = float((Cg * Cg).sum()); cg, sg = fbasis(y, p)
    grid = ((cg.T @ Cg) ** 2 + (sg.T @ Cg) ** 2).sum(1) / (len(y) / 2) / den if den > 0 else np.zeros(K)
    E = fr.E
    o[prefix + 'spec_train'] = (E / E.sum()).tolist() if E.sum() > 0 else E.tolist()
    o[prefix + 'spec_hor2'] = r2.tolist(); o[prefix + 'spec_grid'] = grid.tolist()
    o[prefix + 'top8'] = [int(k) + 1 for k in fr.order[:8]]
    if head is not None:
        o[prefix + 'spec_head'] = weight_spectrum(head.T, p).tolist()
        istr = np.zeros(len(y), bool); istr[tr] = True
        for tag, mask in (('all', np.ones(len(y), bool)), ('ho', ~istr), ('tr', istr)):
            mu = np.zeros((p, M)); cnt = np.bincount(y[mask], minlength=p).astype(float)
            np.add.at(mu, y[mask], H[mask]); mu /= np.maximum(cnt, 1)[:, None]
            score((mu[y] @ head).argmax(1), f'fam_{tag}')
    if embed is not None: o[prefix + 'spec_embed'] = weight_spectrum(embed, p).tolist()
    return o


@torch.no_grad()
def observe(m, heavy, probe=False):
    p = m.p; allr = torch.arange(p * p); y = m.y; tr, te = m.tr, m.te
    F, mlps, resid = m.forward(allr, want=True)
    F = F.double().numpy()
    pred = F.argmax(1)
    other = F.copy(); other[np.arange(p * p), y] = -np.inf; marg = F[np.arange(p * p), y] - other.max(1)
    Ft = F[te]; Z = Ft - Ft.max(1, keepdims=True); lse = np.log(np.exp(Z).sum(1))
    o = dict(train_acc=float((pred[tr] == y[tr]).mean()), test_acc=float((pred[te] == y[te]).mean()),
             train_margin_min=float(marg[tr].min()), test_margin_min=float(marg[te].min()),
             test_loss=float((lse - Z[np.arange(len(te)), y[te]]).mean()))
    nm = m.te_nm
    if len(nm): o['test_acc_nm'] = float((pred[nm] == y[nm]).mean())
    if 'g' in m.P: o['gain'] = float(m.P['g'])
    if heavy:
        L = m.cfg['n_layers']
        for l, h in enumerate(mlps): o[f'AH_full_h{l + 1}'] = ah_full(h.double().numpy(), y, p)
        R = resid.double().numpy()
        o['AH_full_resid'] = ah_full(R, y, p); o['AH_full_logit'] = ah_full(F - F.mean(1, keepdims=True), y, p)
        o['normV'] = float(torch.linalg.norm(m.P['W_U']))
        o['normW'] = float(math.sqrt(sum(float((W * W).sum()) for k, W in m.P.items() if k not in ('W_U', 'g'))))
        for k, W in m.P.items():
            if k not in ('W_U', 'g'): o['normW_' + k] = float(torch.linalg.norm(W))
        n = len(tr); tau = m.cfg['refit_tau'] * n
        for tag, H in (('', R), ('_mlp', mlps[-1].double().numpy())):
            B = ridge(H[tr], y[tr], tau, p); Fr = H @ B
            pr = Fr.argmax(1)
            Tte = np.eye(p)[y[te]] - 1.0 / p
            o['refit' + tag + '_train_acc'] = float((pr[tr] == y[tr]).mean())
            o['refit' + tag + '_test_acc'] = float((pr[te] == y[te]).mean())
            o['refit' + tag + '_Eref'] = float(((Fr[te] - Tte) ** 2).sum() / (len(te) * (p - 1) / p))
            if len(nm): o['refit' + tag + '_test_acc_nm'] = float((pr[nm] == y[nm]).mean())
        if m.cfg.get('observers_fourier'):
            o.update(fourier_observers(R, F, y, tr, te, nm, p, tau, head=m.P['W_U'].double().numpy(),
                                       embed=m.P['W_E'][:p].double().numpy(), prefix='f'))
            o.update(fourier_observers(mlps[-1].double().numpy(), F, y, tr, te, nm, p, tau, prefix='fm', full=False))
        if probe:
            Fp = probe_lr(R[tr], y[tr], R, p, m.cfg['probe_iters'], m.cfg['probe_kappa']); pp = Fp.argmax(1)
            o['probe_train_acc'] = float((pp[tr] == y[tr]).mean()); o['probe_test_acc'] = float((pp[te] == y[te]).mean())
            if len(nm): o['probe_test_acc_nm'] = float((pp[nm] == y[nm]).mean())
    return o


def _save_ckpt(path, lr, t_next, loss, obs, gen_since):
    tmp = path + '.tmp'
    st = dict(t_next=t_next, loss=loss[:t_next].copy(), obs=obs, gen_since=gen_since, svd_fallbacks=list(SVD_FALLBACKS),
              P={k: v.detach().clone() for k, v in lr.m.P.items()},
              opt=(lr.opt.state_dict() if lr.opt is not None else None))
    torch.save(st, tmp); os.replace(tmp, path)
    side = path[:-len('.pt')] + '.json'
    with open(side + '.tmp', 'w') as f: json.dump(dict(t_next=t_next), f)
    os.replace(side + '.tmp', side)


def _generalized_since(obs, acc):
    s = None
    for o in obs:
        if o['test_acc'] >= acc:
            if s is None: s = o['step']
        else:
            s = None
    return s


def run(cfg, out, ckpt_every=2000):
    lr = Learner(cfg); c = lr.cfg; T = c['steps']; t0 = time.time(); ck = out + '.ckpt.pt'
    loss = np.empty(T + 1); obs = []; t_start = 0; SVD_FALLBACKS.clear()
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        t_start = int(st['t_next']); loss[:t_start] = st['loss']; obs = st['obs']; SVD_FALLBACKS[:] = st.get('svd_fallbacks', [])
        with torch.no_grad():
            for k, v in st['P'].items(): lr.m.P[k].copy_(v)
        if lr.opt is not None and st['opt'] is not None: lr.opt.load_state_dict(st['opt'])
    sg = c.get('stop_generalized'); reason = None; stop = False
    gen_since = _generalized_since(obs, sg['acc']) if sg else None
    t = t_start
    for t in range(t_start, T + 1):
        if ckpt_every and t > t_start and t % ckpt_every == 0: _save_ckpt(ck, lr, t, loss, obs, gen_since)
        _STEP[0] = t
        L = lr.m.loss_and_grads(); loss[t] = L
        if not (math.isfinite(L) and all(bool(torch.isfinite(v.grad).all()) for v in lr.m.P.values())):
            stop = True; reason = 'nonfinite'; break
        if t % c['obs_every'] == 0 or t == T:
            heavy = (t % c['heavy_every'] == 0 or t == T)
            probe = heavy and c['probe_every'] > 0 and (t % c['probe_every'] == 0 or t == T)
            o = observe(lr.m, heavy, probe); o['step'] = t; o['loss'] = L
            if sg is not None and t < T:
                if o['test_acc'] >= sg['acc']:
                    if gen_since is None: gen_since = t
                    if t >= max(sg['factor'] * gen_since, gen_since + sg['extra']):
                        if 'normV' not in o: o = observe(lr.m, True, c['probe_every'] > 0); o['step'] = t; o['loss'] = L
                        stop = True; reason = 'generalized'
                else:
                    gen_since = None
            obs.append(o)
        if t == T or stop: break
        try:
            lr.step()
        except NonFinite:
            stop = True; reason = 'nonfinite'; break
    n_loss = t if reason == 'nonfinite' else t + 1
    res = dict(cfg=cfg, cfg_full=c, version=VERSION, loss=loss[:n_loss].tolist(), obs=obs, seconds=time.time() - t0,
               resumed_from=t_start, threads=_THREADS, torch=torch.__version__, svd_fallback_steps=list(SVD_FALLBACKS))
    if stop: res['stopped_at'] = t
    if reason: res['stop_reason'] = reason
    tmp = out + '.tmp'
    with gzip.open(tmp, 'wt') as f: json.dump(res, f)
    os.replace(tmp, out)
    for f in (ck, ck[:-len('.pt')] + '.json'):
        if os.path.exists(f): os.remove(f)
    return res


if __name__ == '__main__':
    ck = int(os.environ.get('CAMPAIGN_CKPT_EVERY', 2000))
    if len(sys.argv) >= 4 and sys.argv[1] == 'one':
        cfg = json.load(open(sys.argv[2])); out = sys.argv[3]
        os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
        run(cfg, out, ck); print('done', out, flush=True)
    else:
        print(__doc__)
