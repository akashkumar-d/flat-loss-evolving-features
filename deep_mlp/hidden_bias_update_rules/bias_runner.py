import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import gzip, json, math, sys, time
import numpy as np

VERSION = 'campaign-J bias_runner 1.1 (2026-10-06)'
BIAS_UPDATES = ('joint', 'separate')
AFTER_KNEE = (None, 'frozen', 'decay')


def fourier_basis(y, p):
    k = np.arange(1, (p - 1) // 2 + 1)
    ang = 2 * np.pi * np.outer(y, k) / p
    return np.concatenate([np.cos(ang), np.sin(ang)], 1)


def orth(M):
    M = M - M.mean(0)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    keep = s > 1e-10 * max(s[0], 1e-300)
    return U[:, keep]


def cca2(Z, B):
    Qz, Qb = orth(Z), orth(B)
    if Qz.shape[1] == 0: return 0.0
    return float((np.linalg.svd(Qz.T @ Qb, compute_uv=False) ** 2).sum() / Qz.shape[1])


def polar_dir(G, rel_cut=1e-7):
    tall = G.shape[0] >= G.shape[1]
    M = G.T @ G if tall else G @ G.T
    ev, Q = np.linalg.eigh(M)
    smax = math.sqrt(max(ev[-1], 0.0))
    if smax == 0: return np.zeros_like(G), 0
    keep = ev > (rel_cut * smax) ** 2
    Qk = Q[:, keep]; sk = np.sqrt(ev[keep])
    P = (G @ Qk) / sk @ Qk.T if tall else Qk @ (Qk.T @ G / sk[:, None])
    return P, int(keep.sum())


class Run:
    def __init__(s, cfg):
        s.cfg = cfg; p = cfg['p']; s.p = p
        a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); y = (a + b) % p
        perm = np.random.default_rng(cfg['split_seed']).permutation(p * p); n = int(cfg['frac'] * p * p)
        s.tr = np.sort(perm[:n]); s.te = np.sort(perm[n:]); s.a, s.b, s.y = a, b, y
        s.Aoh = np.eye(p)[a]; s.Boh = np.eye(p)[b]; s.Y = np.eye(p)[y]
        rng = np.random.default_rng(cfg['init_seed']); D = cfg['depth']
        ws = cfg.get('widths', [cfg['width']] * D)
        c = cfg['init_scale']
        s.Ws = [c * rng.normal(size=(ws[0], 2 * p)) / math.sqrt(p)]
        k = cfg.get('deep_init_scale', 1.0)
        for l in range(1, D): s.Ws.append(k * rng.normal(size=(ws[l], ws[l - 1])) * math.sqrt(2.0 / ws[l - 1]))
        s.V = rng.normal(size=(p, ws[-1])) / (cfg.get('head_init_div', c * k ** (D - 1)) * math.sqrt(ws[-1]))
        s.Bf = fourier_basis(y, p)
        s.bias = bool(cfg.get('bias', False))
        s.bias_step = s.bias and not cfg.get('bias_frozen', False)
        s.bs = [np.zeros(w) for w in ws] if s.bias else None
        s.bias_update = cfg.get('bias_update', 'joint'); s.bias_after_knee = cfg.get('bias_after_knee')
        if s.bias_update not in BIAS_UPDATES or s.bias_after_knee not in AFTER_KNEE:
            raise ValueError(f'bias_update {s.bias_update!r} / bias_after_knee {s.bias_after_knee!r} not supported')
        if s.bias_update != 'joint' or s.bias_after_knee is not None:
            if not s.bias_step: raise ValueError('bias_update / bias_after_knee need bias: true (and biases that are trained)')
            if s.bias_update != 'joint' and s.bias_after_knee is not None: raise ValueError('one campaign-J change per run')
            if cfg.get('hidden_mode', 'polar') != 'polar': raise ValueError('campaign-J keys are defined for the Polar hidden step')
        s.bias_knee = None

    def forward(s, rows):
        W0 = s.Ws[0]; p = s.p
        pre = W0[:, s.a[rows]].T + W0[:, p + s.b[rows]].T
        if s.bias: pre = pre + s.bs[0]
        pres = [pre]; hs = [np.maximum(pre, 0.0)]
        for l, W in enumerate(s.Ws[1:], start=1):
            pre = hs[-1] @ W.T
            if s.bias: pre = pre + s.bs[l]
            pres.append(pre); hs.append(np.maximum(pre, 0.0))
        F = hs[-1] @ s.V.T
        return pres, hs, F

    def loss_and_grads(s):
        tr = s.tr; n = len(tr); cfg = s.cfg
        pres, hs, F = s.forward(tr)
        if cfg['loss'] == 'ce':
            Z = F - F.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            rel = F - F[np.arange(n), s.y[tr]][:, None]; rel[np.arange(n), s.y[tr]] = -np.inf
            mx = rel.max(1); lr = mx + np.log(np.exp(rel - mx[:, None]).sum(1)); L = float(np.logaddexp(0.0, lr).mean())
            dF = (P - s.Y[tr]) / n
        else:
            R = F - s.Y[tr]; L = float((R * R).sum() / n); dF = 2 * R / n
        gV = dF.T @ hs[-1]
        dh = dF @ s.V; gWs = [None] * len(s.Ws); gbs = [None] * len(s.Ws)
        for l in range(len(s.Ws) - 1, -1, -1):
            dpre = dh * (pres[l] > 0)
            if s.bias: gbs[l] = dpre.sum(0)
            if l == 0:
                gWs[0] = np.concatenate([dpre.T @ s.Aoh[tr], dpre.T @ s.Boh[tr]], 1)
            else:
                gWs[l] = dpre.T @ hs[l - 1]; dh = dpre @ s.Ws[l]
        s.gbs = gbs
        return L, gWs, gV

    def step(s, gWs, gV, t=0):
        cfg = s.cfg
        fh = cfg.get('freeze_hidden_after'); fv = cfg.get('freeze_head_after')
        for l, (W, G) in enumerate(zip(s.Ws, gWs)):
            if fh is not None and t >= fh: continue
            if 'eta_layers' in cfg: eta = cfg['eta_layers'][l]
            else: eta = cfg['eta_w'] if l == 0 else cfg.get('eta_deep', cfg['eta_w'])
            if 'lam_layers' in cfg: lam = cfg['lam_layers'][l]
            else: lam = cfg['lam_w'] if l == 0 else cfg.get('lam_deep', cfg['lam_w'])
            if s.bias_step and (s.bias_update == 'separate' or (s.bias_knee is not None and t >= s.bias_knee)):
                Q, r = polar_dir(-G)
                s.Ws[l] = W + eta * ((Q / math.sqrt(r) if r else 0 * W) - lam * W)
                if s.bias_update == 'separate':
                    g = s.gbs[l]; gn = float(np.linalg.norm(g))
                    s.bs[l] = s.bs[l] + eta * ((-g / gn / math.sqrt(r) if (gn > 0 and r) else 0 * g) - lam * s.bs[l])
                elif s.bias_after_knee == 'decay':
                    s.bs[l] = s.bs[l] + eta * (-lam * s.bs[l])
                continue
            if s.bias_step:
                Wa = np.concatenate([W, s.bs[l][:, None]], 1); Ga = np.concatenate([G, s.gbs[l][:, None]], 1)
                if cfg.get('hidden_mode', 'polar') == 'ngd':
                    gnl = float(np.linalg.norm(Ga)); Wa = Wa + eta * ((-Ga / gnl if gnl > 0 else 0 * Wa) - lam * Wa)
                else:
                    Q, r = polar_dir(-Ga)
                    Wa = Wa + eta * ((Q / math.sqrt(r) if r else 0 * Wa) - lam * Wa)
                s.Ws[l] = np.ascontiguousarray(Wa[:, :-1]); s.bs[l] = np.ascontiguousarray(Wa[:, -1])
                continue
            if cfg.get('hidden_mode', 'polar') == 'ngd':
                gnl = float(np.linalg.norm(G)); s.Ws[l] = W + eta * ((-G / gnl if gnl > 0 else 0 * W) - lam * W)
            else:
                Q, r = polar_dir(-G)
                s.Ws[l] = W + eta * ((Q / math.sqrt(r) if r else 0 * W) - lam * W)
        gn = np.linalg.norm(gV)
        if fv is not None and t >= fv: return
        if cfg.get('head_mode', 'ngd') == 'ngd':
            s.V = s.V + cfg['eta_v'] * ((-gV / gn if gn > 0 else 0 * gV) - cfg['lam_v'] * s.V)
        elif cfg['head_mode'] == 'polar':
            Q, r = polar_dir(-gV); s.V = s.V + cfg['eta_v'] * (Q / math.sqrt(max(r, 1)) - cfg['lam_v'] * s.V)

    def observe(s, heavy):
        p = s.p; allr = np.arange(p * p)
        pres, hs, F = s.forward(allr)
        y, tr, te = s.y, s.tr, s.te
        pred = F.argmax(1)
        other = F.copy(); other[allr, y] = -np.inf; marg = F[allr, y] - other.max(1)
        o = dict(train_acc=float((pred[tr] == y[tr]).mean()), test_acc=float((pred[te] == y[te]).mean()),
                 train_margin_min=float(marg[tr].min()), test_margin_min=float(marg[te].min()))
        Ft = F[te]
        if s.cfg['loss'] == 'ce':
            Z = Ft - Ft.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            o['test_loss'] = float(-np.log(np.maximum(P[np.arange(len(te)), y[te]], 1e-300)).mean())
        else:
            o['test_loss'] = float(((Ft - s.Y[te]) ** 2).sum() / len(te))
        if heavy:
            for name, H in [('h%d' % (l + 1), h) for l, h in enumerate(hs)] + [('logit', F - F.mean(1, keepdims=True))]:
                C = H - H.mean(0); den = float((C * C).sum())
                M = np.stack([C[y == c].mean(0) for c in range(p)])
                o['AH_full_' + name] = float((M[y] ** 2).sum() / den) if den > 0 else 0.0
                mu = H[tr].mean(0); Mtr = np.stack([H[tr][y[tr] == c].mean(0) if np.any(y[tr] == c) else mu for c in range(p)])
                sse_c = float(((H[te] - Mtr[y[te]]) ** 2).sum()); sse_0 = float(((H[te] - mu) ** 2).sum())
                o['AH_ho_' + name] = 1 - sse_c / sse_0 if sse_0 > 0 else 0.0
            Hl = hs[-1]; Gm = s.V.T @ s.V; ev, U = np.linalg.eigh(Gm); U = U[:, ::-1]
            rng = np.random.default_rng(12345)
            for r in (4, p - 1):
                Ur = U[:, :r]
                o[f'agop{r}_full'] = cca2(Hl @ Ur, s.Bf)
                o[f'agop{r}_ho'] = cca2(Hl[te] @ Ur, s.Bf[te])
                ctl_f, ctl_h = [], []
                for k in range(8):
                    R_, _ = np.linalg.qr(rng.normal(size=(Hl.shape[1], r)))
                    ctl_f.append(cca2(Hl @ R_, s.Bf)); ctl_h.append(cca2(Hl[te] @ R_, s.Bf[te]))
                o[f'agop{r}_full_rand'] = float(np.mean(ctl_f)); o[f'agop{r}_ho_rand'] = float(np.mean(ctl_h))
            if len(s.Ws) >= 2:
                trr = s.tr; n = len(trr); masks = [(pr[trr] > 0) for pr in pres]
                Gm1 = np.zeros((s.Ws[0].shape[0], s.Ws[0].shape[0]))
                for c in range(p):
                    g = np.broadcast_to(s.V[c], (n, s.V.shape[1])) * masks[-1]
                    for l in range(len(s.Ws) - 1, 0, -1):
                        g = g @ s.Ws[l]
                        if l - 1 >= 1: g = g * masks[l - 1]
                    Gm1 += g.T @ g
                Gm1 /= n; ev1, U1 = np.linalg.eigh(Gm1); U1 = U1[:, ::-1]; H1 = hs[0]
                for r in (4, p - 1):
                    o[f'agopH1_{r}_ho'] = cca2(H1[te] @ U1[:, :r], s.Bf[te]); o[f'agopH1_{r}_full'] = cca2(H1 @ U1[:, :r], s.Bf)
                    ctl = []
                    for k in range(8):
                        R_, _ = np.linalg.qr(rng.normal(size=(H1.shape[1], r))); ctl.append(cca2(H1[te] @ R_, s.Bf[te]))
                    o[f'agopH1_{r}_ho_rand'] = float(np.mean(ctl))
            Htr = Hl[tr]; mu = Htr.mean(0); Hc = Htr - mu; A = Hc.T @ Hc; lam = 1e-3 * np.trace(A) / A.shape[0] + 1e-12
            Bm = np.linalg.solve(A + lam * np.eye(A.shape[0]), Hc.T @ (s.Y[tr] - s.Y[tr].mean(0)))
            o['refit_acc'] = float((((Hl[te] - mu) @ Bm).argmax(1) == y[te]).mean())
            E = sum(np.abs(np.fft.rfft(Wx, axis=1)[:, 1:]) ** 2 for Wx in (s.Ws[0][:, :p], s.Ws[0][:, p:]))
            tot = float(E.sum()); o['W1_four_top1'] = float(E.max(1).sum() / tot); o['W1_four_top4'] = float(np.sort(E.sum(0))[::-1][:4].sum() / tot)
            o['normW'] = [float(np.linalg.norm(W)) for W in s.Ws]; o['normV'] = float(np.linalg.norm(s.V))
            if s.bias: o['normb'] = [float(np.linalg.norm(b)) for b in s.bs]
            o['logit_rms'] = float(np.sqrt((F ** 2).mean()))
            if s.cfg.get('observers_v1'): o.update(observe_v1(s, pres, hs, F))
        return o


_V1_FRAMES = {}


def _v1_orth(M, rel=1e-10):
    if M.size == 0: return M[:, :0]
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    if sv.size == 0 or sv[0] <= 0: return U[:, :0]
    return U[:, sv > rel * sv[0]]


def _v1_overlap(Z, Bc):
    Qz, Qf = _v1_orth(Z), _v1_orth(Bc)
    if Qz.shape[1] == 0 or Qf.shape[1] == 0: return float('nan'), Qz.shape[1], Qf.shape[1]
    return float((np.linalg.svd(Qz.T @ Qf, compute_uv=False) ** 2).sum()), Qz.shape[1], Qf.shape[1]


def _v1_energy(H, labels, p):
    N = H.shape[0]; mu = H.mean(0); C = H - mu
    M = np.zeros((p, H.shape[1]))
    np.add.at(M, labels, C); cnt = np.bincount(labels, minlength=p).astype(float)
    M /= np.maximum(cnt, 1)[:, None]
    Ep = float((M[labels] ** 2).sum() / N); Et = float((C ** 2).sum() / N)
    return Ep, Et, float((mu ** 2).sum())


def _v1_agop(Hl, U, te, Bte_c, q):
    Z = Hl[te] @ U[:, :q]; Z = Z - Z.mean(0)
    return _v1_overlap(Z, Bte_c)


def observe_v1(s, pres, hs, F):
    p = s.p; y = s.y; tr, te = s.tr, s.te; n = len(tr); q = 4; o = {}
    yd = (s.a - s.b) % p
    Bte = s.Bf[te]; Bte_c = Bte - Bte.mean(0)
    Vc = s.V - s.V.mean(0)
    D = len(hs); masks = [(pr[tr] > 0) for pr in pres]
    key = (id(s), tuple(h.shape[1] for h in hs))
    if key not in _V1_FRAMES:
        rng = np.random.default_rng(777)
        _V1_FRAMES[key] = [[np.linalg.qr(rng.normal(size=(h.shape[1], q)))[0] for _ in range(8)] for h in hs]
    frames = _V1_FRAMES[key]
    for name, l in (('last', D - 1), ('first', 0)):
        if name == 'first' and D == 1: continue
        H = hs[l]
        Ep, Et, Edc = _v1_energy(H, y, p); Ed, _, _ = _v1_energy(H, yd, p)
        o[f'v1_{name}_Eadd'] = Ep; o[f'v1_{name}_Etot'] = Et; o[f'v1_{name}_Edc'] = Edc
        o[f'v1_{name}_AH'] = Ep / Et if Et > 0 else float('nan'); o[f'v1_{name}_Adiff'] = Ed / Et if Et > 0 else float('nan')
        if l == D - 1:
            G = Vc.T @ Vc
        else:
            G = np.zeros((H.shape[1], H.shape[1]))
            for c in range(p):
                g = np.broadcast_to(Vc[c], (n, Vc.shape[1])) * masks[-1]
                for j in range(D - 1, l, -1):
                    g = g @ s.Ws[j]
                    if j - 1 > l: g = g * masks[j - 1]
                G += g.T @ g
            G /= n
        ev, U = np.linalg.eigh(G); ev, U = ev[::-1], U[:, ::-1]
        lmax = float(ev[0]) if ev.size else 0.0
        gap = float((ev[q - 1] - ev[q]) / lmax) if (lmax > 0 and ev.size > q) else float('nan')
        O, rz, rf = _v1_agop(H, U, te, Bte_c, q)
        o[f'v1_{name}_agop_purity'] = O / rz if rz else float('nan'); o[f'v1_{name}_agop_req'] = O / q
        o[f'v1_{name}_agop_cov'] = O / rf if rf else float('nan'); o[f'v1_{name}_agop_rZ'] = rz; o[f'v1_{name}_agop_rF'] = rf
        o[f'v1_{name}_agop_gap'] = gap; o[f'v1_{name}_agop_lmax'] = lmax
        rp = []
        for R_ in frames[l]:
            Zr = H[te] @ R_; Zr = Zr - Zr.mean(0); Or, rzr, _ = _v1_overlap(Zr, Bte_c); rp.append(Or / rzr if rzr else float('nan'))
        rp = np.array(rp); o[f'v1_{name}_agop_rand_med'] = float(np.nanmedian(rp)); o[f'v1_{name}_agop_rand_min'] = float(np.nanmin(rp)); o[f'v1_{name}_agop_rand_max'] = float(np.nanmax(rp))
    Hl = hs[-1]; Yc = s.Y - 1.0 / p; alpha = 1e-3
    A = Hl[tr].T @ Hl[tr] / n + alpha * np.eye(Hl.shape[1]); Bm = np.linalg.solve(A, Hl[tr].T @ Yc[tr] / n)
    Pte = Hl[te] @ Bm; Ptr = Hl[tr] @ Bm
    o['v1_ridge_acc_te'] = float((Pte.argmax(1) == y[te]).mean()); o['v1_ridge_acc_tr'] = float((Ptr.argmax(1) == y[tr]).mean())
    o['v1_ridge_mse_te'] = float(((Pte - Yc[te]) ** 2).sum(1).mean()); o['v1_ridge_mse_tr'] = float(((Ptr - Yc[tr]) ** 2).sum(1).mean())
    Fc = F - F.mean(1, keepdims=True); mu = Fc.mean(0)
    My = np.zeros((p, p)); np.add.at(My, y, Fc); My /= np.maximum(np.bincount(y, minlength=p).astype(float), 1)[:, None]
    Fp = My[y] - mu; Rr = Fc - Fp
    Fe, Re, ye = Fp[te], Rr[te], y[te]; idx = np.arange(len(te))
    u = Fe[idx, ye][:, None] - Fe; b = Re - Re[idx, ye][:, None]
    u[idx, ye] = np.inf; b[idx, ye] = -np.inf
    gam = u.min(1); bet = b.max(1); bound = gam - bet
    Fr = F[te]; other = Fr.copy(); other[idx, ye] = -np.inf; marg = Fr[idx, ye] - other.max(1)
    for k_, v_ in (('gamma', gam), ('beta', bet), ('bound', bound), ('margin', marg)):
        qs = np.quantile(v_, [0.0, 0.05, 0.5, 0.95])
        o[f'v1_signed_{k_}_min'], o[f'v1_signed_{k_}_q05'], o[f'v1_signed_{k_}_q50'], o[f'v1_signed_{k_}_q95'] = (float(x) for x in qs)
    o['v1_signed_certified'] = float((bound > 0).mean())
    return o


def _save_ckpt(path, R, t_next, loss, obs):
    tmp = path + '.tmp.npz'
    extra = {f'b{l}': b for l, b in enumerate(R.bs)} if R.bias else {}
    np.savez(tmp, t_next=t_next, loss=loss[:t_next], V=R.V, nW=len(R.Ws), obs=json.dumps(obs),
             **{f'W{l}': W for l, W in enumerate(R.Ws)}, **extra)
    os.replace(tmp, path)


def at_knee(o, lam_v):
    return 'normV' in o and o['normV'] >= 0.99 / lam_v


def knee_of(obs, lam_v):
    return next((o['step'] for o in obs if at_knee(o, lam_v)), None)


def _generalized_since(obs, acc):
    s = None
    for o in obs:
        if o['test_acc'] >= acc:
            if s is None: s = o['step']
        else:
            s = None
    return s


def run(cfg, out, ckpt_every=5000):
    R = Run(cfg); T = cfg['steps']; t0 = time.time(); ck = out + '.ckpt.npz'
    loss = np.empty(T + 1); obs = []; t_start = 0
    if os.path.exists(ck):
        with np.load(ck, allow_pickle=False) as z:
            t_start = int(z['t_next'])
            loss[:t_start] = z['loss']; obs = json.loads(str(z['obs'])); R.V = z['V']; R.Ws = [z[f'W{l}'] for l in range(int(z['nW']))]
            if R.bias: R.bs = [z[f'b{l}'] for l in range(int(z['nW']))]
        if R.bias_after_knee is not None: R.bias_knee = knee_of(obs, cfg['lam_v'])
    sg = cfg.get('stop_generalized'); reason = None
    gen_since = _generalized_since(obs, sg['acc']) if sg else None
    for t in range(t_start, T + 1):
        if ckpt_every and t > t_start and t % ckpt_every == 0: _save_ckpt(ck, R, t, loss, obs)
        L, gWs, gV = R.loss_and_grads(); loss[t] = L
        stop = False
        if t % cfg['obs_every'] == 0 or t == T:
            o = R.observe(heavy=(t % cfg['heavy_every'] == 0 or t == T)); o['step'] = t; o['loss'] = L
            if cfg.get('stop_at_test') is not None and o['test_acc'] >= cfg['stop_at_test'] and t < T:
                if 'normW' not in o:
                    o = R.observe(heavy=True); o['step'] = t; o['loss'] = L
                stop = True
            elif sg is not None and t < T:
                if o['test_acc'] >= sg['acc']:
                    if gen_since is None: gen_since = t
                    if t >= max(sg['factor'] * gen_since, gen_since + sg['extra']):
                        if 'normW' not in o: o = R.observe(heavy=True); o['step'] = t; o['loss'] = L
                        stop = True; reason = 'generalized'
                else:
                    gen_since = None
            obs.append(o)
            if R.bias_after_knee is not None and R.bias_knee is None and at_knee(o, cfg['lam_v']):
                R.bias_knee = t
        if t == T or stop: break
        R.step(gWs, gV, t)
    res = dict(cfg=cfg, loss=loss[:t + 1].tolist(), obs=obs, seconds=time.time() - t0, resumed_from=t_start)
    if stop: res['stopped_at'] = t
    if reason: res['stop_reason'] = reason
    if R.bias_after_knee is not None: res['bias_knee'] = R.bias_knee
    tmp = out + '.tmp'
    with gzip.open(tmp, 'wt') as f: json.dump(res, f)
    os.replace(tmp, out)
    if os.path.exists(ck): os.remove(ck)
    return res


def load_gz(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)


K1 = 0.03 * 0.015
SLOPE = 0.9
ROUND = 0.005
R_MIN, R_MAX = 0.10, 1.20
ETA_UNIT = 0.03 * 0.015 / 0.07


def solve_rate(t10, r0, x_target, slope=SLOPE):
    lo, hi = 0.01, 3.0
    f = lambda r: r * K1 * t10 * math.exp(slope * (r - r0)) - x_target
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0: lo = mid
        else: hi = mid
    r = round(round(0.5 * (lo + hi) / ROUND) * ROUND, 10)
    return min(max(r, R_MIN), R_MAX)


def first_cross(obs, thr):
    return next((o['step'] for o in obs if o.get('test_acc', 0) >= thr), None)


def run_chain(spec, workdir, ckpt_every=2000):
    os.makedirs(workdir, exist_ok=True)
    pcfg = spec['pilot_cfg']; pout = os.path.join(workdir, 'pilot.json.gz')
    if not os.path.exists(pout): run(pcfg, pout, ckpt_every)
    P = load_gz(pout)
    t10 = first_cross(P['obs'], 0.1); censored = t10 is None
    t10u = t10 if t10 is not None else pcfg['steps']
    rate = solve_rate(t10u, spec['r0'], spec['x_target'])
    mcfg = json.loads(json.dumps(spec['main_template'])); mcfg['eta_layers'][-1] = rate * ETA_UNIT
    rec = dict(r0=spec['r0'], x_target=spec['x_target'], t10_pilot=t10, censored=censored, t10_used=t10u, rate=rate,
               x_pred=round(rate * K1 * t10u, 4), eta_last=mcfg['eta_layers'][-1])
    with open(os.path.join(workdir, 'recipe.json'), 'w') as f: json.dump(rec, f, indent=1)
    mout = os.path.join(workdir, 'main.json.gz')
    if not os.path.exists(mout): run(mcfg, mout, ckpt_every)
    return rec


if __name__ == '__main__':
    ck = int(os.environ.get('CAMPAIGN_CKPT_EVERY', 2000))
    if len(sys.argv) >= 4 and sys.argv[1] == 'one':
        cfg = json.load(open(sys.argv[2])); out = sys.argv[3]
        os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
        run(cfg, out, ck); print('done', out, flush=True)
    elif len(sys.argv) >= 4 and sys.argv[1] == 'chain':
        spec = json.load(open(sys.argv[2])); rec = run_chain(spec, sys.argv[3], ck); print('done', json.dumps(rec), flush=True)
    else:
        print(__doc__)
