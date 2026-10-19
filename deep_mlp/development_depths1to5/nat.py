import numpy as np, json, math, sys, time

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
    def forward(s, rows):
        W0 = s.Ws[0]; p = s.p
        pre = W0[:, s.a[rows]].T + W0[:, p + s.b[rows]].T
        pres = [pre]; hs = [np.maximum(pre, 0.0)]
        for W in s.Ws[1:]:
            pre = hs[-1] @ W.T; pres.append(pre); hs.append(np.maximum(pre, 0.0))
        F = hs[-1] @ s.V.T
        return pres, hs, F
    def loss_and_grads(s):
        tr = s.tr; n = len(tr); cfg = s.cfg
        pres, hs, F = s.forward(tr)
        if cfg['loss'] == 'ce':
            Z = F - F.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            L = float(-np.log(np.maximum(P[np.arange(n), s.y[tr]], 1e-300)).mean())
            rel = F - F[np.arange(n), s.y[tr]][:, None]; rel[np.arange(n), s.y[tr]] = -np.inf
            mx = rel.max(1); lr = mx + np.log(np.exp(rel - mx[:, None]).sum(1)); L = float(np.logaddexp(0.0, lr).mean())
            dF = (P - s.Y[tr]) / n
        else:
            R = F - s.Y[tr]; L = float((R * R).sum() / n); dF = 2 * R / n
        gV = dF.T @ hs[-1]
        dh = dF @ s.V; gWs = [None] * len(s.Ws)
        for l in range(len(s.Ws) - 1, -1, -1):
            dpre = dh * (pres[l] > 0)
            if l == 0:
                gWs[0] = np.concatenate([dpre.T @ s.Aoh[tr], dpre.T @ s.Boh[tr]], 1)
            else:
                gWs[l] = dpre.T @ hs[l - 1]; dh = dpre @ s.Ws[l]
        return L, gWs, gV
    def step(s, gWs, gV, t=0):
        cfg = s.cfg; ranks = []
        fh = cfg.get('freeze_hidden_after'); fv = cfg.get('freeze_head_after')
        for l, (W, G) in enumerate(zip(s.Ws, gWs)):
            if fh is not None and t >= fh: ranks.append(-1); continue
            eta = cfg['eta_w'] if l == 0 else cfg.get('eta_deep', cfg['eta_w'])
            lam = cfg['lam_w'] if l == 0 else cfg.get('lam_deep', cfg['lam_w'])
            Q, r = polar_dir(-G); ranks.append(r)
            s.Ws[l] = W + eta * ((Q / math.sqrt(r) if r else 0 * W) - lam * W)
        gn = np.linalg.norm(gV)
        if fv is not None and t >= fv: return ranks
        if cfg.get('head_mode', 'ngd') == 'ngd':
            s.V = s.V + cfg['eta_v'] * ((-gV / gn if gn > 0 else 0 * gV) - cfg['lam_v'] * s.V)
        elif cfg['head_mode'] == 'polar':
            Q, r = polar_dir(-gV); s.V = s.V + cfg['eta_v'] * (Q / math.sqrt(max(r, 1)) - cfg['lam_v'] * s.V)
        return ranks
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
            Z = Ft - Ft.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True); o['test_loss'] = float(-np.log(np.maximum(P[np.arange(len(te)), y[te]], 1e-300)).mean())
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
            o['logit_rms'] = float(np.sqrt((F ** 2).mean()))
        return o

def run(cfg, out):
    R = Run(cfg); T = cfg['steps']; t0 = time.time()
    loss = np.empty(T + 1); obs = []
    for t in range(T + 1):
        L, gWs, gV = R.loss_and_grads(); loss[t] = L
        if t % cfg['obs_every'] == 0 or t == T:
            o = R.observe(heavy=(t % cfg['heavy_every'] == 0 or t == T)); o['step'] = t; o['loss'] = L; obs.append(o)
        if t == T: break
        R.step(gWs, gV, t)
    res = dict(cfg=cfg, loss=loss.tolist(), obs=obs, seconds=time.time() - t0)
    json.dump(res, open(out, 'w'))
    return res

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]); run(cfg, sys.argv[2]); print(sys.argv[2], 'done')
