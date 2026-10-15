import math
import numpy as np

TAU_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]


def freq_basis(v, p):
    k = np.arange(1, (p - 1) // 2 + 1)
    ang = 2 * np.pi * np.outer(v, k) / p
    return np.cos(ang), np.sin(ang)


def train_energy(Htr, ytr, p):
    Hc = Htr - Htr.mean(0); C, S = freq_basis(ytr, p); n = len(ytr)
    gc = Hc.T @ C / n; gs = Hc.T @ S / n
    return (gc * gc + gs * gs).sum(0), gc, gs


def null_threshold(Htr, ytr, p, n_perm=50, q=0.95, seed=0):
    rng = np.random.default_rng(seed); Hc = Htr - Htr.mean(0); n = len(ytr); mx = []
    for _ in range(n_perm):
        C, S = freq_basis(rng.permutation(ytr), p)
        gc = Hc.T @ C / n; gs = Hc.T @ S / n
        mx.append(float((gc * gc + gs * gs).sum(0).max()))
    return float(np.quantile(mx, q))


class FourierRidge:
    def __init__(s, Htr, ytr, p, tau):
        s.p = p; M = Htr.shape[1]
        C, S = freq_basis(ytr, p); K = C.shape[1]; s.K = K
        coef = np.linalg.solve(Htr.T @ Htr + tau * np.eye(M), Htr.T @ np.concatenate([C, S], 1))
        s.coef_c, s.coef_s = coef[:, :K], coef[:, K:]
        s.Cc, s.Cs = freq_basis(np.arange(p), p)

    def predict(s, Hall):
        return Hall @ s.coef_c, Hall @ s.coef_s

    def logits(s, Zc, Zs, ks=None, phase=False):
        ks = np.arange(s.K) if ks is None else np.asarray(ks, int)
        zc, zs = Zc[:, ks], Zs[:, ks]
        if phase:
            r = np.sqrt(zc * zc + zs * zs); r[r == 0] = 1.0; zc, zs = zc / r, zs / r
        return (2.0 / s.p) * (zc @ s.Cc[:, ks].T + zs @ s.Cs[:, ks].T)


def plain_ridge_logits(Htr, ytr, Hall, p, tau):
    T = np.eye(p)[ytr] - 1.0 / p
    B = np.linalg.solve(Htr.T @ Htr + tau * np.eye(Htr.shape[1]), Htr.T @ T)
    return Hall @ B


def _folds(n, k=5, seed=0, groups=None):
    if groups is None: groups = np.arange(n)
    u, inv = np.unique(groups, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(u)); fold_of = np.empty(len(u), int); fold_of[perm] = np.arange(len(u)) % k
    f = fold_of[inv]
    return [np.where(f == i)[0] for i in range(k)]


def cv_topk(Htr, ytr, p, tau_rel, k_max=None, seed=0, groups=None):
    n = len(ytr); K = (p - 1) // 2; k_max = K if k_max is None else k_max
    acc = np.zeros(k_max)
    for va in _folds(n, 5, seed, groups):
        trm = np.ones(n, bool); trm[va] = False
        E, _, _ = train_energy(Htr[trm], ytr[trm], p); order = np.argsort(-E)
        fr = FourierRidge(Htr[trm], ytr[trm], p, tau_rel * trm.sum())
        Zc, Zs = fr.predict(Htr[va])
        for j in range(k_max):
            acc[j] += (fr.logits(Zc, Zs, order[:j + 1]).argmax(1) == ytr[va]).mean()
    return int(np.argmax(acc)) + 1, acc / 5


def cv_tau(Htr, ytr, p, seed=0, groups=None):
    n = len(ytr); acc = np.zeros(len(TAU_GRID))
    for va in _folds(n, 5, seed, groups):
        trm = np.ones(n, bool); trm[va] = False
        for j, tr_ in enumerate(TAU_GRID):
            Fv = plain_ridge_logits(Htr[trm], ytr[trm], Htr[va], p, tr_ * trm.sum())
            acc[j] += (Fv.argmax(1) == ytr[va]).mean()
    return TAU_GRID[int(np.argmax(acc))], acc / 5


def grid_spectrum(H, y, p):
    C = H - H.mean(0); den = float((C * C).sum()); N = H.shape[0]
    cc, ss = freq_basis(y, p)
    return ((cc.T @ C) ** 2 + (ss.T @ C) ** 2).sum(1) / (N / 2) / den


def weight_spectrum(Wt, p):
    cc, ss = freq_basis(np.arange(p), p)
    P = ((cc.T @ Wt) ** 2 + (ss.T @ Wt) ** 2).sum(1)
    return P / P.sum()


def all_readouts(H, y, tr, p, tau_rel=1e-3, probe=None, tops=(1, 2, 4, 8, 16), groups=None):
    Htr, ytr = H[tr], y[tr]; n = len(tr); tau = tau_rel * n
    E, gc, gs = train_energy(Htr, ytr, p); order = np.argsort(-E)
    thr = null_threshold(Htr, ytr, p)
    ks_sig = [int(k) for k in order if E[k] > thr] or [int(order[0])]
    k_cv, cv_curve = cv_topk(Htr, ytr, p, tau_rel, groups=groups); ks_cv = [int(k) for k in order[:k_cv]]
    fr = FourierRidge(Htr, ytr, p, tau); Zc, Zs = fr.predict(H)
    pred = {'ridge': fr.logits(Zc, Zs).argmax(1),
            'filt_sig': fr.logits(Zc, Zs, ks_sig).argmax(1), 'filt_cv': fr.logits(Zc, Zs, ks_cv).argmax(1)}
    for K in tops:
        if K <= len(order): pred[f'filt_top{K}'] = fr.logits(Zc, Zs, order[:K]).argmax(1)
    for K in (4, 8):
        pred[f'phase_top{K}'] = fr.logits(Zc, Zs, order[:K], phase=True).argmax(1)
        ks = order[:K]; G = np.concatenate([gc[:, ks], gs[:, ks]], 1)
        U, sv, _ = np.linalg.svd(G, full_matrices=False); Q = U[:, sv > 1e-10 * sv[0]]
        Hp = (H - Htr.mean(0)) @ Q
        pred[f'proj_top{K}'] = plain_ridge_logits(Hp[tr], ytr, Hp, p, tau).argmax(1)
    tau_cv, _ = cv_tau(Htr, ytr, p, groups=groups); pred['ridge_cv'] = plain_ridge_logits(Htr, ytr, H, p, tau_cv * n).argmax(1)
    if probe is not None: pred['probe'] = probe(Htr, ytr, H).argmax(1)
    info = dict(ks_sig=[k + 1 for k in ks_sig], ks_cv=[k + 1 for k in ks_cv], k_cv=k_cv, tau_cv=tau_cv,
                order=[int(k) + 1 for k in order], E_train=(E / E.sum()).tolist(), E_thr=thr / E.sum(),
                cv_curve=cv_curve.tolist())
    return pred, info
