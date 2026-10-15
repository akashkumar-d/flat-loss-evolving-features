import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import gzip, json, math, sys, time
import numpy as np

VERSION = 't41_runner 1.0 (2026-10-06)'
Q_CLK = 0.9 ** (1 / 32)


def qr_set(p):
    return sorted(set((u * u) % p for u in range(1, p)))


def make_split(cfg):
    p = cfg['p']; a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p)
    if cfg['split'] == 'qr':
        q = np.zeros(p, bool); q[qr_set(p)] = True
        is_tr = q[(b - a) % p]
    elif cfg['split'] == 'random':
        perm = np.random.default_rng(cfg['split_seed']).permutation(p * p); n = int(cfg['frac'] * p * p)
        is_tr = np.zeros(p * p, bool); is_tr[perm[:n]] = True
    else:
        raise ValueError('unknown split ' + str(cfg['split']))
    return np.flatnonzero(is_tr), np.flatnonzero(~is_tr)


def polar_dir(G, rel_cut=1e-7):
    tall = G.shape[0] >= G.shape[1]
    S = G.T @ G if tall else G @ G.T
    ev, Q = np.linalg.eigh(S)
    smax = math.sqrt(max(ev[-1], 0.0))
    if smax == 0: return np.zeros_like(G), 0
    keep = ev > (rel_cut * smax) ** 2
    Qk = Q[:, keep]; sk = np.sqrt(ev[keep])
    P = (G @ Qk) / sk @ Qk.T if tall else Qk @ (Qk.T @ G / sk[:, None])
    return P, int(keep.sum())


class Net:
    def __init__(self, cfg):
        self.cfg = cfg; p = self.p = cfg['p']; M = cfg['width']
        self.a = np.repeat(np.arange(p), p); self.b = np.tile(np.arange(p), p); self.y = (self.a + self.b) % p
        self.tr, self.te = make_split(cfg)
        rng = np.random.default_rng(cfg['init_seed'])
        c = cfg['init_scale']
        self.W = c * rng.normal(size=(M, 2 * p)) / math.sqrt(p)
        self.V = rng.normal(size=(p, M)) / (cfg.get('head_init_div', c) * math.sqrt(M))
        at, bt = self.a[self.tr], self.b[self.tr]
        self._oa = np.argsort(at, kind='stable'); self._ob = np.argsort(bt, kind='stable')
        self._sa = np.searchsorted(at[self._oa], np.arange(p)); self._sb = np.searchsorted(bt[self._ob], np.arange(p))
        assert np.all(np.bincount(at, minlength=p) > 0) and np.all(np.bincount(bt, minlength=p) > 0), 'an operand has no training row'

    def hidden(self, rows):
        p = self.p
        pre = self.W[:, self.a[rows]].T + self.W[:, p + self.b[rows]].T
        return pre, np.maximum(pre, 0.0)

    def loss_and_grads(self):
        tr = self.tr; n = len(tr); yt = self.y[tr]; idx = np.arange(n)
        pre, H = self.hidden(tr)
        F = H @ self.V.T
        mx = F.max(1, keepdims=True); E = np.exp(F - mx); S = E.sum(1, keepdims=True)
        lse = (mx + np.log(S))[:, 0]
        L = float(np.mean(lse - F[idx, yt]))
        Fy = F[idx, yt].copy(); F2 = F.copy(); F2[idx, yt] = -np.inf; marg = Fy - F2.max(1)
        dF = E / S; dF[idx, yt] -= 1.0; dF /= n
        gV = dF.T @ H
        dpre = (dF @ self.V) * (pre > 0)
        gA = np.add.reduceat(dpre[self._oa], self._sa, axis=0); gB = np.add.reduceat(dpre[self._ob], self._sb, axis=0)
        gW = np.concatenate([gA.T, gB.T], 1)
        return L, gW, gV, float((marg > 0).mean()), float(marg.min())

    def step(self, gW, gV):
        cfg = self.cfg
        Q, r = polar_dir(-gW)
        self.W = (1 - cfg['eta_w'] * cfg['lam_w']) * self.W + cfg['eta_w'] * (Q / math.sqrt(r) if r else 0 * self.W)
        if cfg['eta_v'] > 0:
            gn = float(np.linalg.norm(gV))
            self.V = (1 - cfg['eta_v'] * cfg['lam_v']) * self.V + cfg['eta_v'] * (-gV / gn if gn > 0 else 0 * gV)
        elif cfg['lam_v'] > 0:
            raise ValueError('lam_v > 0 needs eta_v > 0')

    def observe(self, heavy):
        p, y, tr, te = self.p, self.y, self.tr, self.te
        o = {}
        _, Hte = self.hidden(te); Fte = Hte @ self.V.T; yte = y[te]; it = np.arange(len(te))
        F2 = Fte.copy(); F2[it, yte] = -np.inf; mte = Fte[it, yte] - F2.max(1)
        o['test_acc'] = float((mte > 0).mean()); o['test_margin_min'] = float(mte.min())
        mx = Fte.max(1, keepdims=True); o['test_loss'] = float(np.mean((mx[:, 0] + np.log(np.exp(Fte - mx).sum(1))) - Fte[it, yte]))
        if not heavy: return o
        allr = np.arange(p * p); _, H = self.hidden(allr)
        C = H - H.mean(0); Ectr = float((C * C).sum())
        Mc = np.zeros((p, H.shape[1])); np.add.at(Mc, y, C); Mc /= p
        Eadd = float(p * (Mc * Mc).sum())
        o['Eadd'] = Eadd; o['Ectr'] = Ectr; o['AH'] = Eadd / Ectr if Ectr > 0 else 0.0
        Htr, Hte = H[tr], H[te]; n = len(tr)
        T = np.eye(p)[y] - 1.0 / p
        A = Htr.T @ Htr; R = float(np.trace(A)); HtT = Htr.T @ T[tr]
        for name, tau in (('canon', 1e-3 * n), ('thm', R / (32 * n))):
            if R == 0:
                o[f'refit_{name}_acc'] = 0.0; continue
            B = np.linalg.solve(A + tau * np.eye(A.shape[0]), HtT)
            Pte = Hte @ B; Ptr = Htr @ B
            P2 = Pte.copy(); P2[it, yte] = -np.inf; mr = Pte[it, yte] - P2.max(1)
            o[f'refit_{name}_acc'] = float((mr > 0).mean()); o[f'refit_{name}_margin_min'] = float(mr.min())
            o[f'refit_{name}_Eref'] = float(((Pte - T[te]) ** 2).sum() / (len(te) * (p - 1) / p))
            o[f'refit_{name}_train_acc'] = float((Ptr.argmax(1) == y[tr]).mean())
        o['tau_thm'] = R / (32 * n)
        o['normW'] = float(np.linalg.norm(self.W)); o['normV'] = float(np.linalg.norm(self.V))
        F = H @ self.V.T; o['logit_rms'] = float(np.sqrt((F * F).mean()))
        Ef = sum(np.abs(np.fft.rfft(Wx, axis=1)[:, 1:]) ** 2 for Wx in (self.W[:, :p], self.W[:, p:]))
        tot = float(Ef.sum()); o['W_four_top1'] = float(Ef.max(1).sum() / tot) if tot > 0 else 0.0
        return o


def _save_ckpt(path, net, t_next, loss, tracc, obs):
    tmp = path + '.tmp.npz'
    np.savez(tmp, t_next=t_next, loss=loss[:t_next], tracc=tracc[:t_next], W=net.W, V=net.V, obs=json.dumps(obs))
    os.replace(tmp, path)


def run(cfg, out, ckpt_every=2000):
    net = Net(cfg); T = cfg['steps']; t0 = time.time(); ck = out + '.ckpt.npz'
    loss = np.empty(T + 1); tracc = np.empty(T + 1); obs = []; t_start = 0
    if os.path.exists(ck):
        with np.load(ck, allow_pickle=False) as z:
            t_start = int(z['t_next']); loss[:t_start] = z['loss']; tracc[:t_start] = z['tracc']
            net.W = z['W']; net.V = z['V']; obs = json.loads(str(z['obs']))
    sg = cfg.get('stop_generalized'); gen_since = None; stop = False
    if sg:
        for o in obs:
            gen_since = (gen_since if gen_since is not None else o['step']) if o['test_acc'] >= sg['acc'] else None
    for t in range(t_start, T + 1):
        if ckpt_every and t > t_start and t % ckpt_every == 0: _save_ckpt(ck, net, t, loss, tracc, obs)
        L, gW, gV, acc_tr, mtr = net.loss_and_grads(); loss[t] = L; tracc[t] = acc_tr
        if t % cfg['obs_every'] == 0 or t == T:
            o = net.observe(heavy=(t % cfg['heavy_every'] == 0 or t == T))
            o.update(step=t, loss=L, train_acc=acc_tr, train_margin_min=mtr)
            if sg is not None and t < T:
                if o['test_acc'] >= sg['acc']:
                    gen_since = t if gen_since is None else gen_since
                    if t >= max(sg['factor'] * gen_since, gen_since + sg['extra']):
                        if 'AH' not in o: o.update(net.observe(heavy=True))
                        stop = True
                else:
                    gen_since = None
            obs.append(o)
        if t == T or stop: break
        net.step(gW, gV)
    res = dict(cfg=cfg, version=VERSION, loss=loss[:t + 1].tolist(), train_acc=tracc[:t + 1].tolist(), obs=obs,
               seconds=time.time() - t0, resumed_from=t_start, n_train=int(len(net.tr)), n_test=int(len(net.te)))
    if stop: res['stopped_at'] = t; res['stop_reason'] = 'generalized'
    tmp = out + '.tmp'
    with gzip.open(tmp, 'wt') as f: json.dump(res, f)
    os.replace(tmp, out)
    if os.path.exists(ck): os.remove(ck)
    return res


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == 'one':
        cfg = json.load(open(sys.argv[2])); out = sys.argv[3]
        os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
        run(cfg, out, int(os.environ.get('CAMPAIGN_CKPT_EVERY', 2000))); print('done', out, flush=True)
    else:
        print(__doc__)
