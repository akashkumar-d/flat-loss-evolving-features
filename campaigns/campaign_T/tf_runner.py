#!/usr/bin/env python3
import os
_THREADS = int(os.environ.get('CAMPAIGN_THREADS', '1'))
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, str(_THREADS))
import gzip, json, math, sys, time
import numpy as np
import torch

torch.set_num_threads(_THREADS)
VERSION = 'campaign-T tf_runner 1.0 (2026-10-07)'
DEFAULTS = dict(p=113, frac=0.3, n_layers=1, d_model=128, n_heads=4, d_mlp=512, dtype='float32', optimizer='polar',
                eta_w=0.01, lam_w=0.1, eta_v=0.01, lam_v=0.2, rel_cut=1e-7, lr=1e-3, wd=1.0, betas=[0.9, 0.98], eps=1e-8,
                head_init_std=None, steps=30000, obs_every=20, heavy_every=500, refit_tau=1e-3)
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
        s.P = {k: v.requires_grad_(True) for k, v in P.items()}
        s.mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))

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
        logits = resid @ P['W_U']
        return (logits, mlps, resid) if want else logits

    def loss_and_grads(s):
        for v in s.P.values(): v.grad = None
        F = s.forward(s.tr_t)
        L = torch.nn.functional.cross_entropy(F, s.Y[s.tr_t])
        L.backward()
        return float(L.detach())


def polar_dir(G, rel_cut):
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
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
            if name == 'W_U':
                gn = float(torch.linalg.norm(G))
                W.add_(c['eta_v'] * ((-G / gn if gn > 0 else torch.zeros_like(G)) - c['lam_v'] * W))
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


@torch.no_grad()
def observe(m, heavy):
    p = m.p; allr = torch.arange(p * p); y = m.y; tr, te = m.tr, m.te
    F, mlps, resid = m.forward(allr, want=True)
    F = F.double().numpy()
    pred = F.argmax(1)
    other = F.copy(); other[np.arange(p * p), y] = -np.inf; marg = F[np.arange(p * p), y] - other.max(1)
    Ft = F[te]; Z = Ft - Ft.max(1, keepdims=True); lse = np.log(np.exp(Z).sum(1))
    o = dict(train_acc=float((pred[tr] == y[tr]).mean()), test_acc=float((pred[te] == y[te]).mean()),
             train_margin_min=float(marg[tr].min()), test_margin_min=float(marg[te].min()),
             test_loss=float((lse - Z[np.arange(len(te)), y[te]]).mean()))
    if heavy:
        L = m.cfg['n_layers']
        for l, h in enumerate(mlps): o[f'AH_full_h{l + 1}'] = ah_full(h.double().numpy(), y, p)
        R = resid.double().numpy()
        o['AH_full_resid'] = ah_full(R, y, p); o['AH_full_logit'] = ah_full(F - F.mean(1, keepdims=True), y, p)
        o['normV'] = float(torch.linalg.norm(m.P['W_U']))
        o['normW'] = float(math.sqrt(sum(float((W * W).sum()) for k, W in m.P.items() if k != 'W_U')))
        for k, W in m.P.items():
            if k != 'W_U': o['normW_' + k] = float(torch.linalg.norm(W))
        n = len(tr); tau = m.cfg['refit_tau'] * n
        for tag, H in (('', R), ('_mlp', mlps[-1].double().numpy())):
            B = ridge(H[tr], y[tr], tau, p); Fr = H @ B
            pr = Fr.argmax(1)
            Tte = np.eye(p)[y[te]] - 1.0 / p
            o['refit' + tag + '_train_acc'] = float((pr[tr] == y[tr]).mean())
            o['refit' + tag + '_test_acc'] = float((pr[te] == y[te]).mean())
            o['refit' + tag + '_Eref'] = float(((Fr[te] - Tte) ** 2).sum() / (len(te) * (p - 1) / p))
    return o


def _save_ckpt(path, lr, t_next, loss, obs, gen_since):
    tmp = path + '.tmp'
    st = dict(t_next=t_next, loss=loss[:t_next].copy(), obs=obs, gen_since=gen_since,
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
    loss = np.empty(T + 1); obs = []; t_start = 0
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        t_start = int(st['t_next']); loss[:t_start] = st['loss']; obs = st['obs']
        with torch.no_grad():
            for k, v in st['P'].items(): lr.m.P[k].copy_(v)
        if lr.opt is not None and st['opt'] is not None: lr.opt.load_state_dict(st['opt'])
    sg = c.get('stop_generalized'); reason = None; stop = False
    gen_since = _generalized_since(obs, sg['acc']) if sg else None
    t = t_start
    for t in range(t_start, T + 1):
        if ckpt_every and t > t_start and t % ckpt_every == 0: _save_ckpt(ck, lr, t, loss, obs, gen_since)
        L = lr.m.loss_and_grads(); loss[t] = L
        if t % c['obs_every'] == 0 or t == T:
            heavy = (t % c['heavy_every'] == 0 or t == T)
            o = observe(lr.m, heavy); o['step'] = t; o['loss'] = L
            if sg is not None and t < T:
                if o['test_acc'] >= sg['acc']:
                    if gen_since is None: gen_since = t
                    if t >= max(sg['factor'] * gen_since, gen_since + sg['extra']):
                        if 'normV' not in o: o = observe(lr.m, True); o['step'] = t; o['loss'] = L
                        stop = True; reason = 'generalized'
                else:
                    gen_since = None
            obs.append(o)
        if t == T or stop: break
        lr.step()
    res = dict(cfg=cfg, cfg_full=c, version=VERSION, loss=loss[:t + 1].tolist(), obs=obs, seconds=time.time() - t0,
               resumed_from=t_start, threads=_THREADS, torch=torch.__version__)
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
