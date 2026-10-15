import json, math, os, sys, tempfile, gzip
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import tf_runner as R

TINY = dict(p=7, frac=0.5, n_layers=2, d_model=8, n_heads=2, d_mlp=16, dtype='float64', init_seed=1, split_seed=2,
            steps=40, obs_every=5, heavy_every=10, eta_w=0.05, lam_w=0.1, eta_v=0.05, lam_v=0.2)


def test_gradients_match_finite_differences():
    m = R.TF(TINY); m.loss_and_grads()
    rng = np.random.default_rng(0)
    for name in ('W_E', 'W_pos', 'W_Q0', 'W_K1', 'W_V1', 'W_O0', 'W_in1', 'W_out0', 'W_U'):
        W = m.P[name]; g = W.grad.clone()
        for _ in range(3):
            idx = tuple(int(rng.integers(n)) for n in W.shape); h = 1e-6
            with torch.no_grad():
                W[idx] += h; lp = float(torch.nn.functional.cross_entropy(m.forward(m.tr_t), m.Y[m.tr_t]))
                W[idx] -= 2 * h; lm = float(torch.nn.functional.cross_entropy(m.forward(m.tr_t), m.Y[m.tr_t]))
                W[idx] += h
            fd = (lp - lm) / (2 * h)
            assert abs(fd - float(g[idx])) <= 1e-6 + 1e-5 * abs(fd), (name, idx, fd, float(g[idx]))


def test_last_position_shortcut_matches_full_computation():
    m = R.TF(TINY); rows = torch.arange(m.p * m.p)
    with torch.no_grad():
        F = m.forward(rows)
        P = m.P; c = m.cfg; d, H = c['d_model'], c['n_heads']; dh = d // H; X = m.X[rows]; N = X.shape[0]
        x = P['W_E'][X] + P['W_pos'][None]
        for l in range(c['n_layers']):
            q = (x @ P[f'W_Q{l}'].T).view(N, 3, H, dh).transpose(1, 2); k = (x @ P[f'W_K{l}'].T).view(N, 3, H, dh).transpose(1, 2)
            v = (x @ P[f'W_V{l}'].T).view(N, 3, H, dh).transpose(1, 2)
            att = ((q @ k.transpose(-1, -2)) / math.sqrt(dh)).masked_fill(~m.mask[None, None], float('-inf')).softmax(-1)
            x = x + (att @ v).transpose(1, 2).reshape(N, 3, d) @ P[f'W_O{l}'].T
            x = x + torch.relu(x @ P[f'W_in{l}'].T) @ P[f'W_out{l}'].T
        F2 = x[:, -1] @ P['W_U']
    assert torch.allclose(F, F2, atol=1e-12)


def test_polar_step_has_unit_frobenius_direction():
    G = torch.tensor(np.random.default_rng(3).normal(size=(6, 4)))
    Q, k = R.polar_dir(G, 1e-7)
    assert k == 4 and abs(float(torch.linalg.norm(Q / math.sqrt(k))) - 1) < 1e-12
    assert torch.allclose(Q.T @ Q, torch.eye(4, dtype=Q.dtype), atol=1e-12)
    G2 = G[:, :2] @ torch.tensor(np.random.default_rng(4).normal(size=(2, 4)))
    Q2, k2 = R.polar_dir(G2, 1e-7)
    assert k2 == 2 and abs(float(torch.linalg.norm(Q2 / math.sqrt(k2))) - 1) < 1e-12


def test_polar_update_rule():
    lr = R.Learner(TINY); W0 = {k: v.detach().clone() for k, v in lr.m.P.items()}
    lr.m.loss_and_grads(); G = {k: v.grad.clone() for k, v in lr.m.P.items()}; lr.step()
    c = lr.cfg
    for k, W in lr.m.P.items():
        if k == 'W_U':
            exp = W0[k] + c['eta_v'] * (-G[k] / torch.linalg.norm(G[k]) - c['lam_v'] * W0[k])
        else:
            Q, r = R.polar_dir(-G[k], c['rel_cut']); exp = W0[k] + c['eta_w'] * (Q / math.sqrt(r) - c['lam_w'] * W0[k])
        assert torch.allclose(W.detach(), exp, atol=1e-12), k


def _run(cfg, fn, ck=0):
    return R.run(cfg, fn, ckpt_every=ck)


def test_resume_is_exact_polar_and_adamw():
    for opt in ('polar', 'adamw'):
        cfg = dict(TINY, optimizer=opt, lr=1e-2, wd=0.5)
        with tempfile.TemporaryDirectory() as d:
            a = _run(cfg, os.path.join(d, 'a.json.gz'))
            cfg2 = dict(cfg); fn = os.path.join(d, 'b.json.gz')
            lr_ = R.Learner(cfg2); loss = np.empty(cfg2['steps'] + 1); obs = []
            for t in range(20):
                L = lr_.m.loss_and_grads(); loss[t] = L
                if t % cfg2['obs_every'] == 0:
                    o = R.observe(lr_.m, t % cfg2['heavy_every'] == 0); o['step'] = t; o['loss'] = L; obs.append(o)
                lr_.step()
            R._save_ckpt(fn + '.ckpt.pt', lr_, 20, loss, obs, None)
            b = _run(cfg2, fn, ck=20)
            assert b['resumed_from'] == 20
            assert np.array_equal(np.array(a['loss']), np.array(b['loss'])), opt
            assert [o['test_acc'] for o in a['obs']] == [o['test_acc'] for o in b['obs']]


def test_output_format_and_observers():
    with tempfile.TemporaryDirectory() as d:
        res = _run(dict(TINY), os.path.join(d, 'r.json.gz'))
        with gzip.open(os.path.join(d, 'r.json.gz'), 'rt') as f: back = json.load(f)
    assert len(back['loss']) == TINY['steps'] + 1
    h = [o for o in back['obs'] if 'normV' in o]
    assert h and all(k in h[0] for k in ('AH_full_h1', 'AH_full_h2', 'AH_full_resid', 'refit_test_acc', 'refit_mlp_Eref',
                                          'train_margin_min', 'test_loss'))
    assert all(0 <= o['AH_full_h2'] <= 1 + 1e-9 for o in h)


def test_ah_full_detects_addition_features():
    p = 11; a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); y = (a + b) % p
    H_add = np.stack([np.cos(2 * np.pi * y / p), np.sin(2 * np.pi * 3 * y / p)], 1)
    H_a = np.stack([np.cos(2 * np.pi * a / p), np.sin(2 * np.pi * a / p)], 1)
    assert abs(R.ah_full(H_add, y, p) - 1) < 1e-12 and R.ah_full(H_a, y, p) < 1e-12


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
    print('ALL TESTS PASS')
