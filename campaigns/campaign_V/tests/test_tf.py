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


def test_svd_fallback_and_nonfinite_stop():
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        _fallback_and_nonfinite(pathlib.Path(d))


def _fallback_and_nonfinite(tmp_path):
    cfg = dict(p=11, frac=0.5, n_layers=1, d_model=16, n_heads=2, d_mlp=32, optimizer='polar', eta_w=0.03, lam_w=0.03, eta_v=0.03,
               lam_v=0.2, steps=20, obs_every=5, heavy_every=10, init_seed=1, split_seed=2)
    real = torch.linalg.svd; n = {'k': 0}
    def flaky(G, full_matrices=True):
        n['k'] += 1
        if G.dtype == torch.float32 and n['k'] == 5: raise torch.linalg.LinAlgError('simulated')
        return real(G, full_matrices=full_matrices)
    R.SVD_FALLBACKS.clear(); torch.linalg.svd = flaky
    try: res = R.run(cfg, str(tmp_path / 'a.json.gz'), ckpt_every=0)
    finally: torch.linalg.svd = real
    assert res['svd_fallback_steps'] == [0] and 'stop_reason' not in res and len(res['loss']) == 21
    base = R.run(cfg, str(tmp_path / 'b.json.gz'), ckpt_every=0)
    assert base['svd_fallback_steps'] == [] and base['loss'][:1] == res['loss'][:1]
    class Boom(R.Learner):
        k = 0
        def step(s):
            Boom.k += 1
            if Boom.k == 7:
                with torch.no_grad(): s.m.P['W_in0'][0, 0] = float('inf')
            return super().step()
    L0 = R.Learner; R.Learner = Boom
    try: res = R.run(cfg, str(tmp_path / 'c.json.gz'), ckpt_every=0)
    finally: R.Learner = L0
    assert res['stop_reason'] == 'nonfinite' and res['stopped_at'] == 7 and len(res['loss']) == 7
    assert all(x == x and abs(x) != float('inf') for x in res['loss'])


def test_final_rms_norm_forward_and_gradients():
    cfg = dict(TINY, final_norm='rms'); m = R.TF(cfg); m0 = R.TF(TINY); rows = torch.arange(m.p * m.p)
    with torch.no_grad():
        _, _, r0 = m0.forward(rows, want=True)
        r = r0 / torch.sqrt((r0 * r0).mean(-1, keepdim=True) + 1e-12)
        F, _, rin = m.forward(rows, want=True)
        assert torch.allclose(rin, r, atol=1e-12) and torch.allclose(F, r @ m.P['W_U'], atol=1e-12)
        assert torch.allclose((rin * rin).mean(-1), torch.ones(len(rows), dtype=rin.dtype), atol=1e-9)
    m.loss_and_grads(); rng = np.random.default_rng(1)
    for name in ('W_E', 'W_Q1', 'W_in0', 'W_out1', 'W_U'):
        W = m.P[name]; g = W.grad.clone()
        idx = tuple(int(rng.integers(n)) for n in W.shape); h = 1e-6
        with torch.no_grad():
            W[idx] += h; lp = float(torch.nn.functional.cross_entropy(m.forward(m.tr_t), m.Y[m.tr_t]))
            W[idx] -= 2 * h; lm = float(torch.nn.functional.cross_entropy(m.forward(m.tr_t), m.Y[m.tr_t]))
            W[idx] += h
        fd = (lp - lm) / (2 * h)
        assert abs(fd - float(g[idx])) <= 1e-6 + 1e-5 * abs(fd), (name, idx, fd, float(g[idx]))


def test_loss_record_is_float64_ce_of_float32_logits():
    cfg = dict(TINY, dtype='float32', steps=3, obs_every=1, heavy_every=1)
    with tempfile.TemporaryDirectory() as d:
        res = _run(cfg, os.path.join(d, 'a.json.gz'))
    m = R.TF(cfg)
    with torch.no_grad():
        F = m.forward(m.tr_t)
        L64 = float(torch.nn.functional.cross_entropy(F.double(), m.Y[m.tr_t]))
        L32 = float(torch.nn.functional.cross_entropy(F, m.Y[m.tr_t]))
    assert res['loss'][0] == L64 and abs(L64 - L32) < 1e-5 and m.P['W_U'].dtype == torch.float32


def test_logistic_probe():
    p = 11; a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); y = (a + b) % p
    H = np.concatenate([np.stack([np.cos(2 * np.pi * w * y / p), np.sin(2 * np.pi * w * y / p)], 1) for w in (1, 2, 3)], 1)
    tr = np.arange(0, p * p, 2)
    for sc in (1.0, 1e3):
        F = R.probe_lr(H[tr] * sc, y[tr], H * sc, p, 100, 1e-4)
        assert (F.argmax(1) == y).mean() == 1.0
    cfg = dict(TINY, probe_every=20, final_norm='rms')
    with tempfile.TemporaryDirectory() as d:
        res = _run(cfg, os.path.join(d, 'p.json.gz'))
    pr = [o for o in res['obs'] if 'probe_test_acc' in o]
    assert [o['step'] for o in pr] == [0, 20, 40] and all(0 <= o['probe_train_acc'] <= 1 for o in pr)
    assert not any('probe_test_acc' in o for o in _run(dict(TINY), os.path.join(tempfile.mkdtemp(), 'q.json.gz'))['obs'])


def test_resume_is_exact_with_rms_and_probe():
    cfg = dict(TINY, final_norm='rms', probe_every=10)
    for opt in ('polar', 'adamw'):
        c = dict(cfg, optimizer=opt, lr=1e-2, wd=0.5)
        with tempfile.TemporaryDirectory() as d:
            a = _run(c, os.path.join(d, 'a.json.gz')); fn = os.path.join(d, 'b.json.gz')
            lr_ = R.Learner(c); loss = np.empty(c['steps'] + 1); obs = []
            for t in range(20):
                L = lr_.m.loss_and_grads(); loss[t] = L
                if t % c['obs_every'] == 0:
                    hv = t % c['heavy_every'] == 0
                    o = R.observe(lr_.m, hv, hv and t % c['probe_every'] == 0); o['step'] = t; o['loss'] = L; obs.append(o)
                lr_.step()
            R._save_ckpt(fn + '.ckpt.pt', lr_, 20, loss, obs, None)
            b = _run(c, fn, ck=20)
            assert b['resumed_from'] == 20 and np.array_equal(np.array(a['loss']), np.array(b['loss'])), opt
            assert [o.get('probe_test_acc') for o in a['obs']] == [o.get('probe_test_acc') for o in b['obs']]


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
    print('ALL TESTS PASS')
