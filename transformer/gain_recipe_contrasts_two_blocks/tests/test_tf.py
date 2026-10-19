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


def test_ngd_update_rule():
    for fn in ('none', 'rms'):
        c0 = dict(TINY, hidden_mode='ngd', final_norm=fn)
        lr = R.Learner(c0); W0 = {k: v.detach().clone() for k, v in lr.m.P.items()}
        lr.m.loss_and_grads(); G = {k: v.grad.clone() for k, v in lr.m.P.items()}; lr.step()
        c = lr.cfg
        for k, W in lr.m.P.items():
            if k == 'W_U':
                exp = W0[k] + c['eta_v'] * (-G[k] / torch.linalg.norm(G[k]) - c['lam_v'] * W0[k])
            else:
                exp = W0[k] + c['eta_w'] * (-G[k] / torch.linalg.norm(G[k]) - c['lam_w'] * W0[k])
            assert torch.allclose(W.detach(), exp, atol=1e-12), (fn, k)


def test_polar_mode_is_the_default_and_unknown_mode_fails():
    a = R.Learner(dict(TINY)); b = R.Learner(dict(TINY, hidden_mode='polar'))
    for lr in (a, b):
        for _ in range(3): lr.m.loss_and_grads(); lr.step()
    assert all(torch.equal(a.m.P[k], b.m.P[k]) for k in a.m.P)
    try:
        R.Learner(dict(TINY, hidden_mode='sgd')); assert False, 'unknown hidden_mode accepted'
    except ValueError:
        pass


def test_no_mirror_held_out_accuracy():
    cfg = dict(TINY, p=11, frac=0.4, heavy_every=5, probe_every=5)
    m = R.TF(cfg); p = m.p
    intr = set(m.tr.tolist()); exp = [i for i in m.te.tolist() if (i % p) * p + i // p not in intr]
    assert m.te_nm.tolist() == exp and 0 < len(exp) < len(m.te)
    o = R.observe(m, True, True)
    with torch.no_grad(): pred = m.forward(torch.arange(p * p)).numpy().argmax(1)
    assert abs(o['test_acc_nm'] - float((pred[exp] == m.y[exp]).mean())) < 1e-12
    assert all(k in o for k in ('refit_test_acc_nm', 'refit_mlp_test_acc_nm', 'probe_test_acc_nm'))


def test_small_head_init():
    m = R.TF(dict(TINY, head_init_std=1e-3)); s = float(m.P['W_U'].std())
    assert 5e-4 < s < 2e-3 and abs(float(R.TF(dict(TINY)).P['W_U'].std()) - 1 / math.sqrt(TINY['d_model'])) < 0.2


def _grid(p):
    a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); return a, b, (a + b) % p


def test_fourier_identity_ridge_is_equal_weight_frequency_sum():
    p = 13; a, b, y = _grid(p); rng = np.random.default_rng(0)
    H = np.concatenate([np.stack([np.cos(2 * np.pi * k * y / p), np.sin(2 * np.pi * k * a / p)], 1) for k in (2, 5)], 1)
    H = H @ rng.normal(size=(4, 12)) + 0.3 * rng.normal(size=(p * p, 12))
    tr = np.sort(rng.permutation(p * p)[:80]); tau = 1e-3 * len(tr)
    fr = R.FourierReadouts(H, tr, y[tr], p, tau)
    plain = H @ R.ridge(H[tr], y[tr], tau, p)
    assert np.abs(plain - (2.0 / p) * fr.logits(np.arange(fr.K))).max() < 1e-10
    Fc = plain - plain.mean(1, keepdims=True)
    assert np.abs(fr.restrict(plain, np.arange(fr.K)) - Fc).max() < 1e-10


def test_fourier_readouts_never_see_held_out_labels():
    p = 11; a, b, y = _grid(p); rng = np.random.default_rng(1)
    H = rng.normal(size=(p * p, 9)); F = rng.normal(size=(p * p, p)); tr = np.sort(rng.permutation(p * p)[:60])
    P1 = R.FourierReadouts(H, tr, y[tr], p, 0.06).predictions(F)
    y2 = y.copy(); te = np.setdiff1d(np.arange(p * p), tr); y2[te] = rng.integers(0, p, len(te))
    P2 = R.FourierReadouts(H, tr, y2[tr], p, 0.06).predictions(F)
    assert P1.keys() == P2.keys() and all(np.array_equal(P1[k], P2[k]) for k in P1)
    import inspect
    assert list(inspect.signature(R.FourierReadouts.__init__).parameters) == ['s', 'H', 'tr', 'ytr', 'p', 'tau']


def test_fourier_planted_frequencies_and_spectra():
    p = 31; a, b, y = _grid(p); rng = np.random.default_rng(2)
    cols = []
    for k in (3, 7):
        w = 2 * np.pi * k / p; cols += [np.cos(w * y), np.sin(w * y), np.cos(w * a), np.sin(w * b)]
    H = np.stack(cols, 1) @ rng.normal(size=(8, 40)) + 0.3 * rng.normal(size=(p * p, 40))
    tr = np.sort(rng.permutation(p * p)[:480]); te = np.setdiff1d(np.arange(p * p), tr)
    F = H @ rng.normal(size=(40, p))
    o = R.fourier_observers(H, F, y, tr, te, None, p, 1e-3 * len(tr), head=rng.normal(size=(40, p)),
                            embed=rng.normal(size=(p, 5)))
    assert sorted(o['ftop8'][:2]) == [3, 7]
    assert o['fref_top2_test_acc'] >= o['fref_all_test_acc'] and o['fref_top2_test_acc'] > 0.9
    assert abs(sum(o['fspec_grid']) - R.ah_full(H, y, p)) < 1e-10 and abs(sum(o['fspec_train']) - 1) < 1e-12
    K = (p - 1) // 2
    for k in ('fspec_train', 'fspec_hor2', 'fspec_grid', 'fspec_head', 'fspec_embed'): assert len(o[k]) == K, k
    assert np.argsort(o['fspec_hor2'])[-2:].tolist() in ([2, 6], [6, 2])
    assert o['ffam_tr_train_acc'] >= 0 and 'fnet_top4_test_acc' in o and 'fref_top16_test_acc' not in o


def test_fourier_observers_are_read_only_and_recorded():
    base = dict(TINY, p=11, n_layers=1, final_norm='rms', heavy_every=10, probe_every=20)
    with tempfile.TemporaryDirectory() as d:
        off = _run(base, os.path.join(d, 'off.json.gz'))
        on = _run(dict(base, observers_fourier=True), os.path.join(d, 'on.json.gz'))
    assert off['loss'] == on['loss'] and len(off['obs']) == len(on['obs'])
    for a_, b_ in zip(off['obs'], on['obs']):
        assert all(a_[k] == b_[k] for k in a_), a_['step']
        extra = set(b_) - set(a_)
        if 'normV' in a_:
            assert {'fref_all_test_acc_nm', 'fref_top4_test_acc', 'fphase_top8_test_acc', 'fnet_top4_test_acc_nm',
                    'fref_oracle_test_acc', 'fnet_oracle_test_acc', 'fspec_hor2', 'fspec_head', 'fspec_embed', 'ftop8',
                    'ffam_all_test_acc_nm', 'ffam_ho_test_acc', 'ffam_tr_test_acc', 'fmref_top4_test_acc'} <= extra
            assert b_['fref_all_test_acc'] == b_['refit_test_acc'] and b_['fmref_all_test_acc'] == b_['refit_mlp_test_acc']
        else:
            assert not extra


def test_resume_is_exact_with_fourier_observers():
    c = dict(TINY, p=11, n_layers=1, final_norm='rms', observers_fourier=True)
    with tempfile.TemporaryDirectory() as d:
        a_ = _run(c, os.path.join(d, 'a.json.gz')); fn = os.path.join(d, 'b.json.gz')
        lr_ = R.Learner(c); loss = np.empty(c['steps'] + 1); obs = []
        for t in range(20):
            L = lr_.m.loss_and_grads(); loss[t] = L
            if t % c['obs_every'] == 0:
                o = R.observe(lr_.m, t % c['heavy_every'] == 0); o['step'] = t; o['loss'] = L; obs.append(o)
            lr_.step()
        R._save_ckpt(fn + '.ckpt.pt', lr_, 20, loss, obs, None)
        b_ = _run(c, fn, ck=20)
    assert np.array_equal(np.array(a_['loss']), np.array(b_['loss']))
    assert [o.get('fref_top4_test_acc') for o in a_['obs']] == [o.get('fref_top4_test_acc') for o in b_['obs']]


def _gain_cfg(**kw):
    return dict(dict(TINY, p=11, n_layers=1, final_norm='rms', gain_init=2.0, eta_g=0.01, lam_g=1.0), **kw)


def test_gain_off_is_14():
    c = dict(TINY, final_norm='rms')
    m = R.TF(c); assert 'g' not in m.P
    with tempfile.TemporaryDirectory() as d:
        a = _run(c, os.path.join(d, 'a.json.gz'))
        b = _run(dict(c, gain_init=1.0, eta_g=0.0, lam_g=0.0), os.path.join(d, 'b.json.gz'))
    assert not any('gain' in o for o in a['obs']) and all(o['gain'] == 1.0 for o in b['obs'])
    assert np.array_equal(np.array(a['loss']), np.array(b['loss']))
    ka = [{k: v for k, v in o.items() if k != 'gain'} for o in a['obs']]; kb = [{k: v for k, v in o.items() if k != 'gain'} for o in b['obs']]
    assert ka == kb


def test_gain_scales_logits_and_needs_rms():
    m = R.TF(_gain_cfg()); rows = torch.arange(20)
    with torch.no_grad():
        F2 = m.forward(rows).clone(); m.P['g'].fill_(1.0); F1 = m.forward(rows)
    assert torch.allclose(F2, 2.0 * F1, rtol=0, atol=1e-13)
    try:
        R.TF(dict(TINY, final_norm='none', gain_init=2.0)); raise AssertionError('accepted gain without rms')
    except ValueError:
        pass


def test_gain_gradient_and_update_rule():
    c = _gain_cfg(); lr_ = R.Learner(c); lr_.m.loss_and_grads()
    g0 = float(lr_.m.P['g']); G = float(lr_.m.P['g'].grad)
    with torch.no_grad():
        h = 1e-6; lr_.m.P['g'] += h; lp = float(torch.nn.functional.cross_entropy(lr_.m.forward(lr_.m.tr_t), lr_.m.Y[lr_.m.tr_t]))
        lr_.m.P['g'] -= 2 * h; lm = float(torch.nn.functional.cross_entropy(lr_.m.forward(lr_.m.tr_t), lr_.m.Y[lr_.m.tr_t]))
        lr_.m.P['g'] += h
    assert abs((lp - lm) / (2 * h) - G) <= 1e-6 + 1e-5 * abs(G)
    W0 = {k: v.detach().clone() for k, v in lr_.m.P.items()}; Gs = {k: v.grad.clone() for k, v in lr_.m.P.items()}
    lr_.step()
    assert math.isclose(float(lr_.m.P['g']), g0 + c['eta_g'] * (-math.copysign(1.0, G) - c['lam_g'] * g0), rel_tol=0, abs_tol=1e-15)
    exp = W0['W_U'] + c['eta_v'] * (-Gs['W_U'] / torch.linalg.norm(Gs['W_U']) - c['lam_v'] * W0['W_U'])
    assert torch.allclose(lr_.m.P['W_U'].detach(), exp, atol=1e-12)


def test_gain_relaxes_to_one_over_lam_g_once_fit():
    c = dict(p=11, frac=0.5, n_layers=1, d_model=16, n_heads=2, d_mlp=32, optimizer='polar', dtype='float64', eta_w=0.1,
             lam_w=0.01, eta_v=0.03, lam_v=0.275, steps=300, obs_every=10, heavy_every=100, init_seed=1, split_seed=2,
             final_norm='rms', gain_init=3.0, eta_g=0.01, lam_g=1.0)
    with tempfile.TemporaryDirectory() as d:
        r = _run(c, os.path.join(d, 'g.json.gz'))
    g = [o['gain'] for o in r['obs']]; fit = [o['train_acc'] == 1.0 for o in r['obs']]
    i = fit.index(True)
    assert all(b <= a + 1e-15 for a, b in zip(g[i:], g[i + 1:])) and min(g[i:]) >= 1.0 - 1e-12 and g[-1] < g[i]


def test_gain_observers_read_the_normalized_input():
    c = _gain_cfg(observers_fourier=True); m = R.TF(c)
    o2 = R.observe(m, True)
    with torch.no_grad(): m.P['g'].fill_(5.0)
    o5 = R.observe(m, True)
    for k in ('AH_full_resid', 'refit_test_acc', 'refit_Eref', 'fref_top4_test_acc', 'fspec_train', 'normW'):
        assert o2[k] == o5[k], k
    assert 'normW_g' not in o2 and o5['gain'] == 5.0


def test_resume_is_exact_with_gain():
    c = _gain_cfg(observers_fourier=True)
    with tempfile.TemporaryDirectory() as d:
        a_ = _run(c, os.path.join(d, 'a.json.gz')); fn = os.path.join(d, 'b.json.gz')
        lr_ = R.Learner(c); loss = np.empty(c['steps'] + 1); obs = []
        for t in range(20):
            L = lr_.m.loss_and_grads(); loss[t] = L
            if t % c['obs_every'] == 0:
                o = R.observe(lr_.m, t % c['heavy_every'] == 0); o['step'] = t; o['loss'] = L; obs.append(o)
            lr_.step()
        R._save_ckpt(fn + '.ckpt.pt', lr_, 20, loss, obs, None)
        b_ = _run(c, fn, ck=20)
    assert b_['resumed_from'] == 20 and np.array_equal(np.array(a_['loss']), np.array(b_['loss']))
    assert [o['gain'] for o in a_['obs']] == [o['gain'] for o in b_['obs']]


def test_v16_adamw_keeps_the_gain_on_its_sign_rule():
    cfg = dict(TINY, final_norm='rms', optimizer='adamw', lr=1e-3, wd=1.0, gain_init=2.0, eta_g=1e-2, lam_g=1.0)
    lr = R.Learner(cfg); m = lr.m
    assert all(p_ is not m.P['g'] for grp in lr.opt.param_groups for p_ in grp['params'])
    assert sum(len(grp['params']) for grp in lr.opt.param_groups) == len(m.P) - 1
    for _ in range(3):
        g0 = m.P['g'].detach().clone(); m.loss_and_grads(); G = m.P['g'].grad.clone(); W0 = m.P['W_E'].detach().clone(); lr.step()
        assert torch.equal(m.P['g'], g0 + 1e-2 * (-torch.sign(G) - 1.0 * g0))
        assert not torch.equal(m.P['W_E'], W0)


def test_v16_adamw_with_gain_resumes_exactly():
    c = _gain_cfg(optimizer='adamw', lr=1e-2, wd=0.5)
    with tempfile.TemporaryDirectory() as d:
        a_ = _run(c, os.path.join(d, 'a.json.gz')); fn = os.path.join(d, 'b.json.gz')
        lr_ = R.Learner(c); loss = np.empty(c['steps'] + 1); obs = []
        for t in range(20):
            L = lr_.m.loss_and_grads(); loss[t] = L
            if t % c['obs_every'] == 0:
                o = R.observe(lr_.m, t % c['heavy_every'] == 0); o['step'] = t; o['loss'] = L; obs.append(o)
            lr_.step()
        R._save_ckpt(fn + '.ckpt.pt', lr_, 20, loss, obs, None)
        b_ = _run(c, fn, ck=20)
    assert b_['resumed_from'] == 20 and np.array_equal(np.array(a_['loss']), np.array(b_['loss']))
    assert [o['gain'] for o in a_['obs']] == [o['gain'] for o in b_['obs']]


def test_v16_matches_v15_update_for_update():
    import importlib.util
    fn = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'one_block_gain_p113', 'tf_runner.py')
    if not os.path.exists(fn): return
    spec = importlib.util.spec_from_file_location('tf_runner_15', fn); R15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R15)
    assert R15.VERSION.startswith('campaign-Y tf_runner 1.5')
    base = dict(TINY, final_norm='rms')
    for cs in (dict(optimizer='polar'), dict(optimizer='polar', hidden_mode='ngd'), dict(optimizer='adamw', lr=1e-3, wd=1.0),
               dict(optimizer='polar', gain_init=2.0, eta_g=1e-2), dict(optimizer='polar', hidden_mode='ngd', gain_init=2.0, eta_g=1e-2),
               dict(optimizer='adamw', lr=1e-3, wd=1.0, final_norm='none')):
        cfg = dict(base, **cs); a = R15.Learner(cfg); b = R.Learner(cfg)
        for _ in range(25):
            assert a.m.loss_and_grads() == b.m.loss_and_grads(); a.step(); b.step()
        assert all(torch.equal(a.m.P[k], b.m.P[k]) for k in a.m.P), cs


def test_manifest_is_campaign_y_g2_plus_the_declared_changes():
    man = json.load(open(os.path.join(os.path.dirname(HERE), 'manifest.json')))
    Y_G2 = dict(d_mlp=512, d_model=128, dtype='float64', eta_v=0.03, eta_w=0.1, final_norm='rms', frac=0.2, heavy_every=100,
                hidden_mode='polar', lam_w=0.01, n_heads=4, n_layers=1, obs_every=20, optimizer='polar', p=113, probe_every=200,
                refit_tau=0.001, steps=30000, stop_generalized=dict(acc=0.99, extra=5000, factor=1.5), observers_fourier=True,
                lam_v=0.275, gain_init=2.0, eta_g=2e-4, lam_g=1.0)
    ADAMW = {'optimizer', 'lr', 'wd', 'betas', 'eps'}; NOGAIN = {'gain_init', 'eta_g', 'lam_g'}
    allowed = {'C_NGD_G2': ({'hidden_mode'}, set(range(11, 21))), 'C_ADAMW_G2': (ADAMW, set(range(11, 21))),
               'C_NGD_NOGAIN': ({'hidden_mode'} | NOGAIN, set(range(11, 16))), 'C_ADAMW_NOGAIN': (ADAMW | NOGAIN, set(range(11, 16)))}
    assert man['campaign'] == 'Z'
    seen = set(); per = {}
    for r in man['runs']:
        c = r['cfg']; cell = r['cell']; exp = dict(Y_G2, init_seed=7100 + r['seed'], split_seed=8100 + r['seed'])
        diff = {k for k in set(c) | set(exp) if c.get(k) != exp.get(k)}
        if cell in allowed:
            keys, seeds = allowed[cell]; assert diff == keys, (r['id'], diff)
            if 'hidden_mode' in diff: assert c['hidden_mode'] == 'ngd'
            if 'optimizer' in diff: assert (c['optimizer'], c['lr'], c['wd'], c['betas'], c['eps']) == ('adamw', 1e-3, 1.0, [0.9, 0.98], 1e-8)
            if 'gain_init' in diff: assert not ({'gain_init', 'eta_g', 'lam_g'} & set(c))
        elif cell.startswith('R_G'):
            assert diff <= {'gain_init', 'eta_g'}, (r['id'], diff); seeds = {11, 12, 13}
            assert c['gain_init'] in (1.5, 2.0, 3.0) and c['eta_g'] in (0.0, 4e-4, 2e-4, 1e-4, 5e-5) and diff
        elif cell.startswith('R_'):
            assert len(diff) == 1 and diff <= {'lam_v', 'eta_w', 'lam_w', 'frac'}, (r['id'], diff); seeds = {11, 12, 13}
        elif cell.startswith('P_'):
            assert diff <= {'p', 'frac', 'lam_v', 'eta_g'} and 'p' in diff and c['p'] != 113, (r['id'], diff); seeds = set(range(11, 16))
        elif cell.startswith('B_'):
            assert diff <= {'n_layers', 'frac', 'lam_v', 'eta_g'} and c['n_layers'] == 2 and r['threads'] == 2, (r['id'], diff)
            seeds = set(range(11, 16))
        else:
            raise AssertionError(f'unknown cell {cell}')
        assert r['id'] == f"Z_{cell}_s{r['seed']}" and r['id'] not in seen; seen.add(r['id'])
        per.setdefault(cell, (set(), seeds))[0].add(r['seed'])
    for cell, (got, want) in per.items(): assert got == want, (cell, got, want)
    assert {'C_NGD_G2', 'C_ADAMW_G2', 'C_NGD_NOGAIN', 'C_ADAMW_NOGAIN'} <= set(per)
    own = {}
    for r in man['runs']: own.setdefault(r['worker'], []).append(r)
    assert man['workers'] == [f'Z{i}' for i in range(1, 25)]
    for w, R in own.items():
        if int(w[1:]) <= 12: assert all(not (x['part'] == 'B' and x['seed'] >= 14) for x in R), w
        else: assert len(R) == 1 and R[0]['part'] == 'B' and R[0]['seed'] in (14, 15), w


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
    print('ALL TESTS PASS')
