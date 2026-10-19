import json, math, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import bias_runner as B

BASE = dict(frac=0.8, width=12, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=10, heavy_every=50, p=11,
            lam_v=0.2, lam_w=0.015, deep_init_scale=8.0, lam_deep=0.07, depth=3, steps=120, init_seed=5, split_seed=6,
            eta_layers=[0.03, 0.006428571428571428, 0.003214285714285714], lam_layers=[0.015, 0.07, 0.07])


def _run(cfg, out=None, ckpt_every=0):
    return B.run(cfg, out or os.path.join(tempfile.mkdtemp(), 'x.json.gz'), ckpt_every=ckpt_every)


def _same(a, b):
    if os.environ.get('EXACT_TESTS') == '1': return json.dumps(a) == json.dumps(b)
    if isinstance(a, dict) and isinstance(b, dict): return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)): return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) and isinstance(b, float): return bool(np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))
    return type(a) is type(b) and a == b


def test_bias_gradients():
    for D in (1, 2, 3):
        cfg = dict(BASE, depth=D, bias=True, eta_layers=BASE['eta_layers'][:D], lam_layers=BASE['lam_layers'][:D])
        R = B.Run(cfg); rng = np.random.default_rng(1)
        R.bs = [0.3 * rng.normal(size=b.shape) for b in R.bs]; R.V = rng.normal(size=R.V.shape) * 0.05
        L, gWs, gV = R.loss_and_grads(); gbs = [g.copy() for g in R.gbs]; eps = 1e-6
        for l in range(D):
            for j in range(4):
                old = R.bs[l][j]
                R.bs[l][j] = old + eps; Lp = R.loss_and_grads()[0]; R.bs[l][j] = old - eps; Lm = R.loss_and_grads()[0]; R.bs[l][j] = old
                fd = (Lp - Lm) / (2 * eps); assert abs(fd - gbs[l][j]) < 1e-6 * max(1.0, abs(fd)), (D, l, j, fd, gbs[l][j])
            i, k = 1, 2; old = R.Ws[l][i, k]
            R.Ws[l][i, k] = old + eps; Lp = R.loss_and_grads()[0]; R.Ws[l][i, k] = old - eps; Lm = R.loss_and_grads()[0]; R.Ws[l][i, k] = old
            fd = (Lp - Lm) / (2 * eps); assert abs(fd - gWs[l][i, k]) < 1e-6 * max(1.0, abs(fd)), (D, l, 'W', fd, gWs[l][i, k])


def test_enlarged_polar_step():
    cfg = dict(BASE, bias=True)
    R = B.Run(cfg); rng = np.random.default_rng(2); R.bs = [0.1 * rng.normal(size=b.shape) for b in R.bs]
    W0 = [W.copy() for W in R.Ws]; b0 = [b.copy() for b in R.bs]
    L, gWs, gV = R.loss_and_grads(); gbs = [g.copy() for g in R.gbs]
    R.step(gWs, gV)
    for l in range(3):
        G = -np.concatenate([gWs[l], gbs[l][:, None]], 1)
        U, s, Zt = np.linalg.svd(G, full_matrices=False); r = int((s > 1e-7 * s[0]).sum())
        eta, lam = cfg['eta_layers'][l], cfg['lam_layers'][l]
        Wa = np.concatenate([W0[l], b0[l][:, None]], 1)
        exp = Wa + eta * ((U[:, :r] @ Zt[:r]) / math.sqrt(r) - lam * Wa)
        got = np.concatenate([R.Ws[l], R.bs[l][:, None]], 1)
        assert np.abs(got - exp).max() < 1e-12, (l, np.abs(got - exp).max())


def test_frozen_bias_equals_bias_free():
    for D in (1, 2, 3):
        c = dict(BASE, depth=D, eta_layers=BASE['eta_layers'][:D], lam_layers=BASE['lam_layers'][:D], observers_v1=True)
        a = _run(c); b = _run(dict(c, bias=True, bias_frozen=True))
        assert a['loss'] == b['loss'], D
        oa = [{k: v for k, v in o.items() if k != 'normb'} for o in b['obs']]
        assert json.dumps(a['obs']) == json.dumps(oa), D


def test_bias_changes_the_run_and_logs_norms():
    a = _run(dict(BASE)); b = _run(dict(BASE, bias=True))
    assert a['loss'][0] == b['loss'][0] and a['loss'][-1] != b['loss'][-1]
    hv = [o for o in b['obs'] if 'normb' in o]; assert len(hv) >= 2 and len(hv[-1]['normb']) == 3 and hv[-1]['normb'][0] > 0


def test_resume_with_bias():
    c = dict(BASE, bias=True, steps=60); d = tempfile.mkdtemp()
    a = B.run(c, os.path.join(d, 'a.json.gz'), ckpt_every=0)
    out = os.path.join(d, 'b.json.gz')
    R = B.Run(c); loss = np.empty(61); obs = []
    for t in range(30):
        L, gWs, gV = R.loss_and_grads(); loss[t] = L
        if t % c['obs_every'] == 0:
            o = R.observe(heavy=(t % c['heavy_every'] == 0)); o['step'] = t; o['loss'] = L; obs.append(o)
        R.step(gWs, gV, t)
    B._save_ckpt(out + '.ckpt.npz', R, 30, loss, obs)
    b = B.run(c, out, ckpt_every=0)
    assert _same(a['loss'], b['loss']) and _same(a['obs'], b['obs'])


def test_recipe():
    for t10, want in ((31940, 0.285), (30120, 0.300), (28160, 0.315), (27320, 0.320)):
        assert abs(B.solve_rate(t10, 0.35, 3.85) - want) < 1e-9


def test_chain():
    tmpl = dict(BASE, bias=True, steps=150)
    spec = dict(r0=0.5, x_target=3.85, pilot_cfg=dict(tmpl, stop_at_test=0.0), main_template=tmpl)
    d = tempfile.mkdtemp(); rec = B.run_chain(spec, d, ckpt_every=0)
    for f in ('pilot.json.gz', 'main.json.gz', 'recipe.json'): assert os.path.exists(os.path.join(d, f)), f
    M = B.load_gz(os.path.join(d, 'main.json.gz'))
    assert abs(M['cfg']['eta_layers'][-1] - rec['rate'] * B.ETA_UNIT) < 1e-15 and rec['rate'] == B.solve_rate(rec['t10_used'], 0.5, 3.85)


TESTS = [test_bias_gradients, test_enlarged_polar_step, test_frozen_bias_equals_bias_free, test_bias_changes_the_run_and_logs_norms,
         test_resume_with_bias, test_recipe, test_chain]

if __name__ == '__main__':
    for f in TESTS: f(); print('ok', f.__name__, flush=True)
    print('ALL TESTS PASS')
