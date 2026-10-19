import sys, os, math, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t41_runner as R


def cfg(**k):
    c = dict(p=11, width=8, split='qr', init_seed=1, split_seed=0, init_scale=2.0, head_init_div=1.0,
             eta_w=0.01, lam_w=1.0, eta_v=0.0, lam_v=0.0, steps=30, obs_every=5, heavy_every=10)
    c.update(k); return c


def _same(a, b):
    if os.environ.get('EXACT_TESTS') == '1': return json.dumps(a) == json.dumps(b)
    if isinstance(a, dict) and isinstance(b, dict): return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)): return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) and isinstance(b, float): return bool(np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))
    return type(a) is type(b) and a == b


def test_split():
    for p in (11, 13, 101):
        tr, te = R.make_split(dict(p=p, split='qr'))
        a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); y = (a + b) % p
        assert len(tr) == p * (p - 1) // 2 and len(te) == p * p - len(tr)
        for arr in (a[tr], b[tr], y[tr]): assert np.all(np.bincount(arr, minlength=p) == (p - 1) // 2)
        assert np.sum(a[te] == b[te]) == p and np.sum(a[tr] == b[tr]) == 0
    tr, te = R.make_split(dict(p=11, split='random', frac=0.8, split_seed=3)); assert len(tr) == int(0.8 * 121)


def test_grad():
    net = R.Net(cfg()); net.V = np.random.default_rng(3).normal(size=net.V.shape)
    L, gW, gV, _, _ = net.loss_and_grads()
    rng = np.random.default_rng(4); eps = 1e-6
    for (Mat, G) in ((net.W, gW), (net.V, gV)):
        for _ in range(8):
            i = tuple(int(rng.integers(0, s)) for s in Mat.shape); old = Mat[i]
            Mat[i] = old + eps; Lp = net.loss_and_grads()[0]; Mat[i] = old - eps; Lm = net.loss_and_grads()[0]; Mat[i] = old
            fd = (Lp - Lm) / (2 * eps); assert abs(fd - G[i]) < 1e-6 * max(1, abs(fd)), (fd, G[i])


def test_step():
    net = R.Net(cfg(eta_v=0.1, lam_v=0.0)); W0, V0 = net.W.copy(), net.V.copy()
    L, gW, gV, _, _ = net.loss_and_grads(); net.step(gW, gV)
    U, s, Zt = np.linalg.svd(-gW, full_matrices=False); r = int((s > 1e-7 * s[0]).sum())
    Wexp = (1 - 0.01) * W0 + 0.01 * (U[:, :r] @ Zt[:r]) / math.sqrt(r)
    assert np.abs(net.W - Wexp).max() < 1e-12
    assert np.abs(net.V - (V0 - 0.1 * gV / np.linalg.norm(gV))).max() < 1e-14


def test_observers():
    net = R.Net(cfg(width=6)); p = net.p; y = net.y; k = np.arange(1, 4)
    Hadd = np.concatenate([np.cos(2 * np.pi * np.outer(y, k) / p), np.sin(2 * np.pi * np.outer(y, k) / p)], 1) + 3.0
    net.hidden = lambda rows: (Hadd[rows], Hadd[rows]); net.V = np.zeros((p, 6))
    o = net.observe(True); assert abs(o['AH'] - 1) < 1e-12, o['AH']
    yd = (net.a - net.b) % p
    Hdif = np.concatenate([np.cos(2 * np.pi * np.outer(yd, k) / p), np.sin(2 * np.pi * np.outer(yd, k) / p)], 1)
    net.hidden = lambda rows: (Hdif[rows], Hdif[rows]); o = net.observe(True); assert o['AH'] < 1e-12, o['AH']
    Hy = np.eye(p)[y]; net.V = np.zeros((p, p)); net.hidden = lambda rows: (Hy[rows], Hy[rows])
    o = net.observe(True)
    assert o['refit_canon_acc'] == 1.0 and o['refit_thm_acc'] == 1.0 and abs(o['AH'] - 1) < 1e-12


def test_resume():
    d = tempfile.mkdtemp(); c = cfg(steps=40, eta_v=0.01)
    a = R.run(c, os.path.join(d, 'a.json.gz'), ckpt_every=0)
    out = os.path.join(d, 'b.json.gz'); net = R.Net(c); loss = np.empty(41); tra = np.empty(41); obs = []
    for t in range(20):
        L, gW, gV, acc, mtr = net.loss_and_grads(); loss[t] = L; tra[t] = acc
        if t % c['obs_every'] == 0:
            o = net.observe(heavy=(t % c['heavy_every'] == 0)); o.update(step=t, loss=L, train_acc=acc, train_margin_min=mtr); obs.append(o)
        net.step(gW, gV)
    R._save_ckpt(out + '.ckpt.npz', net, 20, loss, tra, obs)
    b = R.run(c, out, ckpt_every=0)
    assert _same(a['loss'], b['loss']) and _same(a['obs'], b['obs']), 'resume differs'


def test_stop_generalized():
    calls = {'n': 0}; orig = R.Net.observe
    def fake(self, heavy):
        o = orig(self, heavy); k = calls['n']; calls['n'] += 1
        o['test_acc'] = 1.0 if k >= 4 else 0.0
        return o
    R.Net.observe = fake
    try:
        c = cfg(steps=200, obs_every=5, heavy_every=1000, stop_generalized=dict(acc=0.99, factor=2.0, extra=10))
        r = R.run(c, os.path.join(tempfile.mkdtemp(), 'g.json.gz'), ckpt_every=0)
    finally:
        R.Net.observe = orig
    assert r.get('stopped_at') == 40 and len(r['loss']) == 41 and 'AH' in r['obs'][-1], (r.get('stopped_at'), len(r['loss']))


TESTS = [test_split, test_grad, test_step, test_observers, test_resume, test_stop_generalized]

if __name__ == '__main__':
    for f in TESTS: f(); print('ok', f.__name__, flush=True)
    print('ALL TESTS PASS')
