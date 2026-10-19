import importlib.util, json, math, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import bias_runner as B

CAMPAIGN_I_RUNNER = os.path.join(os.path.dirname(ROOT), 'hidden_bias_depths1_2_4', 'bias_runner.py')
BASE = dict(frac=0.8, width=12, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=10, heavy_every=50, p=11,
            lam_v=0.2, lam_w=0.015, deep_init_scale=8.0, lam_deep=0.07, depth=3, steps=120, init_seed=5, split_seed=6,
            eta_layers=[0.03, 0.006428571428571428, 0.003214285714285714], lam_layers=[0.015, 0.07, 0.07])
KNEE = dict(frac=0.8, width=16, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=5, heavy_every=10, p=11,
            lam_v=2.0, lam_w=0.015, depth=1, steps=160, init_seed=7, split_seed=8, observers_v1=True, bias=True)


def _run(mod, cfg, ckpt_every=0):
    return mod.run(cfg, os.path.join(tempfile.mkdtemp(), 'x.json.gz'), ckpt_every=ckpt_every)


def _load_campaign_I():
    spec = importlib.util.spec_from_file_location('bias_runner_campaign_I', CAMPAIGN_I_RUNNER)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _same(a, b):
    if os.environ.get('EXACT_TESTS') == '1': return json.dumps(a) == json.dumps(b)
    if isinstance(a, dict) and isinstance(b, dict): return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)): return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) and isinstance(b, float): return bool(np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))
    return type(a) is type(b) and a == b


def test_default_equals_campaign_I():
    if not os.path.exists(CAMPAIGN_I_RUNNER):
        print('  skipped: campaign I runner not found at', CAMPAIGN_I_RUNNER); return
    BI = _load_campaign_I()
    cfgs = []
    for D in (1, 2, 3):
        c = dict(BASE, depth=D, eta_layers=BASE['eta_layers'][:D], lam_layers=BASE['lam_layers'][:D], observers_v1=True)
        cfgs += [c, dict(c, bias=True), dict(c, bias=True, bias_frozen=True), dict(c, bias=True, bias_update='joint')]
    cfgs.append(dict(KNEE))
    for c in cfgs:
        a = _run(BI, c); b = _run(B, c)
        assert _same(a['loss'], b['loss']), c
        assert _same(a['obs'], b['obs']), c
        assert 'bias_knee' not in b


def test_separate_step():
    for D in (1, 3):
        cfg = dict(BASE, depth=D, eta_layers=BASE['eta_layers'][:D], lam_layers=BASE['lam_layers'][:D], bias=True,
                   bias_update='separate')
        R = B.Run(cfg); rng = np.random.default_rng(3); R.bs = [0.1 * rng.normal(size=b.shape) for b in R.bs]
        W0 = [W.copy() for W in R.Ws]; b0 = [b.copy() for b in R.bs]
        L, gWs, gV = R.loss_and_grads(); gbs = [g.copy() for g in R.gbs]
        R.step(gWs, gV)
        for l in range(D):
            U, s, Zt = np.linalg.svd(-gWs[l], full_matrices=False); r = int((s > 1e-7 * s[0]).sum())
            eta, lam = cfg['eta_layers'][l], cfg['lam_layers'][l]
            expW = W0[l] + eta * ((U[:, :r] @ Zt[:r]) / math.sqrt(r) - lam * W0[l])
            g = gbs[l]; expb = b0[l] + eta * (-g / np.linalg.norm(g) / math.sqrt(r) - lam * b0[l])
            assert np.abs(R.Ws[l] - expW).max() < 1e-12, (D, l, np.abs(R.Ws[l] - expW).max())
            assert np.abs(R.bs[l] - expb).max() < 1e-12, (D, l, np.abs(R.bs[l] - expb).max())


def test_separate_changes_the_run():
    j = _run(B, dict(KNEE)); s = _run(B, dict(KNEE, bias_update='separate')); f = _run(B, {k: v for k, v in KNEE.items() if k != 'bias'})
    assert j['loss'][0] == s['loss'][0] == f['loss'][0]
    assert s['loss'][-1] != j['loss'][-1] and s['loss'][-1] != f['loss'][-1]
    assert any('normb' in o for o in s['obs'])


def test_after_knee():
    j = _run(B, dict(KNEE))
    kj = B.knee_of(j['obs'], KNEE['lam_v']); assert kj is not None and 0 < kj < KNEE['steps'] - 30, kj
    eta, lam = KNEE['eta_w'], KNEE['lam_w']
    for mode in ('frozen', 'decay'):
        r = _run(B, dict(KNEE, bias_after_knee=mode)); k = r['bias_knee']
        assert k == kj == B.knee_of(r['obs'], KNEE['lam_v']), (mode, k, kj)
        assert r['loss'][:k + 1] == j['loss'][:k + 1], mode
        assert r['loss'][k + 2] != j['loss'][k + 2], mode
        oj = [o for o in j['obs'] if o['step'] <= k]; orr = [o for o in r['obs'] if o['step'] <= k]
        assert json.dumps(oj) == json.dumps(orr), mode
        nb = [(o['step'], o['normb'][0]) for o in r['obs'] if 'normb' in o and o['step'] >= k]
        assert len(nb) >= 3 and nb[0][0] == k
        for t, v in nb:
            want = nb[0][1] if mode == 'frozen' else nb[0][1] * (1 - eta * lam) ** (t - k)
            if mode == 'frozen': assert v == want, (mode, t, v, want)
            else: assert abs(v - want) <= 1e-12 * want, (mode, t, v, want)


def test_decay_equals_folded_bias_free():
    p = KNEE['p']; K = 40; N = 60
    Rd = B.Run(dict(KNEE, bias_after_knee='decay'))
    for t in range(K):
        L, gWs, gV = Rd.loss_and_grads(); Rd.step(gWs, gV, t)
    assert np.linalg.norm(Rd.bs[0]) > 0
    Rd.bias_knee = K
    Rf = B.Run({k: v for k, v in KNEE.items() if k != 'bias'})
    W = Rd.Ws[0].copy(); W[:, :p] += Rd.bs[0][:, None]; Rf.Ws = [W]; Rf.V = Rd.V.copy()
    for t in range(K, K + N):
        Ld, gWd, gVd = Rd.loss_and_grads(); Lf, gWf, gVf = Rf.loss_and_grads()
        assert abs(Ld - Lf) <= 1e-10 * abs(Lf), (t, Ld, Lf)
        Rd.step(gWd, gVd, t); Rf.step(gWf, gVf, t)
    Wd = Rd.Ws[0].copy(); Wd[:, :p] += Rd.bs[0][:, None]
    assert np.abs(Wd - Rf.Ws[0]).max() <= 1e-9 * np.abs(Rf.Ws[0]).max() and np.abs(Rd.V - Rf.V).max() <= 1e-9 * np.abs(Rf.V).max()


def test_resume_after_knee():
    for mode in ('frozen', 'decay'):
        c = dict(KNEE, bias_after_knee=mode); d = tempfile.mkdtemp()
        a = B.run(c, os.path.join(d, 'a.json.gz'), ckpt_every=0)
        k = a['bias_knee']; cut = k + 20
        assert cut % c['obs_every'] == 0 and cut < c['steps']
        out = os.path.join(d, 'b.json.gz'); R = B.Run(c); loss = np.empty(c['steps'] + 1); obs = []
        for t in range(cut):
            L, gWs, gV = R.loss_and_grads(); loss[t] = L
            if t % c['obs_every'] == 0:
                o = R.observe(heavy=(t % c['heavy_every'] == 0)); o['step'] = t; o['loss'] = L; obs.append(o)
                if R.bias_knee is None and B.at_knee(o, c['lam_v']): R.bias_knee = t
            R.step(gWs, gV, t)
        assert R.bias_knee == k
        B._save_ckpt(out + '.ckpt.npz', R, cut, loss, obs)
        b = B.run(c, out, ckpt_every=0)
        assert b['resumed_from'] == cut and b['bias_knee'] == k, mode
        assert _same(a['loss'], b['loss']) and _same(a['obs'], b['obs']), mode


def test_bad_combinations():
    bad = [dict(KNEE, bias_update='separate', bias_after_knee='frozen'), dict(KNEE, bias=False, bias_update='separate'),
           dict(KNEE, bias_frozen=True, bias_after_knee='decay'), dict(KNEE, hidden_mode='ngd', bias_update='separate'),
           dict(KNEE, bias_update='sideways'), dict(KNEE, bias_after_knee='later')]
    for c in bad:
        try: B.Run(c)
        except ValueError: continue
        raise AssertionError(f'accepted {c}')


TESTS = [test_default_equals_campaign_I, test_separate_step, test_separate_changes_the_run, test_after_knee,
         test_decay_equals_folded_bias_free, test_resume_after_knee, test_bad_combinations]

if __name__ == '__main__':
    for f in TESTS: f(); print('ok', f.__name__, flush=True)
    print('ALL TESTS PASS')
