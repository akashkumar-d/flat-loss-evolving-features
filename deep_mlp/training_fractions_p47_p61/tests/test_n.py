import json, math, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import bias_runner as B

UNIT = 0.03 * 0.015 / 0.07
BASE = dict(frac=0.6, width=12, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=10, heavy_every=50, p=11,
            lam_v=0.2, lam_w=0.015, deep_init_scale=8.0, lam_deep=0.07, depth=4, steps=120, init_seed=7001, split_seed=8001,
            eta_layers=[0.03, UNIT, UNIT, 0.13 * UNIT], lam_layers=[0.015, 0.07, 0.07, 0.07])


def test_gradients():
    for D in (3, 4):
        cfg = dict(BASE, depth=D, eta_layers=BASE['eta_layers'][-D:], lam_layers=BASE['lam_layers'][:D])
        R = B.Run(cfg); rng = np.random.default_rng(1); R.V = rng.normal(size=R.V.shape) * 0.05
        L, gWs, gV = R.loss_and_grads(); eps = 1e-6
        for l in range(D):
            for (i, k) in ((1, 2), (3, 0)):
                old = R.Ws[l][i, k]
                R.Ws[l][i, k] = old + eps; Lp = R.loss_and_grads()[0]; R.Ws[l][i, k] = old - eps; Lm = R.loss_and_grads()[0]
                R.Ws[l][i, k] = old
                fd = (Lp - Lm) / (2 * eps); assert abs(fd - gWs[l][i, k]) < 1e-6 * max(1.0, abs(fd)), (D, l, fd, gWs[l][i, k])
        i, k = 2, 3; old = R.V[i, k]
        R.V[i, k] = old + eps; Lp = R.loss_and_grads()[0]; R.V[i, k] = old - eps; Lm = R.loss_and_grads()[0]; R.V[i, k] = old
        fd = (Lp - Lm) / (2 * eps); assert abs(fd - gV[i, k]) < 1e-6 * max(1.0, abs(fd)), ('V', fd, gV[i, k])


def test_polar_and_head_step():
    R = B.Run(BASE); W0 = [W.copy() for W in R.Ws]; V0 = R.V.copy()
    L, gWs, gV = R.loss_and_grads(); R.step(gWs, gV)
    for l in range(4):
        U, s, Zt = np.linalg.svd(-gWs[l], full_matrices=False); r = int((s > 1e-7 * s[0]).sum())
        exp = W0[l] + BASE['eta_layers'][l] * ((U[:, :r] @ Zt[:r]) / math.sqrt(r) - BASE['lam_layers'][l] * W0[l])
        assert np.abs(R.Ws[l] - exp).max() < 1e-12, l
    expV = V0 + BASE['eta_v'] * (-gV / np.linalg.norm(gV) - BASE['lam_v'] * V0)
    assert np.abs(R.V - expV).max() < 1e-12


def test_resume():
    c = dict(BASE, steps=60); d = tempfile.mkdtemp()
    a = B.run(c, os.path.join(d, 'a.json.gz'), ckpt_every=0)
    out = os.path.join(d, 'b.json.gz'); R = B.Run(c); loss = np.empty(61); obs = []
    for t in range(30):
        L, gWs, gV = R.loss_and_grads(); loss[t] = L
        if t % c['obs_every'] == 0:
            o = R.observe(heavy=(t % c['heavy_every'] == 0)); o['step'] = t; o['loss'] = L; obs.append(o)
        R.step(gWs, gV, t)
    B._save_ckpt(out + '.ckpt.npz', R, 30, loss, obs)
    b = B.run(c, out, ckpt_every=0)
    assert a['loss'] == b['loss'] and json.dumps(a['obs']) == json.dumps(b['obs'])


def k_base_cfg(p, D):
    r = 0.32 if (p, D) == (23, 4) else 0.45
    return dict(frac=0.8, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, lam_v=0.2, lam_w=0.015,
                observers_v1=True, stop_generalized=dict(acc=0.99, factor=1.5, extra=10000), p=p, depth=D, width=256,
                deep_init_scale=8.0, lam_deep=0.07, steps=120000 if p == 23 else 100000, heavy_every=1000 if p == 101 else 500,
                eta_layers=[0.03] + [UNIT] * (D - 2) + [r * UNIT], lam_layers=[0.015] + [0.07] * (D - 1))


def test_manifest_is_campaign_m_with_fresh_seeds():
    sys.path.insert(0, ROOT); import make_manifest as MN
    man = json.load(open(os.path.join(ROOT, 'manifest.json')))
    assert man['runs'] == MN.runs() and man['campaign'] == 'N' and man['workers'] == [f'N{i}' for i in range(1, 17)]
    horizon = {0.8: 100000, 0.7: 150000, 0.6: 200000, 0.5: 250000}
    r_sel = {3: {0.8: 0.45, 0.7: 0.30, 0.6: 0.18, 0.5: 0.08}, 4: {0.8: 0.45, 0.7: 0.28, 0.6: 0.13, 0.5: 0.06}}
    mfn = os.path.join(os.path.dirname(ROOT), 'training_fractions_rate_tuning', 'manifest.json')
    M = {}
    if os.path.exists(mfn):
        for x in json.load(open(mfn))['runs']:
            d = x['dial']
            if d['lam_v_rule'] == 'lin': M[(x['depth'], x['p'], d['frac'], d['r'])] = x['cfg']
        assert len(M) == 36
    seen = set(); cells = {}
    for r in man['runs']:
        c = r['cfg']; k = k_base_cfg(r['p'], r['depth']); dl = r['dial']
        diff = {kk for kk in set(c) | set(k) if c.get(kk) != k.get(kk)}
        assert diff <= {'frac', 'steps', 'lam_v', 'eta_layers', 'init_seed', 'split_seed'}, (r['id'], diff)
        assert r['p'] in (47, 61) and r['depth'] in (3, 4) and r['threads'] == 1
        assert c['frac'] == dl['frac'] and c['steps'] == horizon[c['frac']] and dl['r'] == r_sel[r['depth']][c['frac']]
        assert c['eta_layers'][:-1] == k['eta_layers'][:-1] and abs(c['eta_layers'][-1] - dl['r'] * UNIT) < 1e-15
        assert dl['lam_v_rule'] == 'lin' and c['lam_v'] == dl['lam_v'] == round(0.2 * 31 / r['p'], 4)
        assert c['lam_w'] == 0.015 and c['lam_layers'] == k['lam_layers'] and 'bias' not in c
        assert (c['init_seed'], c['split_seed']) == (9000 + r['seed'], 9500 + r['seed']) and r['seed'] in range(1, 11)
        if M:
            m = M[(r['depth'], r['p'], c['frac'], dl['r'])]
            assert {kk for kk in set(c) | set(m) if c.get(kk) != m.get(kk)} == {'init_seed', 'split_seed'}, r['id']
        assert r['id'] not in seen; seen.add(r['id'])
        cells.setdefault((r['depth'], r['p'], c['frac']), set()).add(r['seed'])
    assert len(seen) == 160 and len(cells) == 16 and all(v == set(range(1, 11)) for v in cells.values())
    per = {}
    for r in man['runs']: per[r['worker']] = per.get(r['worker'], 0) + 1
    assert set(per) == set(man['workers'])


def test_splits_are_nested():
    for p, seed in ((47, 8001), (61, 8002), (47, 8003)):
        sets = []
        for f in (0.8, 0.7, 0.6, 0.5):
            R = B.Run(dict(BASE, p=p, frac=f, split_seed=seed, depth=3, eta_layers=BASE['eta_layers'][:3], lam_layers=BASE['lam_layers'][:3]))
            S = set(map(int, R.tr)); assert len(S) == int(f * p * p), (p, f); sets.append(S)
        assert all(sets[i + 1] <= sets[i] for i in range(3)), p


TESTS = [test_gradients, test_polar_and_head_step, test_resume, test_manifest_is_campaign_m_with_fresh_seeds, test_splits_are_nested]

if __name__ == '__main__':
    for f in TESTS: f(); print('ok', f.__name__, flush=True)
    print('ALL TESTS PASS')
