import json, math, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import bias_runner as B

BASE = dict(frac=0.5, width=12, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=10, heavy_every=50, p=11,
            lam_v=0.1108, lam_w=0.025, deep_init_scale=8.0, lam_deep=0.07, depth=4, steps=120, init_seed=7001, split_seed=8001,
            eta_layers=[0.03, 0.006428571428571428, 0.006428571428571428, 0.0028928571428571423], lam_layers=[0.025, 0.07, 0.07, 0.07])


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
    unit = 0.03 * 0.015 / 0.07; r = 0.32 if (p, D) == (23, 4) else 0.45
    return dict(frac=0.8, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, lam_v=0.2, lam_w=0.015,
                observers_v1=True, stop_generalized=dict(acc=0.99, factor=1.5, extra=10000), p=p, depth=D, width=256,
                deep_init_scale=8.0, lam_deep=0.07, steps=120000 if p == 23 else 100000, heavy_every=1000 if p == 101 else 500,
                eta_layers=[0.03] + [unit] * (D - 2) + [r * unit], lam_layers=[0.015] + [0.07] * (D - 1))


def test_manifest_changes_only_the_planned_keys():
    sys.path.insert(0, ROOT); import make_manifest as MM
    man = json.load(open(os.path.join(ROOT, 'manifest.json')))
    assert man['runs'] == MM.runs()
    seen = set(); per_cell = {}
    for r in man['runs']:
        c = r['cfg']; k = k_base_cfg(r['p'], r['depth'])
        diff = {kk for kk in set(c) | set(k) if c.get(kk) != k.get(kk)}
        assert diff <= {'frac', 'steps', 'lam_v', 'lam_w', 'lam_layers', 'init_seed', 'split_seed'}, (r['id'], diff)
        assert c['frac'] == 0.5 and c['steps'] == MM.HORIZON[r['p']] and c['steps'] >= 2 * k['steps']
        assert c['lam_layers'] == [c['lam_w']] + k['lam_layers'][1:]
        assert (c['lam_v'], c['lam_w']) in [(lv, lw) for _, lv, _, lw in MM.settings(r['p'])]
        assert (c['init_seed'], c['split_seed']) == (7000 + r['seed'], 8000 + r['seed']) and r['seed'] in (1, 2, 3)
        assert 'bias' not in c and r['id'] not in seen; seen.add(r['id'])
        per_cell.setdefault((r['p'], r['depth']), set()).add((c['lam_v'], c['lam_w']))
    assert len(seen) == 120 and all(len(v) == 5 for v in per_cell.values()) and len(per_cell) == 8
    assert all((0.2, 0.015) in v for v in per_cell.values())


def test_half_split_is_nested_in_campaign_k_split():
    for p, seed in ((23, 8001), (47, 8002), (101, 8003)):
        a = B.Run(dict(BASE, p=p, frac=0.8, split_seed=seed, depth=3, eta_layers=BASE['eta_layers'][:3], lam_layers=BASE['lam_layers'][:3]))
        b = B.Run(dict(BASE, p=p, frac=0.5, split_seed=seed, depth=3, eta_layers=BASE['eta_layers'][:3], lam_layers=BASE['lam_layers'][:3]))
        A, Bs = set(map(int, a.tr)), set(map(int, b.tr))
        assert len(Bs) == int(0.5 * p * p) and len(A) == int(0.8 * p * p) and Bs <= A, p


TESTS = [test_gradients, test_polar_and_head_step, test_resume, test_manifest_changes_only_the_planned_keys,
         test_half_split_is_nested_in_campaign_k_split]

if __name__ == '__main__':
    for f in TESTS: f(); print('ok', f.__name__, flush=True)
    print('ALL TESTS PASS')
