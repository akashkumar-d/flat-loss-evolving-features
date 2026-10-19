import importlib.util, json, os, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import bias_runner as B

SHARED = os.path.dirname(ROOT)
SMALL = dict(deep_init_scale=8.0, depth=2, eta_deep=0.003214285714285714, eta_v=0.03, eta_w=0.03, frac=0.8, heavy_every=50,
             init_scale=24.0, lam_deep=0.07, lam_v=0.2, lam_w=0.015, loss='ce', obs_every=10, p=11, steps=150, width=16,
             init_seed=1001, split_seed=2001)


def _run(cfg):
    with tempfile.TemporaryDirectory() as d:
        return B.run(cfg, os.path.join(d, 'r.json.gz'), ckpt_every=0)


def _same(a, b):
    if os.environ.get('EXACT_TESTS') == '1': return json.dumps(a) == json.dumps(b)
    if isinstance(a, dict) and isinstance(b, dict): return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)): return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) and isinstance(b, float): return bool(np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))
    return type(a) is type(b) and a == b


def test_canonical_observers_never_change_a_depth2_run():
    for extra in ({}, dict(bias=True)):
        a = _run(dict(SMALL, **extra)); b = _run(dict(SMALL, observers_v1=True, **extra))
        assert _same(a['loss'], b['loss']), extra
        for oa, ob in zip(a['obs'], b['obs']):
            assert _same({k: ob[k] for k in oa}, oa), extra
        assert any(k.startswith('v1_') for k in b['obs'][-1]) and not any(k.startswith('v1_') for k in a['obs'][-1])
        assert 'v1_ridge_acc_te' in b['obs'][-1]


def test_manifest_is_depth2_on_the_evaluation_seeds():
    man = json.load(open(os.path.join(ROOT, 'manifest.json')))
    assert man['campaign'] == 'QD' and man['workers'] == ['QD1', 'QD2', 'QD3', 'QD4'] and len(man['runs']) == 80
    D2 = dict(SMALL, heavy_every=500, obs_every=20, p=31, steps=40000, width=256)
    fn = os.path.join(SHARED, 'hidden_bias_depths1_2_4', 'manifest.json'); I = None
    if os.path.exists(fn):
        Im = json.load(open(fn))['runs']
        I = [r['cfg'] for r in Im if r['family'] == 'D2'][0]
        assert sorted((r['init_seed'], r['split_seed']) for r in Im if r['family'] == 'D4') == [(1000 + i, 2000 + i) for i in range(1, 41)]
    seen = {}
    for r in man['runs']:
        i, c = r['seed'], r['cfg']; fam = r['family']
        exp = dict(D2, init_seed=1000 + i, split_seed=2000 + i, observers_v1=True, **(dict(bias=True) if fam == 'B_D2' else {}))
        assert c == exp, (r['id'], {k for k in set(c) | set(exp) if c.get(k) != exp.get(k)})
        assert r['id'] == f'QD_{fam}_s{i:03d}' and r['threads'] == 1 and r['kind'] == 'one' and r['worker'] in man['workers']
        if I is not None:
            diff = {k for k in set(c) | set(I) if c.get(k) != I.get(k)}
            assert diff == {'init_seed', 'split_seed', 'observers_v1'} | ({'bias'} if fam == 'D2' else set()), (r['id'], diff)
        seen.setdefault(fam, set()).add(i)
    assert seen == {'D2': set(range(1, 41)), 'B_D2': set(range(1, 41))}
    per = {}
    for r in man['runs']: per.setdefault(r['worker'], []).append(r)
    assert sorted(len(v) for v in per.values()) == [20, 20, 20, 20]


def _suite(fn):
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(fn))[0], fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return [v for k, v in vars(m).items() if k.startswith('test_') and callable(v)]


TESTS = _suite(os.path.join(HERE, 'test_bias.py')) + [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]

if __name__ == '__main__':
    import time
    for f in TESTS:
        t = time.time(); f(); print(f'ok {f.__module__}.{f.__name__} ({time.time() - t:.1f}s)', flush=True)
    print(f'ALL {len(TESTS)} TESTS PASS')
