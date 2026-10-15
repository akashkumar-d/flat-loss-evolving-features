import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); SHARED = os.path.dirname(ROOT)


def test_manifest_is_campaign_h_eta_v_3e4_with_seeds_7_to_10():
    man = json.load(open(os.path.join(ROOT, 'manifest.json')))
    assert man['campaign'] == 'QH' and man['workers'] == ['QH1'] and [r['seed'] for r in man['runs']] == [7, 8, 9, 10]
    fn = os.path.join(SHARED, 'campaign_H', 'manifest.json'); H = {}
    if os.path.exists(fn): H = {r['id']: r['cfg'] for r in json.load(open(fn))['runs']}
    for r in man['runs']:
        c, s = r['cfg'], r['seed']
        assert r['id'] == f'QH_T41_ev3e-4_s{s}' and c['init_seed'] == 9100 + s and c['eta_v'] == 0.0003 and r['threads'] == 1
        assert c['p'] == 101 and c['width'] == 512 and c['split'] == 'qr' and c['lam_v'] == 0.0 and c['steps'] == 40000
        if H:
            a, b = H['T41_ev3e-4_s1'], H[f'T41_ev1e-3_s{s}']
            assert {k for k in set(c) | set(a) if c.get(k) != a.get(k)} == {'init_seed'}, r['id']
            assert {k for k in set(c) | set(b) if c.get(k) != b.get(k)} == {'eta_v'}, r['id']
            assert f'T41_ev3e-4_s{s}' not in H


def _suite(fn):
    sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(fn))[0], fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return [v for k, v in vars(m).items() if k.startswith('test_') and callable(v)]


TESTS = _suite(os.path.join(HERE, 'test_t41.py')) + [test_manifest_is_campaign_h_eta_v_3e4_with_seeds_7_to_10]

if __name__ == '__main__':
    import time
    for f in TESTS:
        t = time.time(); f(); print(f'ok {f.__module__}.{f.__name__} ({time.time() - t:.1f}s)', flush=True)
    print(f'ALL {len(TESTS)} TESTS PASS')
