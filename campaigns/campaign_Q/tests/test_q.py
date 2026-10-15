import gzip, importlib.util, json, os, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import tf_runner as R

SHARED = os.path.dirname(os.path.dirname(HERE))
TINY = dict(p=7, frac=0.5, n_layers=1, d_model=8, n_heads=2, d_mlp=16, dtype='float64', init_seed=1, split_seed=2, steps=40,
            obs_every=5, heavy_every=10, eta_w=0.05, lam_w=0.1, eta_v=0.05, lam_v=0.2, final_norm='rms', gain_init=2.0, eta_g=1e-2)


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, fn); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _r17():
    m = _load('ref_tf_runner_17', os.path.join(HERE, 'ref_tf_runner_17.py'))
    assert m.VERSION.startswith('campaign-ZM tf_runner 1.7')
    return m


def _run(mod, cfg, d, name):
    return mod.run(cfg, os.path.join(d, name), ckpt_every=0)


def test_v18_without_late_matches_v17_update_for_update():
    R17 = _r17()
    cases = (dict(), dict(n_layers=2, observers_fourier=True, probe_every=10), dict(hidden_mode='ngd'),
             dict(optimizer='adamw', lr=1e-3, wd=1.0), dict(split_mode='mirror_free'), dict(gain_init=None, eta_g=0.0),
             dict(eta_groups=dict(mlp=0.02), lam_groups=dict(attn=0.3)))
    for cs in cases:
        cfg = dict(TINY, **cs)
        with tempfile.TemporaryDirectory() as d:
            a = _run(R17, cfg, d, 'a.json.gz'); b = _run(R, cfg, d, 'b.json.gz')
        assert a['loss'] == b['loss'], cs
        assert a['obs'] == b['obs'], cs
        assert {k: v for k, v in b['cfg_full'].items() if k not in ('late_at', 'late')} == a['cfg_full'], cs
        assert b['cfg_full']['late_at'] is None and b['cfg_full']['late'] == {} and b['version'].startswith('campaign-Q tf_runner 1.8')


def test_late_rates_switch_after_the_update_at_late_at():
    t0 = 17
    for late in (dict(eta_w=0.01), dict(lam_w=0.3), dict(eta_v=0.01), dict(lam_v=0.5), dict(eta_g=0.0), dict(lam_g=2.0)):
        with tempfile.TemporaryDirectory() as d:
            a = _run(R, TINY, d, 'a.json.gz'); b = _run(R, dict(TINY, late_at=t0, late=late), d, 'b.json.gz')
        assert a['loss'][:t0 + 1] == b['loss'][:t0 + 1], late
        assert a['loss'][t0 + 1] != b['loss'][t0 + 1], late
        assert [o for o in a['obs'] if o['step'] <= t0] == [o for o in b['obs'] if o['step'] <= t0], late
        assert b['cfg_full']['late_at'] == t0 and b['cfg_full']['late'] == late


def test_late_rates_equal_a_run_configured_with_them():
    late = dict(eta_w=0.02, lam_w=0.3, eta_v=0.04, lam_v=0.5, eta_g=3e-3, lam_g=1.5)
    for cs in (dict(), dict(n_layers=2), dict(hidden_mode='ngd')):
        a = R.Learner(dict(TINY, late_at=5, late=late, **cs)); b = R.Learner(dict(TINY, **late, **cs))
        with torch.no_grad():
            for k in a.m.P: b.m.P[k].copy_(a.m.P[k])
        for t in (5, 6, 7):
            a.phase(t); b.phase(t); assert a.m.loss_and_grads() == b.m.loss_and_grads(); a.step(); b.step()
        assert all(torch.equal(a.m.P[k], b.m.P[k]) for k in a.m.P), cs
        a.phase(4); assert a.rates == a.rates_base and a.rates_base == {k: R.full_cfg(dict(TINY, **cs))[k] for k in R.LATE_KEYS}


def test_late_eta_g_zero_freezes_the_gain():
    t0 = 10
    with tempfile.TemporaryDirectory() as d:
        r = _run(R, dict(TINY, late_at=t0, late=dict(eta_g=0.0)), d, 'r.json.gz')
    g = {o['step']: o['gain'] for o in r['obs']}
    later = [v for s, v in g.items() if s >= t0]
    assert len(later) >= 3 and len(set(later)) == 1, g
    assert g[t0 - 5] != g[t0], g


def test_late_rates_resume_exactly_across_the_switch():
    t0 = 20
    c = dict(TINY, late_at=t0, late=dict(eta_w=0.01, eta_g=0.0), observers_fourier=True)
    with tempfile.TemporaryDirectory() as d:
        a_ = _run(R, c, d, 'a.json.gz')
        for t_ck in (10, 25):
            fn = os.path.join(d, f'b{t_ck}.json.gz')
            lr_ = R.Learner(c); loss = np.empty(c['steps'] + 1); obs = []
            for t in range(t_ck):
                L = lr_.m.loss_and_grads(); loss[t] = L
                if t % c['obs_every'] == 0:
                    o = R.observe(lr_.m, t % c['heavy_every'] == 0); o['step'] = t; o['loss'] = L; obs.append(o)
                lr_.phase(t); lr_.step()
            R._save_ckpt(fn + '.ckpt.pt', lr_, t_ck, loss, obs, None)
            b_ = R.run(c, fn, ckpt_every=t_ck)
            assert b_['resumed_from'] == t_ck and a_['loss'] == b_['loss'] and a_['obs'] == b_['obs'], t_ck


def test_late_rates_are_validated():
    bad = (dict(late=dict(eta_w=0.01)), dict(late_at=5), dict(late_at=-1, late=dict(eta_w=0.01)),
           dict(late_at=5, late=dict(rel_cut=0.1)), dict(late_at=5, late=dict(eta_w=0.01), eta_groups=dict(mlp=0.02)),
           dict(late_at=5, late=dict(lam_w=0.01), lam_groups=dict(attn=0.3)),
           dict(late_at=5, late=dict(eta_w=0.01), optimizer='adamw', lr=1e-3, wd=1.0))
    for cs in bad:
        try:
            R.Learner(dict(TINY, **cs))
        except ValueError:
            continue
        raise AssertionError(f'{cs} must raise')
    R.Learner(dict(TINY, late_at=5, late=dict(eta_g=0.0), optimizer='adamw', lr=1e-3, wd=1.0))
    R.Learner(dict(TINY, late_at=5, late=dict(eta_v=0.01), eta_groups=dict(mlp=0.02)))


def test_run_record_carries_version_and_late_config():
    with tempfile.TemporaryDirectory() as d:
        R.run(dict(TINY, late_at=8, late=dict(lam_w=0.2)), os.path.join(d, 'r.json.gz'), ckpt_every=0)
        with gzip.open(os.path.join(d, 'r.json.gz'), 'rt') as f: rec = json.load(f)
    assert rec['version'] == 'campaign-Q tf_runner 1.8 (2026-10-13)'
    assert rec['cfg']['late_at'] == 8 and rec['cfg']['late'] == dict(lam_w=0.2) and rec['cfg_full']['late'] == dict(lam_w=0.2)


Y_G2 = dict(d_mlp=512, d_model=128, dtype='float64', eta_v=0.03, eta_w=0.1, final_norm='rms', frac=0.2, heavy_every=100,
            hidden_mode='polar', lam_w=0.01, n_heads=4, n_layers=1, obs_every=20, optimizer='polar', p=113, probe_every=200,
            refit_tau=0.001, steps=30000, stop_generalized=dict(acc=0.99, extra=5000, factor=1.5), observers_fourier=True,
            lam_v=0.275, gain_init=2.0, eta_g=2e-4, lam_g=1.0)
P_WANT = {'P_p61_f0.33_lam_v0.45': dict(p=61, frac=0.33, lam_v=0.45),
          'M_p61_f0.55_lam_v0.45': dict(p=61, frac=0.55, lam_v=0.45, split_mode='mirror_free'),
          'P_p89_f0.25_lam_v0.33': dict(p=89, frac=0.25, lam_v=0.33),
          'P_p149_f0.16_lam_v0.23': dict(p=149, frac=0.16, lam_v=0.23)}
B0 = dict(n_layers=2, eta_g=1e-4)
B_WANT = {'B_f0.2_lam_v0.275_tau10k': ({}, range(16, 21)),
          'B_f0.2_lam_v0.275_tau10k_F12': (dict(late_at=12000, late=dict(eta_g=0.0)), range(11, 14)),
          'B_f0.2_lam_v0.275_tau10k_E12': (dict(late_at=12000, late=dict(eta_w=0.03)), range(11, 14)),
          'B_f0.2_lam_v0.275_tau10k_V12': (dict(late_at=12000, late=dict(eta_v=0.01)), range(11, 14)),
          'B_f0.175_lam_v0.275_tau10k': (dict(frac=0.175), range(11, 14)),
          'B_f0.225_lam_v0.275_tau10k': (dict(frac=0.225), range(11, 14))}


def _diff(a, b): return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_manifest_is_the_planned_cells_and_seeds():
    man = json.load(open(os.path.join(os.path.dirname(HERE), 'manifest.json')))
    assert man['campaign'] == 'Q' and man['workers'] == [f'Q{i}' for i in range(1, 25)] and len(man['runs']) == 40
    zp_fn, z_fn = os.path.join(SHARED, 'campaign_ZP', 'manifest.json'), os.path.join(SHARED, 'campaign_Z', 'manifest.json')
    ZP = {r['cell']: r['cfg'] for r in json.load(open(zp_fn))['runs'] if r['seed'] == 11} if os.path.exists(zp_fn) else {}
    ZB = [r['cfg'] for r in json.load(open(z_fn))['runs'] if r['id'] == 'Z_B_f0.2_lam_v0.275_tau10k_s11'] if os.path.exists(z_fn) else []
    seen, per, own = set(), {}, {}
    for r in man['runs']:
        c, i = r['cfg'], r['seed']
        assert r['id'] == f"Q_{r['cell']}_s{i}" and r['id'] not in seen and r['kind'] == 'one'; seen.add(r['id'])
        assert c['init_seed'] == 7100 + i and c['split_seed'] == 8100 + i and r['init_seed'] == 7100 + i
        seeds = dict(init_seed=7100 + i, split_seed=8100 + i)
        if r['part'] == 'P':
            ch = P_WANT[r['cell']]; exp = dict(Y_G2, **seeds, **ch)
            assert _diff(c, exp) == set() and r['threads'] == 1 and r['worker'] in ('Q1', 'Q2', 'Q3', 'Q4'), r['id']
            assert i in range(16, 21)
            if ZP:
                ref = dict(ZP['P_p61_f0.33_lam_v0.45'], frac=0.55, split_mode='mirror_free') if r['cell'].startswith('M_') else ZP[r['cell']]
                assert _diff(c, ref) == {'init_seed', 'split_seed'}, r['id']
        else:
            ch, want_seeds = B_WANT[r['cell']]; exp = dict(Y_G2, **B0, **seeds, **ch)
            assert _diff(c, exp) == set() and r['threads'] == 4 and i in want_seeds, r['id']
            if ZB:
                assert _diff(c, dict(ZB[0], **ch)) <= {'init_seed', 'split_seed'}, r['id']
                assert (_diff(c, ZB[0]) - {'init_seed', 'split_seed'}) == set(ch), r['id']
        per.setdefault(r['cell'], set()).add(i); own.setdefault(r['worker'], []).append(r)
    assert per == {**{k: set(range(16, 21)) for k in P_WANT}, **{k: set(v[1]) for k, v in B_WANT.items()}}, per
    assert set(own) == set(man['workers'])
    for w, v in own.items():
        assert (len(v) <= 6 and all(r['part'] == 'P' for r in v)) if w in ('Q1', 'Q2', 'Q3', 'Q4') else (len(v) == 1 and v[0]['part'] == 'B'), w


def _suite(fn):
    m = _load(os.path.splitext(os.path.basename(fn))[0], fn)
    return [v for k, v in vars(m).items() if k.startswith('test_') and callable(v)]


TESTS = (_suite(os.path.join(HERE, 'test_tf_16suite.py')) + _suite(os.path.join(HERE, 'test_zm_suite.py'))
         + [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)])

if __name__ == '__main__':
    import time
    for f in TESTS:
        t = time.time(); f(); print(f'ok {f.__module__}.{f.__name__} ({time.time() - t:.1f}s)', flush=True)
    print(f'ALL {len(TESTS)} TESTS PASS')
