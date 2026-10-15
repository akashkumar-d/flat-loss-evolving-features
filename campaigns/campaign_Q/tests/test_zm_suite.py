import gzip, json, os, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import tf_runner as R
import importlib.util

ZP = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'campaign_ZP')
TINY = dict(p=7, frac=0.5, n_layers=1, d_model=8, n_heads=2, d_mlp=16, dtype='float64', init_seed=1, split_seed=2, steps=40,
            obs_every=5, heavy_every=10, eta_w=0.05, lam_w=0.1, eta_v=0.05, lam_v=0.2, final_norm='rms', gain_init=2.0, eta_g=1e-2)


def _r16():
    fn = os.path.join(ZP, 'tf_runner.py')
    if not os.path.exists(fn): return None
    spec = importlib.util.spec_from_file_location('tf_runner_16', fn); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.VERSION.startswith('campaign-Z tf_runner 1.6')
    return m


def test_random_split_is_16s():
    R16 = _r16()
    if R16 is None: return
    for p, frac, seed in ((7, 0.5, 2), (61, 0.33, 8111), (61, 0.35, 8113), (89, 0.25, 8112), (149, 0.16, 8115)):
        cfg = dict(TINY, p=p, frac=frac, split_seed=seed)
        a, b = R16.TF(cfg), R.TF(cfg)
        for k in ('tr', 'te', 'te_nm'): assert np.array_equal(getattr(a, k), getattr(b, k)), (p, frac, k)
        assert R.full_cfg(cfg)['split_mode'] == 'random'


def test_mirror_free_split():
    for p, frac in ((7, 0.5), (11, 0.3), (61, 0.33), (61, 0.55), (89, 0.25)):
        for seed in (1, 8111):
            m = R.TF(dict(TINY, p=p, frac=frac, split_seed=seed, split_mode='mirror_free'))
            n = int(frac * p * p); assert len(m.tr) == n and len(set(m.tr.tolist())) == n
            assert np.array_equal(np.sort(np.concatenate([m.tr, m.te])), np.arange(p * p))
            intr = np.zeros(p * p, bool); intr[m.tr] = True
            mirror = (p * (m.tr % p) + m.tr // p)
            assert intr[mirror].all(), 'every training pair has its mirror in the training set'
            assert not intr[p * (m.te % p) + m.te // p].any(), 'no held-out pair has its mirror in the training set'
            assert np.array_equal(m.te_nm, m.te)
        a = R.mirror_free_train(p, frac, 1); b = R.mirror_free_train(p, frac, 1); c = R.mirror_free_train(p, frac, 2)
        assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_unknown_split_mode_fails():
    try:
        R.TF(dict(TINY, split_mode='mirror'))
    except ValueError:
        return
    raise AssertionError('an unknown split_mode must raise')


def test_v17_default_matches_v16_update_for_update():
    R16 = _r16()
    if R16 is None: return
    for cs in (dict(optimizer='polar'), dict(optimizer='polar', hidden_mode='ngd'), dict(optimizer='adamw', lr=1e-3, wd=1.0),
               dict(optimizer='polar', observers_fourier=True)):
        cfg = dict(TINY, **cs)
        with tempfile.TemporaryDirectory() as d:
            a = R16.run(cfg, os.path.join(d, 'a.json.gz'), ckpt_every=0); b = R.run(cfg, os.path.join(d, 'b.json.gz'), ckpt_every=0)
        assert a['loss'] == b['loss'], cs
        assert [{k: v for k, v in o.items()} for o in a['obs']] == b['obs'], cs
        assert {k: v for k, v in b['cfg_full'].items() if k not in ('split_mode', 'late_at', 'late')} == a['cfg_full']


def test_mirror_free_run_records_equal_held_out_definitions():
    cfg = dict(TINY, split_mode='mirror_free', observers_fourier=True, probe_every=10)
    with tempfile.TemporaryDirectory() as d:
        r = R.run(cfg, os.path.join(d, 'm.json.gz'), ckpt_every=0)
        with gzip.open(os.path.join(d, 'm.json.gz'), 'rt') as f: assert json.load(f)['version'].startswith('campaign-Q tf_runner 1.8')
    for o in r['obs']:
        assert o['test_acc'] == o['test_acc_nm']
        for k in o:
            if k.endswith('_test_acc') and k + '_nm' in o: assert o[k] == o[k + '_nm'], k


def test_mirror_free_resume_is_exact():
    c = dict(TINY, split_mode='mirror_free')
    with tempfile.TemporaryDirectory() as d:
        a_ = R.run(c, os.path.join(d, 'a.json.gz'), ckpt_every=0); fn = os.path.join(d, 'b.json.gz')
        lr_ = R.Learner(c); loss = np.empty(c['steps'] + 1); obs = []
        for t in range(20):
            L = lr_.m.loss_and_grads(); loss[t] = L
            if t % c['obs_every'] == 0:
                o = R.observe(lr_.m, t % c['heavy_every'] == 0); o['step'] = t; o['loss'] = L; obs.append(o)
            lr_.step()
        R._save_ckpt(fn + '.ckpt.pt', lr_, 20, loss, obs, None)
        b_ = R.run(c, fn, ckpt_every=20)
    assert b_['resumed_from'] == 20 and np.array_equal(np.array(a_['loss']), np.array(b_['loss']))
    assert a_['obs'] == b_['obs']


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f): f(); print('ok', k)
