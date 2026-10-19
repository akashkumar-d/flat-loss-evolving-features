import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import refits as RF


def _toy(p=31, M=40, n_frac=0.5, seed=0, ks=(3, 7)):
    rng = np.random.default_rng(seed); a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); y = (a + b) % p
    cols = []
    for k in ks:
        w = 2 * np.pi * k / p
        cols += [np.cos(w * y), np.sin(w * y), np.cos(w * a), np.sin(w * b)]
    H = np.stack(cols, 1) @ rng.normal(size=(len(cols), M)) + 0.3 * rng.normal(size=(p * p, M))
    tr = np.sort(rng.permutation(p * p)[:int(n_frac * p * p)])
    return H, y, tr, p


def test_identity():
    H, y, tr, p = _toy(); tau = 1e-3 * len(tr)
    A = RF.plain_ridge_logits(H[tr], y[tr], H, p, tau)
    fr = RF.FourierRidge(H[tr], y[tr], p, tau); Zc, Zs = fr.predict(H); B = fr.logits(Zc, Zs)
    assert np.abs(A - B).max() < 1e-10 * max(1.0, np.abs(A).max()), np.abs(A - B).max()


def test_no_leak():
    H, y, tr, p = _toy(); y2 = y.copy(); te = np.setdiff1d(np.arange(len(y)), tr)
    y2[te] = np.random.default_rng(5).integers(0, p, len(te))
    P1, I1 = RF.all_readouts(H, y, tr, p); P2, I2 = RF.all_readouts(H, y2, tr, p)
    for k in P1: assert np.array_equal(P1[k], P2[k]), k
    assert I1['ks_sig'] == I2['ks_sig'] and I1['k_cv'] == I2['k_cv']


def test_finds_planted_frequencies():
    H, y, tr, p = _toy(ks=(3, 7)); P, I = RF.all_readouts(H, y, tr, p)
    assert sorted(I['ks_sig']) == [3, 7], I['ks_sig']
    te = np.setdiff1d(np.arange(len(y)), tr)
    assert (P['filt_sig'][te] == y[te]).mean() >= (P['ridge'][te] == y[te]).mean()


def test_spectrum_sums_to_AH():
    import tf_runner as T
    H, y, tr, p = _toy(); assert abs(RF.grid_spectrum(H, y, p).sum() - T.ah_full(H, y, p)) < 1e-10


def test_matches_tf_runner(run_dir):
    import torch, tf_runner as T, score as S
    snaps = sorted(f for f in os.listdir(os.path.join(run_dir, 'snaps')) if f.endswith('.npz') and '.tmp' not in f)
    cfg = json.load(open(os.path.join(HERE, 'cfgs', 'W_R5_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.275_s1.json')))
    z = np.load(os.path.join(run_dir, 'snaps', snaps[-1]))
    m, F, tabs, _ = S.tf_tables(cfg, z)
    o = T.observe(m, True)
    for name, tag in (('resid', ''), ('mlp', '_mlp')):
        pr = RF.plain_ridge_logits(tabs[name][m.tr], m.y[m.tr], tabs[name], m.p, 1e-3 * len(m.tr)).argmax(1)
        assert abs((pr[m.te] == m.y[m.te]).mean() - o['refit' + tag + '_test_acc']) < 1e-12, name
        assert abs((pr[m.te_nm] == m.y[m.te_nm]).mean() - o['refit' + tag + '_test_acc_nm']) < 1e-12, name
    assert abs((F.argmax(1)[m.te] == m.y[m.te]).mean() - o['test_acc']) < 1e-12


def test_matches_bias_runner(run_dir):
    import bias_runner as B, score as S
    snaps = sorted(f for f in os.listdir(os.path.join(run_dir, 'snaps')) if f.endswith('.npz') and '.tmp' not in f)
    cfg = json.load(open(os.path.join(HERE, 'cfgs', 'F_H_p31_lw0.0150_s10.json')))
    z = np.load(os.path.join(run_dir, 'snaps', snaps[-1]))
    R, F, tabs, _ = S.mlp_tables(cfg, z); o = R.observe(True)
    pr = RF.plain_ridge_logits(tabs['last'][R.tr], R.y[R.tr], tabs['last'], R.p, 1e-3 * len(R.tr)).argmax(1)
    assert abs((pr[R.te] == R.y[R.te]).mean() - o['v1_ridge_acc_te']) < 1e-12, ((pr[R.te] == R.y[R.te]).mean(), o['v1_ridge_acc_te'])


if __name__ == '__main__':
    for f in (test_identity, test_no_leak, test_finds_planted_frequencies, test_spectrum_sums_to_AH): f(); print('ok', f.__name__, flush=True)
    if len(sys.argv) > 1: test_matches_tf_runner(sys.argv[1]); print('ok test_matches_tf_runner', flush=True)
    if len(sys.argv) > 2: test_matches_bias_runner(sys.argv[2]); print('ok test_matches_bias_runner', flush=True)
    print('ALL TESTS PASS')
