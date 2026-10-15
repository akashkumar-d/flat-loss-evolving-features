#!/usr/bin/env python3
import gzip, json, os, sys
import numpy as np
AQ, AZM10, RZM10, AQD, F8, OUT = sys.argv[1:7]
SHARED = os.environ.get('SHARED', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'campaigns'))
sys.path.insert(0, os.path.join(SHARED, 'campaign_ZM'))
sys.path.insert(0, os.path.join(SHARED, 'campaign_Q'))
import freqs_ZM as FZ
import analyze_X as AX
import importlib.util
_spec = importlib.util.spec_from_file_location('tf_runner_q', os.path.join(SHARED, 'campaign_Q', 'tf_runner.py'))
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)

Q = json.load(open(AQ))['cells']; ZM = json.load(open(AZM10))['cells']; RO = json.load(open(RZM10))
rep = {}
rng = lambda xs: [min(xs), max(xs)] if xs else None
med = lambda xs: float(np.median(xs)) if xs else None


def load(fn): return AX.load(fn)


def row(c):
    R_ = c['runs']; S = c['summary']
    nm_only = sum(bool(r['nm'].get('window', {}).get('ok_strict')) and r['all']['final'] >= 0.99 for r in R_)
    return dict(n=S['n'], strict=S['pass_strict'], loose=S['pass_loose'], nm_only=nm_only, n_train=rng([r['n_train'] for r in R_]),
                dL_median=S['ce_change']['median'], dL_range=S['ce_change']['range'], dL_n=S['ce_change']['n'],
                final=S['final_all'], why=S['why'], seeds=sorted(r['seed'] for r in R_),
                max_mis_window=rng([r['all'].get('window', {}).get('max_mis') for r in R_ if r['all'].get('window')]))


rep['table6'] = {k: row(Q[k]) for k in ('P_p61', 'P_p61_mf', 'P_p89', 'P_p113', 'P_p149')}


def unordered(c):
    p = c['p']
    if c.get('split_mode', 'random') == 'mirror_free': tr = R.mirror_free_train(p, c['frac'], c['split_seed'])
    else: tr = np.sort(np.random.default_rng(c['split_seed']).permutation(p * p)[:int(c['frac'] * p * p)])
    a, b = tr // p, tr % p
    return len({(min(x, y), max(x, y)) for x, y in zip(a.tolist(), b.tolist())})


def mirror_stats(cell):
    out = dict(runs=[])
    for r in sorted(cell['runs'], key=lambda r: r['seed']):
        d = load(r['file']); c = d.get('cfg_full') or d['cfg']; obs = d['obs']
        s, va = AX.series(obs, 'test_acc'); _, vn = AX.series(obs, 'test_acc_nm')
        m90 = None
        if r['n_mirror'] > 0:
            vm = (r['n_heldout'] * va - r['n_nm'] * vn) / r['n_mirror']; m90 = AX.interp_cross(s, vm, 0.9)
        W = r['all'].get('window', {}).get('W')
        out['runs'].append(dict(id=r['id'], seed=r['seed'], n_train=r['n_train'], n_heldout=r['n_heldout'], n_mirror=r['n_mirror'],
                                unordered=unordered(c), mirror_t90=m90, W=W, a0=r.get('a0_all'), q_all=r['all'].get('window', {}).get('q_te'),
                                passed=r['pass_strict'], why=r['why'], ce_min_t=r['ce_min']['t'], nm01=r['nm']['net']['0.1'],
                                nm05=r['nm']['net']['0.5'], nm09=r['nm']['net']['0.9'], final_all=r['all']['final']))
    R_ = out['runs']
    out.update(n=len(R_), n_mirror=rng([x['n_mirror'] for x in R_]), n_heldout=rng([x['n_heldout'] for x in R_]),
               unordered=rng([x['unordered'] for x in R_]), mirror_t90=rng([x['mirror_t90'] for x in R_ if x['mirror_t90'] is not None]),
               W0=rng([x['W'][0] for x in R_ if x['W']]), a0=rng([x['a0'] for x in R_ if x['a0'] is not None]),
               q_pass=sorted(x['q_all'] for x in R_ if x['passed']), q_fail=sorted((x['q_all'], x['why']) for x in R_ if not x['passed']),
               ce_min_t_median=med([x['ce_min_t'] for x in R_]), nm01_median=med([x['nm01'] for x in R_]),
               nm05_median=med([x['nm05'] for x in R_]), nm09_median=med([x['nm09'] for x in R_]), passes=sum(x['passed'] for x in R_))
    return out


rep['mirror'] = {k: mirror_stats(ZM[k]) for k in ('ZP_random_f0.33', 'ZM_f0.55')}


COLS = {'p61 mirror-free f0.55 (ZM)': 'ZM_f0.55', 'p61 random f0.33 (ZP)': 'ZP_random_f0.33', 'p89 random f0.25 (ZP)': 'P_p89_f0.25_lam_v0.33',
        'p113 random f0.2 (Y G2)': 'Y_G2', 'p149 random f0.16 (ZP)': 'P_p149_f0.16_lam_v0.23'}
ROWS = (('Refit, all frequencies', 'paper refit (all 30 or 56 freq.)'), ('Fourier refit, top 8', 'Fourier refit top 8'),
        ('Fourier refit, top 16', 'Fourier refit top 16'))
t7 = {}
for col, key in COLS.items():
    t7[col] = {}
    for name, rk in ROWS:
        L = RO[col][rk]
        for j, q in ((1, '0.5'), (2, '0.9')):
            xs = [x['lead'][j] for x in L if x['lead'][j] is not None]
            t7[col][f'{name} {q}'] = dict(median_k=round(med(xs) / 1000, 2) if xs else None, n_lead=sum(x > 0 for x in xs), n=len(L), n_cross=len(xs))
        t7[col][f'{name} final_median'] = med([x['final'] for x in L])
    S = ZM[key]['summary']['nm']
    for q in ('0.5', '0.9'):
        x = S[f'lead_probe_{q}']; t7[col][f'Logistic probe {q}'] = dict(median_k=round(x['median'] / 1000, 2), n_lead=x['n_lead'], n=x['n'])
        y = S[f'lead_paper_refit_{q}']; t7[col][f'(summary) refit {q}'] = dict(median_k=round(y['median'] / 1000, 2), n_lead=y['n_lead'], n=y['n'])
        y = S[f'lead_fourier_top8_{q}']; t7[col][f'(summary) top8 {q}'] = dict(median_k=round(y['median'] / 1000, 2), n_lead=y['n_lead'], n=y['n'])
    t7[col]['runs'] = ZM[key]['summary']['n']
rep['table7'] = t7

fq = {}
for col, key in COLS.items():
    rows = []
    for r in ZM[key]['runs']:
        obs = load(r['file'])['obs']; s, v = AX.series(obs, 'test_acc_nm')
        H = [o for o in obs if 'fspec_hor2' in o and 'fspec_train' in o]
        rows.append({name: FZ.at(H, t) for name, t in (('net 0.5', AX.interp_cross(s, v, 0.5)), ('net 0.9', AX.interp_cross(s, v, 0.9)))})
    fq[col] = {name: dict(K=rows[0][name]['K'], n_sig=rng([x[name]['n_sig'] for x in rows]), in_top8=rng([x[name]['in_top8'] for x in rows]),
                          in_top16=rng([x[name]['in_top16'] for x in rows]), all_in_top16=sum(x[name]['in_top16'] == x[name]['n_sig'] for x in rows))
               for name in ('net 0.5', 'net 0.9')}
rep['freqs'] = fq


def first_cross(s, v, thr):
    k = np.flatnonzero(v >= thr); return int(s[k[0]]) if len(k) else None


def blocks(cell):
    out = []
    for r in sorted(cell['runs'], key=lambda r: r['seed']):
        d = load(r['file']); obs = d['obs']; ntr = r['n_train']
        st = np.array([o['step'] for o in obs if 'train_acc' in o]); tr = np.array([o['train_acc'] for o in obs if 'train_acc' in o])
        mis = np.round((1 - tr) * ntr).astype(int)
        ff = r['first_fit']; after = st >= (ff if ff is not None else 0)
        k = np.flatnonzero(after & (mis > 0)); flip = int(st[k[0]]) if len(k) else None
        s, va = AX.series(obs, 'test_acc'); t99 = first_cross(s, va, 0.99)
        w = r['nm'].get('window', {}); wa = r['all'].get('window', {})
        L = r.get('late', {})
        out.append(dict(id=r['id'], seed=r['seed'], first_fit=ff, nm01=r['nm']['net']['0.1'], nm09=r['nm']['net']['0.9'],
                        ce_change=r.get('rise', {}).get('ce_change'), rise_max_mis=r.get('rise', {}).get('max_mis'),
                        q_nm=w.get('q_te'), four_gain_nm=w.get('fourier_top8_gain'), W_nm=w.get('W'), W_all=wa.get('W'),
                        max_mis_window_all=wa.get('max_mis'), t99_all=t99, first_flip=flip, flip_after_99=None if (flip is None or t99 is None) else flip - t99,
                        max_mis_after_fit=int(mis[after].max()) if after.any() else None, final=r['all']['final'], steps=r['steps'],
                        strict=r['pass_strict'], loose=r['pass_loose'], why=r['why'], ce_min_t=r['ce_min']['t'],
                        late_first_flip=L.get('first_flip'), late_max_mis=L.get('max_mis_late'), late_min_margin=L.get('min_margin_late'),
                        equal_to_base_until_12k=L.get('equal_to_base_until_12k'), gain_12k=L.get('gain_12k'), gain_end=L.get('gain_end'),
                        wout1_growth=L.get('normW_W_out1_growth')))
    return out


B = {k: blocks(Q[k]) for k in ('B_base', 'B_F12', 'B_E12', 'B_V12', 'B_f0.175', 'B_f0.225')}
rep['blocks'] = B
summ = {}
for k, runs in B.items():
    g = lambda f: [x[f] for x in runs if x[f] is not None]
    summ[k] = dict(n=len(runs), strict=sum(x['strict'] for x in runs), loose=sum(x['loose'] for x in runs), nm01=rng(g('nm01')), nm09=rng(g('nm09')),
                   ce_change=rng(g('ce_change')), rise_max_mis=rng(g('rise_max_mis')), q_nm=rng(g('q_nm')), four_gain_nm=rng(g('four_gain_nm')),
                   t99_all=rng(g('t99_all')), first_flip=rng(g('first_flip')), n_flip=sum(x['first_flip'] is not None for x in runs),
                   flip_after_99=rng(g('flip_after_99')), max_mis_after_fit=rng(g('max_mis_after_fit')), final=rng(g('final')),
                   below_099=sum(x['final'] < 0.99 for x in runs), why=sorted({w for x in runs for w in x['why']}),
                   ce_min_t=rng(g('ce_min_t')), equal_to_base=[x['equal_to_base_until_12k'] for x in runs])
rep['blocks_summary'] = summ

qd = json.load(open(AQD)); rep['QD'] = qd['summary']
rep['QD_misses'] = {k: (v['why_not'], round(v['share'], 3), v['final_test']) for k, v in qd['runs'].items() if not v['primary']}
f8 = json.load(open(F8)); rep['fig8'] = {}
for arm, v in f8.items():
    R_ = v['runs'].values(); g = lambda f: [x[f] for x in R_ if x.get(f) is not None]
    rep['fig8'][arm] = dict(n=v['n'], v_1k_T=rng(g('v_1k_T')), v_refit_rise=rng(g('v_refit_rise')), AH10=rng(g('AH_at_refit10')),
                            AH90=rng(g('AH_at_refit90')), test_at_refit90=rng(g('test_at_refit90')), test_T=rng(g('test_T')),
                            refit_T=rng(g('refit_T')), train_acc_at_refit10=rng([1 - m / 5050 for m in g('mis_at_refit10')]),
                            mis_T=rng(g('mis_T')), CE_over_logp_mean=rng(g('CE_over_logp_mean')))
json.dump(rep, open(OUT, 'w'), indent=1, default=float)


def show(k, v, ind=''):
    if isinstance(v, dict) and len(json.dumps(v, default=float)) > 150:
        print(f'{ind}{k}:')
        for kk, vv in v.items(): show(kk, vv, ind + '  ')
    else: print(f'{ind}{k}: {json.dumps(v, default=float)}')


for k in ('table6', 'mirror', 'table7', 'freqs', 'blocks_summary', 'QD', 'QD_misses', 'fig8'):
    v = rep[k]
    if k == 'mirror': v = {kk: {x: y for x, y in vv.items() if x != 'runs'} for kk, vv in v.items()}
    show(k, v)
print('blocks per run:')
for k, runs in B.items():
    for x in runs:
        print(' ', k, x['id'], {f: x[f] for f in ('first_fit', 'nm01', 'nm09', 'ce_change', 'q_nm', 'four_gain_nm', 't99_all', 'first_flip',
                                                   'flip_after_99', 'max_mis_after_fit', 'final', 'strict', 'why', 'late_first_flip',
                                                   'late_max_mis', 'equal_to_base_until_12k', 'gain_12k', 'gain_end', 'ce_min_t')})
