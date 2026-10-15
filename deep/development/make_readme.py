import glob, math, json, os, numpy as np
from anat import load, series, knee_step, first_persistent, at
from plateau import onset
from dtable import first_cross
from flatwin import longest_flat

LP = math.log(31)
def cmv(L): return (L[:-1] + L[1:]) / 2
def f2(a, b, d=2): return f"{a:.{d}f} → {b:.{d}f}"
def pct(x): return f"{x:.1f}%"
def cfg(fn): return json.load(open(fn))['cfg']

_stats_cache = {}
def stats(fn):
    key = (fn, os.path.getmtime(fn))
    if key in _stats_cache: return _stats_cache[key]
    o = _stats_uncached(fn); _stats_cache[key] = o; return o

def _stats_uncached(fn):
    d, L, obs, st = load(fn); c = d['cfg']; T = len(L) - 1; cm = cmv(L); D = c['depth']; r = c['p'] - 1
    k = knee_step(d, obs) if c['lam_v'] > 0 else None
    o = dict(fn=fn, name=os.path.basename(fn)[:-5], D=D, T=T, knee=k, lp=math.log(c['p']))
    o['t10'] = first_cross(obs, 'test_acc', 0.1); o['t90'] = first_persistent(obs, 'test_acc', 0.9)
    s_, te = series(obs, 'test_acc'); o['final'] = float(te[-1]); o['test'] = lambda t: float(at(obs, 'test_acc', t))
    o['on2'] = onset(L, k, 5000, 0.02) if k is not None else None
    if o['t10'] is not None:
        g0, g1 = o['t10'], (o['t90'] or T); seg = cm[g0:g1 + 1]
        o['g'] = (g0, g1); o['g_lev'] = seg.mean(); o['g_R'] = 100 * (seg.max() - seg.min()) / seg.mean(); o['g_D'] = 100 * (seg[-1] - seg[0]) / seg.mean()
        lp = o['lp'] - seg; o['g_Rl'] = 100 * (lp.max() - lp.min()) / lp.mean()
    if k is not None:
        a, b = longest_flat(cm, k, 0.03); seg = cm[a:b + 1]; o['fw'] = (a, b); o['fw_lev'] = seg.mean(); o['fw_R'] = 100 * (seg.max() - seg.min()) / seg.mean()
    o['m'] = lambda key, t: float(at(obs, key, t))
    o['r'] = r
    return o

def row_grok(o, extra=''):
    g0, g1 = o['g']
    return (f"| {o['t10']:,} → {('%s' % format(o['t90'], ',')) if o['t90'] else 'not reached'} | {o['g_lev']:.2f} ({o['g_lev'] / o['lp']:.2f}) | {pct(o['g_R'])} | {o['g_D']:+.1f}% | {pct(o['g_Rl'])} | "
            f"{f2(o['m']('agop%d_ho' % o['r'], g0), o['m']('agop%d_ho' % o['r'], g1))} | {o['final']:.2f} |")

def fw_block(o):
    a, b = o['fw']; m = o['m']; r = o['r']; D = o['D']
    ah = ' / '.join(f2(m(f'AH_full_h{l + 1}', a), m(f'AH_full_h{l + 1}', b)) for l in range(D))
    return (f"| {a:,}–{b:,} ({b - a:,}) | {o['fw_lev']:.2f} ({o['fw_lev'] / o['lp']:.2f}) | {f2(m('test_acc', a), m('test_acc', b))} | {f2(m('refit_acc', a), m('refit_acc', b))} | "
            f"{f2(m('agop%d_ho' % r, a), m('agop%d_ho' % r, b))} | {f2(m('agopH1_4_ho', a), m('agopH1_4_ho', b))} ({m('agopH1_4_ho_rand', b):.2f}) | {ah} | {f2(m('W1_four_top1', a), m('W1_four_top1', b))} |")

def gain(fn):
    d, L, obs, st = load(fn); hv = [o for o in obs if 'normW' in o]; T = len(L) - 1
    t10 = first_cross(obs, 'test_acc', 0.1); t90 = first_persistent(obs, 'test_acc', 0.9) or T
    atv = lambda t: min(hv, key=lambda o: abs(o['step'] - t)); pr = lambda o: o['normV'] * np.prod(o['normW'])
    a, b = atv(t10), atv(t90)
    return pr(b) / pr(a), (b['logit_rms'] / pr(b)) / (a['logit_rms'] / pr(a)), b['logit_rms'] / a['logit_rms']

if __name__ == '__main__':
    out = []
    W = out.append
    S = {}
    def st(fn):
        if fn not in S: S[fn] = stats(fn)
        return S[fn]
    seeds31 = sorted(glob.glob('runs/seed_p31_lw0.015_lv0.2_s*.json')); seeds23 = sorted(glob.glob('runs/seed_p23_lw0.01_lv0.2_s*.json'))
    widths = [('runs/h1_p31_w128_lw0.015.json', 128), ('runs/h1_p31_w256_lw0.015.json', 256), ('runs/s2_p31_lw0.015_lv0.2.json', 512), ('runs/h1_p31_w1024_lw0.015.json', 1024)]
    he2 = [('runs/d14_w256_k1_lw0.022_ld0.05.json', 0.022), ('runs/d18_w256_k1_lw0.025_ld0.05.json', 0.025), ('runs/d18_w256_k1_lw0.027_ld0.05.json', 0.027), ('runs/d14_w256_k1_lw0.03_ld0.05.json', 0.03)]
    he3 = [('runs/d19_D3_w256_k1_lw0.03_ld0.05.json', 0.03), ('runs/d19_D3_w256_k1_lw0.022_ld0.05.json', 0.022), ('runs/d22_D3_w256_k1_lw0.017_ld0.05.json', 0.017), ('runs/d22_D3_w256_k1_lw0.013_ld0.05.json', 0.013)]
    rdial = [('runs/d12_w256_k8_ld0.07.json', '1 (same as layer 1)'), ('runs/d24_w256_k8_ld0.07_slowW2_0.7.json', '0.7'), ('runs/d25_w256_k8_ld0.07_r0.5_T40k.json', '0.5'),
             ('runs/d26_w256_k8_ld0.07_r0.45_T40k.json', '0.45'), ('runs/d25_w256_k8_ld0.07_r0.4_T40k.json', '0.4')]
    seedsD2 = sorted(glob.glob('runs/seedD2_r0.5_s*.json'))
    json.dump(dict(seeds31=seeds31, seedsD2=seedsD2), open('readme_inputs.json', 'w'))

    T = {}
    h = '| run | test 0.1 → 0.9 (steps) | CE in that window (× log p) | CE range | CE drift | learned-part range | AGOP–Fourier, top-(p−1), held-out | final test |\n|---|---|---|---|---|---|---|---|'
    T['1hl'] = h + '\n' + '\n'.join(f"| p = {cfg(fn)['p']}, seed {i} " + row_grok(st(fn)) for i, fn in enumerate(seeds31 + seeds23)).replace('| p = 23, seed 5', '| p = 23, seed 0').replace('| p = 23, seed 6', '| p = 23, seed 1').replace('| p = 23, seed 7', '| p = 23, seed 2').replace('| p = 23, seed 8', '| p = 23, seed 3').replace('| p = 23, seed 9', '| p = 23, seed 4')
    h = '| width | test 0.1 → 0.9 (steps) | CE in that window (× log p) | CE range | CE drift | learned-part range | AGOP–Fourier, top-30, held-out | final test |\n|---|---|---|---|---|---|---|---|'
    T['width'] = h + '\n' + '\n'.join(f"| {w} " + row_grok(st(fn)) for fn, w in widths)
    h = '| λ_W1 | test 0.1 → 0.9 (steps) | CE in that window (× log p) | CE range | CE drift | learned-part range | AGOP–Fourier (last layer), top-30 | final test | first-layer AGOP, top-4 |\n|---|---|---|---|---|---|---|---|---|'
    def he_row(fn, lw):
        o = st(fn); g0, g1 = o['g']; return f"| {lw} " + row_grok(o) + f" {f2(o['m']('agopH1_4_ho', g0), o['m']('agopH1_4_ho', g1))} |"
    T['he2'] = h + '\n' + '\n'.join(he_row(fn, lw) for fn, lw in he2)
    T['he3'] = h + '\n' + '\n'.join(he_row(fn, lw) for fn, lw in he3)
    h = '| r | test 0.1 → 0.9 (steps) | CE in that window (× log p) | CE range | CE drift | learned-part range | AGOP–Fourier (last layer), top-30 | final test | first-layer AGOP, top-4 |\n|---|---|---|---|---|---|---|---|---|'
    T['rdial'] = h + '\n' + '\n'.join(he_row(fn, r) for fn, r in rdial)
    h = ('| seed | flat window, steps (length) | CE (× log p) | test accuracy | ridge refit | AGOP last layer, top-30 | AGOP first layer, top-4 (random) | A_H layer 1 / layer 2 | layer-1 weight Fourier share |\n'
         '|---|---|---|---|---|---|---|---|---|')
    T['d2fw'] = h + '\n' + '\n'.join(f"| {i} " + fw_block(st(fn)) for i, fn in enumerate(seedsD2))
    h = '| seed | ±2% onset (test there) | test 0.1 → 0.9 (steps) | CE range over test 0.1 → 0.9 | CE drift | CE at 40k vs flat-window mean | test at 40k |\n|---|---|---|---|---|---|---|'
    def d2g(i, fn):
        o = st(fn); d, L, obs, s_ = load(fn); cm = cmv(L)
        return (f"| {i} | {o['on2']:,} ({o['test'](o['on2']):.2f}) | {o['t10']:,} → {format(o['t90'], ',') if o['t90'] else 'not reached'} | {pct(o['g_R'])} | {o['g_D']:+.1f}% | "
                f"{100 * (cm[-1] / o['fw_lev'] - 1):+.1f}% | {o['final']:.2f} |")
    T['d2g'] = h + '\n' + '\n'.join(d2g(i, fn) for i, fn in enumerate(seedsD2))
    gl = [('one hidden layer, width 512', 'runs/s2_p31_lw0.015_lv0.2.json'), ('depth 2, He deep layer (λ_W1 0.027)', 'runs/d18_w256_k1_lw0.027_ld0.05.json'),
          ('depth 3, He deep layers (λ_W1 0.022)', 'runs/d19_D3_w256_k1_lw0.022_ld0.05.json'), ('depth 2, large-init deep layer, r = 0.5', 'runs/d25_w256_k8_ld0.07_r0.5_T40k.json')]
    h = '| model (tuning seed) | product of norms | normalized gain | logit RMS |\n|---|---|---|---|'
    T['gain'] = h + '\n' + '\n'.join(f"| {lab} | ×{a:.2f} | ×{b:.2f} | ×{c:.2f} |" for lab, fn in gl for a, b, c in [gain(fn)])
    json.dump({k: v for k, v in T.items()}, open('readme_tables.json', 'w'), indent=1)
    for k, v in T.items(): print('\n###', k); print(v)

def controls_table():
    rows = [('runs/seedD2_r0.5_s0.json', 'primary design (fresh seed 0, for reference)'),
            ('runs/d25_w256_k8_ld0.07_r0.5_T40k.json', 'primary design, tuning seed'),
            ('runs/ctlD2_r0.5_freezehidden16000.json', 'tuning seed; both hidden layers frozen at step 16,000 (head keeps training)'),
            ('runs/ctlD2_r0.5_nodecay.json', 'tuning seed; no weight decay anywhere (λ = 0)')]
    h = '| run | CE at 16k → 26k | test at 16k → 26k → 40k | first-layer AGOP, top-4, 16k → 26k | layer-1 weight Fourier share, 16k → 26k |\n|---|---|---|---|---|'
    out = [h]
    for fn, lab in rows:
        d, L, obs, st = load(fn); cm = cmv(L)
        m = lambda k, t: float(at(obs, k, t))
        out.append(f"| {lab} | {cm[16000]:.3g} → {cm[26000]:.3g} | {m('test_acc', 16000):.2f} → {m('test_acc', 26000):.2f} → {m('test_acc', 40000):.2f} | "
                   f"{m('agopH1_4_ho', 16000):.2f} → {m('agopH1_4_ho', 26000):.2f} | {m('W1_four_top1', 16000):.2f} → {m('W1_four_top1', 26000):.2f} |")
    return '\n'.join(out)

def assemble(d3_summary, d3_section, out='README.md'):
    import subprocess
    T = json.load(open(_here_or_cwd('readme_tables.json')))
    ctl = controls_table().split('\n')
    ctl = '\n'.join([l for l in ctl if 'fresh seed 0' not in l])
    body = open(_here_or_cwd('README_body.md')).read()
    for k, v in T.items(): body = body.replace(f'<<T:{k}>>', v)
    import glob as _g
    body = body.replace('<<T:w512>>', w512_table(sorted(_g.glob('runs/T2_D2_w512_r0.5_s*.json'))))
    body = body.replace('<<T:ctl>>', ctl).replace('<<D3_SUMMARY>>', d3_summary).replace('<<D3_SECTION>>', d3_section)
    assert '<<' not in body, body[body.index('<<'):body.index('<<') + 40]
    open(out, 'w').write(body); return out

def d3_table(items):
    h = ('| r (steps) | flat window, steps (length) | CE (× log p) | test in window | ridge refit | A_H layers 1 / 2 / 3 | layer-1 weight Fourier share | test 0.1 → 0.9 | CE range, drift over test 0.1 → 0.9 | final test |\n'
         '|---|---|---|---|---|---|---|---|---|---|')
    rows = [h]
    for fn, lab in items:
        o = stats(fn); a, b = o['fw']; m = o['m']
        ah = ' / '.join(f2(m(f'AH_full_h{l + 1}', a), m(f'AH_full_h{l + 1}', b)) for l in range(3))
        g = f"{o['t10']:,} → {format(o['t90'], ',') if o['t90'] else 'not reached'}" if o['t10'] else 'not reached'
        gr = f"{o['g_R']:.1f}%, {o['g_D']:+.1f}%" if o['t10'] else '–'
        rows.append(f"| {lab} | {a:,}–{b:,} ({b - a:,}) | {o['fw_lev']:.2f} ({o['fw_lev'] / o['lp']:.2f}) | {f2(m('test_acc', a), m('test_acc', b))} | {f2(m('refit_acc', a), m('refit_acc', b))} | {ah} | "
                    f"{f2(m('W1_four_top1', a), m('W1_four_top1', b))} | {g} | {gr} | {o['final']:.2f} |")
    return '\n'.join(rows)

def w512_table(fns):
    h = ('| seed | flat window, steps (length) | CE (× log p) | test accuracy | share of the 0.1 → 0.9 rise inside | AGOP last layer, top-30 | AGOP first layer, top-4 | A_H layer 1 / layer 2 | CE range over test 0.1 → 0.9 | CE at 40k vs flat mean |\n'
         '|---|---|---|---|---|---|---|---|---|---|')
    rows = [h]
    for i, fn in enumerate(fns):
        o = stats(fn); a, b = o['fw']; m = o['m']; d, L, obs, st = load(fn); cm = cmv(L)
        ta, tb = m('test_acc', a), m('test_acc', b); share = (min(tb, 0.9) - max(ta, 0.1)) / 0.8
        rows.append(f"| {i} | {a:,}–{b:,} ({b - a:,}) | {o['fw_lev']:.2f} ({o['fw_lev'] / o['lp']:.2f}) | {f2(ta, tb)} | {100 * share:.0f}% | {f2(m('agop30_ho', a), m('agop30_ho', b))} | "
                    f"{f2(m('agopH1_4_ho', a), m('agopH1_4_ho', b))} | {f2(m('AH_full_h1', a), m('AH_full_h1', b))} / {f2(m('AH_full_h2', a), m('AH_full_h2', b))} | {o['g_R']:.1f}% | {100 * (cm[-1] / o['fw_lev'] - 1):+.1f}% |")
    return '\n'.join(rows)

_nd_cache = {}
def nondec(fn, delta=0.01):
    key = (fn, os.path.getmtime(fn), delta)
    if key not in _nd_cache:
        from nondec import nondec_window
        a, b, sh, lev, d = nondec_window(fn, delta=delta); _nd_cache[key] = (a, b, sh)
    return _nd_cache[key]

_bw_cache = {}
def bestwin(fn):
    key = (fn, os.path.getmtime(fn))
    if key not in _bw_cache: _bw_cache[key] = _bestwin(fn)
    return _bw_cache[key]

def _bestwin(fn):
    from flatwin import best_grok_window
    d, L, obs, st = load(fn); cm = cmv(L); k = knee_step(d, obs); s_, te = series(obs, 'test_acc')
    a, b, sh = best_grok_window(cm, s_, te, k); seg = cm[a:b + 1]
    return a, b, sh, float(seg.mean()), float(100 * (seg.max() - seg.min()) / seg.mean())

def share_of(o, a, b):
    t = lambda s: min(max(o['m']('test_acc', s), 0.1), 0.9)
    return (t(b) - t(a)) / 0.8

def grok_s(o):
    return f"{o['t10']:,} → {format(o['t90'], ',') if o['t90'] else 'not reached'}" if o['t10'] is not None else 'not reached'

def moduli_table(items):
    h = ('| p, seed | flat window, steps (length) | CE (× log p) | test accuracy | share of the 0.1 → 0.9 rise inside | AGOP last layer, top-(p−1) | AGOP first layer, top-4 | A_H layer 1 / layer 2 | layer-1 weight Fourier share | test 0.1 → 0.9 (steps) | CE range over it | final test |\n'
         '|---|---|---|---|---|---|---|---|---|---|---|---|')
    rows = [h]
    for fn, lab in items:
        o = stats(fn); a, b = o['fw']; m = o['m']; r = o['r']
        rows.append(f"| {lab} | {a:,}–{b:,} ({b - a:,}) | {o['fw_lev']:.2f} ({o['fw_lev'] / o['lp']:.2f}) | {f2(m('test_acc', a), m('test_acc', b))} | {100 * share_of(o, a, b):.0f}% | "
                    f"{f2(m('agop%d_ho' % r, a), m('agop%d_ho' % r, b))} | {f2(m('agopH1_4_ho', a), m('agopH1_4_ho', b))} | {f2(m('AH_full_h1', a), m('AH_full_h1', b))} / {f2(m('AH_full_h2', a), m('AH_full_h2', b))} | "
                    f"{f2(m('W1_four_top1', a), m('W1_four_top1', b))} | {grok_s(o)} | {o['g_R']:.1f}% | {o['final']:.2f} |")
    return '\n'.join(rows)

def dial_table(items):
    h = ('| deep-layer shrink rates, × layer 1 (steps) | test 0.1 → 0.9 (steps) | CE range, drift over it | ≤3% window holding most of that rise: steps (length) | CE there (× log p) | test there | share of the rise inside | flat-or-rising window (≤3%, no drop >1%): steps (share) | longest ≤3% window, loss alone: steps (× log p; share of the rise) | final test |\n'
         '|---|---|---|---|---|---|---|---|---|---|')
    rows = [h]
    for fn, lab in items:
        o = stats(fn); m = o['m']; a, b, sh, lev, R = bestwin(fn); fa, fb = o['fw']
        gr = f"{o['g_R']:.1f}%, {o['g_D']:+.1f}%" if o['t10'] is not None else '–'
        na, nb, nsh = nondec(fn)
        rows.append(f"| {lab} | {grok_s(o)} | {gr} | {a:,}–{b:,} ({b - a:,}) | {lev:.2f} ({lev / o['lp']:.2f}) | {f2(m('test_acc', a), m('test_acc', b))} | {100 * sh:.0f}% | "
                    f"{na:,}–{nb:,} ({100 * nsh:.0f}%) | {fa:,}–{fb:,} ({o['fw_lev'] / o['lp']:.2f}; {100 * share_of(o, fa, fb):.0f}%) | {o['final']:.2f} |")
    return '\n'.join(rows)

def inside_table(items):
    rows = []
    for fn, lab in items:
        o = stats(fn); m = o['m']; D = o['D']; a, b, sh, lev, R = bestwin(fn)
        ah = ' / '.join(f2(m(f'AH_full_h{l + 1}', a), m(f'AH_full_h{l + 1}', b)) for l in range(D))
        rows.append(f"| {lab} | {a:,}–{b:,} | {f2(m('test_acc', a), m('test_acc', b))} | {f2(m('refit_acc', a), m('refit_acc', b))} | {f2(m('agopH1_4_ho', a), m('agopH1_4_ho', b))} ({f2(m('agopH1_4_ho_rand', a), m('agopH1_4_ho_rand', b))}) | "
                    f"{f2(m('agop30_ho', a), m('agop30_ho', b))} ({f2(m('agop30_ho_rand', a), m('agop30_ho_rand', b))}) | {ah} | {f2(m('W1_four_top1', a), m('W1_four_top1', b))} |")
    Dmax = max(stats(fn)['D'] for fn, _ in items)
    h = (f"| run | window, steps | test accuracy | ridge refit | AGOP first layer, top-4 (random frames) | AGOP last layer, top-30 (random frames) | A_H layers {' / '.join(str(l + 1) for l in range(Dmax))} | layer-1 weight Fourier share |\n"
         '|---|---|---|---|---|---|---|---|')
    return h + '\n' + '\n'.join(rows)

def norms_table(items):
    rows = ['| run | ‖W_l‖ change per hidden layer | product of norms | normalized gain | logit RMS |\n|---|---|---|---|---|']
    for fn, lab in items:
        d, L, obs, st = load(fn); hv = [o for o in obs if 'normW' in o]; T = len(L) - 1
        t10 = first_cross(obs, 'test_acc', 0.1); t90 = first_persistent(obs, 'test_acc', 0.9) or T
        atv = lambda t: min(hv, key=lambda o: abs(o['step'] - t)); a, b = atv(t10), atv(t90)
        per = ' / '.join(f"×{y / x:.2f}" for x, y in zip(a['normW'], b['normW'])); g = gain(fn)
        rows.append(f"| {lab} | {per} | ×{g[0]:.2f} | ×{g[1]:.2f} | ×{g[2]:.2f} |")
    return '\n'.join(rows)

D3_DIAL = [('runs/d28_D3_w256_k8_ld0.07_r0.7_T40k.json', 'uniform 0.7, 0.7 (40k)'), ('runs/d28_D3_w256_k8_ld0.07_r0.5_T60k.json', 'uniform 0.5, 0.5 (60k)'),
           ('runs/d29_D3_w256_k8_ld0.07_r0.4_T80k.json', 'uniform 0.4, 0.4 (80k)'), ('runs/d29_D3_w256_k8_ld0.07_r0.35_T80k.json', 'uniform 0.35, 0.35 (80k)'),
           ('runs/T5_D3_cascade_r0.5_0.25_tune_T100k.json', 'cascade 0.5, 0.25 (100k)'), ('runs/T5_D3_cascade_r0.7_0.35_tune_T100k.json', 'cascade 0.7, 0.35 (100k)'),
           ('runs/T7_D3_cascade_r0.8_0.4_tune_T100k.json', 'cascade 0.8, 0.4 (100k)'), ('runs/T7_D3_cascade_r0.9_0.45_tune_T100k.json', 'cascade 0.9, 0.45 (100k)'),
           ('runs/T8_D3_lastslow_r1.0_0.5_tune_T100k.json', 'last-layer-slow 1.0, 0.5 (100k)')]
MODULI = [('runs/T3_D2_p47_r0.5_tune.json', '47, tuning'), ('runs/T3_D2_p47_r0.5_s0.json', '47, seed 0'),
          ('runs/T3_D2_p23_lw0.01_r0.5_tune.json', '23 (λ_W1 0.01), tuning'), ('runs/T3_D2_p23_lw0.01_r0.5_s0.json', '23 (λ_W1 0.01), seed 0'),
          ('runs/T9_D2_p23_lw0.015_r0.5_tune.json', '23 (λ_W1 0.015), tuning'), ('runs/T9_D2_p23_lw0.015_r0.5_s0.json', '23 (λ_W1 0.015), seed 0'),
          ('runs/T12_D2_p23_lw0.015_r0.4_tune.json', '23 (λ_W1 0.015, r = 0.4), tuning'), ('runs/T12_D2_p23_lw0.015_r0.35_tune.json', '23 (λ_W1 0.015, r = 0.35), tuning')] + \
         [(f'runs/T16_D2_p23_lw0.015_r{r}_{sd}_T80k.json', f"23 (λ_W1 0.015, r = {r}), {'tuning' if sd == 'tune' else 'seed ' + sd[1:]}, 80k") for r in ('0.4', '0.45') for sd in ('tune', 's0', 's1')]

def _here_or_cwd(name):
    return name if os.path.exists(name) else os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

def build_v11(texts, extra_tables=None, out='README.md'):
    import glob as _g
    T = json.load(open(_here_or_cwd('readme_tables.json')))
    body = open(_here_or_cwd('README_body.md')).read()
    for _ in range(2):
        for k, v in texts.items(): body = body.replace(f'<<{k}>>', v)
    for k, v in T.items(): body = body.replace(f'<<T:{k}>>', v)
    ctl = '\n'.join([l for l in controls_table().split('\n') if 'fresh seed 0' not in l])
    live = {'w512': w512_table(sorted(_g.glob('runs/T2_D2_w512_r0.5_s*.json'))), 'ctl': ctl,
            'moduli': moduli_table([x for x in MODULI + (extra_tables or {}).get('moduli_extra', []) if os.path.exists(x[0])]),
            'd3dial': dial_table([x for x in D3_DIAL if os.path.exists(x[0])]),
            'd3inside': inside_table([D3_DIAL[3], D3_DIAL[6], D3_DIAL[7], D3_DIAL[8]]),
            'norms': norms_table([('runs/d25_w256_k8_ld0.07_r0.5_T40k.json', 'depth 2, r = 0.5'), D3_DIAL[0], D3_DIAL[2], D3_DIAL[4], D3_DIAL[5], D3_DIAL[7], D3_DIAL[8]] + (extra_tables or {}).get('norms_extra', []))}
    t6 = sorted(_g.glob('runs/T6_D3_cascade_r0.9_0.45_s*_T100k.json'))
    if t6:
        seeds = [(D3_DIAL[7][0], 'cascade 0.9, 0.45, tuning seed')] + [(fn, 'cascade 0.9, 0.45, seed ' + fn.split('_s')[-1][0]) for fn in t6]
        t13 = sorted(_g.glob('runs/T13_D3_lastslow_r1.0_0.5_s*_T100k.json'))
        seeds += [(D3_DIAL[8][0], 'last-layer-slow 1.0, 0.5, tuning seed')] + [(fn, 'last-layer-slow 1.0, 0.5, seed ' + fn.split('_s')[-1][0]) for fn in t13] if t13 else []
        live['d3seeds'] = dial_table(seeds); live['d3seeds_inside'] = inside_table(seeds); live['d3seeds_norms'] = norms_table(seeds)
    w5 = [(f, ('cascade 0.9, 0.45' if 'cascade' in f else 'last-layer-slow 1.0, 0.5') + ', width 512, ' + ('tuning seed' if '_tune_' in f else 'seed ' + f.split('_s')[-1][0]) + ' (60k)')
          for f in sorted(_g.glob('runs/T10_D3_w512_*.json') + _g.glob('runs/T15_D3_w512_*.json'), key=lambda f: ('lastslow' in f, 0 if '_tune_' in f else 1 + int(f.split('_s')[-1][0])))]
    if w5:
        live['d3w512'] = dial_table(w5); live['d3w512_inside'] = inside_table(w5); live['d3w512_norms'] = norms_table(w5)
    d4 = [('runs/T4_D4_r0.5_tune_T120k.json', 'uniform 0.5, 0.5, 0.5'), ('runs/T4_D4_r0.35_tune_T120k.json', 'uniform 0.35, 0.35, 0.35'),
          ('runs/T8_D4_lastslow_r1.0_1.0_0.5_tune_T120k.json', 'last-layer-slow 1.0, 1.0, 0.5')]
    for fn in sorted(_g.glob('runs/T11_D4_*_tune_T120k.json')) + sorted(_g.glob('runs/T14_D4_*_T120k.json')) + sorted(_g.glob('runs/T17_D4_*_T120k.json')) + sorted(_g.glob('runs/T18_D4_*_T120k.json')):
        tag = fn.split('_r', 1)[1].split('_T120k')[0]; rates, sd = tag.rsplit('_', 1)
        pre = 'slower last layer ' if rates.startswith('1.0_1.0_') else 'cascade '
        lab = pre + ', '.join(rates.split('_')) + ('' if sd == 'tune' else ', seed ' + sd[1:])
        d4.append((fn, lab))
    d4 = [x for x in d4 if os.path.exists(x[0])]
    if d4:
        live['d4dial'] = dial_table(d4); live['d4norms'] = norms_table(d4)
        bal = [x for x in d4 if bestwin(x[0])[2] >= 0.8]
        live['d4inside'] = inside_table(bal)
    for k, v in (extra_tables or {}).items():
        if isinstance(v, str): live[k] = v
    for k, v in live.items(): body = body.replace(f'<<T:{k}>>', v)
    assert '<<' not in body, body[body.index('<<'):body.index('<<') + 60]
    open(out, 'w').write(body); return out
