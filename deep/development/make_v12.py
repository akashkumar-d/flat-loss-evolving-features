import glob, json, os, numpy as np
from make_readme import stats, bestwin, dial_table, inside_table, norms_table, build_v11

TASKS = next((d for d in (os.environ.get('NATPLAT_TASKS', ''),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tasks')) if d and os.path.isdir(d)), 'tasks')
MINUS = lambda s: s.replace('-', '−')

def cfg_for(name):
    for f in sorted(glob.glob(os.path.join(TASKS, '*.json'))):
        for j in json.load(open(f)):
            if j['name'] == name: return j['cfg']
    raise KeyError(name)

def ckpt_json(ck):
    z = np.load(ck, allow_pickle=False); t = int(z['t_next'])
    name = os.path.basename(ck).split('.json.gz.ckpt.npz')[0]
    os.makedirs('runs_ckpt', exist_ok=True); out = f'runs_ckpt/{name}_ck{t}.json'
    if not os.path.exists(out):
        obs = [o for o in json.loads(str(z['obs'])) if o['step'] < t]
        json.dump(dict(cfg=cfg_for(name), loss=z['loss'][:t].tolist(), obs=obs, partial_at=t), open(out, 'w'))
    return out

def d4_candidates(rate, sd):
    r = f'r1.0_1.0_{rate}_{sd}_T120k'
    return [f'runs/T14_D4_{r}.json', f'runs/T17_D4_{r}.json', f'runs/T18_D4_{r}.json',
            f'runs/T11_D4_lastslow_{r}.json', f'runs/T8_D4_lastslow_{r}.json']

def d4_find(rate, sd, ckpt_dirs=('runs_T14',)):
    for c in d4_candidates(rate, sd):
        if os.path.exists(c): return c, False
    for d in ckpt_dirs:
        for ck in glob.glob(f'{d}/T1[478]_D4_r1.0_1.0_{rate}_{sd}_T120k.json.gz.ckpt.npz'):
            return ckpt_json(ck), True
    return None, False

def cell(fn):
    o = stats(fn); a, b, sh, lev, R = bestwin(fn)
    return sh, (o['g_D'] if o['t10'] is not None else None), o

def rate_grid(rates, seeds, finder, pending=()):
    vals = {}
    for r in rates:
        for s in seeds:
            fn, ck = finder(r, s)
            vals[r, s] = (cell(fn), ck) if fn else None
    best = {s: max([vals[r, s][0][0] for r in rates if vals[r, s]] or [0]) for s in seeds}
    lab = {'tune': 'tuning seed'}
    rows = ['| last-layer rate | ' + ' | '.join(lab.get(s, 'seed ' + s[1:]) for s in seeds) + ' |', '|---' * (len(seeds) + 1) + '|']
    anyck = False
    for r in rates:
        cs = []
        for s in seeds:
            v = vals[r, s]
            if v is None: cs.append('running' if (r, s) in pending else '–'); continue
            (sh, dr, o), ck = v; anyck |= ck
            txt = f'{100 * sh:.0f}%'
            if sh >= 0.8 and abs(sh - best[s]) < 1e-9: txt = f'**{txt}**'
            txt += f' ({dr:+.1f}%)' if dr is not None else ' (no grokking)'
            cs.append(MINUS(txt) + ('*' if ck else ''))
        rows.append(f'| {r} | ' + ' | '.join(cs) + ' |')
    return '\n'.join(rows), anyck, vals

def d4_items_sorted():
    import glob as _g
    items = [('runs/T4_D4_r0.5_tune_T120k.json', 'uniform 0.5, 0.5, 0.5'), ('runs/T4_D4_r0.35_tune_T120k.json', 'uniform 0.35, 0.35, 0.35')]
    runs = {}
    for fn in _g.glob('runs/T8_D4_lastslow_r1.0_1.0_0.5_tune_T120k.json') + _g.glob('runs/T11_D4_*_tune_T120k.json') + _g.glob('runs/T14_D4_*_T120k.json') + _g.glob('runs/T17_D4_*_T120k.json') + _g.glob('runs/T18_D4_*_T120k.json'):
        tag = fn.split('_r', 1)[1].split('_T120k')[0]; rates, sd = tag.rsplit('_', 1); rs = rates.split('_')
        casc = rs[1] != '1.0'
        lab = ('cascade ' if casc else 'slower last layer ') + ', '.join(rs) + (', tuning seed' if sd == 'tune' else ', seed ' + sd[1:])
        runs[(0 if sd == 'tune' else 1 + int(sd[1:]), casc, float(rs[-1]))] = (fn, lab)
    return [x for x in items if os.path.exists(x[0])] + [runs[k] for k in sorted(runs)]

def d4w512_items():
    fs = sorted(glob.glob('runs/T19_D4_w512_*.json') + glob.glob('runs/T20_D4_w512_*.json'),
                key=lambda f: (0 if '_tune_' in f else 1 + int(f.split('_s')[-1][0]), f.split('_r1.0_1.0_')[1].split('_')[0]))
    return [(f, 'slower last layer 1.0, 1.0, ' + f.split('_r1.0_1.0_')[1].split('_')[0] + ', width 512, ' +
             ('tuning seed' if '_tune_' in f else 'seed ' + f.split('_s')[-1][0]) + ' (60k)') for f in fs]

def d5_items(tasks=('T21', 'T22', 'T23')):
    fs = sorted([f for t in tasks for f in glob.glob(f'runs/{t}_D5_*.json')], key=lambda f: (os.path.basename(f)[:3], 0 if '_tune_' in f else 1 + int(f.split('_s')[-1][0]), f))
    return [(f, 'slower last layer 1.0, 1.0, 1.0, ' + f.split('_r1.0_1.0_1.0_')[1].split('_')[0] + ', ' +
             ('tuning seed' if '_tune_' in f else 'seed ' + f.split('_s')[-1][0]) + f" ({f.split('_T')[-1][:-5]}, {os.path.basename(f)[:3]})") for f in fs]

def extra_tables():
    ex = {}
    grid, anyck, _ = rate_grid(['0.25', '0.3', '0.35', '0.4', '0.45', '0.5'], ['tune', 's0', 's1', 's2'], d4_find,
                               pending={('0.3', 'tune'), ('0.35', 's2'), ('0.4', 's0'), ('0.5', 's1'), ('0.5', 's2'),
                                        ('0.4', 'tune'), ('0.45', 'tune'), ('0.5', 's0'), ('0.45', 's0')})
    d4 = d4_items_sorted()
    ex['d4dial'] = dial_table(d4); ex['d4norms'] = norms_table(d4); ex['d4inside'] = inside_table([x for x in d4 if bestwin(x[0])[2] >= 0.8])
    ex['d4rate'] = grid + ('\n\n\\* scored from a checkpoint of a run still in progress; the full 120k-step run is on the Lightning Studio.' if anyck else '')
    w = d4w512_items()
    if w:
        ex['d4w512'] = dial_table(w); ex['d4w512_inside'] = inside_table(w); ex['d4w512_norms'] = norms_table(w)
    d5 = d5_items()
    if d5:
        ex['d5dial'] = dial_table(d5); ex['d5norms'] = norms_table(d5)
        bal = [x for x in d5 if bestwin(x[0])[2] >= 0.8]
        ex['d5inside'] = inside_table(bal) if bal else '(no depth-5 run holds 80% of the rise in one window yet)'
    return ex

if __name__ == '__main__':
    import sys
    ex = extra_tables()
    for k, v in ex.items(): print(f'== {k}\n{v}\n')

RELAX_RUNS = [('runs/seedD2_r0.5_s0.json', 'depth 2, 0.5×, seed 0'), ('runs/T13_D3_lastslow_r1.0_0.5_s0_T100k.json', 'depth 3, rule 1.0/0.5, seed 0'),
              ('runs/T7_D3_cascade_r0.9_0.45_tune_T100k.json', 'depth 3, cascade 0.9/0.45, tuning'), ('runs/T11_D4_lastslow_r1.0_1.0_0.25_tune_T120k.json', 'depth 4, 0.25×, tuning'),
              ('runs/T11_D4_lastslow_r1.0_1.0_0.35_tune_T120k.json', 'depth 4, 0.35×, tuning'), ('runs/T18_D4_r1.0_1.0_0.4_s1_T120k.json', 'depth 4, 0.4×, seed 1'),
              ('runs/T18_D4_r1.0_1.0_0.45_s2_T120k.json', 'depth 4, 0.45×, seed 2'), ('runs/T8_D4_lastslow_r1.0_1.0_0.5_tune_T120k.json', 'depth 4, 0.5×, tuning')]

def relax_rows():
    from anat import load
    out = []
    for fn, lab in RELAX_RUNS:
        if not os.path.exists(fn): continue
        d, L, obs, st = load(fn); c = d['cfg']
        kap = c['eta_layers'][-1] * c['lam_layers'][-1] if 'eta_layers' in c else c['eta_deep'] * c['lam_deep']
        hv = [o for o in obs if 'normW' in o]; s = np.array([o['step'] for o in hv]); N = np.array([o['normW'][-1] for o in hv])
        tau = 1 / kap; v = [float(np.interp(k * tau, s, N)) for k in range(5)]; Nend = float(N[-1])
        out.append((lab, tau, v[0], Nend, [(x - Nend) / (v[0] - Nend) for x in v[1:]]))
    return out

def relax_table():
    rows = ['| run | τ = 1/(η_L λ_L), steps | ‖W_L‖ at start → end | excess left after 1τ | 2τ | 3τ | 4τ |', '|---|---|---|---|---|---|---|']
    for lab, tau, n0, ne, fr in relax_rows():
        rows.append(f'| {lab} | {tau:,.0f} | {n0:.0f} → {ne:.1f} | ' + ' | '.join(f'{100 * f:.1f}%' for f in fr) + ' |')
    return '\n'.join(rows)

REG = (-17.1, 799.0)
import datetime as _dt
REG_T = _dt.datetime(2026, 9, 27, 22, 50, tzinfo=_dt.timezone.utc).timestamp()
def arrived_after(name): return os.path.getmtime('runs/' + name + '.json') > REG_T

def rule_fit():
    import math, subprocess, sys
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rate_rule.py')], check=True, capture_output=True)
    rows = json.load(open('rate_rule_rows.json'))
    base = [r for r in rows if r[2] == 31 and r[1] == 256 and not r[3].startswith('d24_w256_k8_ld0.07_slowW2_0.5') and not arrived_after(r[3])]
    x = np.array([r[7] for r in base]); y = np.array([r[8] for r in base]); D = np.array([r[0] for r in base])
    reg = REG[0] + REG[1] * np.exp(-x); ss = ((y - reg) ** 2).sum(); r2 = 1 - ss / ((y - y.mean()) ** 2).sum(); rmse = float(np.sqrt(ss / len(x)))
    Ds = sorted(set(D.tolist())); A2 = np.stack([(D == d).astype(float) for d in Ds] + [np.exp(-x)], 1); c, *_ = np.linalg.lstsq(A2, y, rcond=None)
    pr = A2 @ c; r2D = 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum(); rmseD = float(np.sqrt(((y - pr) ** 2).mean()))
    bal = [r for r in base if abs(r[8]) <= 3]
    return dict(n=len(base), r2=r2, rmse=rmse, x0=math.log(REG[1] / -REG[0]), Ds=Ds, aD=c[:-1].tolist(), bD=float(c[-1]), r2D=r2D, rmseD=rmseD,
                x0D=[math.log(c[-1] / -a) for a in c[:-1]], bal_x=(min(r[7] for r in bal), max(r[7] for r in bal)), bal_share=(min(r[9] for r in bal), max(r[9] for r in bal)), nbal=len(bal),
                rows=rows)

def rule_facts(F):
    from anat import load
    rows = F['rows']; reg = lambda x: REG[0] + REG[1] * np.exp(-x)
    out = {}
    by = {'shallow': [], 'deep': []}
    for r in rows:
        if r[2] == 31 and r[1] == 256 and abs(r[8]) <= 3 and r[0] <= 4 and not r[3].startswith(('T19', 'T20', 'T21', 'T22', 'T23')):
            d, L, obs, st = load('runs/' + r[3] + '.json'); hv = [o for o in obs if 'normW' in o]
            s = np.array([o['step'] for o in hv]); N = np.array([o['normW'][-1] for o in hv])
            n10 = np.interp(r[5], s, N); n90 = np.interp(r[6] or s[-1], s, N)
            by['shallow' if r[0] == 2 else 'deep'].append((n10 / N[-1], n90 / n10, N.min() / N[-1]))
    for k, v in by.items():
        if v: out[k] = dict(above=(min(a for a, b, c in v) - 1, max(a for a, b, c in v) - 1), shrink=(min(b for a, b, c in v), max(b for a, b, c in v)), under=(1 - max(c for a, b, c in v), 1 - min(c for a, b, c in v)))
    rr = M_relax = relax_rows()
    fr = np.median(np.array([[1.0] + x[4] for x in rr]), 0)
    left = lambda xx: float(np.exp(np.interp(xx, np.arange(5), np.log(np.maximum(fr, 1e-6)))))
    out['left'] = {d: left(x0) for d, x0 in zip(F['Ds'], F['x0D'])}
    hand = [('0.35', 'tune'), ('0.4', 's1'), ('0.4', 's2'), ('0.45', 's0')]
    xs = []
    for rate, sd in hand:
        fn, ck = d4_find(rate, sd)
        if fn:
            c = json.load(open(fn))['cfg']; o = stats(fn); xs.append((c['eta_layers'][-1] * c['lam_layers'][-1] * o['t10'], ck))
    out['d4_hand_x'] = (min(x for x, _ in xs), max(x for x, _ in xs)); out['d4_hand_ck'] = any(ck for _, ck in xs)
    t = []
    for rate in ('0.35', '0.4', '0.45'):
        fn, ck = d4_find(rate, 's1')
        if fn: t.append((rate, stats(fn)['t10']))
    out['s1_t10'] = t
    def dev(cond): return [r[8] - reg(r[7]) for r in rows if cond(r)]
    out['dev_w512'] = dev(lambda r: r[1] == 512 and r[2] == 31 and not r[3].startswith(('T19', 'T20')))
    out['dev_p23'] = dev(lambda r: r[2] == 23 and r[3].find('lw0.015') >= 0)
    out['dev_p23_01'] = dev(lambda r: r[2] == 23 and r[3].find('lw0.01_') >= 0)
    out['x_p47'] = [r[7] for r in rows if r[2] == 47]
    return out


def rule_test_rows(F):
    aD = dict(zip(F['Ds'], F['aD'])); out = []
    for r in F['rows']:
        if not arrived_after(r[3]): continue
        x, dr = r[7], r[8]; reg = REG[0] + REG[1] * np.exp(-x)
        dep = aD[r[0]] + F['bD'] * np.exp(-x) if (r[0] in aD and r[1] == 256 and r[2] == 31) else None
        out.append(dict(name=r[3], D=r[0], w=r[1], p=r[2], x=x, drift=dr, reg=reg, dev=dr - reg, dep=dep, share=r[9]))
    return out

def label_of(name):
    t = name.split('_')
    if name.startswith(('T21', 'T22', 'T23')): return f"depth 5, last {name.split('_r1.0_1.0_1.0_')[1].split('_')[0]}×, " + ('tuning seed' if '_tune_' in name else 'seed ' + name.split('_s')[-1][0])
    w = ', width 512' if '_w512_' in name else ''
    rates = name.split('_r', 1)[1].split('_T')[0]; rs, sd = rates.rsplit('_', 1)
    return f"depth {name.split('_D')[1][0]}{w}, {'/'.join(rs.split('_'))}, " + ('tuning seed' if sd == 'tune' else 'seed ' + sd[1:])

def rule_test_table(F):
    rows = ['| run (arrived after 22:50 UTC) | x | CE drift | registered curve | deviation | per-depth curve | share in one window |', '|---|---|---|---|---|---|---|']
    for q in sorted(rule_test_rows(F), key=lambda q: (q['D'], q['w'], q['x'])):
        f = lambda v: f'{v:+.1f}%'.replace('-', '−')
        rows.append(f"| {label_of(q['name'])} | {q['x']:.2f} | {f(q['drift'])} | {f(q['reg'])} | {q['dev']:+.1f}".replace('-', '−') + f" | {f(q['dep']) if q['dep'] is not None else '–'} | {100 * q['share']:.0f}% |")
    return '\n'.join(rows)
