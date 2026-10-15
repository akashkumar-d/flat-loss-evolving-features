import argparse, gzip, hashlib, io, json, math, os, sys, tarfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import endpoints_i as E

ARMS = ['K', 'KD', 'S']
ARM_NAME = dict(K='bias frozen from the knee on', KD='bias decays only, from the knee on (bias-free optimizer after the knee)',
                S='bias updated outside W\'s Polar step')
RHO_CUTS = (1 / 3, 2 / 3)
BAND_EXPL = 1.05


class Sources:
    def __init__(s, srcs):
        s.dirs = [x for x in srcs if os.path.isdir(x)]; s.tars = []
        for x in srcs:
            if not os.path.isdir(x):
                t = tarfile.open(x); s.tars.append((t, {m.name: m for m in t.getmembers() if m.isfile()}))

    def read(s, rid, want_sha=None):
        tail = f'{rid}/run.json.gz'
        for d in s.dirs:
            fn = os.path.join(d, rid, 'run.json.gz')
            if os.path.exists(fn): raw = open(fn, 'rb').read(); break
        else:
            raw = None
            for t, mem in s.tars:
                hit = [n for n in mem if n == tail or n.endswith('/' + tail)]
                if hit: raw = t.extractfile(mem[hit[0]]).read(); break
        if raw is None: raise FileNotFoundError(rid)
        h = hashlib.sha256(raw).hexdigest()
        if want_sha is not None and h != want_sha: raise ValueError(f'{rid}: sha256 {h} != {want_sha}')
        return json.loads(gzip.decompress(raw)), h


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if o.get(key) is not None]
    return np.array([a for a, _ in s], float), np.array([b for _, b in s], float)


def nearest(sv, t):
    s, v = sv
    return float(v[int(np.argmin(np.abs(s - t)))]) if len(s) else float('nan')


def share_at(d, f, band):
    L = np.asarray(d['loss'], float); cm = (L[:-1] + L[1:]) / 2; ts = f['t_peak']; ok = cm >= cm[ts] / band
    a = ts
    while a > 0 and ok[a - 1]: a -= 1
    b = ts
    while b < len(cm) - 1 and ok[b + 1]: b += 1
    st = series(d['obs'], 'test_acc'); clip = lambda x: min(max(nearest(st, x), 0.1), 0.9)
    return (clip(b) - clip(a)) / 0.8


def row(d):
    f = E.fields(d); L = np.asarray(d['loss'], float); cm = (L[:-1] + L[1:]) / 2; lp = math.log(d['cfg']['p'])
    out = dict(knee=f.get('knee'), t_peak=f.get('t_peak'), W=f.get('W'), share=f.get('share'), t10=f.get('t10'), t90=f.get('t90'),
               final=f['final_test'], dAH=f.get('dAH_last'), fit=f.get('fit_strict'), ce_W=f.get('CE_peak_over_logp'),
               passed=bool(f['primary']), why=f.get('why_not'), horizon=len(L) - 1, stopped_at=d.get('stopped_at'))
    if f.get('t_peak') is not None and f.get('t90') is not None:
        t90 = min(f['t90'], len(cm) - 1); tp = f['t_peak']
        out['fall'] = float((cm[tp] - cm[t90]) / cm[tp])
        out['ce_t90'] = float(cm[t90] / lp); out['band_end_minus_t90'] = int(f['W'][1] - f['t90'])
        out['margin_t90'] = nearest(series(d['obs'], 'train_margin_min'), f['t90'])
        out['share_5pct'] = float(share_at(d, f, BAND_EXPL))
    nb = [(o['step'], o['normb'][0]) for o in d['obs'] if o.get('normb') is not None]
    if nb:
        s = np.array([a for a, _ in nb], float); v = np.array([b for _, b in nb], float)
        out['normb'] = dict(max=float(v.max()), at_max=int(s[v.argmax()]), knee=nearest((s, v), f['knee']) if f.get('knee') else None,
                            t90=nearest((s, v), f['t90']) if f.get('t90') else None, final=float(v[-1]))
    return out


def r7(x): return float(f'{x:.7g}')


def pre_knee_check(arm_d, joint_d, knee):
    A, B = arm_d['loss'], joint_d['loss']; n = min(knee + 1, len(A), len(B))
    eq = sum(r7(A[t]) == r7(B[t]) for t in range(n)); first = next((t for t in range(n) if r7(A[t]) != r7(B[t])), None)
    rel = max(abs(A[t] - B[t]) / abs(B[t]) for t in range(n))
    return dict(steps=n, equal_7sf=eq, first_difference=first, max_rel=rel)


def sign_test(d):
    d = [x for x in d if x != 0]; n = len(d); k = sum(x > 0 for x in d); m = min(k, n - k)
    return dict(n=n, positive=k, p=min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0)


def rho_of(fa, fj, ft):
    den = np.median(np.asarray(fj) - np.asarray(ft))
    return float(np.median(np.asarray(fa) - np.asarray(ft)) / den) if den != 0 else float('nan')


def bootstrap_rho(fa, fj, ft, B=10000, seed=20261006):
    rng = np.random.default_rng(seed); n = len(fa); fa, fj, ft = map(np.asarray, (fa, fj, ft)); out = []
    for _ in range(B):
        i = rng.integers(0, n, n); out.append(rho_of(fa[i], fj[i], ft[i]))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def reading(rho):
    return 'removes most of the effect' if rho <= RHO_CUTS[0] else ('keeps most of the effect' if rho >= RHO_CUTS[1] else 'partial')


def _plain(o):
    if isinstance(o, (np.integer, np.bool_)): return o.item()
    if isinstance(o, np.floating): return float(o)
    raise TypeError(type(o))


def report(srcs, outdir):
    man = json.load(open(os.path.join(HERE, 'manifest.json'))); comp = {c['seed']: c for c in json.load(open(os.path.join(HERE, 'comparison.json')))['seeds']}
    S = Sources(srcs); sums = {}
    for src in srcs:
        side = os.path.join(os.path.dirname(os.path.abspath(src)), 'SHA256SUMS_runs')
        if os.path.exists(side):
            for l in open(side):
                h, n = l.split(maxsplit=1); sums[n.strip()] = h
    refs = {}
    for i, c in comp.items():
        for k in ('joint', 'twin'):
            d, h = S.read(c[k]['id'], c[k]['sha256']); refs[(k, i)] = (d, row(d))
    runs = {}
    for r in man['runs']:
        rec_fn = None
        for src in srcs:
            if os.path.isdir(src) and os.path.exists(os.path.join(src, r['id'], 'record.json')): rec_fn = os.path.join(src, r['id'], 'record.json')
        want = json.load(open(rec_fn))['file']['sha256'] if rec_fn else sums.get(f"{r['id']}/run.json.gz")
        try: d, h = S.read(r['id'], want)
        except FileNotFoundError: continue
        x = row(d); x.update(arm=r['arm'], seed=r['seed'], sha256=h, bias_knee=d.get('bias_knee'))
        if r['arm'] in ('K', 'KD'):
            jd = refs[('joint', r['seed'])][0]; x['pre_knee'] = pre_knee_check(d, jd, d['bias_knee'])
            x['knee_equals_joint'] = d['bias_knee'] == refs[('joint', r['seed'])][1]['knee']
        runs[r['id']] = x
    out = dict(campaign='J', plan='PLAN.md', runs=runs, references={f'{k}_s{i}': v[1] for (k, i), v in refs.items()}, arms={})
    seeds = sorted(comp)
    fj = [refs[('joint', i)][1]['fall'] for i in seeds]; ft = [refs[('twin', i)][1]['fall'] for i in seeds]
    out['campaign_I_effect'] = dict(median_fall_joint=float(np.median(fj)), median_fall_twin=float(np.median(ft)),
                                    median_paired_difference=float(np.median(np.subtract(fj, ft))), sign_test=sign_test(np.subtract(fj, ft)))
    for arm in ARMS:
        rs = {r['seed']: r for r in runs.values() if r['arm'] == arm}
        if len(rs) < len(seeds): out['arms'][arm] = dict(n=len(rs), complete=False); continue
        fa = [rs[i]['fall'] for i in seeds]
        rho = rho_of(fa, fj, ft); k = sum(rs[i]['passed'] for i in seeds); lo, hi = E.wilson(k, len(seeds))
        A = dict(n=len(seeds), complete=True, name=ARM_NAME[arm],
                 fall=dict(median=float(np.median(fa)), per_seed=fa), rho=rho, rho_boot95=bootstrap_rho(fa, fj, ft), reading=reading(rho),
                 vs_joint=sign_test(np.subtract(fa, fj)), vs_twin=sign_test(np.subtract(fa, ft)),
                 ratio_per_seed=[float((a - t) / (j - t)) for a, j, t in zip(fa, fj, ft)],
                 passed=k, wilson=[lo, hi], why={},
                 share=dict(median=float(np.median([rs[i]['share'] for i in seeds])), ge075=sum(rs[i]['share'] >= 0.75 for i in seeds)),
                 share_5pct=dict(median=float(np.median([rs[i]['share_5pct'] for i in seeds])), ge075=sum(rs[i]['share_5pct'] >= 0.75 for i in seeds)),
                 final_ge099=sum(rs[i]['final'] >= 0.99 for i in seeds))
        for i in seeds:
            for w in (rs[i]['why'] or '').split(', '):
                if w: A['why'][w] = A['why'].get(w, 0) + 1
        for q in ('t10', 't90', 'band_end_minus_t90', 'margin_t90', 'ce_W', 'ce_t90', 'final', 'dAH'):
            a = [rs[i][q] for i in seeds]; j = [refs[('joint', i)][1][q] for i in seeds]; t = [refs[('twin', i)][1][q] for i in seeds]
            A[q] = dict(median=float(np.median(a)), median_joint=float(np.median(j)), median_twin=float(np.median(t)),
                        vs_joint=sign_test(np.subtract(a, j)), vs_twin=sign_test(np.subtract(a, t)))
        A['W_end'] = dict(median=float(np.median([rs[i]['W'][1] for i in seeds])), median_joint=float(np.median([refs[('joint', i)][1]['W'][1] for i in seeds])),
                          median_twin=float(np.median([refs[('twin', i)][1]['W'][1] for i in seeds])))
        nb = [rs[i].get('normb') for i in seeds]
        if all(nb): A['normb'] = {k_: float(np.median([b[k_] for b in nb])) for k_ in ('max', 'knee', 't90', 'final')}
        if arm in ('K', 'KD'):
            A['pre_knee'] = dict(all_knees_equal_joint=all(rs[i]['knee_equals_joint'] for i in seeds),
                                 max_rel=max(rs[i]['pre_knee']['max_rel'] for i in seeds),
                                 steps_equal_7sf=f"{sum(rs[i]['pre_knee']['equal_7sf'] for i in seeds)}/{sum(rs[i]['pre_knee']['steps'] for i in seeds)}")
        out['arms'][arm] = A
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'analysis.json'), 'w') as f: json.dump(out, f, indent=1, sort_keys=True, default=_plain)
    with open(os.path.join(outdir, 'ANALYSIS.md'), 'w') as f: f.write(markdown(out, seeds))
    print(markdown(out, seeds))


def markdown(o, seeds):
    e = o['campaign_I_effect']; L = []
    p = lambda x: f'{100 * x:.1f}%'
    L.append('# Campaign J: analysis\n')
    L.append('Written by `analyze_J.py report` from the run files (every file checked against its SHA-256). PLAN.md fixes the '
             'measures and the reading before any evaluation run.\n')
    L.append(f"**Campaign I's effect (the references).** The training CE falls from its post-knee peak to held-out t₉₀ by a median "
             f"{p(e['median_fall_joint'])} with the joint step and {p(e['median_fall_twin'])} without biases; median paired difference "
             f"{100 * e['median_paired_difference']:.2f} points, larger with the joint step in {e['sign_test']['positive']}/{e['sign_test']['n']} pairs.\n")
    L.append('## Primary: how much of the effect each arm keeps\n')
    L.append('ρ = median(fall_arm − fall_twin) / median(fall_joint − fall_twin): 0 is the bias-free run, 1 is campaign I. '
             f'Reading fixed in PLAN.md: ρ ≤ 1/3 removes most of the effect, ρ ≥ 2/3 keeps most, otherwise partial.\n')
    L.append('| arm | update of b | median fall | ρ (bootstrap 95%) | reading | lower than joint | higher than twin |')
    L.append('|---|---|---|---|---|---|---|')
    for arm in ARMS:
        A = o['arms'].get(arm, {})
        if not A.get('complete'): L.append(f'| {arm} | {ARM_NAME[arm]} | incomplete ({A.get("n", 0)} runs) | | | | |'); continue
        vj, vt = A['vs_joint'], A['vs_twin']
        L.append(f"| {arm} | {A['name']} | {p(A['fall']['median'])} | {A['rho']:.2f} ({A['rho_boot95'][0]:.2f} to {A['rho_boot95'][1]:.2f}) | "
                 f"{A['reading']} | {vj['n'] - vj['positive']}/{vj['n']} (p = {vj['p']:.3g}) | {vt['positive']}/{vt['n']} (p = {vt['p']:.3g}) |")
    L.append(f"| joint | campaign I | {p(e['median_fall_joint'])} | 1 | | | |")
    L.append(f"| twin | no bias | {p(e['median_fall_twin'])} | 0 | | | |\n")
    L.append('## Secondary: campaign C\'s endpoint and timing (medians; joint and twin in brackets)\n')
    L.append('| arm | pass (Wilson) | failed checks | share ≥ 0.75 | share median | band end | t₉₀ | band end − t₉₀ | min margin at t₉₀ | CE on W / log p | final ≥ 0.99 | share ≥ 0.75 at a 5% band (exploratory) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for arm in ARMS:
        A = o['arms'].get(arm, {})
        if not A.get('complete'): continue
        why = ', '.join(f'{k} {v}' for k, v in sorted(A['why'].items())) or '-'
        g = lambda q, f='{:.0f}': f"{f.format(A[q]['median'])} [{f.format(A[q]['median_joint'])}, {f.format(A[q]['median_twin'])}]"
        L.append(f"| {arm} | {A['passed']}/10 ({A['wilson'][0]:.2f}–{A['wilson'][1]:.2f}) | {why} | {A['share']['ge075']}/10 | {A['share']['median']:.2f} | "
                 f"{A['W_end']['median']:.0f} [{A['W_end']['median_joint']:.0f}, {A['W_end']['median_twin']:.0f}] | {g('t90')} | {g('band_end_minus_t90')} | "
                 f"{g('margin_t90', '{:.3f}')} | {g('ce_W', '{:.3f}')} | {A['final_ge099']}/10 | {A['share_5pct']['ge075']}/10 |")
    L.append('')
    L.append('## Checks\n')
    for arm in ('K', 'KD'):
        A = o['arms'].get(arm, {})
        if A.get('complete'):
            L.append(f"- **{arm}:** every knee equals campaign I's: {A['pre_knee']['all_knees_equal_joint']}; up to the knee the losses equal campaign I's "
                     f"at 7 significant digits on {A['pre_knee']['steps_equal_7sf']} updates (largest relative difference {A['pre_knee']['max_rel']:.1e}).")
    for arm in ARMS:
        A = o['arms'].get(arm, {})
        if A.get('complete') and 'normb' in A:
            nb = A['normb']; L.append(f"- **‖b‖ ({arm}):** median peak {nb['max']:.2f}, at the knee {nb['knee']:.2f}, at t₉₀ {nb['t90']:.2f}, at the end {nb['final']:.2f}.")
    L.append('\n## Per seed: fall (%)\n')
    L.append('| seed | twin | joint | ' + ' | '.join(ARMS) + ' |'); L.append('|---|---|---|' + '---|' * len(ARMS))
    for i in seeds:
        cells = [f"{100 * o['references'][f'twin_s{i}']['fall']:.2f}", f"{100 * o['references'][f'joint_s{i}']['fall']:.2f}"]
        for arm in ARMS:
            r = next((r for r in o['runs'].values() if r['arm'] == arm and r['seed'] == i), None)
            cells.append(f"{100 * r['fall']:.2f}" if r else '-')
        L.append(f'| {i} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    rp = sp.add_parser('report'); rp.add_argument('--from', dest='srcs', nargs='+', required=True); rp.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'report': report(a.srcs, a.out)
