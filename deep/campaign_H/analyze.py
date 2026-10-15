#!/usr/bin/env python3
import argparse, gzip, json, math, os, sys, tarfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T_A = 1000


def wilson(k, n, z=1.959963984540054):
    if n == 0: return (float('nan'), float('nan'))
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if o.get(key) is not None]
    return np.array([a for a, _ in s], float), np.array([b for _, b in s], float)


def at(s, v, t):
    return float(v[int(np.argmin(np.abs(s - t)))])


def first(s, v, thr):
    i = np.where(v >= thr)[0]
    return int(s[i[0]]) if len(i) else None


def fields(d):
    L = np.asarray(d['loss'], float); T = len(L) - 1; obs = d['obs']; p = d['cfg']['p']
    seg = L[T_A:T + 1]
    v = float((seg.max() - seg.min()) / seg.min())
    sa, ah = series(obs, 'AH'); sr, rf = series(obs, 'refit_thm_acc'); sc, rc = series(obs, 'refit_canon_acc')
    st, te = series(obs, 'test_acc'); s1, tr = series(obs, 'train_acc')
    ah_a, ah_T = at(sa, ah, T_A), at(sa, ah, T)
    r50, r90 = first(sr, rf, 0.5), first(sr, rf, 0.9); n50, n90 = first(st, te, 0.5), first(st, te, 0.9)
    J1 = v <= 0.01; J2 = (ah_T - ah_a) >= 0.02
    J3 = at(sr, rf, T_A) <= 0.1 and r90 is not None
    J4 = r50 is not None and (n50 is None or n50 - r50 >= 1000)
    fit_from_r50 = bool(r50 is not None and np.all(tr[s1 >= r50] >= 1.0))
    tr1 = first(s1, tr, 1.0)
    return dict(T=T, v=v, CE_mean_over_logp=float(seg.mean() / math.log(p)), CE_end=float(L[-1]), logp=math.log(p),
                AH_ta=ah_a, AH_T=ah_T, dAH=ah_T - ah_a, refit_ta=at(sr, rf, T_A), refit_T=float(rf[-1]),
                canon_T=float(rc[-1]) if len(rc) else None, test_T=float(te[-1]), train_T=float(tr[-1]),
                t_refit50=r50, t_refit90=r90, t_test50=n50, t_test90=n90, t_train1=tr1, train1_from_refit50=fit_from_r50,
                J1=bool(J1), J2=bool(J2), J3=bool(J3), J4=bool(J4), joint=bool(J1 and J2 and J3 and J4))


def runs(src):
    out = {}
    for e in sorted(os.listdir(src)):
        p = os.path.join(src, e)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'run.json.gz')):
            with gzip.open(os.path.join(p, 'run.json.gz'), 'rt') as f: out[e] = json.load(f)
        elif e.endswith('.tar'):
            with tarfile.open(p) as t:
                for m in t.getmembers():
                    if m.isfile() and m.name.endswith('/run.json.gz'):
                        out[m.name.split('/')[-2]] = json.loads(gzip.decompress(t.extractfile(m).read()).decode())
    return out


def report(src, out_md, out_json):
    man = json.load(open(os.path.join(HERE, 'manifest.json'))); entry = {r['id']: r for r in man['runs']}
    R = runs(src); rows = {}
    for rid in sorted(R):
        d = R[rid]; e = entry[rid]; assert d['cfg'] == e['cfg'], rid
        f = fields(d); f.update(arm=e['arm'], seed=e['seed'], eta_v=d['cfg']['eta_v'], eta_w=d['cfg']['eta_w'], lam_w=d['cfg']['lam_w'],
                                stopped_at=d.get('stopped_at'))
        rows[rid] = f
    arms = {}
    for rid, f in rows.items(): arms.setdefault(f['arm'], []).append(f)
    summ = {}
    for a, fs in arms.items():
        n = len(fs); s = {'n': n}
        for J in ('J1', 'J2', 'J3', 'J4', 'joint', 'train1_from_refit50'):
            k = sum(f[J] for f in fs); s[J] = [k, n, *[round(x, 3) for x in wilson(k, n)]]
        for q in ('v', 'dAH', 'AH_T', 'refit_T', 'canon_T', 'test_T', 'train_T', 'CE_mean_over_logp', 't_refit50', 't_refit90', 't_test50',
                  't_test90', 't_train1', 'T'):
            vals = [f[q] for f in fs if f[q] is not None]
            s['median_' + q] = float(np.median(vals)) if vals else None; s['n_' + q] = len(vals)
        summ[a] = s
    json.dump(dict(rows=rows, arms=summ), open(out_json, 'w'), indent=1)
    order = [r['arm'] for r in man['runs']]; order = sorted(summ, key=order.index)
    eta_v = {r['arm']: r['cfg']['eta_v'] for r in man['runs']}
    rng = lambda q, fs, f='.3f': (lambda v: '–' if not v else (format(min(v), f) if format(min(v), f) == format(max(v), f)
                                                                else f'{min(v):{f}}–{max(v):{f}}'))([x[q] for x in fs if x[q] is not None])
    kn = lambda s, J: f"{s[J][0]}/{s[J][1]}"
    L = ["# Campaign H: Theorem 4.1's configuration at p = 101 from an iid initialization", '',
         f"{len(rows)}/{len(man['runs'])} runs. Arms, seeds and endpoints: PLAN.md (fixed before any campaign run). J1: v([1000, T]) <= 1%; "
         "J2: ΔA_H >= 0.02 on [1000, T]; J3: Theorem 4.1's refit <= 0.1 held-out at update 1000 and reaches 0.9; J4: the refit reaches 0.5 "
         "at least 1000 updates before the network.", '',
         '## Per arm', '',
         '| arm | η_V | runs | J1 | J2 | J3 | J4 | joint (Wilson 95%) | v on [1000, T] | refit 0.9 at (median) | network held-out at T | training accuracy at T |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for a in order:
        s = summ[a]; fs = arms[a]; t90 = s['median_t_refit90']
        L.append(f"| {a} | {eta_v[a]:g} | {s['n']} | {kn(s, 'J1')} | {kn(s, 'J2')} | {kn(s, 'J3')} | {kn(s, 'J4')} | "
                 f"{kn(s, 'joint')} ({s['joint'][2]:.2f}–{s['joint'][3]:.2f}) | {rng('v', fs, '.4f')} | "
                 f"{'never' if t90 is None else f'{t90:.0f}'} | {rng('test_T', fs)} | {rng('train_T', fs, '.4f')} |")
    L += ['', '## Every run', '',
          '| run | v | ΔA_H | refit at 1000 | refit 0.5 / 0.9 at | network 0.5 / 0.9 at | refit at T | network at T | training at T | J1 J2 J3 J4 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    iv = lambda x: '–' if x is None else str(x)
    for a in order:
        for f in sorted(arms[a], key=lambda f: f['seed']):
            rid = next(k for k, v in rows.items() if v is f)
            L.append(f"| {rid} | {f['v']:.4f} | {f['dAH']:.3f} | {f['refit_ta']:.3f} | {iv(f['t_refit50'])} / {iv(f['t_refit90'])} | "
                     f"{iv(f['t_test50'])} / {iv(f['t_test90'])} | {f['refit_T']:.3f} | {f['test_T']:.3f} | {f['train_T']:.4f} | "
                     + ' '.join('yes' if f[J] else 'no' for J in ('J1', 'J2', 'J3', 'J4')) + ' |')
    open(out_md, 'w').write('\n'.join(L) + '\n')
    for a in order:
        s = summ[a]
        print(f"{a:7s} n={s['n']:2d} J1 {s['J1'][0]} J2 {s['J2'][0]} J3 {s['J3'][0]} J4 {s['J4'][0]} joint {s['joint'][0]}/{s['n']} | "
              f"median v {s['median_v']:.4f}, refit 0.9 at {s['median_t_refit90']}, network held-out at T {s['median_test_T']:.3f}")
    print('wrote', out_md, 'and', out_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('report'); p.add_argument('--from', dest='src', required=True)
    p.add_argument('--out', default=os.path.join(HERE, 'ANALYSIS.md')); p.add_argument('--json', default=os.path.join(HERE, 'analysis.json'))
    a = ap.parse_args()
    report(a.src, a.out, a.json)


if __name__ == '__main__':
    main()
