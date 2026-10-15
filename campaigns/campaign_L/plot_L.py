#!/usr/bin/env python3
import json, math, os, statistics as S, sys
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import endpoints_i as E
import analyze_L as AL
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

PRIMES = (23, 47, 61, 101)
SETTINGS = (('paper', 0.015), ('sqrt', 0.015), ('sqrt', 0.025), ('lin', 0.015), ('lin', 0.025))
RULE_TXT = dict(paper='$\\lambda_V$ = 0.2 (paper)', sqrt='$\\lambda_V = 0.2\\sqrt{31/p}$', lin='$\\lambda_V = 0.2\\cdot 31/p$')


def lam_v(p, rule): return 0.2 if rule == 'paper' else (round(0.2 * math.sqrt(31 / p), 4) if rule == 'sqrt' else round(0.2 * 31 / p, 4))


def setting_id(D, p, rule, lw, pre='L'): return f'{pre}_D{D}_p{p}_lv{lam_v(p, rule):.4f}_lw{lw:.3f}'


def key_of(r):
    return ('paper' if r['dial']['lam_v_rule'] == 'paper' else ('sqrt' if r['dial']['lam_v_rule'] == 'sqrt' else 'lin'), r['dial']['lam_w'])


def board(R, out, stem, kind):
    rows = [(D, p) for D in (3, 4) for p in PRIMES]
    heads = [f'{RULE_TXT[r]}\n$\\lambda_W$ = {lw:g}' for r, lw in SETTINGS]
    fig, ax = plt.subplots(figsize=(7.6, 5.2), layout='constrained')
    ax.set_xlim(0, len(heads)); ax.set_ylim(len(rows), -0.05); ax.axis('off')
    green = np.array(to_rgb(A.K['AH'])); white = np.ones(3)
    for j, h in enumerate(heads): ax.text(j + 0.5, -0.12, h, ha='center', va='bottom', fontsize=7.2)
    cell_runs = {}
    for r in R['runs'].values(): cell_runs.setdefault((r['depth'], r['p'], key_of(r)), []).append(r)
    name = dict(share='share', final='final acc.', F02='$A_H$ rise', fit='fit', refit='refit', **{'held-out': 'held-out'})
    for i, (D, p) in enumerate(rows):
        ax.text(-0.06, i + 0.5, f'depth {D}, p = {p}', ha='right', va='center', fontsize=8)
        for j, (rule, lw) in enumerate(SETTINGS):
            rs = sorted(cell_runs.get((D, p, (rule, lw)), []), key=lambda r: r['seed'])
            fin = [r for r in rs if r['state'] == 'finished']; nrun = sum(r['state'] == 'running' for r in rs); nleft = sum(r['state'] == 'not started' for r in rs)
            if kind == 'endpoint':
                k = sum(bool(r['L']['endpoint']['primary']) for r in fin)
                checks = sorted({w.strip() for r in fin if r['L']['endpoint']['why_not'] for w in r['L']['endpoint']['why_not'].split(',')})
                sub = ('pass' if fin and not checks else ', '.join(name[x] for x in ('fit', 'share', 'F02', 'final') if x in checks)) if fin else ''
                if rule == 'paper': ref = R['paper_frac08'].get(f'D{D}_p{p}', []); kref = sum(bool(x['endpoint']['primary']) for x in ref)
                else: ref = [r['K'] for r in fin if 'K' in r]; kref = sum(bool(x['endpoint']['primary']) for x in ref)
            else:
                k = sum(bool(r['L']['refit']['ok']) for r in fin); k3 = sum(bool(r['L']['refit']['ok3']) for r in fin)
                fails = sorted({w for r in fin for w in r['L']['refit']['why']})
                ftxt = 'all three' if len(fails) == 3 else ', '.join(fails)
                sub = (f'pass3 {k3}/{len(fin)}' + ('; fails: ' + ftxt if fails else '')) if fin else ''
                if rule == 'paper': ref = R['paper_frac08'].get(f'D{D}_p{p}', []); kref = sum(bool(x['refit']['ok']) for x in ref)
                else: ref = [r['K'] for r in fin if 'K' in r]; kref = sum(bool(x['refit']['ok']) for x in ref)
            n = len(fin); frac = k / n if n else 0
            col = white * (1 - 0.85 * frac) + green * 0.85 * frac if frac > 0 else np.array([0.955] * 3)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.90, color=col, lw=0))
            tc = 'white' if frac > 0.6 else 'black'
            ax.text(j + 0.5, i + 0.24, f'{k}/{n}' if n else 'no run finished', ha='center', va='center', fontsize=8.5 if n else 6.6,
                    fontweight='bold' if n else 'normal', color=tc)
            if sub: ax.text(j + 0.5, i + 0.47, sub, ha='center', va='center', fontsize=5.9, color=tc)
            pend = ', '.join(x for x in ((f'{nrun} running' if nrun else ''), (f'{nleft} to start' if nleft else '')) if x)
            ref_txt = (f'frac 0.8: {kref}/{len(ref)}' + (' (F/G, 10 seeds)' if rule == 'paper' else ' (K, same seeds)')) if ref else ''
            ax.text(j + 0.5, i + 0.69, pend, ha='center', va='center', fontsize=5.6, color='#444444' if tc == 'black' else tc)
            ax.text(j + 0.5, i + 0.86, ref_txt, ha='center', va='center', fontsize=5.4, color='#666666' if tc == 'black' else tc)
        if i == 3: ax.plot([-1.2, len(heads)], [4, 4], color='#BBBBBB', lw=0.6, clip_on=False)
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'plot_L.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig)


def draw2(cols, out, stem):
    orig = A.plt.subplots
    A.plt.subplots = lambda *a, **k: orig(*a, **dict(k, squeeze=False))
    try: A.draw2(cols, out, stem, rows=('ce', 'acc', 'err', 'ah'), height=5.6)
    finally: A.plt.subplots = orig


def depth_figures(res, R, out, kind):
    for D in (3, 4):
        for rule, lw in SETTINGS:
            cols = []
            for p in PRIMES:
                st = setting_id(D, p, rule, lw); ids = [f'{st}_s{i}' for i in (1, 2, 3) if R['runs'][f'{st}_s{i}']['state'] == 'finished']
                if not ids: continue
                runs = []
                for rid in ids:
                    d = E.load(os.path.join(res, rid, 'run.json.gz'))
                    if kind == 'endpoint':
                        f = R['runs'][rid]['L']['endpoint']
                        if f.get('W') is None: w = A.window_record(d); W, ok = w['W2'], False
                        else: W, ok = f['W'], bool(f['primary'])
                        r = A.deep_run(d, rid, (W, ok, 0.0, 0.0)); r['verdict'] = "campaign C's endpoint"
                    else:
                        x = R['runs'][rid]['L']['refit']; r = A.deep_run(d, rid, (x['W'], x['ok'], x['g_te'], x['g_rf'])); r['verdict'] = 'refit criterion'
                    runs.append(r); del d
                c = A.column2(runs, f'p = {p}', ids)
                k = sum(r['primary'] for r in runs)
                done = '' if len(ids) == 3 else f' ({len(ids)} of 3 done)'
                c['title'] = f'p = {p}: {k}/{len(ids)} pass{done}\n$\\lambda_V$ = {lam_v(p, rule):g}'
                cols.append(c)
            if not cols: continue
            stem = f"L{'_refit' if kind == 'refit' else ''}_depth{D}_{rule}_lw{lw}"
            draw2(cols, out, stem)
            print('drew', stem, [c['title'].split('\n')[0] for c in cols], flush=True)


def med_curves(fns, xmax, step=500):
    g = np.arange(0, xmax + 1, step, dtype=float); ce, te, rf = [], [], []
    for fn in fns:
        d = E.load(fn); lp = math.log(d['cfg']['p']); L = np.asarray(d['loss'], float) / lp; cm = (L[:-1] + L[1:]) / 2
        s, t = E.series(d['obs'], 'test_acc'); sr, r = E.series(d['obs'], 'v1_ridge_acc_te')
        n = len(cm); gg = g[g < n]
        ce.append(np.concatenate([np.interp(gg, np.arange(n), cm), np.full(len(g) - len(gg), np.nan)]))
        te.append(np.where(g <= s[-1], np.interp(g, s, t), np.nan)); rf.append(np.where(g <= sr[-1], np.interp(g, sr, r), np.nan)); del d
    with np.errstate(all='ignore'):
        return g, np.nanmedian(np.array(ce), 0), np.nanmedian(np.array(te), 0), np.nanmedian(np.array(rf), 0), np.sum(np.isfinite(np.array(te)), 0)


def vs_K(lres, kres, paper_root, R, out, D, p, settings):
    present = [(rule, lw) for rule, lw in settings if any(R['runs'][f'{setting_id(D, p, rule, lw)}_s{i}']['state'] == 'finished' for i in (1, 2, 3))]
    if not present: return
    pap = AL.paper_runs(paper_root).get(f'D{D}_p{p}', [])
    fig, axs = plt.subplots(2, len(present), figsize=(1.6 * len(present) + 1.3, 3.7), layout='constrained', squeeze=False, sharey='row')
    for j, (rule, lw) in enumerate(present):
        st = setting_id(D, p, rule, lw); ids = [f'{st}_s{i}' for i in (1, 2, 3) if R['runs'][f'{st}_s{i}']['state'] == 'finished']
        lf = [os.path.join(lres, i, 'run.json.gz') for i in ids]
        kf = pap if rule == 'paper' else [os.path.join(kres, 'K_' + i[2:], 'run.json.gz') for i in ids]
        xmax = max(len(E.load(f)['loss']) for f in lf) - 1
        for fns, light, lab in ((kf, True, '0.8'), (lf, False, '0.5')):
            if not fns: continue
            g, ce, te, rf, nn = med_curves(fns, xmax)
            al = 0.45 if light else 1.0
            axs[0, j].plot(g, ce, color=A.K['train'], lw=1.3, alpha=al, ls='--' if light else '-')
            axs[1, j].plot(g, te, color=A.K['test'], lw=1.3, alpha=al, ls='--' if light else '-')
            axs[1, j].plot(g, rf, color=A.K['refit'], lw=1.2, alpha=al, ls='--' if light else '-')
        nk = len(kf)
        axs[0, j].set_title(f"{RULE_TXT[rule].split(' (')[0]}, $\\lambda_W$ {lw:g}\n$\\lambda_V$ = {lam_v(p, rule):g}; 0.5: {len(ids)} runs, 0.8: {nk}", fontsize=6.6)
        for ax in axs[:, j]:
            ax.set_xlim(0, xmax); ax.xaxis.set_major_formatter(A.KFMT); ax.grid(True, axis='y', color=A.K['grid'], lw=0.6)
        axs[0, j].set_ylim(0, 1.05); axs[1, j].set_ylim(-0.02, 1.04); axs[1, j].set_xlabel('Updates (thousands)')
    axs[0, 0].set_ylabel('Training CE / log p\n(median, cycle mean)'); axs[1, 0].set_ylabel('Accuracy (median)')
    h = [Line2D([], [], color=A.K['train'], lw=1.3), Line2D([], [], color=A.K['test'], lw=1.3), Line2D([], [], color=A.K['refit'], lw=1.2),
         Line2D([], [], color='#555555', lw=1.3), Line2D([], [], color='#555555', lw=1.3, ls='--', alpha=0.45)]
    fig.legend(h, ['Train CE', 'Held-out', 'Refit', 'frac 0.5 (L)', 'frac 0.8 (K same seeds; paper setting F/G)'],
               loc='outside lower center', ncol=5 if len(present) >= 4 else 3, frameon=False, fontsize=6.6)
    fig.suptitle(f'Depth {D}, p = {p}: training fraction 0.5 against 0.8', fontsize=8)
    stem = f'L_vs_K_D{D}_p{p}'
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'plot_L.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig); print('drew', stem, flush=True)


def main(lres, kres, paper_root, runs_json, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); R = json.load(open(runs_json))
    board(R, out, 'L_scoreboard', 'endpoint'); board(R, out, 'L_refit_scoreboard', 'refit')
    depth_figures(lres, R, out / 'figures', 'endpoint'); depth_figures(lres, R, out / 'figures_refit', 'refit')
    for D in (3, 4):
        for p in PRIMES: vs_K(lres, kres, paper_root, R, out, D, p, SETTINGS)


if __name__ == '__main__':
    main(*sys.argv[1:6])
