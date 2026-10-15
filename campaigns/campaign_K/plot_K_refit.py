#!/usr/bin/env python3
"""Campaign K and the paper's deep runs scored with the refit criterion (claude/1hl-figures, 2026-10-08, Akash's criterion):

    "in a flat or rising training loss, the refit gets better, and held-out accuracy goes up in the same window or later".

    python3 -I plot_K_refit.py K_RESULTS PAPER_RUNS OUT_DIR

K_RESULTS holds campaign K's <run>/run.json.gz; PAPER_RUNS holds the unpacked campaign D, F and G tars (any depth of folders;
only depth-3 and depth-4 runs at p = 23, 47, 61, 101 with lam_v = 0.2 and the paper's r are used: 10 seeds per cell).

Per run, with W = [a, b] the paper's post-hoc window (Appendix F.4; acrossp_figs.window_record): the stretch after the knee in
which the training CE (cycle mean) never falls more than 3% below its running maximum, the one with the largest CE rise.
  pass   refit accuracy rises by >= 0.2 inside W, held-out accuracy rises by >= 0.2 from a to the end of the run, and the final
         held-out accuracy is >= 0.99.
  pass3  pass, and training accuracy stays >= 0.97 at every logged step in W (at most 3% of training pairs misclassified).
Also recorded: when the refit and the held-out accuracy first reach 0.5 (the refit's lead).

Writes K_refit_scoreboard.{pdf,png}, K_refit_depth{3,4}_<setting>.{pdf,png} (shaded: median W), K_refit_depth{3,4}_paper.{pdf,png}
and K_refit_stats.json. Drawn with the paper's renderer (acrossp_figs.py next to this file, unchanged).
"""
import glob, json, math, os, statistics as S, sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import endpoints_i as E
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

GAIN, FINAL, TRAIN3 = 0.2, 0.99, 0.97
PRIMES = (23, 47, 61, 101)
SETTINGS = (('sqrt', 0.015), ('sqrt', 0.025), ('lin', 0.015), ('lin', 0.025))
RULE_TXT = dict(sqrt='$\\lambda_V = 0.2\\sqrt{31/p}$', lin='$\\lambda_V = 0.2\\cdot 31/p$')


def lam_v(p, rule): return round(0.2 * math.sqrt(31 / p), 4) if rule == 'sqrt' else round(0.2 * 31 / p, 4)


def at(s, v, t): return float(v[int(np.argmin(np.abs(s - t)))])


def first(s, v, thr):
    i = np.where(v >= thr)[0]; return int(s[i[0]]) if len(i) else None


def score(d):
    w = A.window_record(d); a, b = w['W2']
    s, te = E.series(d['obs'], 'test_acc'); st, tr = E.series(d['obs'], 'train_acc')
    sr, rf = E.series(d['obs'], 'v1_ridge_acc_te')
    k = (st >= a) & (st <= b); minacc = float(tr[k].min()) if k.any() else float('nan')
    final = float(te[-1]); rise = final - at(s, te, a)
    ok = bool(w['g_rf'] >= GAIN and rise >= GAIN and final >= FINAL)
    return dict(W=[int(a), int(b)], g_rf=w['g_rf'], g_te=w['g_te'], heldout_rise_from_a=rise, final=final, min_train_acc_in_W=minacc,
                ok=ok, ok3=bool(ok and minacc >= TRAIN3), t_refit50=first(sr, rf, 0.5), t_heldout50=first(s, te, 0.5),
                why=[n for n, c in (('refit', w['g_rf'] >= GAIN), ('held-out', rise >= GAIN), ('final', final >= FINAL)) if not c])


def drawn(d, rid, x):
    r = A.deep_run(d, rid, (x['W'], x['ok'], x['g_te'], x['g_rf'])); r['verdict'] = 'refit criterion'; return r


def paper_runs(root):
    out = {}
    for fn in sorted(glob.glob(os.path.join(root, '**', 'run.json.gz'), recursive=True)):
        d = E.load(fn); c = d['cfg']
        if c.get('lam_v') != 0.2 or c['p'] not in PRIMES or c['depth'] not in (3, 4): continue
        r = round(c['eta_layers'][-1] / (0.03 * 0.015 / 0.07), 2); want = 0.32 if (c['depth'] == 4 and c['p'] == 23) else 0.45
        if abs(r - want) > 1e-6: continue
        out.setdefault((c['depth'], c['p']), []).append((os.path.basename(os.path.dirname(fn)), fn))
    return out


def figure(cells, out, stem, sub):
    cols = []
    for p, runs in cells:
        rs = [r for _, r, _ in runs]
        c = A.column2(rs, f'p = {p}', [rid for rid, _, _ in runs])
        k = sum(x['ok'] for _, _, x in runs); c['title'] = f'p = {p}  ({k}/{len(runs)} pass)\n{sub(p)}'
        cols.append(c)
    A.draw2(cols, out, stem, rows=('ce', 'acc', 'err', 'ah'), height=5.6)


def scoreboard(table, out):
    rows = [(D, p) for D in (3, 4) for p in PRIMES]
    keys = ['paper'] + [f'{r}_{lw}' for r, lw in SETTINGS]
    heads = ['Paper setting\n$\\lambda_V$ = 0.2, $\\lambda_W$ = 0.015\n10 seeds'] + \
            [f'{RULE_TXT[r]}\n$\\lambda_W$ = {lw:g}\n3 seeds' for r, lw in SETTINGS]
    fig, ax = plt.subplots(figsize=(6.75, 4.6), layout='constrained')
    ax.set_xlim(0, len(heads)); ax.set_ylim(len(rows), -0.05); ax.axis('off')
    green = np.array(to_rgb(A.K['AH'])); white = np.ones(3)
    for j, h in enumerate(heads): ax.text(j + 0.5, -0.12, h, ha='center', va='bottom', fontsize=7.5)
    for i, (D, p) in enumerate(rows):
        ax.text(-0.06, i + 0.5, f'depth {D}, p = {p}', ha='right', va='center', fontsize=8)
        for j, key in enumerate(keys):
            xs = table[(D, p)][key]; n = len(xs); k = sum(x['ok'] for x in xs); k3 = sum(x['ok3'] for x in xs)
            frac = k / n; col = white * (1 - 0.85 * frac) + green * 0.85 * frac if frac > 0 else np.array([0.955] * 3)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.90, color=col, lw=0))
            tc = 'white' if frac > 0.6 else 'black'
            ax.text(j + 0.5, i + 0.27, f'{k}/{n}', ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
            ax.text(j + 0.5, i + 0.56, f'train err. $\\leq$ 3%: {k3}/{n}', ha='center', va='center', fontsize=6.6, color=tc)
            fails = sorted({w for x in xs for w in x['why']})
            if fails: ax.text(j + 0.5, i + 0.80, 'fails: ' + ', '.join(fails), ha='center', va='center', fontsize=6.2, color=tc)
        if i == 3: ax.plot([-1.2, len(heads)], [4, 4], color='#BBBBBB', lw=0.6, clip_on=False)
    fig.savefig(out / 'K_refit_scoreboard.pdf', metadata={'Title': 'K_refit_scoreboard', 'Author': '', 'Creator': 'plot_K_refit.py', 'CreationDate': None})
    fig.savefig(out / 'K_refit_scoreboard.png', dpi=200); plt.close(fig)


def main(kres, paper, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(os.path.join(HERE, 'manifest.json')))
    table, stats = {}, dict(criterion=[b for b in __doc__.split('\n\n') if b.startswith('Per run')][0], runs={})
    loaded = {}
    for r in man['runs']:
        d = E.load(os.path.join(kres, r['id'], 'run.json.gz')); x = score(d); stats['runs'][r['id']] = x
        key = f"{'sqrt' if r['dial']['lam_v_rule'] == 'sqrt' else 'lin'}_{r['dial']['lam_w']}"
        table.setdefault((r['depth'], r['p']), {}).setdefault(key, []).append(x); loaded.setdefault((r['depth'], r['p'], key), []).append((r['id'], drawn(d, r['id'], x), x)); del d
    for (D, p), lst in sorted(paper_runs(paper).items()):
        for rid, fn in lst:
            d = E.load(fn); x = score(d); stats['runs'][rid] = x
            table.setdefault((D, p), {}).setdefault('paper', []).append(x); loaded.setdefault((D, p, 'paper'), []).append((rid, drawn(d, rid, x), x)); del d
    scoreboard(table, out)
    for D in (3, 4):
        figure([(p, loaded[(D, p, 'paper')]) for p in PRIMES], out, f'K_refit_depth{D}_paper', lambda p: 'paper setting, $\\lambda_V$ = 0.2')
        for rule, lw in SETTINGS:
            key = f'{rule}_{lw}'
            figure([(p, loaded[(D, p, key)]) for p in PRIMES], out, f'K_refit_depth{D}_{rule}_lw{lw}', lambda p, r=rule: f'$\\lambda_V$ = {lam_v(p, r):g}')
    summ = {}
    for (D, p), t in sorted(table.items()):
        for key, xs in t.items():
            lead = [x['t_heldout50'] - x['t_refit50'] for x in xs if x['t_heldout50'] is not None and x['t_refit50'] is not None]
            summ[f'D{D}_p{p}_{key}'] = dict(n=len(xs), passes=sum(x['ok'] for x in xs), passes_train3=sum(x['ok3'] for x in xs),
                                           median_refit_lead=float(np.median(lead)) if lead else None,
                                           median_W=[float(np.median([x['W'][0] for x in xs])), float(np.median([x['W'][1] for x in xs]))],
                                           min_train_acc_in_W=float(min(x['min_train_acc_in_W'] for x in xs)))
    stats['summary'] = summ
    (out / 'K_refit_stats.json').write_text(json.dumps(stats, indent=1, default=float) + '\n')
    for k_, v in summ.items(): print(k_, v)


if __name__ == '__main__':
    main(*sys.argv[1:4])
