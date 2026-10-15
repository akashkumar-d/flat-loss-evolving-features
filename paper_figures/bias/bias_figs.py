#!/usr/bin/env python3
import argparse, glob, gzip, hashlib, json, math, os, sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

K, KFMT, LAST, FIRST = A.K, A.KFMT, A.LAST, A.FIRST
BIAS_LS = (0, (1.0, 1.1))
BIAS_LW = 1.5


class Store:
    def __init__(self, data, dev2, bias):
        self.path = {}
        for kind, sub, name in (('C', 'C', 'main.json.gz'), ('C', 'C5', 'main.json.gz'), ('D', 'D', 'run.json.gz'),
                                ('F', 'F', 'run.json.gz'), ('F', 'F_D2', 'run.json.gz'), ('G', 'G', 'run.json.gz')):
            for d in sorted(glob.glob(os.path.join(data, sub, '*'))):
                f = os.path.join(d, name)
                if os.path.exists(f): self._add(os.path.basename(d), kind, f)
        for f in sorted(glob.glob(os.path.join(dev2, 'seedD2_r0.5_s*.json'))): self._add(Path(f).stem, 'dev', f)
        for d in sorted(glob.glob(os.path.join(bias, 'B_*'))):
            f = os.path.join(d, 'main.json.gz' if os.path.exists(os.path.join(d, 'main.json.gz')) else 'run.json.gz')
            self._add(os.path.basename(d), 'bias', f)
        self.used = {}

    def _add(self, rid, kind, f):
        assert rid not in self.path, rid
        self.path[rid] = (kind, f)

    def load(self, rid):
        kind, f = self.path[rid]; raw = open(f, 'rb').read(); self.used[rid] = dict(kind=kind, sha256=hashlib.sha256(raw).hexdigest())
        return json.loads(gzip.decompress(raw) if f.endswith('.gz') else raw)


def scored(store, V, rid, cache):
    if rid not in cache:
        d = store.load(rid)
        if rid in V: r = A.deep_run(d, rid, V[rid]); r['verdict'] = 'v2 settings.jsonl'
        else:
            w = A.window_record(d); r = A.deep_run(d, rid, (w['W2'], w['ok2'], w['g_te'], w['g_rf'])); r['verdict'] = 'scored here'
        cache[rid] = r
    return cache[rid]


def bias_medians(store, ids, g):
    S = {k: [] for k in ('ce', 'test', 'refit', 'ahL', 'ah1')}; kinds = set()
    for rid in ids:
        d = store.load(rid); r = A.run_series(d); L = r['L']; c = (L[:-1] + L[1:]) / 2 / math.log(r['p'])
        S['ce'].append(A.on_grid(np.arange(len(c)), c, g)); kinds.add(r['refit_kind'])
        for k in ('test', 'refit', 'ahL'): S[k].append(A.on_grid(*r[k], g))
        if r['ah1'] is not None: S['ah1'].append(A.on_grid(*r['ah1'], g))
    out = {k: np.median(np.array(v), 0) for k, v in S.items() if v}
    out['n'] = len(ids); out['refit_kind'] = sorted(kinds)
    return out


def draw(cols, out, stem, rows, height, legend=True, overlay=None, creator='bias_figs.py'):
    overlay = overlay or {}
    n = len(cols)
    fig, axs = plt.subplots(len(rows), n, figsize=(6.75, height), sharey='row', layout='constrained')
    axs = np.atleast_2d(axs)
    ah_top = max(float(np.nanmax(c['S']['ahL'])) for c in cols) * 1.08
    err_top = max(c['ntr'] for c in cols) * 1.5
    for j, c in enumerate(cols):
        g, S, B = c['g'], c['S'], overlay.get(j)
        for i, row in enumerate(rows):
            ax = axs[i, j]
            if c['band'][1] - c['band'][0] > 0.004 * c['xmax']: ax.axvspan(*c['band'], color=K['window'], lw=0, zorder=0)
            ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_xlim(0, c['xmax'])
            ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(2 if n > 5 else 3, integer=True))
            thin = dict(lw=0.45, alpha=0.28, zorder=1)
            dot = dict(lw=BIAS_LW, ls=BIAS_LS, zorder=6)
            if row == 'ce':
                for y in S['ce']: ax.plot(g, y, color=K['train'], **thin)
                ax.plot(g, np.median(S['ce'], 0), color=K['train'], lw=1.4, zorder=3)
                if B is not None: ax.plot(g, B['ce'], color=K['train'], **dot)
                ax.set_ylim(0, 1.05)
            elif row == 'acc':
                for y in S['test']: ax.plot(g, y, color=K['test'], **thin)
                ax.plot(g, np.median(S['train'], 0), color=K['train_acc'], lw=1.0, ls='--', zorder=3)
                ax.plot(g, np.median(S['refit'], 0), color=K['refit'], lw=1.4, zorder=4)
                ax.plot(g, np.median(S['test'], 0), color=K['test'], lw=1.4, zorder=5); ax.set_ylim(-0.02, 1.04)
                if B is not None:
                    ax.plot(g, B['refit'], color=K['refit'], **dot); ax.plot(g, B['test'], color=K['test'], **dict(dot, zorder=7))
            elif row == 'ah':
                for y in S['ahL']: ax.plot(g, y, color=K['AH'], **thin)
                ax.plot(g, np.median(S['ahL'], 0), color=K['AH'], lw=1.4, ls=LAST, zorder=3)
                if S['ah1'] is not None: ax.plot(g, np.median(S['ah1'], 0), color=K['AH'], lw=1.2, ls=FIRST, zorder=3)
                if B is not None: ax.plot(g, B['ahL'], color=K['AH'], **dot)
                ax.set_ylim(0, ah_top)
            if i < len(rows) - 1: ax.tick_params(labelbottom=False)
        axs[0, j].set_title(c['title'], loc='center')
    lab = dict(ce='Training CE / log p\n(cycle mean)', acc='Accuracy', ah='Hidden Fourier\n$A_H$')
    if height < 4: lab.update(ce='Train CE\n/ log p', ah='Hidden\nFourier $A_H$')
    for i, row in enumerate(rows): axs[i, 0].set_ylabel(lab[row])
    for j in ((0, 1) if n == 2 else (n // 2,)): axs[-1, j].set_xlabel('Updates (thousands)')
    h = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
         Line2D([], [], color=K['test'], lw=1.4), Line2D([], [], color=K['refit'], lw=1.4)]
    l = ['Train CE', 'Train acc.', 'Held-out', 'Refit']
    if 'ah' in rows and any(c['S']['ah1'] is not None for c in cols):
        h += [Line2D([], [], color=K['layer'], lw=1.4, ls=LAST), Line2D([], [], color=K['layer'], lw=1.2, ls=FIRST)]
        l += ['Last hidden', 'First hidden']
    elif 'ah' in rows: h.append(Line2D([], [], color=K['AH'], lw=1.4)); l.append('$A_H$')
    if overlay: h.append(Line2D([], [], color=K['layer'], lw=BIAS_LW, ls=BIAS_LS)); l.append('With biases')
    if legend: fig.legend(h, l, loc='outside lower center', ncol=len(h) if len(h) <= 7 else (len(h) + 1) // 2, frameon=False,
                          handlelength=1.8, columnspacing=0.9, fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': creator, 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig)


H = lambda p, lw: [f'H_p{p}_lw{lw:.4f}_s{i}' for i in range(1, 11)]
C2 = lambda st, D: sorted(k for k in st.path if st.path[k][0] == 'C' and (k.startswith(f'C1_D{D}_s') or k.startswith(f'C2_D{D}_s')))
SPECS = {
    'fig2_acrossp_1hl': (lambda st: [('p = 23', H(23, 0.01), None),
                                     ('p = 31', H(31, 0.015), [f'B_H_p31_lw0.0150_s{i}' for i in range(1, 11)]),
                                     ('p = 47', H(47, 0.02), None), ('p = 61', H(61, 0.02), None), ('p = 101', H(101, 0.02), None)],
                         ('ce', 'acc', 'ah'), 2.6, 'p101_swap_20261006/fig2_acrossp_1hl', 'acrossp_figs.py'),
    'fig3_depths_2to5': (lambda st: [('Depth 2', [f'seedD2_r0.5_s{i}' for i in range(10)], [f'B_D2_r0.5_s{i}' for i in range(10)]),
                                     ('Depth 3', C2(st, 3), None),
                                     ('Depth 4', C2(st, 4), [f'B_D4_s{i:03d}' for i in range(1, 41)]),
                                     ('Depth 5', C2(st, 5), None)],
                         ('ce', 'acc', 'ah'), 2.6, 'restructure_20261005/fig3_depths_2to5', 'scheme2.py'),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--winv2', required=True)
    ap.add_argument('--bias', required=True); ap.add_argument('--dev2', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--no-bias', action='store_true'); ap.add_argument('--check-against')
    a = ap.parse_args()
    st = Store(a.data, a.dev2, a.bias); V = A.v2_verdicts(a.winv2); cache, report = {}, {}
    for stem, (spec, rows, height, paper, creator) in SPECS.items():
        cols, overlay, colstats = [], {}, []
        for j, (title, ids, bids) in enumerate(spec(st)):
            ids = [i for i in ids if i in st.path]
            c = A.column2([scored(st, V, i, cache) for i in ids], title, ids); cols.append(c)
            entry = dict(c['stats'])
            if bids and not a.no_bias:
                B = bias_medians(st, bids, c['g']); overlay[j] = B
                entry['with_bias'] = dict(n=B['n'], ids=bids, refit=B['refit_kind'])
            colstats.append(entry)
        name = stem if a.no_bias else f'{stem}_bias'
        draw(cols, a.out, name, rows, height, overlay=overlay or None, creator=creator if a.no_bias else 'bias_figs.py')
        report[name] = dict(paper_file=paper, columns=colstats)
        print(name, [(c['title'], c['stats']['passes'], c['stats']['n'], (overlay.get(j) or {}).get('n')) for j, c in enumerate(cols)], flush=True)
    (a.out / 'bias_figs_stats.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    (a.out / 'SOURCES.json').write_text(json.dumps(dict(note='SHA-256 of every run file read', runs=st.used), indent=1, sort_keys=True) + '\n')
    if a.check_against:
        for stem, (_, _, _, paper, _) in SPECS.items():
            mine, ref = a.out / f'{stem}.pdf', Path(a.check_against) / f'{paper}.pdf'
            if mine.exists() and ref.exists():
                same = mine.read_bytes() == ref.read_bytes()
                print(f'{stem}.pdf against the paper file {paper}.pdf: {"byte-identical" if same else "DIFFERENT"}')


if __name__ == '__main__':
    main()
