#!/usr/bin/env python3
import argparse, glob, gzip, hashlib, io, json, math, os, sys, tarfile, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import endpoints as E

C = dict(test='#009E73', train='#0072B2', refit='#E69F00', four='#7B4AB5', ce='#333333', window='#DDE3E8',
         grid='#E5E7EB', muted='#6B7280', text='#111827')
PS = [23, 31, 47, 61, 97]
CAMPAIGN_C_PASS = {'D3': '40/40', 'D4': '39/40'}
MLP_PLATEAU_P31 = (0.46, 0.58)
E_ENV = '#C4C9D1'
E_REF = '#E9B8BE'
plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 10, 'axes.labelsize': 9.5, 'legend.fontsize': 9, 'xtick.labelsize': 8.5,
                     'ytick.labelsize': 8.5, 'axes.edgecolor': '#9CA3AF', 'axes.linewidth': 0.8, 'figure.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})


class Runs:
    def __init__(self, root, kind):
        self.root, self.kind, self.idx, self.dups, self.chains = root, kind, {}, {}, {}
        if not root: return
        for d in sorted(glob.glob(os.path.join(root, '*'))):
            if os.path.isdir(d):
                for fn in ('run.json.gz', 'main.json.gz'):
                    if os.path.exists(os.path.join(d, fn)): self._add(os.path.basename(d), ('file', os.path.join(d, fn)))
                if os.path.exists(os.path.join(d, 'chain.json')): self.chains[os.path.basename(d)] = ('file', os.path.join(d, 'chain.json'))
        for tf in sorted(glob.glob(os.path.join(root, '*.tar')) + glob.glob(os.path.join(root, '*.tar.gz'))):
            with tarfile.open(tf) as t:
                for m in t.getmembers():
                    if not m.isfile(): continue
                    parts = m.name.split('/')
                    if parts[-1] == 'chain.json' and len(parts) >= 2: self.chains.setdefault(parts[-2], ('tar', tf, m.name))
                    if parts[-1] in ('run.json.gz', 'main.json.gz') and len(parts) >= 2: key = parts[-2]
                    elif m.name.endswith(('.json', '.json.gz')) and parts[-1] != 'chain.json': key = parts[-1].replace('.json.gz', '').replace('.json', '')
                    else: continue
                    self._add(key, ('tar', tf, m.name))
    def _add(self, key, src):
        if key in self.idx and self.idx[key] != src: self.dups.setdefault(key, [self.idx[key]]).append(src)
        else: self.idx.setdefault(key, src)
    def has(self, rid): return rid in self.idx
    def chain(self, rid):
        src = self.chains.get(rid)
        if src is None: return None
        if src[0] == 'file': return json.load(open(src[1]))
        with tarfile.open(src[1]) as t: return json.loads(t.extractfile(src[2]).read().decode())
    def load(self, rid):
        if rid in self.dups: raise SystemExit(f'ambiguous run {rid}: found in {self.dups[rid]}; point the option at one archive folder')
        src = self.idx[rid]
        if src[0] == 'file': raw = open(src[1], 'rb').read(); name = src[1]
        else:
            with tarfile.open(src[1]) as t: raw = t.extractfile(src[2]).read(); name = f'{os.path.basename(src[1])}:{src[2]}'
        if raw[:2] == b'\x1f\x8b': raw = gzip.decompress(raw)
        d = json.loads(raw.decode()); d['_source'] = dict(where=name, sha256=hashlib.sha256(raw).hexdigest())
        return d


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if key in o and o[key] is not None]
    return np.array([x[0] for x in s], float), np.array([x[1] for x in s], float)

def depth_of(cfg): return cfg.get('depth', cfg.get('n_layers'))

def dial_of(r): return r['dial'].get('lam_w', r['dial'].get('r'))


def d_rows(runs_d, analysis=None):
    if analysis and os.path.exists(analysis):
        rows = json.load(open(analysis))['rows']; rows = list(rows.values()) if isinstance(rows, dict) else rows
        return [r for r in rows if r.get('status') == 'done']
    out = []
    for rid in sorted(runs_d.idx):
        d = runs_d.load(rid); c = d['cfg']
        stage = 'H' if rid.startswith('H_') else 'HW' if rid.startswith('HW_') else rid.split('_')[0]
        dial = dict(lam_w=c['lam_w']) if stage in ('H', 'HW') else dict(r=round(c['eta_layers'][-1] * c['lam_layers'][-1] / E.K1, 4))
        out.append(dict(id=rid, stage=stage, p=c['p'], depth=c['depth'], width=c['width'], dial=dial, seed=int(rid.rsplit('_s', 1)[1]),
                        status='done', f=E.run_fields(d)))
    return out


def best_dials(rows, stage):
    out = {}
    for p in sorted({r['p'] for r in rows if r['stage'] == stage}):
        groups = {}
        for r in rows:
            if r['stage'] == stage and r['p'] == p: groups.setdefault(dial_of(r), []).append(r['f'])
        key = lambda d: (np.mean([f.get('share', 0) for f in groups[d]]), sum(bool(f.get('primary')) for f in groups[d]),
                         np.mean([f['final_test'] for f in groups[d]]))
        out[p] = max(groups, key=key)
    return out


def reasons(fs):
    got = {w.strip() for f in fs for w in (f.get('why_not') or '').split(',') if w.strip()}
    return ', '.join(o for o in ['share', 'fit', 'final', 'F02', 'no knee'] if o in got)


def sig(x, n=4): return [float(f'{float(v):.{n}g}') for v in x]


def chunk_stats(cm, xmax, st):
    k = max(1, xmax // st); seg = cm[:k * st].reshape(k, st)
    return list(range(0, k * st, st)), np.median(seg, axis=1), seg.min(axis=1), seg.max(axis=1)


def column_data(run, f, title, rid, log_ce=False):
    cfg = run['cfg']; D = depth_of(cfg); p = cfg['p']
    L = np.asarray(run['loss'], float); cm = 0.5 * (L[:-1] + L[1:])
    obs = run['obs']; s, te = series(obs, 'test_acc'); _, tr = series(obs, 'train_acc'); sr, rf = series(obs, 'refit_acc')
    sa, ahL = series(obs, f'AH_full_h{D}'); _, ah1 = series(obs, 'AH_full_h1')
    ntr = int(cfg.get('frac', 0.8) * p * p)
    a, b = f['W'] if f.get('W') else (None, None); t90 = f.get('t90') or f.get('t90_first')
    xmax = min(len(L) - 1, int(max(b or 0, t90 or 0, f.get('t10') or 0) * 1.15 + 1500))
    st = max(1, xmax // 3000); m = s <= xmax; mr = sr <= xmax; ma = sa <= xmax
    lst = lambda x: [round(float(v), 7) for v in x]
    if log_ce:
        ch = max(1, xmax // 400); cs, med, lo, hi = chunk_stats(cm / math.log(p), xmax, ch)
        ce = dict(step=cs, ce_over_logp=sig(med), lo=sig(lo), hi=sig(hi), chunk=ch,
                  stat=f'per chunk of {ch} updates: median (ce_over_logp), min (lo) and max (hi) of the two-update mean / log p')
    else:
        ce = dict(step=list(range(0, xmax, st)), ce_over_logp=lst(cm[:xmax:st] / math.log(p)))
    return dict(title=title, run=rid, source=run.get('_source'), p=p, depth=D, width=cfg.get('width', cfg.get('d_model')), xmax=xmax,
                band=[a, b], fields={k: f.get(k) for k in ('share', 'primary', 'why_not', 't10', 't90', 'CE_peak_over_logp', 'final_test',
                                                          'dAH_last', 'T_fit_first', 'knee', 'knee_kind', 'range_transition')},
                ce=ce,
                acc=dict(step=lst(s[m]), test=lst(te[m]), train=lst(tr[m]), train_errors=[int(v) for v in np.rint((1 - tr[m]) * ntr)], n_train=ntr),
                refit=dict(step=lst(sr[mr]), test=lst(rf[mr])),
                fourier=dict(step=lst(sa[ma]), last=lst(ahL[ma]), first=lst(ah1[ma]) if D and D > 1 else None))


def draw_grid(cols, out, title, note, log_ce=False):
    n = len(cols); W = 3.35 * n + 1.2
    fig, axs = plt.subplots(4, n, figsize=(W, 11.4), sharey='row', squeeze=False)
    fig.subplots_adjust(left=0.075 if n > 3 else 0.095, right=0.99, top=0.79 if log_ce else 0.83,
                        bottom=0.11 if log_ce else 0.085 if n > 3 else 0.14, hspace=0.32, wspace=0.12)
    deep = any((c['depth'] or 1) > 1 for c in cols)
    for j, c in enumerate(cols):
        a, b = c['band']; xmax = c['xmax']
        narrow = log_ce and a is not None and (b - a) < xmax / 150
        for i in range(4):
            ax = axs[i, j]
            if a is not None and not narrow: ax.axvspan(a, b, color=C['window'], alpha=0.75, lw=0, zorder=0)
            if narrow: ax.axvline(a, color='#7C8DA0', lw=0.9, ls=(0, (3, 2)), zorder=0.6)
            ax.grid(color=C['grid'], lw=0.5); ax.set_axisbelow(True); ax.set_xlim(0, xmax)
            if log_ce: ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1000:.0f}k'))
        if log_ce:
            ax = axs[0, j]; ce = c['ce']
            ax.axhspan(*MLP_PLATEAU_P31, color=E_REF, lw=0, zorder=0.5)
            ax.fill_between(ce['step'], ce['lo'], ce['hi'], color=E_ENV, lw=0, zorder=1, step='post')
            ax.plot(ce['step'], ce['ce_over_logp'], color=C['ce'], lw=0.8, zorder=2, drawstyle='steps-post')
            ax.set_yscale('log'); ax.set_ylim(1e-10, 3e4); ax.set_yticks([1e-9, 1e-6, 1e-3, 1, 1e3])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if narrow: ax.plot([a], [1], transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes), marker='v',
                               ms=7, color='#7C8DA0', clip_on=False, zorder=5)
        else:
            axs[0, j].plot(c['ce']['step'], c['ce']['ce_over_logp'], color=C['ce'], lw=1.3); axs[0, j].set_ylim(0, 1.05)
        ax = axs[1, j]
        ax.plot(c['acc']['step'], c['acc']['test'], color=C['test'], lw=1.5); ax.plot(c['acc']['step'], c['acc']['train'], color=C['train'], lw=1.3, ls='--')
        ax.plot(c['refit']['step'], c['refit']['test'], color=C['refit'], lw=1.4, ls=':'); ax.set_ylim(-0.02, 1.03)
        ax = axs[2, j]; ax.plot(c['acc']['step'], c['acc']['train_errors'], color=C['train'], lw=1.1)
        ax.set_yscale('symlog', linthresh=1, linscale=0.6); ax.set_ylim(-0.3, 12000)
        ax.set_yticks([0, 1, 10, 100, 1000, 10000]); ax.set_yticklabels(['0', '1', '10', '100', '1k', '10k'])
        ax = axs[3, j]; ax.plot(c['fourier']['step'], c['fourier']['last'], color=C['four'], lw=1.6)
        if c['fourier']['first'] is not None: ax.plot(c['fourier']['step'], c['fourier']['first'], color=C['four'], lw=1.2, ls='--')
        ax.set_xlabel('updates')
        f = c['fields']; ok = bool(f.get('primary'))
        axs[0, j].set_title(c['title'] + f' · share {100 * (f.get("share") or 0):.0f}%\n' + ('PASS' if ok else 'fail: ' + str(f.get('why_not')))
                            + (f"\n{c['band_note']}" if c.get('band_note') else ''),
                            fontsize=9.5, color='#8A5A00' if ok else C['text'], fontweight='bold' if ok else 'normal')
    for i, lab in enumerate(['training CE / log p\n(two-update mean)', 'accuracy', 'misclassified\ntraining pairs', r'Fourier fraction $A_H$']):
        axs[i, 0].set_ylabel(lab)
    if log_ce:
        h = [Line2D([], [], color=C['ce'], lw=1.1, label='training CE / log p: median over each chunk of updates'),
             Patch(fc=E_ENV, label='its min-max within the chunk (the loss swings over decades)'),
             Patch(fc=E_REF, label=f'MLP plateau level at p = 31, depths 1-5 ({MLP_PLATEAU_P31[0]:.2f}-{MLP_PLATEAU_P31[1]:.2f})')]
    else:
        h = [Line2D([], [], color=C['ce'], lw=1.3, label='training CE / log p')]
    h += [Line2D([], [], color=C['test'], lw=1.5, label='held-out accuracy (network)'),
          Line2D([], [], color=C['train'], lw=1.3, ls='--', label='training accuracy / misclassified training pairs'),
          Line2D([], [], color=C['refit'], lw=1.4, ls=':', label='held-out accuracy of a train-only ridge refit'),
          Line2D([], [], color=C['four'], lw=1.6, label=r'$A_H$, last hidden layer' if deep else r'$A_H$')]
    if deep: h.append(Line2D([], [], color=C['four'], lw=1.2, ls='--', label=r'$A_H$, first hidden layer'))
    h.append(Patch(fc=C['window'], alpha=0.75, label='loss-only peak band (cycle-mean CE within 3% of its peak)'))
    if log_ce: h.append(Line2D([], [], color='#7C8DA0', marker='v', ms=7, lw=0.9, ls=(0, (3, 2)), label='the same band, where narrower than a line'))
    fig.legend(handles=h, loc='lower center', ncol=5 if log_ce else 4 if n > 3 else 3, frameon=False, bbox_to_anchor=(0.53, 0.0),
               fontsize=9 if n > 3 else 8.3)
    x0 = 0.075 if n > 3 else 0.095
    fig.suptitle(title, x=x0, ha='left', fontsize=13, fontweight='bold')
    fig.text(x0, 0.94, '\n'.join(textwrap.fill(par, int(W * 12.8)) for par in note.split('\n')), fontsize=8.8, color=C['muted'], va='top')
    save(fig, out)


def save(fig, out):
    fig.savefig(out + '.png', dpi=200); fig.savefig(out + '.pdf'); plt.close(fig)


def fig_D_overview(rows, out):
    fig = plt.figure(figsize=(15.5, 11.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.32], hspace=0.34, wspace=0.16, left=0.05, right=0.985, top=0.855, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0]); H = [r for r in rows if r['stage'] == 'H']
    lams = sorted({dial_of(r) for r in H}); ramp = LinearSegmentedColormap.from_list('lam', ['#C9C3E6', '#2E1F73']); ends = []; data = dict(panel_a=[])
    for i, lam in enumerate(lams):
        col = ramp(i / max(1, len(lams) - 1)); xs, ys, ok = [], [], []
        for p in PS:
            fs = [r['f'] for r in H if r['p'] == p and dial_of(r) == lam]
            if not fs: continue
            xs.append(p); ys.append(float(np.mean([f['CE_peak_over_logp'] for f in fs]))); ok.append(any(f.get('primary') for f in fs))
            data['panel_a'].append(dict(lam_w=lam, p=p, ce_over_logp_mean=ys[-1], seeds=len(fs), any_pass=ok[-1]))
        ax.plot(xs, ys, '-', color=col, lw=1.8, zorder=2)
        for x, y, o in zip(xs, ys, ok): ax.plot(x, y, 'o', ms=8 if o else 6, mfc=col if o else 'white', mec=col, mew=1.6, zorder=3)
        ends.append([ys[-1], ys[-1], xs[-1], lam, col])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)): ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.042)
    shift = max(0.0, ends[-1][1] - 0.985)
    for e in ends:
        ax.annotate(rf'$\lambda_W$ = {e[3]:g}', (e[2], e[0]), xytext=(e[2] + 4, e[1] - shift), textcoords='data', va='center', fontsize=8.5,
                    color=C['text'], arrowprops=dict(arrowstyle='-', color=e[4], lw=0.7, shrinkA=0, shrinkB=3))
    ax.set_xticks(PS); ax.set_xlim(19, 118); ax.set_ylim(0, 1.0); ax.set_xlabel('prime p'); ax.set_ylabel('training CE at the band peak  /  log p')
    ax.grid(color=C['grid'], lw=0.6); ax.set_axisbelow(True)
    ax.set_title('(a) One hidden layer (width 512): the plateau sits higher as p grows', loc='left', fontweight='bold')
    ax.text(0.0, -0.17, 'Each point: mean of 2 seeds. log p is the CE of uniform predictions; the band peak is the top of the plateau.',
            transform=ax.transAxes, fontsize=8.5, color=C['muted'])
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=8, mfc='#4B3F8F', mec='#4B3F8F', label='at least one seed passes'),
                       Line2D([], [], ls='', marker='o', ms=6, mfc='white', mec='#4B3F8F', mew=1.6, label='no seed passes')], loc='lower right', frameon=False)
    cmap = LinearSegmentedColormap.from_list('share', ['#F3F6FB', '#9DB9E0', '#2A5DA8', '#0F2C5C'])
    for title, stage, cell in ((r'(b) One hidden layer: hidden decay $\lambda_W$', 'H', gs[0, 1]), ('(c) Depth 3 (width 256): last-layer rate r', 'D3', gs[1, 0]),
                               ('(d) Depth 4 (width 256): last-layer rate r', 'D4', gs[1, 1])):
        ax = fig.add_subplot(cell); cols = [(p, None) for p in PS] + ([(97, 1024), (97, 2048)] if stage == 'H' else [])
        R = [r for r in rows if r['stage'] == stage or (stage == 'H' and r['stage'] == 'HW')]
        dials = sorted({dial_of(r) for r in R}); ax.set_xlim(0, len(cols)); ax.set_ylim(len(dials), 0); cells = []
        for j, (p, w) in enumerate(cols):
            if stage in CAMPAIGN_C_PASS and p == 31:
                ax.add_patch(plt.Rectangle((j, 0), 1, len(dials), fc='#EEF0F3', ec='white', lw=2, hatch='///', zorder=1))
                ax.text(j + 0.5, len(dials) / 2, f'campaign C\nrecipe rate\n{CAMPAIGN_C_PASS[stage]} pass', ha='center', va='center', fontsize=9,
                        zorder=3, bbox=dict(fc='white', ec='none', pad=2.5))
                cells.append(dict(p=31, campaign_C=CAMPAIGN_C_PASS[stage])); continue
            for i, dv in enumerate(dials):
                fs = [r['f'] for r in R if r['p'] == p and dial_of(r) == dv and ((w is None and r['stage'] != 'HW') or r.get('width') == w and r['stage'] == 'HW')]
                if not fs: ax.add_patch(plt.Rectangle((j, i), 1, 1, fc='white', ec='#F3F4F6', lw=1)); continue
                sh = float(np.mean([f.get('share', 0) for f in fs])); k = sum(bool(f.get('primary')) for f in fs)
                cells.append(dict(p=p, width=w, dial=dv, share_mean=sh, passes=k, runs=len(fs), failed_checks=reasons(fs)))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=cmap(sh), ec='white', lw=2)); tc = 'white' if sh > 0.55 else C['text']
                ax.text(j + 0.5, i + 0.36, f'{100 * sh:.0f}%', ha='center', va='center', fontsize=10, fontweight='bold', color=tc)
                ax.text(j + 0.5, i + 0.70, f'{k}/{len(fs)} pass' if k else reasons(fs), ha='center', va='center', fontsize=7.6, color=tc,
                        fontweight='bold' if k else 'normal')
                if k: ax.add_patch(plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc='none', ec='#E69F00', lw=2.4, zorder=4))
        data[f'panel_{stage}'] = cells
        ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels([f'p = {p}' if w is None else f'p = {p}\nwidth {w}' for p, w in cols])
        ax.set_yticks(np.arange(len(dials)) + 0.5); ax.set_yticklabels([f'{d:g}' for d in dials])
        ax.set_ylabel(r'$\lambda_W$' if stage == 'H' else 'r'); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, loc='left', fontweight='bold')
    fig.suptitle('Campaign D (development sweep, 127 runs, none failed): where the loss-only plateau band holds the held-out rise',
                 x=0.05, ha='left', fontsize=13, fontweight='bold')
    note = ('Cells: mean share of the held-out 10%→90% rise that falls inside the loss-only peak band (cycle-mean training CE within 3% of its '
            'peak), over 2 seeds (1 seed at p = 97 for depths 3-4 and for widths 1024/2048). Orange frame: at least one run passes campaign C\'s '
            'per-run endpoint (share ≥ 75%, final held-out accuracy ≥ 0.99, last-layer ΔA_H ≥ 0.02, training set fit throughout the band). '
            'Otherwise the failed checks are listed: share; fit (some training pair misclassified in the band); final; F02 (ΔA_H too small).')
    fig.text(0.05, 0.94, textwrap.fill(note, 200), fontsize=8.8, color=C['muted'], va='top')
    save(fig, out); return data


def fig_D_traj(rows, runs_d, stage, out, title, note):
    best = best_dials(rows, stage); sel = [(p, stage, best[p]) for p in PS if p in best]
    if stage == 'H':
        hw = [r for r in rows if r['stage'] == 'HW' and r.get('width') == 2048]
        sel.append((97, 'HW', max(hw, key=lambda r: r['f'].get('share', 0))['dial']['lam_w']))
    cols = []
    for p, st, dv in sel:
        r = sorted([x for x in rows if x['stage'] == st and x['p'] == p and dial_of(x) == dv and (st != 'HW' or x.get('width') == 2048)],
                   key=lambda x: x['seed'])[0]
        dial = rf'$\lambda_W$ = {dv:g}' if st in ('H', 'HW') else f'r = {dv:g}'
        cols.append(column_data(runs_d.load(r['id']), r['f'], f"p = {p}{', width 2048' if st == 'HW' else ''}\n{dial} · seed {r['seed']}", r['id']))
    draw_grid(cols, out, title, note); return dict(columns=cols, selection='seed 1 at the setting with the highest mean share at each p')


def rate_r(cfg):
    if 'eta_layers' in cfg: return cfg['eta_layers'][-1] * cfg['lam_layers'][-1] / E.K1
    if 'eta_deep' in cfg: return cfg['eta_deep'] * cfg.get('lam_deep', cfg['lam_w']) / E.K1
    return None

def fig_depths_p31(rows, runs_d, runs_c, runs_dev, out):
    H = [r for r in rows if r['id'] == 'H_p31_lw0.0150_s1'][0]
    cols = [column_data(runs_d.load(H['id']), H['f'], 'depth 1 (width 512), campaign D\n' + r'$\lambda_W$ = 0.015 · seed 1', H['id'])]
    d2 = runs_dev.load('seedD2_r0.5_s0'); cols.append(column_data(d2, E.run_fields(d2), 'depth 2, development runs\nr = 0.5 · fresh seed 0', 'seedD2_r0.5_s0'))
    for D in (3, 4, 5):
        rid = f'C2_D{D}_s001'; run = runs_c.load(rid)
        cols.append(column_data(run, E.run_fields(run), f'depth {D}, campaign C\nrecipe r = {rate_r(run["cfg"]):.2f} · seed 1', rid))
    draw_grid(cols, out, 'Depths 1 to 5 at p = 31: the same learner, one run per depth',
              'p = 31, random 80% split. Depth 1: campaign D (width 512; 2/2 runs pass at this decay). Depth 2: development run (width 256; layer 2 '
              'shrinks at r = 0.5 times layer 1\'s rate; seed 0 of five fresh seeds, 4 of which pass the same per-run endpoint). Depths 3-5: campaign C, '
              'the lowest-index paired seed as in paper Figure 3 (width 256; last-layer rate r set per run from a pilot; 40/40, 39/40 and 38/40 runs pass).\n'
              'Row 3 counts misclassified training pairs on a log scale: 0 means the training set is fit.')
    return dict(columns=cols)

def fig_depth2_across_p(runs_dev, out):
    best, best_sh = None, -1
    for r in ('0.4', '0.45'):
        m = float(np.mean([E.run_fields(runs_dev.load(f'T16_D2_p23_lw0.015_r{r}_{sd}_T80k')).get('share', 0) for sd in ('s0', 's1')]))
        if m > best_sh: best, best_sh = r, m
    c23 = runs_dev.load(f'T16_D2_p23_lw0.015_r{best}_s0_T80k'); d2 = runs_dev.load('seedD2_r0.5_s0'); c47 = runs_dev.load('T3_D2_p47_r0.5_s0')
    cols = [column_data(c23, E.run_fields(c23), f'p = 23\nr = {float(best):g} · fresh seed 0 · 80k updates', f'T16_D2_p23_lw0.015_r{best}_s0_T80k'),
            column_data(d2, E.run_fields(d2), 'p = 31\nr = 0.5 · fresh seed 0', 'seedD2_r0.5_s0'),
            column_data(c47, E.run_fields(c47), 'p = 47\nr = 0.5 · fresh seed 0', 'T3_D2_p47_r0.5_s0')]
    draw_grid(cols, out, 'Depth 2 (width 256) across p: development runs',
              'Not a campaign: the rate r was tuned on a separate tuning seed at each p, and seed 0 is the first fresh seed. Layer 1 decay 0.015. '
              'p = 31 passes on 4 of 5 fresh seeds; p = 47 fails only the fit check (a training pair flips back to wrong in the band), as at depths 3-4 '
              'in campaign D; p = 23 fails the share. Depth 2 was not run at p = 61 or 97.\n'
              'Row 3 counts misclassified training pairs on a log scale: 0 means the training set is fit.')
    return dict(columns=cols, p23_rate_choice=f'r = {best} (higher mean share over seeds 0 and 1)')


E_ARMS = ['paper', 'hybrid', 'adamw']
E_ARM_NAME = {'paper': "paper's learner", 'hybrid': 'hybrid', 'adamw': 'AdamW'}

def e_cells(rows): return sorted({r['cell'] for r in rows}, key=lambda c: (c[:2], E_ARMS.index(c.split('_', 1)[1])))

def e_cell_label(c):
    D, arm = c.split('_', 1); return f"{D[1]} layer{'' if D[1] == '1' else 's'} · {E_ARM_NAME[arm]}"

def e_setting_label(rid):
    s, d = rid.split('_')[3:5]
    return f'init ×{s[1:]} · ' + (rf'$\lambda_W$ {d[2:]}' if d.startswith('lw') else f'wd {d[2:]}')

def e_rise(run, f):
    if f.get('t10') is None: return None
    L = np.asarray(run['loss'], float); cm = 0.5 * (L[:-1] + L[1:])
    end = f['t90_first'] if f.get('t90_first') is not None else len(cm) - 1
    seg = cm[f['t10']:end + 1] / math.log(run['cfg']['p'])
    return dict(t10=f['t10'], t90_first=f.get('t90_first'), end=int(end), median=float(np.median(seg)), q10=float(np.quantile(seg, 0.1)),
                q90=float(np.quantile(seg, 0.9)))

def e_rows(runs_e, manifest):
    import tfm_analyze as A
    man = json.load(open(manifest)); rows = []
    for ch in man['chains']:
        r = dict(id=ch['id'], stage=ch['stage'], cell=ch['cell'], opt=ch['opt'], n_layers=ch['n_layers'], seed=ch['seed'], status='missing')
        if runs_e.has(ch['id']):
            run = runs_e.load(ch['id']); f = A.fields(run); r.update(status='done', f=f, rise=e_rise(run, f))
        else:
            meta = runs_e.chain(ch['id'])
            if meta and meta.get('skipped'): r.update(status='skipped', reason=meta.get('reason'))
        rows.append(r)
    return rows, man

def e_dev(rows, c): return {r['id']: r for r in rows if r['cell'] == c and r['stage'] == 'dev' and r['status'] == 'done'}

def e_pick(rows, c):
    import tfm_analyze as A
    dev = e_dev(rows, c); return A.select_setting({k: v['f'] for k, v in dev.items()}) if dev else (None, False)

def e_band_note(f):
    a, b = f['W'] if f.get('W') else (None, None)
    if a is None: return 'no band (no knee)'
    where = f'update {a:,}' if a == b else f'updates {a:,}-{b:,}'
    if f.get('knee') == 0: return f'band: {where} (knee at update 0)'
    return f"band: {where} · CE {f['CE_peak_over_logp']:.2g} log p"

def fig_E_overview(rows, out):
    cells = e_cells(rows); ys, groups, y = {}, [], 0.0; data = dict(runs=[], cells=[], mlp_plateau_p31=list(MLP_PLATEAU_P31))
    for c in cells:
        dev = e_dev(rows, c); chosen, passed = e_pick(rows, c); fr = [r for r in rows if r['cell'] == c and r['stage'] == 'fresh']
        data['cells'].append(dict(cell=c, chosen=chosen, chosen_passed=passed, fresh_done=sum(r['status'] == 'done' for r in fr),
                                  fresh_pass=sum(1 for r in fr if r['status'] == 'done' and r['f'].get('primary')),
                                  fresh_skipped=sum(r['status'] == 'skipped' for r in fr), fresh_total=len(fr)))
        yh = y; y += 1.05
        for rid in sorted(dev): ys[rid] = y; y += 1
        groups.append((c, yh)); y += 0.45
    H = 11.2; fig = plt.figure(figsize=(15.5, H))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 1.0], wspace=0.05, left=0.125, right=0.995, top=0.835, bottom=0.1)
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1], sharey=axA); axC = fig.add_subplot(gs[2], sharey=axA)
    axA.axvspan(*MLP_PLATEAU_P31, color=E_REF, lw=0, zorder=0)
    axA.text(0.36, groups[0][1], 'MLP plateau at p = 31 →', ha='right', va='center', fontsize=8.5, color='#9B3D4A')
    for c, yh in groups:
        axA.annotate(e_cell_label(c), xy=(0.006, yh), xycoords=('figure fraction', 'data'), va='center', ha='left', fontsize=10,
                     fontweight='bold', color=C['text'])
        if yh > 0:
            for ax in (axA, axB, axC): ax.axhline(yh - 0.72, color='#D1D5DB', lw=0.9, zorder=0)
        for rid in sorted(e_dev(rows, c)):
            r = [x for x in rows if x['id'] == rid][0]; f = r['f']; rs = r['rise']; yy = ys[rid]; chosen = rid == e_pick(rows, c)[0]
            if rs:
                axA.plot([rs['q10'], rs['q90']], [yy, yy], color=C['ce'], lw=1.3, solid_capstyle='butt', zorder=2)
                axA.plot(rs['median'], yy, 'o', ms=6, color=C['ce'], zorder=3)
                t90 = rs['t90_first']; x1 = t90 if t90 is not None else 40000
                axB.plot([rs['t10'], x1], [yy, yy], color=C['test'], lw=2.2, solid_capstyle='butt', zorder=2)
                axB.plot(rs['t10'], yy, 'o', ms=5, color=C['test'], zorder=3)
                axB.plot(x1, yy, 'o' if t90 is not None else '>', ms=5 if t90 is not None else 6, color=C['test'],
                         mfc=C['test'] if t90 is not None else 'white', mew=1.4, zorder=3)
                if t90 is None: axB.text(x1 * 1.35, yy, f"90% not reached\n(final {f['final_test']:.2f})", va='center', fontsize=7.6, color=C['text'])
            if f.get('T_fit_first') is not None: axB.plot(f['T_fit_first'], yy, marker='|', ms=11, mew=2.0, color=C['train'], ls='', zorder=4)
            verdict = 'PASS' if f.get('primary') else 'fails: ' + str(f.get('why_not'))
            axC.text(0.02, yy, f"share {100 * (f.get('share') or 0):.0f}% · {verdict}", va='center', fontsize=8.4, color=C['text'])
            axC.text(0.63, yy, f"final {f['final_test']:.2f}", va='center', fontsize=8.4, color=C['text'])
            if chosen: axC.text(0.8, yy, "← rule's pick", va='center', ha='left', fontsize=8.4, color=C['text'], fontweight='bold')
            data['runs'].append(dict(id=rid, cell=c, y=yy, chosen=chosen, rise=rs, **{k: f.get(k) for k in (
                'share', 'primary', 'why_not', 'final_test', 't10', 't90', 't90_first', 'T_fit_first', 'knee', 'knee_kind', 'W', 'CE_peak_over_logp',
                'range_transition', 'refit_first', 'dAH_last')}))
    axA.set_yticks([ys[k] for k in ys]); axA.set_yticklabels([e_setting_label(k) for k in ys], fontsize=8.4); axA.set_ylim(y - 0.2, -0.6)
    axA.set_xscale('log'); axA.set_xlim(1e-9, 3); axA.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axA.set_xticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    axA.set_xlabel('training CE / log p while held-out accuracy rises from 10% to 90%\n(two-update mean; dot: median, line: 10%-90% quantiles)')
    axB.set_xscale('log'); axB.set_xlim(20, 4.5e5); axB.set_xticks([100, 1000, 10000, 40000]); axB.set_xticklabels(['100', '1k', '10k', '40k'])
    axB.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); axB.set_xlabel('updates (log scale)')
    axB.legend(handles=[Line2D([], [], color=C['test'], lw=2.2, marker='o', ms=5, label='held-out accuracy 10% → 90% (first crossings)'),
                        Line2D([], [], color=C['train'], marker='|', ms=11, mew=2.0, ls='', label='training set first fit')],
               loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=2, frameon=False, fontsize=8.4)
    for ax in (axA, axB): ax.grid(axis='x', color=C['grid'], lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0)
    plt.setp(axB.get_yticklabels(), visible=False); plt.setp(axC.get_yticklabels(), visible=False)
    axC.set_xlim(0, 1); axC.set_xticks([]); axC.tick_params(axis='y', length=0)
    for sp in axC.spines.values(): sp.set_visible(False)
    for ax in (axA, axB): ax.spines['left'].set_visible(False)
    axA.set_title('(a) Training loss during the held-out rise', loc='left', fontweight='bold')
    axB.set_title('(b) When held-out accuracy rises', loc='left', fontweight='bold')
    axC.set_title("(c) Campaign C's per-run endpoint", loc='left', fontweight='bold')
    nsk = sum(d['fresh_skipped'] for d in data['cells']); ntot = sum(d['fresh_total'] for d in data['cells'])
    fig.suptitle('Campaign E (transformer pilot, p = 31): no development run shows a loss plateau while held-out accuracy rises',
                 x=0.006, y=1 - 0.2 / H, ha='left', fontsize=13, fontweight='bold')
    note = ('Each row is one development run (tuning seed): 1 or 2 layers × three optimizers × four settings (initialization scale ×1 or ×4; hidden '
            'decay λ_W for the paper\'s learner and the hybrid, weight decay wd for AdamW). (a) The MLP runs at p = 31 (depths 1-5) sit on plateaus at '
            f'{MLP_PLATEAU_P31[0]:.2f}-{MLP_PLATEAU_P31[1]:.2f} × log p (shaded). (b) First fit: the first logged update at which every training pair is '
            'classified correctly. (c) Knee at the head-norm cap for the paper\'s learner (already met at initialization here) and at the first fit '
            f'otherwise. No run passes, so the rule ran no fresh seeds: {nsk} of {ntot} fresh chains recorded a skip.')
    fig.text(0.006, 1 - 0.62 / H, textwrap.fill(note, 215), fontsize=8.8, color=C['muted'], va='top')
    save(fig, out); return data

def fig_E_traj(rows, runs_e, out):
    cols = []
    for c in e_cells(rows):
        fr = sorted([r for r in rows if r['cell'] == c and r['stage'] == 'fresh' and r['status'] == 'done'], key=lambda r: r['seed'])
        if fr: r = fr[0]; lab = f'fresh seed {r["seed"]}'
        else:
            chosen, _ = e_pick(rows, c)
            if chosen is None: continue
            r = [x for x in rows if x['id'] == chosen][0]; lab = "rule's pick"
        col = column_data(runs_e.load(r['id']), r['f'], f"{e_cell_label(c)}\n{e_setting_label(r['id'])} · {lab}", r['id'], log_ce=True)
        col.update(band_note=e_band_note(r['f']), rise=r.get('rise')); cols.append(col)
    draw_grid(cols, out, 'Campaign E (transformer pilot, p = 31): one run per cell, at the setting the rule picks',
              'No development run passed, so no fresh seeds were run and each column is the development run the preregistered rule picks (every share is '
              '0%, so the tie-break decides: the first setting in sorted order among the runs that end at >= 0.99 held-out accuracy, else among all). Row 1 '
              'is on a log scale: once the training set is fit, the training CE drops far below the MLP plateau level (pink); it then swings over '
              'several decades from one update to the next (paper\'s learner) or spikes every 1.2k-1.5k updates (hybrid, AdamW). Fourier fraction of the '
              'last block\'s MLP activations at "=" (dashed: first block); refit on the final residual stream at "=".\n'
              'Row 3 counts misclassified training pairs on a log scale.', log_ce=True)
    return dict(columns=cols)

def fig_E_dev_all(rows, runs_e, out):
    cells = e_cells(rows); data = dict(runs=[], mlp_plateau_p31=list(MLP_PLATEAU_P31)); H = 2.55 * len(cells) + 2.1
    fig = plt.figure(figsize=(15.5, H))
    outer = fig.add_gridspec(len(cells), 1, hspace=0.34, left=0.095, right=0.985, top=1 - 2.0 / H, bottom=0.035)
    first = None; ch_used = None
    for i, c in enumerate(cells):
        inner = outer[i].subgridspec(2, 4, hspace=0.08, wspace=0.1, height_ratios=[1.15, 1]); chosen = e_pick(rows, c)[0]
        for j, rid in enumerate(sorted(e_dev(rows, c))):
            r = [x for x in rows if x['id'] == rid][0]; run = runs_e.load(rid); p = run['cfg']['p']
            L = np.asarray(run['loss'], float); cm = 0.5 * (L[:-1] + L[1:]); xmax = len(cm); ch = max(1, xmax // 400); ch_used = ch
            cs, med, lo, hi = chunk_stats(cm / math.log(p), xmax, ch)
            s, te = series(run['obs'], 'test_acc'); _, tr = series(run['obs'], 'train_acc'); sr, rf = series(run['obs'], 'refit_acc')
            a1 = fig.add_subplot(inner[0, j], sharex=first) if first else fig.add_subplot(inner[0, j]); first = first or a1
            a2 = fig.add_subplot(inner[1, j], sharex=first)
            a1.axhspan(*MLP_PLATEAU_P31, color=E_REF, lw=0, zorder=0)
            a1.fill_between(cs, lo, hi, color=E_ENV, lw=0, step='post', zorder=1); a1.plot(cs, med, color=C['ce'], lw=0.8, drawstyle='steps-post', zorder=2)
            a1.set_yscale('log'); a1.set_ylim(1e-10, 3e4); a1.set_yticks([1e-8, 1e-4, 1]); a1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            a2.plot(s, te, color=C['test'], lw=0.9); a2.plot(s, tr, color=C['train'], lw=0.8, ls='--'); a2.plot(sr, rf, color=C['refit'], lw=1.1, ls=':')
            a2.set_ylim(-0.03, 1.05); a2.set_yticks([0, 0.5, 1])
            for ax in (a1, a2):
                ax.set_xlim(0, xmax); ax.grid(color=C['grid'], lw=0.5); ax.set_axisbelow(True)
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1000:.0f}k' if v else '0'))
            plt.setp(a1.get_xticklabels(), visible=False)
            if i < len(cells) - 1: plt.setp(a2.get_xticklabels(), visible=False)
            else: a2.set_xlabel('updates')
            if j: plt.setp(a1.get_yticklabels(), visible=False); plt.setp(a2.get_yticklabels(), visible=False)
            else: a1.set_ylabel('CE / log p', fontsize=8.5); a2.set_ylabel('accuracy', fontsize=8.5)
            f = r['f']; tt = e_setting_label(rid) + (" · rule's pick" if rid == chosen else '') + f" · final {f['final_test']:.2f}"
            a1.set_title(tt, fontsize=9, loc='left', fontweight='bold' if rid == chosen else 'normal')
            lst = lambda x: [round(float(v), 7) for v in x]
            data['runs'].append(dict(id=rid, cell=c, chosen=rid == chosen, source=run.get('_source'), p=p,
                                     ce=dict(step=cs, ce_over_logp=sig(med), lo=sig(lo), hi=sig(hi), chunk=ch),
                                     acc=dict(step=lst(s), test=lst(te), train=lst(tr)), refit=dict(step=lst(sr), test=lst(rf)),
                                     fields={k: f.get(k) for k in ('share', 'primary', 'why_not', 'final_test', 't10', 't90', 't90_first', 'T_fit_first')}))
        bb = outer[i].get_position(fig)
        fig.text(0.006, (bb.y0 + bb.y1) / 2, e_cell_label(c), rotation=90, va='center', ha='left', fontsize=10, fontweight='bold', color=C['text'])
    fig.suptitle('Campaign E (transformer pilot, p = 31): all 24 development runs', x=0.006, y=1 - 0.2 / H, ha='left', fontsize=13, fontweight='bold')
    note = ('Tuning seed; 40k updates each. Upper panels: two-update mean training CE / log p on a log scale, with the MLP plateau level at p = 31 '
            f'(depths 1-5) for reference. Lower panels: accuracies. Every run fails campaign C\'s per-run endpoint (E_overview).')
    fig.text(0.006, 1 - 0.55 / H, textwrap.fill(note, 215), fontsize=8.8, color=C['muted'], va='top')
    h = [Line2D([], [], color=C['ce'], lw=1.1, label=f'training CE / log p: median over each {ch_used} updates'),
         Patch(fc=E_ENV, label='its min-max over those updates'), Patch(fc=E_REF, label='MLP plateau level at p = 31'),
         Line2D([], [], color=C['test'], lw=1.3, label='held-out accuracy'), Line2D([], [], color=C['train'], lw=1.1, ls='--', label='training accuracy'),
         Line2D([], [], color=C['refit'], lw=1.3, ls=':', label='held-out accuracy of the train-only ridge refit')]
    fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.095, 1 - 0.9 / H), ncol=3, frameon=False, fontsize=8.8, handlelength=1.8, columnspacing=2.0)
    data['chunk'] = ch_used
    save(fig, out); return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--campaign-c', help='folder with campaignC_C2_D{3,4,5}.tar (campaign C results archive)')
    ap.add_argument('--campaign-d', help='folder with the campaign D archives (campaignD_*.tar) or extracted runs, and its analysis.json')
    ap.add_argument('--development', help='folder with the development archives (*.tar.gz holding seedD2_*, T3_D2_*, T16_D2_* runs)')
    ap.add_argument('--campaign-e', help='folder with the campaign E archives or extracted runs (optional)')
    ap.add_argument('--e-manifest', help='campaign E manifest.json (default: <campaign-e>/manifest.json)')
    ap.add_argument('--out', required=True); ap.add_argument('--only', default='', help='comma-separated figure ids')
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, 'plotdata'), exist_ok=True)
    want = set(a.only.split(',')) - {''}
    go = lambda fid: not want or fid in want
    exported = {}
    def export(fid, data):
        data = dict(figure=fid, generated_by='make_figs_campaigns.py', definitions='lib/endpoints.py (campaigns C-E), lib/tfm_analyze.py (E)', **data)
        with open(os.path.join(a.out, 'plotdata', fid + '.json'), 'w') as f: json.dump(data, f, default=float)
        exported[fid] = os.path.join('plotdata', fid + '.json')
    rd = Runs(a.campaign_d, 'D'); rc = Runs(a.campaign_c, 'C'); rv = Runs(a.development, 'dev')
    rows = d_rows(rd, os.path.join(a.campaign_d, 'analysis.json')) if a.campaign_d else []
    note = ('Seed 1 at the setting with the highest mean share at each p (no selection by outcome among seeds). Row 3 counts misclassified '
            'training pairs on a log scale: 0 means the training set is fit.')
    if rows and go('D_overview'): export('D_overview', fig_D_overview(rows, os.path.join(a.out, 'D_overview')))
    if rows and go('D_traj_1hl'):
        export('D_traj_1hl', fig_D_traj(rows, rd, 'H', os.path.join(a.out, 'D_traj_1hl'), 'Campaign D, one hidden layer (width 512): best hidden decay at each p',
               note + '\nPasses only at p = 23 and 31, where the training set is fit within the first few hundred updates. At p = 47 and 61, tens to '
               'hundreds of training pairs are still wrong when the band starts; at p = 97 (last two columns; the last is width 2048) the held-out '
               'rise starts before the training set is fit.'))
    if rows and go('D_traj_depth3'):
        export('D_traj_depth3', fig_D_traj(rows, rd, 'D3', os.path.join(a.out, 'D_traj_depth3'), 'Campaign D, depth 3 (width 256): best last-layer rate at each p',
               note + '\np = 31 at depth 3 is campaign C (40/40 pass; paper Figure 3). At p = 47 and 61 the training set is fit by about 3-4k updates, '
               'then some pairs flip back to wrong (here 1 pair at p = 47, until the band starts; about 100 at p = 61, into the band). At p = 97 the '
               'training set is not fully fit until long after the held-out rise.'))
    if rows and go('D_traj_depth4'):
        export('D_traj_depth4', fig_D_traj(rows, rd, 'D4', os.path.join(a.out, 'D_traj_depth4'), 'Campaign D, depth 4 (width 256): best last-layer rate at each p',
               note + '\np = 31 at depth 4 is campaign C (39/40 pass; paper Figure 3). At p = 47 the only failed check is one training pair that is wrong '
               'for part of the band; at p = 61 about 100 pairs flip back to wrong through the band; at p = 97 the training set is never fully fit.'))
    if rows and a.campaign_c and a.development and go('depths_1to5_p31'):
        export('depths_1to5_p31', fig_depths_p31(rows, rd, rc, rv, os.path.join(a.out, 'depths_1to5_p31')))
    if a.development and go('depth2_across_p'): export('depth2_across_p', fig_depth2_across_p(rv, os.path.join(a.out, 'depth2_across_p')))
    if a.campaign_e:
        sys.path.insert(0, os.path.join(HERE, 'lib'))
        re_ = Runs(a.campaign_e, 'E'); er, _ = e_rows(re_, a.e_manifest or os.path.join(a.campaign_e, 'manifest.json'))
        if any(r['status'] == 'done' for r in er):
            if go('E_overview'): export('E_overview', fig_E_overview(er, os.path.join(a.out, 'E_overview')))
            if go('E_traj'): export('E_traj', fig_E_traj(er, re_, os.path.join(a.out, 'E_traj')))
            if go('E_dev_all'): export('E_dev_all', fig_E_dev_all(er, re_, os.path.join(a.out, 'E_dev_all')))
    json.dump(exported, open(os.path.join(a.out, 'plotdata', 'INDEX.json'), 'w'), indent=1)
    print('wrote', ', '.join(sorted(exported)))


if __name__ == '__main__':
    main()
