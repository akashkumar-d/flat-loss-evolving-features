import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory as blend
from anat import load, series
from make_readme import bestwin, stats, share_of

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = json.load(open(os.path.join(HERE, 'plot_style_v1.json')))
S, EV, LW = STYLE['series'], STYLE['events'], STYLE['linewidths']
WINC = STYLE['window']['color']; GRID = STYLE['figure']['grid_color']
IQR_A = STYLE['bands']['iqr_alpha']; IND_A = 0.32
NEUTRAL, THRESH, BAR = S['total_or_neutral']['color'], S['random_or_threshold']['color'], '#8C99A6'
PURPLE_RAMP = ['#D9CCF0', '#B9A2E0', '#9B79CF', '#7B4AB5', '#5E3491', '#43226E', '#2B1449']

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.labelsize': 10.5, 'legend.fontsize': 9.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#444444', 'figure.facecolor': 'white'})

def _d4(rate, sd):
    from make_v12 import d4_candidates
    return next((c for c in d4_candidates(rate, sd) if os.path.exists(c)), d4_candidates(rate, sd)[0])

COHORTS = {
    'D2': dict(title='Depth 2, five fresh seeds', runs=[f'runs/seedD2_r0.5_s{s}.json' for s in range(5)], labels=[f'seed {s}' for s in range(5)]),
    'D3': dict(title='Depth 3, the last-layer-slow rule (1.0, 0.5)',
               runs=['runs/T8_D3_lastslow_r1.0_0.5_tune_T100k.json'] + [f'runs/T13_D3_lastslow_r1.0_0.5_s{s}_T100k.json' for s in range(3)],
               labels=['tuning seed', 'seed 0', 'seed 1', 'seed 2']),
    'D4': dict(title='Depth 4, each seed at its balancing rate',
               runs=[_d4('0.35', 'tune'), _d4('0.45', 's0'), _d4('0.4', 's1'), _d4('0.4', 's2')],
               labels=['tuning seed (0.35×)', 'seed 0 (0.45×)', 'seed 1 (0.4×)', 'seed 2 (0.4×)'],
               note='each last-layer rate found by dialing 0.3–0.5× and explained afterwards by the clock'),
    'D4c': dict(title='Depth 4, one rate for all seeds (0.35×)',
                runs=[_d4('0.35', 'tune'), _d4('0.35', 's0'), _d4('0.35', 's1'), _d4('0.35', 's2')],
                labels=['tuning seed', 'seed 0', 'seed 1', 'seed 2'], note='the rate that balances the tuning seed, applied to fresh seeds'),
    'T23': dict(title='Depth 5, recipe rates (T23)',
                runs=[f'runs/T23_D5_r1.0_1.0_1.0_{r}_s{s}_T70k.json' for r, s in (('0.285', 1), ('0.300', 2), ('0.315', 3), ('0.320', 4))],
                labels=['seed 1 (0.285×)', 'seed 2 (0.300×)', 'seed 3 (0.315×)', 'seed 4 (0.320×)'],
                note='each last-layer rate set before the run by the registered recipe, from the seed\'s t₁₀ at 0.35× in T22'),
    'T22': dict(title='Depth 5, one rate for all seeds (0.35×, T22)',
                runs=[f'runs/T22_D5_r1.0_1.0_1.0_0.35_s{s}_T70k.json' for s in (1, 2, 3, 4)], labels=[f'seed {s}' for s in (1, 2, 3, 4)]),
    'D3w512': dict(title='Depth 3, width 512, the cascade (0.9, 0.45)',
                   runs=['runs/T10_D3_w512_cascade_r0.9_0.45_tune_T60k.json', 'runs/T10_D3_w512_cascade_r0.9_0.45_s0_T60k.json',
                         'runs/T15_D3_w512_cascade_r0.9_0.45_s1_T60k.json', 'runs/T15_D3_w512_cascade_r0.9_0.45_s2_T60k.json'],
                   labels=['tuning seed', 'seed 0', 'seed 1', 'seed 2'], note='the rule 1.0 / 0.5 on three more seeds behaves the same, README §3.3'),
    'D4w512': dict(title='Depth 4, width 512, 120k steps (T24)',
                   runs=['runs/T24_D4_w512_r1.0_1.0_0.4_tune_T120k.json', 'runs/T24_D4_w512_r1.0_1.0_0.4_s0_T120k.json'], labels=['tuning seed', 'seed 0']),
}
ORDER = ['D2', 'D3', 'D4', 'T23', 'D3w512', 'D4w512', 'D4c', 'T22']

def rel_rates(c):
    D = c['depth']
    if 'eta_layers' in c:
        k = [e * l for e, l in zip(c['eta_layers'], c['lam_layers'])]; return [x / k[0] for x in k]
    return [1.0] + [c['eta_deep'] * c['lam_deep'] / (c['eta_w'] * c['lam_w'])] * (D - 1)

def auto_sub(fns, note=''):
    cf = [load(f)[0]['cfg'] for f in fns]; c = cf[0]; D = c['depth']
    R = [rel_rates(x) for x in cf]; f_ = lambda v: f'{v:.3g}'
    mids = sorted({round(v, 6) for r in R for v in r[1:-1]}); last = sorted({round(r[-1], 6) for r in R})
    if D == 2: rates = f"layer 2 at {f_(last[0])}" + (f"–{f_(last[-1])}" if len(last) > 1 else '') + "× layer 1's shrink rate η·λ"
    else:
        mid = ("layer 1's shrink rate η·λ" if mids == [1.0] else ' / '.join(f_(v) for v in R[0][1:-1]) + "× layer 1's shrink rate η·λ")
        rates = (f"layer 2 at {mid}" if D == 3 else f"layers 2–{D - 1} at {mid}") + f", the last at {f_(last[0])}" + (f"–{f_(last[-1])}" if len(last) > 1 else '') + '×'
    steps = sorted({len(load(f)[1]) - 1 for f in fns}); st = '/'.join(f'{s // 1000}k' for s in steps)
    return (f"Dense ReLU · depth {D} · width {c.get('width', 256)} · p = {c['p']} · {round(100 * c.get('frac', 0.8))}% split · large init (c = {c['init_scale']:g}, deep layers {c.get('deep_init_scale', 8):g}× He) · "
            f"{st} steps · {len(fns)}/{len(fns)} runs complete · support-Polar hidden and normalized head updates with fixed decay · no loss balancing · {rates}" + (f' · {note}' if note else ''))

def heavy(obs, key):
    s, v = series(obs, key); return s, v

def layer_series(obs, key, l):
    hv = [o for o in obs if key in o]
    return np.array([o['step'] for o in hv]), np.array([o[key][l] for o in hv], dtype=float)

def nearest(s, v, t):
    return float(v[int(np.argmin(np.abs(s - t)))])

def first_cross(s, v, thr, after=-1):
    i = np.where((v >= thr) & (s > after))[0]
    return int(s[i[0]]) if len(i) else None

def run_fields(fn):
    d, L, obs, st = load(fn); c = d['cfg']; T = len(L) - 1; D = c['depth']; p = c['p']
    s, te = series(obs, 'test_acc'); _, tra = series(obs, 'train_acc'); _, mtr = series(obs, 'train_margin_min')
    sh, rf = series(obs, 'refit_acc'); ah = {l: series(obs, f'AH_full_h{l}') for l in range(1, D + 1)}
    pos = mtr > 0
    Tfit = int(s[np.argmax(pos)]) if pos.any() else None
    relapse = s[(s > (Tfit or 0)) & ~pos]
    Tfit_persist = int(s[np.where(~pos)[0].max() + 1]) if (~pos).any() and np.where(~pos)[0].max() + 1 < len(s) else Tfit
    Dly = max(1000, math.ceil(0.02 * T))
    t10 = first_cross(s, te, 0.1); tg = first_cross(s, te, 0.9)
    early90 = bool(tg is not None and np.any(te[s < tg] >= 0.9))
    Tref = first_cross(sh, rf, 0.9)
    def bad_prefix(t_start):
        if t_start is None: return False
        m = (s >= t_start) & (s <= t_start + Dly)
        return bool(m.any() and np.all(te[m] <= 0.2) and s[m].max() >= t_start + Dly - 20)
    eps = []
    for t_ in relapse:
        if eps and t_ - eps[-1][1] <= 20: eps[-1][1] = int(t_)
        else: eps.append([int(t_), int(t_)])
    out = dict(run=os.path.basename(fn)[:-5], p=p, depth=D, width=c['width'], horizon=T, T_fit_first=Tfit, fit_relapses_after_first=int(len(relapse)), fit_relapse_episodes=eps,
               last_relapse=int(relapse.max()) if len(relapse) else None, T_fit_persistent=Tfit_persist, bad_test_delay_required=Dly,
               bad_test_delay_from_first_fit=bad_prefix(Tfit), bad_test_delay_from_persistent_fit=bad_prefix(Tfit_persist),
               t10=t10, T_grok_first90=tg, earlier_90=early90, T_refit90=Tref, final_test=float(te[-1]))
    wins = {}
    a, b, share, lev, R = bestwin(fn); wins['selected'] = (a, b)
    o_ = stats(fn); wins['loss_alone'] = tuple(o_['fw'])
    if t10 is not None and tg is not None: wins['transition'] = (t10, tg)
    elif t10 is not None: wins['transition_to_end'] = (t10, T)
    for name, (a, b) in wins.items():
        seg = L[a:b + 1]; cm = (seg[:-1] + seg[1:]) / 2
        m = (s >= a) & (s <= b)
        dAH = nearest(*ah[D], b) - nearest(*ah[D], a); dAH1 = nearest(*ah[1], b) - nearest(*ah[1], a)
        r0, r1 = nearest(sh, rf, a), nearest(sh, rf, b)
        Tr_in = first_cross(sh, rf, 0.9, after=a)
        useful = bool(r0 <= 0.70 and Tr_in is not None and Tr_in <= b and nearest(sh, rf, Tr_in) - r0 >= 0.20)
        vmin = (seg.max() - seg.min()) / seg.min(); vmean = (seg.max() - seg.min()) / seg.mean()
        fit_strict = bool(np.all(mtr[m] > 0))
        grok_in = bool(tg is not None and a <= tg <= b)
        test0 = nearest(s, te, a)
        persists = bool(Tfit is not None and not np.any(~pos[(s > Tfit) & (s <= b)]))
        core = bool(vmin <= 0.10 and dAH >= 0.02 and fit_strict and test0 <= 0.20 and grok_in and not early90)
        joint = bool(core and persists and bad_prefix(Tfit))
        joint_pers = bool(core and bad_prefix(Tfit_persist))
        lead = bool(useful and Tr_in is not None and tg is not None and a < Tr_in < tg and nearest(s, te, Tr_in) <= 0.20)
        rep_first = bool(joint and lead); rep_first_pers = bool(joint_pers and lead)
        wins[name] = dict(t0=a, t1=b, share_of_rise=float(share_of(o_, a, b)), CE_min=float(seg.min()), CE_mean=float(seg.mean()), CE_max=float(seg.max()), CE_over_logp=float(seg.mean() / math.log(p)),
                          v_min=float(vmin), v_mean=float(vmean), v_min_cycle=float((cm.max() - cm.min()) / cm.min()),
                          drift=float((seg[-1] - seg[0]) / seg.mean()), P10_raw=bool(vmin <= 0.10), P03_raw=bool(vmin <= 0.03), P01_raw=bool(vmin <= 0.01),
                          AH_last_start=nearest(*ah[D], a), AH_last_end=nearest(*ah[D], b), delta_AH_last=float(dAH), delta_AH_first=float(dAH1),
                          F02=bool(dAH >= 0.02), F05=bool(dAH >= 0.05), F10=bool(dAH >= 0.10), F15=bool(dAH >= 0.15),
                          test_start=test0, test_end=nearest(s, te, b), refit_start=r0, refit_end=r1, fit_strict=fit_strict,
                          T_grok_inside=grok_in, first_fit_persists=persists, useful_refit=useful, T_refit90_inside=Tr_in if (Tr_in is not None and Tr_in <= b) else None,
                          test_at_T_refit90=nearest(s, te, Tr_in) if Tr_in is not None else None,
                          joint_PAG10_F02=joint, joint_PAG10_F02_from_persistent_fit=joint_pers,
                          representation_first=rep_first, representation_first_from_persistent_fit=rep_first_pers,
                          agop_last_gain=nearest(*series(obs, 'agop4_ho'), b) - nearest(*series(obs, 'agop4_ho'), a),
                          agop_last_rand_gain=nearest(*series(obs, 'agop4_ho_rand'), b) - nearest(*series(obs, 'agop4_ho_rand'), a),
                          agop_first_gain=(nearest(*series(obs, 'agopH1_4_ho'), b) - nearest(*series(obs, 'agopH1_4_ho'), a)) if D >= 2 else None,
                          agop_first_rand_gain=(nearest(*series(obs, 'agopH1_4_ho_rand'), b) - nearest(*series(obs, 'agopH1_4_ho_rand'), a)) if D >= 2 else None)
    out['windows'] = {k: v for k, v in wins.items() if isinstance(v, dict)}
    out['observers_not_measured'] = ['train-only ridge held-out MSE (only its accuracy was logged)', 'signed rule / residual logit margins',
                                     'difference-mode Fourier fraction', 'fixed-axis AGOP', 'class-centered canonical AGOP (legacy raw-logit V^T V logged)']
    return out

def style_ax(ax, title, xlabel='optimization step (thousands)'):
    ax.set_title(title, loc='center', pad=8)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel)

def kx(s): return np.asarray(s) / 1000.0

def cohort_lines(ax, xs, ys, key, med=True, lw_med=None, label=None, zorder=3, ls=None, color=None):
    col = color or S[key]['color']; ls = ls or S[key]['linestyle']
    for x, y in zip(xs, ys):
        ax.plot(kx(x), y, color=col, lw=LW['individual'], alpha=IND_A, ls='-', zorder=zorder - 1)
    if med and len(ys) > 1:
        n = min(len(y) for y in ys); Y = np.stack([y[:n] for y in ys]); x = xs[0][:n]
        q1, q2, q3 = np.percentile(Y, [25, 50, 75], axis=0)
        ax.fill_between(kx(x), q1, q3, color=col, alpha=IQR_A, lw=0, zorder=zorder - 1)
        ax.plot(kx(x), q2, color=col, lw=lw_med or LW['main'], ls=ls, label=label, zorder=zorder)
    elif len(ys) == 1:
        ax.plot(kx(xs[0]), ys[0], color=col, lw=lw_med or LW['main'], ls=ls, label=label, zorder=zorder)

def win_bars(ax, wins, y0=0.975, dy=0.028, label=False):
    tr = blend(ax.transData, ax.transAxes)
    for i, (a, b) in enumerate(wins):
        y = y0 - i * dy
        ax.plot([a / 1000, b / 1000], [y, y], transform=tr, color=BAR, lw=2.4, solid_capstyle='butt', zorder=6, clip_on=False)
        for t in (a, b): ax.plot([t / 1000] * 2, [y - 0.009, y + 0.009], transform=tr, color=BAR, lw=1.0, zorder=6, clip_on=False)
    if label:
        ax.text(wins[0][1] / 1000 + 0.8, y0, 'per-run windows', transform=tr, fontsize=8.5, color='#5B6773', va='center', zorder=6)

def not_measured(ax, title, why):
    style_ax(ax, title, xlabel=None); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.5, 0.5, 'not measured\n\n' + why, ha='center', va='center', fontsize=10, color='#5B6773', transform=ax.transAxes, wrap=True)

def header(fig, title, sub, sub2):
    import textwrap
    fig.suptitle(title, fontsize=16, y=0.99)
    fig.text(0.5, 0.953, '\n'.join(textwrap.wrap(sub, 200) + textwrap.wrap(sub2, 200)), ha='center', va='top', fontsize=9.2, color='#5B6773', linespacing=1.4)

def footer(fig, lines, width=205):
    import textwrap
    out = [w for l in lines for w in textwrap.wrap(l, width)]
    fig.text(0.5, 0.010, '\n'.join(out), ha='center', va='bottom', fontsize=8.6, color='#5B6773', linespacing=1.4)
    return len(out)

def sheets(key):
    C = dict(COHORTS[key]); fns = [f for f in C['runs'] if os.path.exists(f)]
    C.setdefault('sub', auto_sub(fns, C.get('note', '')))
    runs = [load(f) for f in fns]; F = [run_fields(f) for f in fns]
    D = runs[0][0]['cfg']['depth']; p = runs[0][0]['cfg']['p']; lp = math.log(p)
    wins = [(f['windows']['selected']['t0'], f['windows']['selected']['t1']) for f in F]
    T = max(len(r[1]) - 1 for r in runs)
    sub2 = ('Bars: each run\'s ≤3% CE window holding most of its test rise, chosen with test accuracy in view (exploratory) · sheet 2, panel F also outlines the longest ≤3% window '
            'from the loss alone · thin: runs; thick: medians; bands: IQR')

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); ax = axs.ravel()
    header(fig, f'{C["title"]}: loss, Fourier alignment and test recovery', C['sub'], sub2)
    a = ax[0]
    cohort_lines(a, [np.arange(len(r[1])) for r in runs], [r[1] for r in runs], 'actual_train', label='actual training CE')
    ts = [series(r[2], 'test_loss') for r in runs]
    cohort_lines(a, [t[0] for t in ts], [t[1] for t in ts], 'actual_test', label='actual held-out CE')
    a.axhline(lp, color=THRESH, lw=1.3, ls=':', zorder=2); a.text(1.01, lp, 'log p', transform=blend(a.transAxes, a.transData), ha='left', va='center', fontsize=8.5, color='#5B6773')
    a.set_yscale('log'); a.set_ylabel('cross-entropy (log scale; no floor)'); style_ax(a, 'A. Raw losses'); a.legend(loc='lower left', frameon=False)
    win_bars(a, wins)
    a = ax[1]
    tr = [series(r[2], 'train_acc') for r in runs]; te = [series(r[2], 'test_acc') for r in runs]; rf = [series(r[2], 'refit_acc') for r in runs]
    cohort_lines(a, [t[0] for t in tr], [t[1] for t in tr], 'actual_train', label='actual training')
    cohort_lines(a, [t[0] for t in te], [t[1] for t in te], 'actual_test', label='actual held-out')
    cohort_lines(a, [t[0] for t in rf], [t[1] for t in rf], 'refit_test', label='ridge refit, held-out')
    a.axhline(1 / p, color=THRESH, lw=1.0, ls=':', zorder=2, label='chance 1/p')
    a.set_ylim(-0.02, 1.02); a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{100 * v:.0f}%'))
    a.set_ylabel('accuracy'); style_ax(a, 'B. Actual predictions and train-only refit'); a.legend(loc='center left', bbox_to_anchor=(0.07, 0.56), frameon=False)
    win_bars(a, wins, label=False)
    a = ax[2]
    hL = [layer_like(r[2], f'AH_full_h{D}') for r in runs]; hl = [layer_like(r[2], 'AH_full_logit') for r in runs]
    cohort_lines(a, [h[0] for h in hL], [h[1] for h in hL], 'fourier_addition', label=f'hidden A_H, layer {D} (primary)')
    cohort_lines(a, [h[0] for h in hl], [h[1] for h in hl], 'total_or_neutral', ls='-.', label='actual-logit fraction')
    a.set_ylim(0, 1); a.set_ylabel('fraction of centered energy'); style_ax(a, 'C. Hidden and output Fourier fractions'); a.legend(loc='lower right', frameon=False)
    win_bars(a, wins, label=False)
    a = ax[3]
    g = [layer_like(r[2], 'agop4_ho') for r in runs]; gr = [layer_like(r[2], 'agop4_ho_rand') for r in runs]
    cohort_lines(a, [x[0] for x in g], [x[1] for x in g], 'agop_current', label='current axes (top-4 AGOP eigvecs)')
    cohort_lines(a, [x[0] for x in gr], [x[1] for x in gr], 'random_or_threshold', ls=':', label='matched random 4-frames (mean of 8)')
    a.set_ylim(0, 1); a.set_ylabel('q = 4 held-out-row purity'); style_ax(a, f'D. AGOP–Fourier, layer {D} (legacy)'); a.legend(loc='upper left', frameon=False, bbox_to_anchor=(0, 0.93))
    win_bars(a, wins, label=False)
    a = ax[4]
    mt = [series(r[2], 'train_margin_min') for r in runs]; me = [series(r[2], 'test_margin_min') for r in runs]
    cohort_lines(a, [x[0] for x in mt], [x[1] for x in mt], 'actual_train', label='minimum training margin')
    cohort_lines(a, [x[0] for x in me], [x[1] for x in me], 'actual_test', label='minimum held-out margin')
    a.axhline(0, color='#C9707A', lw=1.0, zorder=2); a.set_yscale('symlog', linthresh=1.0)
    a.set_ylabel('correct − strongest rival logit (symlog)'); style_ax(a, 'E. Actual minimum margins'); a.legend(loc='lower right', frameon=False)
    win_bars(a, wins, label=False)
    a = ax[5]
    nx, ny = [], []
    for r, f in zip(runs, F):
        s_, v_ = layer_series(r[2], 'normW', D - 1); nx.append(s_); ny.append(v_ / v_[-1])
    cohort_lines(a, nx, ny, 'total_or_neutral', label=f'‖W{sub_(D)}‖ / its final value')
    for r, f, x_, y_ in zip(runs, F, nx, ny):
        for t, mk in ((f['t10'], 'o'), (f['T_grok_first90'], 's')):
            if t is not None: a.plot(t / 1000, np.interp(t, x_, y_), mk, ms=5.5, mfc=S['actual_test']['color'] if mk == 's' else 'white', mec=S['actual_test']['color'], mew=1.4, zorder=5)
    a.plot([], [], 'o', ms=5.5, mfc='white', mec=S['actual_test']['color'], mew=1.4, label='held-out 10% (t₁₀)')
    a.plot([], [], 's', ms=5.5, mfc=S['actual_test']['color'], mec=S['actual_test']['color'], label='held-out 90% (T_grok)')
    a.set_yscale('log'); a.set_ylabel('norm ratio (log scale)'); style_ax(a, f'F. Layer {D} still shrinking through grokking'); a.legend(loc='center right', bbox_to_anchor=(1.0, 0.62), frameon=False)
    win_bars(a, wins, label=False)
    for a in ax: a.set_xlim(0, T / 1000)
    footer(fig, ['Every recorded step is plotted: training CE every update, held-out CE / accuracy / margins every 20 updates, Fourier / AGOP / ridge every 500. No smoothing, thinning or loss floor.',
                 f'A_H: full-grid hidden activations centered over the grid, energy in functions of (a+b) mod p. Actual-logit fraction: the same on class-centered logits. Legacy AGOP: Gram Vᵀ V of the raw logits (not class-centered), '
                 'top-4 eigenvectors mapped on held-out rows, purity O/r_Z.',
                 'Ridge refit: last hidden layer, training rows only, centered features and one-hot targets, penalty 10⁻³·tr(HᵀH)/m (a trace-relative rule, not the protocol\'s fixed α); it never replaces the head. '
                 'Not measured: ridge held-out MSE, signed rule / residual margins, difference-mode fraction, fixed-axis and class-centered AGOP.'])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.815, bottom=0.185, wspace=0.26, hspace=0.44)
    out1 = f'figP_{key}_overview.png'; fig.savefig(out1, dpi=STYLE['figure']['png_dpi']); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); ax = axs.ravel()
    header(fig, f'{C["title"]}: raw-loss zoom and diagnostics', C['sub'], sub2)
    lo = max(0, min(w[0] for w in wins) - 4000); hi = min(T, max(w[1] for w in wins) + 4000)
    a = ax[0]
    xs = [np.arange(lo, min(hi, len(r[1]) - 1) + 1) for r in runs]; ys = [r[1][lo:x[-1] + 1] for r, x in zip(runs, xs)]
    cohort_lines(a, xs, ys, 'actual_train', label='actual training CE')
    for r, (w0, w1) in zip(runs, wins):
        a.plot(np.arange(w0, w1 + 1) / 1000, r[1][w0:w1 + 1], color=S['actual_train']['color'], lw=1.1, alpha=0.75, zorder=4)
    a.set_xlim(lo / 1000, hi / 1000); a.set_ylabel('raw cross-entropy (linear)'); style_ax(a, 'A. Training CE, linear zoom')
    a.legend(loc='lower left', frameon=False); win_bars(a, wins)
    sec = a.secondary_yaxis('right', functions=(lambda v: v / lp, lambda v: v * lp)); sec.set_ylabel('× log p', color='#5B6773')
    a = ax[1]
    rel = []
    for r, (w0, w1) in zip(runs, wins):
        seg = r[1][w0:w1 + 1]; y = 100 * (seg / seg[0] - 1); rel.append(y)
        a.plot(np.arange(len(seg)) / 1000, y, color=S['actual_train']['color'], lw=LW['individual'] * 1.6, alpha=0.55, zorder=3)
    n = min(len(y) for y in rel); Y = np.stack([y[:n] for y in rel])
    a.fill_between(np.arange(n) / 1000, *np.percentile(Y, [25, 75], axis=0), color=S['actual_train']['color'], alpha=IQR_A, lw=0)
    a.plot(np.arange(n) / 1000, np.median(Y, 0), color=S['actual_train']['color'], lw=LW['main'], ls='--', label='median of per-run changes')
    a.axhline(0, color='#C9707A', lw=1.0); a.set_ylabel('100 × [L(t) / L(t₀) − 1]  (%)')
    style_ax(a, 'B. Relative change inside each window', xlabel='steps since the run\'s window start (thousands)'); a.legend(loc='lower left', frameon=False)
    a = ax[2]
    yy = np.arange(len(F))[::-1]
    v1 = [100 * f['windows']['selected']['v_min'] for f in F]
    v2 = [100 * (f['windows'].get('transition') or f['windows'].get('transition_to_end') or {}).get('v_min', np.nan) for f in F]
    open_end = ['transition' not in f['windows'] and 'transition_to_end' in f['windows'] for f in F]
    a.barh(yy + 0.19, v1, height=0.36, color=S['actual_train']['color'], alpha=0.85, label='selected ≤3% window')
    a.barh(yy - 0.19, v2, height=0.36, color=S['actual_train']['color'], alpha=0.35, hatch='///', edgecolor=S['actual_train']['color'], lw=0, label='test 10% → 90% transition')
    for y_, v_ in zip(yy, v1): a.text(v_ + 0.15, y_ + 0.19, f'{v_:.2f}%', va='center', fontsize=8.8)
    for y_, v_, oe in zip(yy, v2, open_end): a.text(min(v_, 12.2) + 0.15, y_ - 0.19, f'{v_:.2f}%' + (' (t₁₀ → end; never 90%)' if oe else ''), va='center', fontsize=8.8) if np.isfinite(v_) else None
    a.axvline(3, color=THRESH, ls=':', lw=1.3, label='3% (P03)'); a.axvline(10, color='#B5364A', ls='--', lw=1.3, label='10% (P10)')
    a.set_yticks(yy); a.set_yticklabels(C['labels'][:len(F)]); a.set_xlim(0, 12.5); a.set_ylim(-0.6, len(F) - 0.4 + 1.1)
    style_ax(a, 'C. Flatness of each run', xlabel='100 × (max CE − min CE) / min CE, every update'); a.legend(loc='upper center', ncol=2, frameon=False, fontsize=8.8)
    a.grid(axis='y', visible=False)
    a = ax[3]
    ramp = PURPLE_RAMP[1:] if D <= 6 else PURPLE_RAMP
    for l in range(1, D + 1):
        h = [layer_like(r[2], f'AH_full_h{l}') for r in runs]; col = ramp[min(l - 1, len(ramp) - 1)] if D > 1 else S['fourier_addition']['color']
        cohort_lines(a, [x[0] for x in h], [x[1] for x in h], 'fourier_addition', color=col, lw_med=LW['main'] if l == D else 1.6)
        n_ = min(len(x[1]) for x in h); a.text(T / 1000 * 0.985, float(np.median([x[1][n_ - 1] for x in h])) + 0.015, f'layer {l}' + (' (primary)' if l == D else ''), fontsize=8.8, ha='right', va='bottom', color=col)
    a.set_ylim(0, 1); a.set_xlim(0, T / 1000); a.set_ylabel('centered hidden addition fraction A_H'); style_ax(a, 'D. A_H in every hidden layer')
    win_bars(a, wins, label=False)
    a = ax[4]
    g = [layer_like(r[2], 'agopH1_4_ho') for r in runs]; gr = [layer_like(r[2], 'agopH1_4_ho_rand') for r in runs]
    cohort_lines(a, [x[0] for x in g], [x[1] for x in g], 'agop_current', label='current axes (top-4 hidden-coordinate AGOP)')
    cohort_lines(a, [x[0] for x in gr], [x[1] for x in gr], 'random_or_threshold', ls=':', label='matched random 4-frames (mean of 8)')
    a.set_ylim(0, 1); a.set_xlim(0, T / 1000); a.set_ylabel('q = 4 held-out-row purity'); style_ax(a, 'E. AGOP–Fourier, layer 1 (legacy)')
    a.legend(loc='upper left', frameon=False, bbox_to_anchor=(0, 0.93)); win_bars(a, wins, label=False)
    a = ax[5]
    for i, (f, (w0, w1)) in enumerate(zip(F, wins)):
        y = len(F) - 1 - i
        a.plot([w0 / 1000, w1 / 1000], [y, y], color=BAR, lw=7, solid_capstyle='butt', alpha=0.55, zorder=2)
        la_, lb_ = f['windows']['loss_alone']['t0'], f['windows']['loss_alone']['t1']
        a.add_patch(plt.Rectangle((la_ / 1000, y - 0.36), (lb_ - la_) / 1000, 0.12, fill=False, ec='#5B6773', lw=1.0, zorder=2))
        if f['T_fit_first'] is not None: a.plot(f['T_fit_first'] / 1000, y, marker='|', ms=14, mew=2, color=EV['train_fit']['color'], zorder=4)
        if f['T_refit90'] is not None: a.plot(f['T_refit90'] / 1000, y, marker='D', ms=6.5, color=EV['useful_refit']['color'], zorder=5)
        if f['t10'] is not None: a.plot(f['t10'] / 1000, y, marker='o', ms=7, mfc='white', mec=EV['actual_recovery']['color'], mew=1.6, zorder=5)
        if f['T_grok_first90'] is not None: a.plot(f['T_grok_first90'] / 1000, y, marker='o', ms=7, color=EV['actual_recovery']['color'], zorder=5)
    a.plot([], [], color=BAR, lw=7, alpha=0.55, label='selected window'); a.add_patch(plt.Rectangle((0, 0), 0, 0, fill=False, ec='#5B6773', lw=1.0, label='longest ≤3%, loss alone')); a.plot([], [], marker='|', ms=12, mew=2, ls='', color=EV['train_fit']['color'], label='training fitted (first)')
    a.plot([], [], marker='D', ls='', ms=6.5, color=EV['useful_refit']['color'], label='ridge refit ≥ 90%')
    a.plot([], [], marker='o', ls='', ms=7, mfc='white', mec=EV['actual_recovery']['color'], mew=1.6, label='actual held-out 10%')
    a.plot([], [], marker='o', ls='', ms=7, color=EV['actual_recovery']['color'], label='actual held-out 90%')
    a.set_yticks(range(len(F))); a.set_yticklabels(C['labels'][:len(F)][::-1]); a.set_ylim(-0.7, len(F) - 0.3 + 1.6); a.set_xlim(0, T / 1000)
    style_ax(a, 'F. Chronology of each run'); a.legend(loc='upper left', ncol=2, frameon=False, fontsize=8.6); a.grid(axis='y', visible=False)
    W = [f['windows']['selected'] for f in F]; Tw = [f['windows'].get('transition') for f in F]; La = [f['windows']['loss_alone'] for f in F]
    cnt = lambda ws, k: f"{sum(bool(w and w[k]) for w in ws)}/{len(ws)}"
    def rng(ws, k, s=100, fm='.2f'):
        v = [w[k] for w in ws if w and w.get(k) is not None]
        if not v: return 'n/a'
        lo_, hi_ = format(s * min(v), fm), format(s * max(v), fm)
        return lo_ if lo_ == hi_ else f'{lo_}–{hi_}'
    lines = [f'Protocol v1 fields, selected windows: P10_raw {cnt(W, "P10_raw")} · P03_raw {cnt(W, "P03_raw")} (v_min {rng(W, "v_min")}%; the windows were cut at 3% of the mean) · '
             f'share of the test rise inside {rng(W, "share_of_rise", fm=".0f")}% (longest ≤3% window from the loss alone: {rng(La, "share_of_rise", fm=".0f")}%) · '
             f'F02 on layer {D} {cnt(W, "F02")} (ΔA_H {rng(W, "delta_AH_last", 1)}) · fit in window {cnt(W, "fit_strict")} · 90% inside {cnt(W, "T_grok_inside")} · useful refit {cnt(W, "useful_refit")} '
             f'(refit at t₀ {rng(W, "refit_start", fm=".0f")}%) · joint_PAG10_F02 {cnt(W, "joint_PAG10_F02")} strict, {cnt(W, "joint_PAG10_F02_from_persistent_fit")} from the persistent fit.']
    Tv = [w for w in Tw if w]
    if Tv:
        tr_ = [w['T_refit90_inside'] for w in Tv if w['T_refit90_inside']]
        lines.append(f'Transition windows [t₁₀, T_grok] ({len(Tv)} of {len(F)} runs reach 90%; test-defined, a separate analysis): P10_raw {cnt(Tv, "P10_raw")} (v_min {rng(Tv, "v_min")}%) · '
                     f'P03_raw {cnt(Tv, "P03_raw")} · F02 {cnt(Tv, "F02")} (ΔA_H {rng(Tv, "delta_AH_last", 1)}) · useful refit {cnt(Tv, "useful_refit")} · '
                     f'joint_PAG10_F02 {cnt(Tv, "joint_PAG10_F02")} strict, {cnt(Tv, "joint_PAG10_F02_from_persistent_fit")} from the persistent fit · '
                     f'representation_first (from the persistent fit) {cnt(Tv, "representation_first_from_persistent_fit")}'
                     + (f': the ridge refit reaches 90% at {min(tr_):,}–{max(tr_):,} steps, when the actual head is at {rng(Tv, "test_at_T_refit90", fm=".0f")}% (the endpoint requires ≤ 20%).' if tr_ else '.'))
    rel = [f for f in F if f['fit_relapses_after_first']]
    early = [(f, e) for f in rel for e in f['fit_relapse_episodes'] if e[0] - f['T_fit_first'] <= 1000]
    late = [(f, e) for f in rel for e in f['fit_relapse_episodes'] if e[0] - f['T_fit_first'] > 1000]
    txt = f'Strict first-fit persistence: {len(F) - len(rel)}/{len(F)}'
    if early: txt += (f'; {len({id(f) for f, _ in early})} run(s) lose fit briefly right after first fitting at {min(f["T_fit_first"] for f, _ in early):,}–{max(f["T_fit_first"] for f, _ in early):,} steps '
                      f'(≤{max(e[1] - f["T_fit_first"] for f, e in early) + 20} steps)')
    if late: txt += '; later losses of training fit: ' + ', '.join(f'{C["labels"][F.index(f)]} {e[0]:,}–{e[1]:,}' for f, e in late)
    elif rel: txt += ', and none later'
    lines.append(txt + f'. Raw and 2-step cycle-mean ranges agree to {max(abs(w["v_min"] - w["v_min_cycle"]) for w in W) * 100:.2f} points. Fields per run: figP_{key}_fields.json.')
    footer(fig, lines)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.815, bottom=0.215, wspace=0.36, hspace=0.46)
    out2 = f'figP_{key}_diagnostics.png'; fig.savefig(out2, dpi=STYLE['figure']['png_dpi']); plt.close(fig)
    json.dump(dict(protocol='README_EXPERIMENT_PLOT_PROTOCOL.md v1.0-proposed; plot_style_v1.json sha256 1f75ad7541bbb69b...', cohort=key, runs=F),
              open(f'figP_{key}_fields.json', 'w'), indent=1)
    return out1, out2

def sub_(n): return str(n).translate(str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉'))

def layer_like(obs, key):
    s, v = series(obs, key); return s, v

SC_LABEL = {'D2': 'D2', 'D3': 'D3', 'D4': 'D4', 'T23': 'D5', 'D3w512': 'D3', 'D4w512': 'D4', 'D4c': 'D4', 'T22': 'D5'}
SC_GROUPS = [(('D2', 'D3', 'D4', 'T23'), 'width 256; per-seed rate at D4, recipe at D5'), (('D3w512', 'D4w512'), 'width 512'), (('D4c', 'T22'), 'width 256, one rate for all seeds')]

def scorecard(keys=None, out='figP_depth_scorecard.png'):
    keys = keys or ORDER
    data = {k: [run_fields(f) for f in COHORTS[k]['runs'] if os.path.exists(f)] for k in keys}
    xs = {k: i + (0.6 if i >= 4 else 0) + (0.6 if i >= 6 else 0) for i, k in enumerate(keys)}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); ax = axs.ravel()
    fig.suptitle('How good is the plateau at each depth? One dot per run', fontsize=16, y=0.99)
    import textwrap
    fig.text(0.5, 0.952, '\n'.join(textwrap.wrap('All runs: dense ReLU, p = 31, 80% split, cross-entropy, large init, every layer trained throughout with fixed hyperparameters, no loss balancing. '
             'Columns: the main cohort at each depth (left), width 512 (middle), one shared last-layer rate for all seeds (right, for contrast). '
             'Filled: the ≤3% CE window holding most of the test rise (chosen with test accuracy in view); hollow: the longest ≤3% window from the loss alone, or a run that never reaches 90%.', 185)),
             ha='center', va='top', fontsize=9.4, color='#5B6773', linespacing=1.45)
    rng_ = np.random.default_rng(0)
    def dots(a, k, vals, color, hollow=None, dx=0.0):
        n = len(vals); jit = (np.arange(n) - (n - 1) / 2) * 0.07
        for j, v in enumerate(vals):
            if v is None or not np.isfinite(v): continue
            h = hollow[j] if hollow is not None else False
            a.plot(xs[k] + dx + jit[j], v, 'o', ms=8, mfc='white' if h else color, mec=color, mew=1.6, zorder=4)
    blue, green, purple, mag, amber = S['actual_train']['color'], S['actual_test']['color'], S['fourier_addition']['color'], S['agop_current']['color'], S['refit_test']['color']
    tr = lambda f: f['windows'].get('transition') or f['windows'].get('transition_to_end')
    a = ax[0]
    for k, F in data.items():
        dots(a, k, [100 * f['windows']['selected']['share_of_rise'] for f in F], green, dx=-0.12)
        dots(a, k, [100 * f['windows']['loss_alone']['share_of_rise'] for f in F], green, hollow=[True] * len(F), dx=0.12)
    a.set_ylim(-3, 105); a.set_ylabel('% of the test 0.1 → 0.9 rise inside'); style_ax(a, 'A. Grokking inside one ≤3% CE window', xlabel=None)
    a.plot([], [], 'o', ms=8, color=green, label='window chosen with test in view'); a.plot([], [], 'o', ms=8, mfc='white', mec=green, mew=1.6, label='longest window, loss alone')
    a.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='none', fontsize=8.8)
    a = ax[1]
    for k, F in data.items():
        dots(a, k, [100 * tr(f)['v_min'] if tr(f) else None for f in F], blue, hollow=['transition' not in f['windows'] for f in F])
    a.axhline(3, color=THRESH, ls=':', lw=1.3); a.axhline(10, color='#B5364A', ls='--', lw=1.3)
    a.text(xs[keys[-1]] + 0.45, 3, 'P03', va='bottom', ha='right', fontsize=8.5, color='#5B6773'); a.text(xs[keys[-1]] + 0.45, 10, 'P10', va='bottom', ha='right', fontsize=8.5, color='#B5364A')
    a.set_yscale('log'); a.set_yticks([2, 3, 5, 10, 20, 40]); a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}%')); a.yaxis.set_minor_formatter(plt.NullFormatter())
    a.set_ylabel('raw CE range, (max − min) / min'); style_ax(a, 'B. CE flatness over the whole test 10% → 90%', xlabel=None)
    a = ax[2]
    for k, F in data.items():
        dots(a, k, [tr(f)['delta_AH_last'] if tr(f) else None for f in F], purple, hollow=['transition' not in f['windows'] for f in F], dx=-0.12)
        dots(a, k, [tr(f)['delta_AH_first'] if tr(f) else None for f in F], purple, hollow=[True] * len(F), dx=0.12)
    a.axhline(0.02, color=THRESH, ls=':', lw=1.3); a.text(xs[keys[-1]] + 0.45, 0.02, 'F02', va='bottom', ha='right', fontsize=8.5, color='#5B6773'); a.axhline(0, color='#C9707A', lw=0.8)
    a.plot([], [], 'o', ms=8, color=purple, label='last hidden layer (primary)'); a.plot([], [], 'o', ms=8, mfc='white', mec=purple, mew=1.6, label='first hidden layer')
    a.set_ylabel('ΔA_H over test 10% → 90%'); style_ax(a, 'C. Fourier fraction gained over grokking', xlabel=None); a.legend(loc='upper center', bbox_to_anchor=(0.62, 1.0), frameon=True, framealpha=0.92, edgecolor='none', fontsize=8.8)
    a = ax[3]
    for k, F in data.items():
        dots(a, k, [tr(f)['agop_last_gain'] - tr(f)['agop_last_rand_gain'] if tr(f) else None for f in F], mag, dx=-0.12)
        dots(a, k, [(tr(f)['agop_first_gain'] - tr(f)['agop_first_rand_gain']) if (tr(f) and tr(f)['agop_first_gain'] is not None) else None for f in F], mag, hollow=[True] * len(F), dx=0.12)
    a.axhline(0, color='#C9707A', lw=0.8)
    a.plot([], [], 'o', ms=8, color=mag, label='last hidden layer'); a.plot([], [], 'o', ms=8, mfc='white', mec=mag, mew=1.6, label='first hidden layer')
    a.set_ylabel('gain − random-frame gain (q = 4, legacy)'); style_ax(a, 'D. AGOP gain above random frames', xlabel=None); a.legend(loc='upper right', frameon=False, fontsize=8.8)
    a = ax[4]
    for k, F in data.items():
        dots(a, k, [(f['T_grok_first90'] - f['T_refit90']) / 1000 if (f['T_grok_first90'] and f['T_refit90']) else None for f in F], amber)
    a.axhline(0, color='#C9707A', lw=0.8); a.set_ylabel('T_grok − (ridge refit ≥ 90%), thousand steps'); style_ax(a, 'E. Ridge refit leads the network', xlabel=None)
    a = ax[5]
    for k, F in data.items():
        dots(a, k, [100 * f['final_test'] for f in F], green, hollow=[f['T_grok_first90'] is None for f in F])
    a.set_ylim(40, 102); a.set_ylabel('final held-out accuracy, %'); style_ax(a, 'F. Final test accuracy', xlabel=None)
    for a in ax:
        a.set_xticks([xs[k] for k in keys]); a.set_xticklabels([f"{SC_LABEL[k]} ({len(data[k])})" for k in keys], fontsize=9)
        a.set_xlim(min(xs.values()) - 0.5, max(xs.values()) + 0.5)
        for xv in (xs[keys[4]] - 0.8, xs[keys[6]] - 0.8) if len(keys) > 6 else ():
            a.axvline(xv, color='#D0D5DB', lw=1.0)
        trf = blend(a.transData, a.transAxes)
        for grp, lab in SC_GROUPS:
            g = [xs[k] for k in grp if k in xs]
            if g: a.text(np.mean(g), -0.16, '\n'.join(textwrap.wrap(lab, 22 if len(g) > 2 else 17)), transform=trf, ha='center', va='top', fontsize=8.2, color='#5B6773')
    footer(fig, ['Windows and scores follow README §2 and the protocol-v1 definitions: flatness is (max − min)/min of the raw training CE over every update; ΔA_H is the change in the centered hidden addition fraction between the '
                 'first test-accuracy crossings of 10% and 90% (hollow in B and C: a run that never reaches 90%, measured to its last step); AGOP is the legacy raw-logit q = 4 held-out purity minus the same change for random 4-frames.',
                 'Rates: the depth-4 per-seed rates were found by dialing and explained afterwards by the clock; the depth-5 recipe rates were set from each seed\'s t₁₀ before the runs (registered). '
                 'Per-run numbers: figP_<cohort>_fields.json.'])
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.2, wspace=0.28, hspace=0.66)
    fig.savefig(out, dpi=STYLE['figure']['png_dpi']); plt.close(fig)
    return out

def atlas(out='natural_plateau_depth_atlas.pdf', keys=None):
    from PIL import Image
    keys = keys or ORDER
    pages = ['figP_depth_scorecard.png'] + [f'figP_{k}_{s}.png' for k in keys for s in ('overview', 'diagnostics')]
    ims = [Image.open(p_).convert('RGB') for p_ in pages if os.path.exists(p_)]
    ims[0].save(out, save_all=True, append_images=ims[1:], resolution=200.0)
    return out, len(ims)

if __name__ == '__main__':
    args = sys.argv[1:] or ['T23']
    if args == ['all']:
        for k in ORDER: print(sheets(k))
        print(scorecard()); print(atlas())
    else:
        for k in args: print(sheets(k) if k in COHORTS else (scorecard() if k == 'scorecard' else atlas()))
