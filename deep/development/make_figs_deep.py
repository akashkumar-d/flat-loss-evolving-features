import sys, glob, os
from dfigs import deep_headline, ratio_dial
from figs import C, RAMP
L1 = 'layer 1 c24, step 0.03, decay 0.015'
HEAD = 'head step 0.03, decay 0.2'
FIGS = {}
def fig(f): FIGS[f.__name__] = f; return f

@fig
def D2_seeds():
    deep_headline(sorted(glob.glob('runs/seedD2_r0.5_s*.json')), 'fig_deep_D2_seeds.png',
        'Depth 2, p = 31, width 256, cross-entropy, 80% split, five fresh seeds; every layer trains throughout with fixed hyperparameters (no rescale, freeze or controller).\n'
        'Layer 1: c24 init, Polar step 0.03, decay 0.015 (the 1HL learner). Layer 2: 8×He init, Polar step 0.0032, decay 0.07 (shrinks at half layer 1\'s rate). Head: normalized step 0.03, decay 0.2.', h1_top=4)
@fig
def D2_w512_seeds():
    deep_headline(sorted(glob.glob('runs/T2_D2_w512_r0.5_s*.json')), 'fig_deep_D2_w512_seeds.png',
        f'Depth 2, width 512, p = 31, three fresh seeds (runs by account B, T2); same learner as the width-256 design: {L1};\n'
        f'layer 2 8×He init, decay 0.07, shrinks at half layer 1\'s rate (step 0.0032); {HEAD}. Every layer trains throughout.', h1_top=4)
@fig
def D2_p47():
    deep_headline(['runs/T3_D2_p47_r0.5_tune.json', 'runs/T3_D2_p47_r0.5_s0.json'], 'fig_deep_D2_p47.png',
        'Depth 2, p = 47, width 256, tuning seed and fresh seed 0 (runs by account B, T3). Same learner and hyperparameters as the p = 31 depth-2 design:\n'
        f'{L1}; layer 2 8×He init, decay 0.07, half layer 1\'s shrink rate; {HEAD}.', h1_top=4)
@fig
def D3_cascade():
    deep_headline(['runs/T7_D3_cascade_r0.9_0.45_tune_T100k.json'], 'fig_deep_D3_cascade_0.9_0.45.png',
        f'Depth 3, p = 31, width 256, tuning seed, 100k steps. {L1[0].upper() + L1[1:]}. Layers 2 and 3: 8×He init, decay 0.07, shrinking at 0.9× and 0.45× layer 1\'s rate.\n'
        f'{HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout.', win='grok', h1_top=4, zoom_x=(12000, 60000), ymax_agop=0.8)
@fig
def D3_lastslow():
    fn = 'runs/T8_D3_lastslow_r1.0_0.5_tune_T100k.json'
    if not os.path.exists(fn): return 'missing'
    deep_headline([fn], 'fig_deep_D3_lastslow_1.0_0.5.png',
        f'Depth 3, p = 31, width 256, tuning seed, 100k steps: the last-layer-slow rule. {L1[0].upper() + L1[1:]}. Layer 2: 8×He init, decay 0.07, same shrink rate as layer 1.\n'
        f'Layer 3: 8×He init, decay 0.07, half layer 1\'s shrink rate. {HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout.', win='grok', h1_top=4, zoom_x=(12000, 60000), ymax_agop=0.8)
@fig
def D3_seeds():
    fns = sorted(glob.glob('runs/T6_D3_cascade_r0.9_0.45_s*_T100k.json'))
    if not fns: return 'missing'
    deep_headline(fns, 'fig_deep_D3_cascade_seeds.png',
        f'Depth 3, p = 31, width 256, fresh seeds (runs by account B, T6), 100k steps. {L1[0].upper() + L1[1:]}. Layers 2 and 3: 8×He init, decay 0.07,\n'
        f'shrinking at 0.9× and 0.45× layer 1\'s rate. {HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout.', win='grok', h1_top=4, zoom_x=(12000, 60000), ymax_agop=0.8)
@fig
def D4_lastslow():
    fn = 'runs/T8_D4_lastslow_r1.0_1.0_0.5_tune_T120k.json'
    if not os.path.exists(fn): return 'missing'
    deep_headline([fn], 'fig_deep_D4_lastslow_1.0_1.0_0.5.png',
        f'Depth 4, p = 31, width 256, tuning seed, 120k steps: the last-layer-slow rule. {L1[0].upper() + L1[1:]}. Layers 2 and 3: 8×He init, decay 0.07, same shrink rate as layer 1.\n'
        f'Layer 4: 8×He init, decay 0.07, half layer 1\'s shrink rate. {HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout.', win='grok', h1_top=4, zoom_x=(12000, 80000), ymax_agop=0.8)
@fig
def D4_balanced():
    fns = [f for f in ['runs/T11_D4_lastslow_r1.0_1.0_0.35_tune_T120k.json'] + sorted(glob.glob('runs/T14_D4_r1.0_1.0_0.35_s*_T120k.json')) + sorted(glob.glob('runs/T17_D4_r1.0_1.0_0.35_s*_T120k.json')) if os.path.exists(f)]
    if not fns: return 'missing'
    deep_headline(fns, 'fig_deep_D4_lastslow_1.0_1.0_0.35.png',
        f'Depth 4, p = 31, width 256, 120k steps, {len(fns)} run(s) (tuning seed first). {L1[0].upper() + L1[1:]}. Layers 2 and 3: 8×He init, decay 0.07, same shrink rate as layer 1.\n'
        f'Layer 4: 8×He init, decay 0.07, 0.35× layer 1\'s shrink rate. {HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout.', win='grok', h1_top=4, zoom_x=(12000, 80000), ymax_agop=0.8)
@fig
def D3_dial():
    items = [('runs/d29_D3_w256_k8_ld0.07_r0.35_T80k.json', 'uniform 0.35 / 0.35'), ('runs/T5_D3_cascade_r0.7_0.35_tune_T100k.json', 'cascade 0.7 / 0.35'),
             ('runs/T7_D3_cascade_r0.8_0.4_tune_T100k.json', 'cascade 0.8 / 0.4'), ('runs/T7_D3_cascade_r0.9_0.45_tune_T100k.json', 'cascade 0.9 / 0.45')]
    if os.path.exists('runs/T8_D3_lastslow_r1.0_0.5_tune_T100k.json'):
        items.append(('runs/T8_D3_lastslow_r1.0_0.5_tune_T100k.json', 'last-layer-slow 1.0 / 0.5'))
    cols = [C['orange']] + RAMP[1:len(items)]
    ratio_dial(items, 'fig_deep_D3_cascade_dial.png',
        'Depth 3, width 256, tuning seed: shrink rates of layers 2 / 3 relative to layer 1. A slower LAST layer with a fast middle layer keeps the CE flat through most of grokking.', xmax=80000, colors=cols, legend_loc='lower right')

def _grok_point(fn):
    from anat import load, first_persistent
    from dtable import first_cross
    d, L, obs, st = load(fn); hv = [o for o in obs if 'normW' in o]; T = len(L) - 1
    t10 = first_cross(obs, 'test_acc', 0.1); t90 = first_persistent(obs, 'test_acc', 0.9)
    if t10 is None or t90 is None: return None
    atv = lambda t: min(hv, key=lambda o: abs(o['step'] - t)); a, b = atv(t10), atv(t90)
    cm = (L[:-1] + L[1:]) / 2; g = cm[t10:t90 + 1]
    return dict(last=b['normW'][-1] / a['normW'][-1], drift=100 * (g[-1] - g[0]) / g.mean(), logit=b['logit_rms'] / a['logit_rms'], D=d['cfg']['depth'], p=d['cfg']['p'], w=d['cfg'].get('width', 256))

@fig
def mechanism_scatter():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fns = (sorted(glob.glob('runs/seedD2_r0.5_s*.json')) + ['runs/d25_w256_k8_ld0.07_r0.5_T40k.json', 'runs/d24_w256_k8_ld0.07_slowW2_0.7.json', 'runs/d26_w256_k8_ld0.07_r0.45_T40k.json',
            'runs/d25_w256_k8_ld0.07_r0.4_T40k.json', 'runs/d12_w256_k8_ld0.07.json'] + sorted(glob.glob('runs/T2_*.json')) + sorted(glob.glob('runs/T3_*.json'))
           + sorted(glob.glob('runs/T9_*.json')) + sorted(glob.glob('runs/T12_*.json')) + sorted(glob.glob('runs/T16_*.json'))
           + ['runs/d28_D3_w256_k8_ld0.07_r0.7_T40k.json', 'runs/d28_D3_w256_k8_ld0.07_r0.5_T60k.json', 'runs/d29_D3_w256_k8_ld0.07_r0.4_T80k.json', 'runs/d29_D3_w256_k8_ld0.07_r0.35_T80k.json']
           + sorted(glob.glob('runs/T5_*.json')) + sorted(glob.glob('runs/T6_*.json')) + sorted(glob.glob('runs/T7_*.json')) + sorted(glob.glob('runs/T8_*.json')) + sorted(glob.glob('runs/T13_*.json'))
           + sorted(glob.glob('runs/T4_*.json')) + sorted(glob.glob('runs/T11_*.json')) + sorted(glob.glob('runs/T14_*.json')) + sorted(glob.glob('runs/T17_*.json')) + sorted(glob.glob('runs/T18_*.json'))
           + sorted(glob.glob('runs/T10_*.json')) + sorted(glob.glob('runs/T15_*.json')) + sorted(glob.glob('runs/T19_*.json'))
           + sorted(glob.glob('runs/T20_*.json')) + sorted(glob.glob('runs/T21_*.json')) + sorted(glob.glob('runs/T22_*.json')) + sorted(glob.glob('runs/T23_*.json')) + sorted(glob.glob('runs/T24_*.json')))
    pts = [q for q in (_grok_point(f) for f in fns if os.path.exists(f)) if q is not None]
    cols = {2: C['blue'], 3: C['orange'], 4: C['aqua'], 5: C['violet']}; mk = {31: 'o', 23: '^', 47: 's'}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for a, key, lab in [(ax[0], 'last', 'last hidden layer: ‖W_last‖ at test 0.9 / at test 0.1'), (ax[1], 'logit', 'logit RMS at test 0.9 / at test 0.1')]:
        a.axhspan(-3, 3, color=C['blue'], alpha=0.07, lw=0); a.axhline(0, color=C['axis'], lw=0.8); a.axvline(1, color=C['axis'], lw=0.8, ls=(0, (3, 2)))
        for q in pts:
            if q['w'] >= 512:
                a.scatter(q[key], q['drift'], s=46, marker=mk[q['p']], facecolor='white', edgecolor=cols[q['D']], linewidth=1.6, zorder=4)
            else:
                a.scatter(q[key], q['drift'], s=46, marker=mk[q['p']], color=cols[q['D']], edgecolor='white', linewidth=1.0, zorder=3)
        a.set_xlabel(lab); a.grid(True, color=C['grid'], lw=0.6)
    ax[0].set_ylabel('CE drift over test 0.1 → 0.9 (% of mean)')
    for D in sorted({q['D'] for q in pts}): ax[1].scatter([], [], color=cols[D], marker='o', s=46, label=f'depth {D}')
    for p_, m_ in ((31, 'o'), (23, '^'), (47, 's')): ax[1].scatter([], [], color=C['muted'], marker=m_, s=46, label=f'p = {p_}')
    ax[1].scatter([], [], facecolor='white', edgecolor=C['muted'], linewidth=1.6, marker='o', s=46, label='width 512 (hollow)')
    ax[1].legend(fontsize=8, loc='upper right', frameon=False)
    ax[0].set_title('a  CE drift through grokking vs how much the last hidden layer shrinks', loc='left', fontsize=10)
    ax[1].set_title('b  CE drift vs change of the logit scale', loc='left', fontsize=10)
    inb = [q['last'] for q in pts if abs(q['drift']) <= 3 and q['w'] < 512]; flat0 = [q['drift'] for q in pts if q['last'] >= 0.98 and q['w'] < 512]
    fig.suptitle(f'All {len(pts)} large-init runs that finish grokking (depths 2–{max(q["D"] for q in pts)}, p = 23/31/47, widths 256 and 512). Band: |drift| ≤ 3%.\n'
                 f'At width 256 the CE stays flat through grokking only when the last hidden layer is still shrinking during it (×{min(inb):.2f}–{max(inb):.2f} inside the band);\n'
                 f'with no shrinkage it falls {-max(flat0):.0f}–{-min(flat0):.0f}%. At width 512 (hollow markers) a smaller drop of the logit scale keeps the CE flat (panel b).', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.85)); fig.savefig('fig_mechanism_lastlayer.png', dpi=130); plt.close(fig)

@fig
def D3_w512():
    fns = [f for f in ['runs/T10_D3_w512_cascade_r0.9_0.45_tune_T60k.json', 'runs/T10_D3_w512_cascade_r0.9_0.45_s0_T60k.json'] + sorted(glob.glob('runs/T15_D3_w512_cascade_r0.9_0.45_s*_T60k.json')) if os.path.exists(f)]
    if not fns: return 'missing'
    deep_headline(fns, 'fig_deep_D3_w512_cascade.png',
        f'Depth 3 at width 512, p = 31, 60k steps, {len(fns)} run(s) (tuning seed first). {L1[0].upper() + L1[1:]}. Layers 2 and 3: 8×He init, decay 0.07, shrinking at 0.9× and 0.45× layer 1\'s rate.\n'
        f'{HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout. Panel b uses the longest flat window found from the loss alone.', h1_top=4, ymax_agop=0.8)

@fig
def D4_w512():
    fns = [f for f in sorted(glob.glob('runs/T19_D4_w512_*_tune_T60k.json')) + sorted(glob.glob('runs/T19_D4_w512_*_s*_T60k.json')) + sorted(glob.glob('runs/T20_D4_w512_*_T60k.json')) if os.path.exists(f)]
    if not fns: return 'missing'
    rates = sorted({f.split('_r1.0_1.0_')[1].split('_')[0] for f in fns})
    deep_headline(fns, 'fig_deep_D4_w512.png',
        f'Depth 4 at width 512, p = 31, 60k steps, {len(fns)} run(s) (tuning seed first). {L1[0].upper() + L1[1:]}. Layers 2 and 3: 8×He init, decay 0.07, same shrink rate as layer 1.\n'
        f'Layer 4: 8×He init, decay 0.07, {" or ".join(rates)}× layer 1\'s shrink rate. {HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout. Panel b: longest flat window, loss alone.', h1_top=4, ymax_agop=0.8)
@fig
def D4_w512_long():
    fns = [f for f in ['runs/T24_D4_w512_r1.0_1.0_0.4_s0_T120k.json', 'runs/T24_D4_w512_r1.0_1.0_0.4_tune_T120k.json'] if os.path.exists(f)]
    if not fns: return 'missing'
    from make_readme import stats
    o = stats(fns[0]); span = (o['t10'], o['t90'] or o['T'])
    deep_headline(fns, 'fig_deep_D4_w512_long.png',
        f'Depth 4 at width 512, p = 31, 120k steps (T24): seed 0, then the tuning seed. {L1[0].upper() + L1[1:]}. Layers 2 and 3: 8×He init, decay 0.07, same shrink rate as layer 1; layer 4 at 0.4×.\n'
        f'{HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout. Panel b: black bars hold the most of the rise (chosen knowing test accuracy); grey bars are the longest ≤3% windows (loss alone).',
        win='grok', h1_top=4, ymax_agop=0.85, shade_span=span, shade_label='seed 0, test accuracy 0.1 → 0.9', zoom_x=(0, o['T']))
@fig
def D5_dial():
    items = [(f'runs/T21_D5_r1.0_1.0_1.0_{r}_tune_T150k.json', f'last layer {r}×') for r in ('0.25', '0.3', '0.35', '0.4')]
    items = [x for x in items if os.path.exists(x[0])]
    if not items: return 'missing'
    ratio_dial(items, 'fig_deep_D5_dial.png',
        'Depth 5, width 256, tuning seed, 150k steps: layers 2–4 shrink at layer 1\'s rate, the last hidden layer (5) at 0.25–0.4× of it.', xmax=150000, legend_loc='upper left')
@fig
def D5_best():
    from make_readme import bestwin
    fns = sorted(glob.glob('runs/T21_D5_*_T150k.json'))
    if not fns: return 'missing'
    best = {}
    for f in fns:
        sd = f.split('_')[-2]; sh = bestwin(f)[2]
        if sh > best.get(sd, (None, -1))[1]: best[sd] = (f, sh)
    sel = [best[k][0] for k in sorted(best, key=lambda k: (k != 'tune', k))]
    rates = ', '.join(f.split('_r1.0_1.0_1.0_')[1].split('_')[0] + '×' for f in sel)
    deep_headline(sel, 'fig_deep_D5_best.png',
        f'Depth 5, p = 31, width 256, 150k steps: for each seed the last-layer rate that holds the most of the rise in one window ({rates}; tuning seed first).\n'
        f'{L1[0].upper() + L1[1:]}. Layers 2–4: 8×He init, decay 0.07, same shrink rate as layer 1; layer 5 slower. {HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout.', win='grok', h1_top=4, ymax_agop=0.8)

@fig
def rate_rule():
    import json, math, subprocess, sys, numpy as np
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import make_v12 as M
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rate_rule.py')], check=True, capture_output=True)
    rows = json.load(open('rate_rule_rows.json'))
    REG = (-17.1, 799.0)
    base = [r for r in rows if r[2] == 31 and r[1] == 256 and not r[3].startswith('d24_w256_k8_ld0.07_slowW2_0.5') and not M.arrived_after(r[3])]
    x = np.array([r[7] for r in base]); y = np.array([r[8] for r in base]); Dv = np.array([r[0] for r in base])
    late = [r for r in rows if M.arrived_after(r[3])]
    reg = REG[0] + REG[1] * np.exp(-x); rmse = float(np.sqrt(((y - reg) ** 2).mean())); r2 = 1 - ((y - reg) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    Ds = sorted(set(Dv.tolist())); A2 = np.stack([(Dv == d).astype(float) for d in Ds] + [np.exp(-x)], 1); cD, *_ = np.linalg.lstsq(A2, y, rcond=None)
    r2D = 1 - ((y - A2 @ cD) ** 2).sum() / ((y - y.mean()) ** 2).sum(); x0 = math.log(REG[1] / -REG[0])
    cols = {2: C['blue'], 3: C['orange'], 4: C['aqua'], 5: C['violet']}; mk = {31: 'o', 23: '^', 47: 's'}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.9))
    xs = np.linspace(2.4, 9.6, 300)
    ax[0].axhspan(-3, 3, color=C['blue'], alpha=0.07, lw=0); ax[0].axhline(0, color=C['axis'], lw=0.8)
    ax[0].plot(xs, REG[0] + REG[1] * np.exp(-xs), color=C['ink2'], lw=1.5, zorder=2)
    for k, d in enumerate(Ds): ax[0].plot(xs, cD[k] + cD[-1] * np.exp(-xs), color=cols[d], lw=1.0, ls=(0, (4, 2)), zorder=2)
    for a in ax: a.axvline(x0, color=C['ink2'], lw=0.8, ls=(0, (3, 2)))
    for a, j in ((ax[0], 8), (ax[1], 9)):
        for r in rows:
            D, w, p = r[0], r[1], r[2]; val = r[j] * (100 if j == 9 else 1)
            if w >= 512: a.scatter(r[7], val, s=46, marker=mk[p], facecolor='white', edgecolor=cols[D], linewidth=1.6, zorder=4)
            else: a.scatter(r[7], val, s=46, marker=mk[p], color=cols[D], edgecolor='white', linewidth=1.0, zorder=3)
            if M.arrived_after(r[3]): a.scatter(r[7], val, s=150, marker='o', facecolor='none', edgecolor=C['ink'], linewidth=1.0, zorder=5)
        a.set_xlabel('x = η_L·λ_L·t₁₀: time constants of the last hidden layer\'s relaxation\nelapsed when test accuracy first reaches 0.1')
        a.grid(True, color=C['grid'], lw=0.6); a.set_xlim(2.3, 9.7)
    ax[0].set_ylabel('CE drift over test 0.1 → 0.9 (% of mean)'); ax[0].set_ylim(-52, 64)
    ax[1].set_ylabel('share of the rise in one ≤3% window (%)'); ax[1].set_ylim(0, 105)
    ax[0].text(x0 + 0.08, 57, f'x = {x0:.2f}', fontsize=8, color=C['ink2'])
    for D in sorted({r[0] for r in rows}): ax[1].scatter([], [], color=cols[D], marker='o', s=46, label=f'depth {D}')
    for p_, m_ in ((31, 'o'), (23, '^'), (47, 's')): ax[1].scatter([], [], color=C['muted'], marker=m_, s=46, label=f'p = {p_}')
    ax[1].scatter([], [], facecolor='white', edgecolor=C['muted'], linewidth=1.6, marker='o', s=46, label='width 512 (hollow)')
    ax[1].plot([], [], color=C['ink2'], lw=1.5, label='curve registered at 22:50 UTC (panel a)'); ax[1].plot([], [], color=C['muted'], lw=1.0, ls=(0, (4, 2)), label='same slope, one offset per depth')
    if late: ax[1].scatter([], [], s=150, marker='o', facecolor='none', edgecolor=C['ink'], linewidth=1.0, label=f'ringed: {len(late)} runs that arrived later (test)')
    ax[1].legend(fontsize=8, loc='upper right', frameon=False)
    ax[0].set_title('a  CE drift through grokking', loc='left', fontsize=10); ax[1].set_title('b  How much of grokking one flat window holds', loc='left', fontsize=10)
    reg0 = f'{REG[0]:.1f}'.replace('-', '−')
    fig.suptitle(f'All {len(rows)} large-init deep runs that finish grokking. At p = 31, width 256 ({len(base)} runs before 22:50 UTC: depths {min(Ds)}–{max(Ds)}, seeds, cascades, uniform and last-layer-slow designs) the CE drift\n'
                 f'through grokking is one function of x: drift = {reg0}% + {REG[1]:.0f}%·e^(−x) (R² = {r2:.2f}, RMSE {rmse:.1f} points), zero at x = {x0:.2f}; one offset per depth raises R² to {r2D:.2f}.\n'
                 f'Width 512 (hollow) sits above the curve; p = 47 balances at a smaller x; p = 23 with layer-1 decay 0.01 falls far below.', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig('fig_rate_rule.png', dpi=130); plt.close(fig)
    return f'registered curve on n={len(base)}: R2={r2:.3f} rmse={rmse:.2f}; per-depth offsets {np.round(cD[:-1], 1).tolist()} b={cD[-1]:.0f} R2={r2D:.3f}'

@fig
def D4_per_seed():
    import make_v12 as M
    rates = ['0.25', '0.3', '0.35', '0.4', '0.45', '0.5']; seeds = ['tune', 's0', 's1', 's2']
    _, _, vals = M.rate_grid(rates, seeds, lambda r, s: (M.d4_candidates(r, s)[0] if False else next((c for c in M.d4_candidates(r, s) if os.path.exists(c)), None), False))
    sel = []
    for sd in seeds:
        best = max((r for r in rates if vals[r, sd]), key=lambda r: vals[r, sd][0][0])
        sel.append((next(c for c in M.d4_candidates(best, sd) if os.path.exists(c)), best, sd))
    fns = [x[0] for x in sel]; lab = ', '.join(f"{'tuning' if sd == 'tune' else 'seed ' + sd[1:]} {r}×" for _, r, sd in sel)
    deep_headline(fns, 'fig_deep_D4_per_seed.png',
        f'Depth 4, p = 31, width 256, 120k steps: each seed at its own balancing last-layer rate ({lab}). {L1[0].upper() + L1[1:]}.\n'
        f'Layers 2 and 3: 8×He init, decay 0.07, same shrink rate as layer 1; layer 4 slower. {HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout.', win='grok', h1_top=4, zoom_x=(12000, 60000), ymax_agop=0.8)

@fig
def D3_w512_rule():
    fns = [f for f in ['runs/T10_D3_w512_lastslow_r1.0_0.5_tune_T60k.json'] + sorted(glob.glob('runs/T15_D3_w512_lastslow_r1.0_0.5_s*_T60k.json')) if os.path.exists(f)]
    if not fns: return 'missing'
    deep_headline(fns, 'fig_deep_D3_w512_rule.png',
        f'Depth 3 at width 512, p = 31, 60k steps, {len(fns)} run(s) (tuning seed first): the untuned last-layer-slow rule. {L1[0].upper() + L1[1:]}. Layer 2: 8×He init, decay 0.07, same shrink rate as layer 1;\n'
        f'layer 3: half layer 1\'s rate. {HEAD[0].upper() + HEAD[1:]}. Every layer trains throughout. Panel b uses the longest flat window found from the loss alone.', h1_top=4, ymax_agop=0.8)

@fig
def overview():
    import math, numpy as np
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from anat import load, series
    from make_readme import stats, bestwin, share_of
    runs = [('runs/seedD2_r0.5_s2.json', 'depth 2, width 256', 'layer 2 at 0.5×, seed 2', 'grok'),
            ('runs/T6_D3_cascade_r0.9_0.45_s2_T100k.json', 'depth 3, width 256', 'layers 2/3 at 0.9×/0.45×, seed 2', 'grok'),
            ('runs/T18_D4_r1.0_1.0_0.4_s1_T120k.json', 'depth 4, width 256', 'last layer at 0.4×, seed 1', 'grok'),
            ('runs/T21_D5_r1.0_1.0_1.0_0.35_s0_T150k.json', 'depth 5, width 256', 'last layer at 0.35×, seed 0', 'grok'),
            ('runs/T15_D3_w512_lastslow_r1.0_0.5_s0_T60k.json', 'depth 3, width 512', 'layers 2/3 at 1.0×/0.5×, seed 0', 'longest')]
    runs = [r for r in runs if os.path.exists(r[0])]
    fig, ax = plt.subplots(3, len(runs), figsize=(3.6 * len(runs), 8.6), sharey='row')
    for j, (fn, head, sub, mode) in enumerate(runs):
        d, L, obs, st = load(fn); o = stats(fn); p = d['cfg']['p']; lp = math.log(p); cmv = (L[:-1] + L[1:]) / 2
        if mode == 'grok': a, b, sh, lev, R = bestwin(fn)
        else: a, b = o['fw']; sh = share_of(o, a, b)
        t10, t90 = o['t10'], (o['t90'] or o['T'])
        x0, x1 = max(0, min(a, t10) - 8000), min(o['T'], max(b, t90) + 12000)
        t = np.arange(len(cmv))
        for i in range(3): ax[i, j].axvspan(a, b, color=C['blue'], alpha=0.09, lw=0); ax[i, j].set_xlim(x0, x1); ax[i, j].grid(True, color=C['grid'], lw=0.6)
        mu = cmv[a:b + 1].mean()
        ax[0, j].plot(t, 100 * (cmv / mu - 1), color=C['blue'], lw=1.3)
        ax[0, j].axhspan(-1.5, 1.5, color=C['blue'], alpha=0.10, lw=0); ax[0, j].axhline(0, color=C['axis'], lw=0.8)
        ax[0, j].set_ylim(-12, 12)
        ax[0, j].text(0.03, 0.05, f'window mean {mu / lp:.2f}·log p', transform=ax[0, j].transAxes, fontsize=8, color=C['ink2'])
        s_, v = series(obs, 'train_acc'); ax[1, j].plot(s_, v, color=C['muted'], lw=1.0)
        s_, v = series(obs, 'test_acc'); ax[1, j].plot(s_, v, color=C['orange'], lw=1.4)
        for key, col in (('agopH1_4_ho', C['aqua']), ('agop30_ho', C['violet'])):
            s_, v = series(obs, key); ax[2, j].plot(s_, v, color=col, lw=1.3)
            s_, v = series(obs, key + '_rand'); ax[2, j].plot(s_, v, color=col, lw=0.9, ls=(0, (3, 2)))
        s_, v = series(obs, 'W1_four_top1'); ax[2, j].plot(s_, v, color=C['yellow'], lw=1.3)
        for i in range(3):
            for tt in (t10, t90): ax[i, j].axvline(tt, color=C['ink2'], lw=0.7, ls=(0, (2, 2)))
        ax[0, j].set_title(f'{head}\n{sub}\n{100 * sh:.0f}% of the rise in the shaded window', loc='left', fontsize=9)
        ax[2, j].set_xlabel('step')
    ax[0, 0].set_ylabel('training CE, % from the window mean\n(cycle mean; band ±1.5%)'); ax[1, 0].set_ylabel('accuracy'); ax[2, 0].set_ylabel('alignment / Fourier share')
    ax[1, 0].set_ylim(-0.02, 1.02); ax[2, 0].set_ylim(0, 0.9)
    ax[1, -1].plot([], [], color=C['orange'], lw=1.4, label='test'); ax[1, -1].plot([], [], color=C['muted'], lw=1.0, label='train'); ax[1, -1].legend(fontsize=8, loc='lower right', frameon=False)
    for key, col, lab in (('a', C['aqua'], 'first-layer AGOP, top-4'), ('b', C['violet'], 'last-layer AGOP, top-(p−1)'), ('c', C['yellow'], 'layer-1 weight Fourier share')):
        ax[2, -1].plot([], [], color=col, lw=1.3, label=lab)
    ax[2, -1].plot([], [], color=C['muted'], lw=0.9, ls=(0, (3, 2)), label='dashed: random frames'); ax[2, -1].legend(fontsize=7.5, loc='upper left', frameon=False)
    fig.suptitle('Natural training-loss plateaus through grokking, p = 31: one run per depth at its balancing last-layer rate. Every layer trains throughout with fixed hyperparameters.\n'
                 'Shaded: the ≤3% CE window holding most of the test 0.1 → 0.9 rise (width 256: chosen knowing test accuracy; width 512: the longest flat window, from the loss alone). '
                 'Dotted lines: test 0.1 and 0.9.', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig('fig_overview_depths.png', dpi=130); plt.close(fig)

@fig
def D5_recipe():
    import math, numpy as np
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from anat import load, series
    from make_readme import stats, bestwin, nondec
    pairs = []
    for k in (1, 2, 3, 4):
        s1 = glob.glob(f'runs/T22_D5_*_s{k}_T70k.json'); s2 = glob.glob(f'runs/T23_D5_*_s{k}_T70k.json')
        if s1 and s2: pairs.append((k, s1[0], s2[0]))
    if not pairs: return 'missing'
    fig, ax = plt.subplots(2, len(pairs), figsize=(3.9 * len(pairs), 6.6), sharey='row')
    for j, (k, f1, f2) in enumerate(pairs):
        for fn, col, lw in ((f1, C['muted'], 1.2), (f2, C['blue'], 1.6)):
            d, L, obs, st = load(fn); lp = math.log(d['cfg']['p']); cmv = (L[:-1] + L[1:]) / 2
            ax[0, j].plot(np.arange(len(cmv)), cmv / lp, color=col, lw=lw)
            s_, v = series(obs, 'test_acc'); ax[1, j].plot(s_, v, color=col, lw=lw)
        na, nb, nsh = nondec(f2)
        for i in range(2): ax[i, j].axvspan(na, nb, color=C['blue'], alpha=0.09, lw=0); ax[i, j].set_xlim(15000, 65000); ax[i, j].grid(True, color=C['grid'], lw=0.6)
        o1, o2 = stats(f1), stats(f2); r1 = f1.split('_r1.0_1.0_1.0_')[1].split('_')[0]; r2 = f2.split('_r1.0_1.0_1.0_')[1].split('_')[0]
        n1 = nondec(f1)[2]
        mn = lambda v: f'{v:+.1f}'.replace('-', '−')
        ax[0, j].set_title(f'seed {k}: last layer {r1}× → {r2}×\nCE drift over grokking {mn(o1["g_D"])}% → {mn(o2["g_D"])}%\nflat-or-rising share {100 * n1:.0f}% → {100 * nsh:.0f}%', loc='left', fontsize=9)
        ax[1, j].set_xlabel('step')
    ax[0, 0].set_ylabel('training CE / log p (cycle mean)'); ax[1, 0].set_ylabel('test accuracy'); ax[0, 0].set_ylim(0.45, 0.66); ax[1, 0].set_ylim(-0.02, 1.02)
    ax[0, -1].plot([], [], color=C['muted'], lw=1.2, label='stage 1: 0.35× for every seed (T22)'); ax[0, -1].plot([], [], color=C['blue'], lw=1.6, label='stage 2: rate from the seed\'s t₁₀ (T23)')
    ax[0, -1].legend(fontsize=8, loc='lower right', frameon=False)
    fig.suptitle('Depth 5, width 256, four fresh seeds: setting the last layer\'s rate from each seed\'s grokking onset (target x = η_L·λ_L·t₁₀ = 3.85, registered before stage 2 ran).\n'
                 'Shaded: stage 2\'s flat-or-rising window (CE range ≤ 3%, no drop > 1% below its running peak).', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig('fig_deep_D5_recipe.png', dpi=130); plt.close(fig)

@fig
def he_dial_D2():
    items = [('runs/d14_w256_k1_lw0.022_ld0.05.json', 'λ_W1 = 0.022'), ('runs/d18_w256_k1_lw0.025_ld0.05.json', 'λ_W1 = 0.025'),
             ('runs/d18_w256_k1_lw0.027_ld0.05.json', 'λ_W1 = 0.027'), ('runs/d14_w256_k1_lw0.03_ld0.05.json', 'λ_W1 = 0.03')]
    ratio_dial(items, 'fig_deep_he_dial_D2.png',
        'Depth 2 with a He-initialized deep layer (width 256, tuning seed): the first-layer decay λ_W1 moves grokking relative to the end of the first layer\'s norm relaxation.\n'
        'At the balance (0.025–0.027) the CE still rises and falls by about 7%; the first-layer AGOP alignment falls to its random level.', xmax=30000)
@fig
def he_dial_D3():
    items = [('runs/d22_D3_w256_k1_lw0.013_ld0.05.json', 'λ_W1 = 0.013'), ('runs/d22_D3_w256_k1_lw0.017_ld0.05.json', 'λ_W1 = 0.017'),
             ('runs/d19_D3_w256_k1_lw0.022_ld0.05.json', 'λ_W1 = 0.022'), ('runs/d19_D3_w256_k1_lw0.03_ld0.05.json', 'λ_W1 = 0.03')]
    ratio_dial(items, 'fig_deep_he_dial_D3.png',
        'Depth 3 with He-initialized deep layers (width 256, tuning seed): same dial. The CE falls 17–19% during grokking unless λ_W1 is small (0.013),\n'
        'where the drift is small but the range is still 12%.', xmax=30000)
@fig
def ratio_dial_D2():
    items = [('runs/d25_w256_k8_ld0.07_r0.4_T40k.json', 'r = 0.4'), ('runs/d26_w256_k8_ld0.07_r0.45_T40k.json', 'r = 0.45'),
             ('runs/d25_w256_k8_ld0.07_r0.5_T40k.json', 'r = 0.5'), ('runs/d24_w256_k8_ld0.07_slowW2_0.7.json', 'r = 0.7'),
             ('runs/d12_w256_k8_ld0.07.json', 'r = 1 (same shrink rate)')]
    ratio_dial(items, 'fig_deep_ratio_dial.png',
        'Depth 2, width 256, tuning seed: r = (layer-2 shrink rate η₂λ₂) / (layer-1 shrink rate η₁λ₁). Slower deep-layer relaxation keeps the norm product shrinking through grokking.\n'
        'Layer 1: c24, step 0.03, decay 0.015; layer 2: 8×He init, decay 0.07, step r·0.00643; head: step 0.03, decay 0.2.', xmax=40000)
@fig
def gain():
    from dfigs import gain_panels
    gain_panels([('runs/s2_p31_lw0.015_lv0.2.json', 'a  1 hidden layer, width 512'), ('runs/d18_w256_k1_lw0.027_ld0.05.json', 'b  depth 2, He-init deep layer'),
                 ('runs/d19_D3_w256_k1_lw0.022_ld0.05.json', 'c  depth 3, He-init deep layers'), ('runs/d25_w256_k8_ld0.07_r0.5_T40k.json', 'd  depth 2, large-init slow deep layer')],
                'fig_gain.png', 'Logit scale = product of layer norms × normalized gain, each relative to its value when test accuracy reaches 0.1 '
                '(shaded: test accuracy 0.1 → 0.9; tuning seed)')

if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGS)
    for n in names: print(n, FIGS[n]() or 'ok')
