#!/usr/bin/env python3
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CAMPAIGN_Z_DIR', os.path.join(HERE, '..', 'campaign_Z')))
import plot_Z as PZ
import matplotlib.pyplot as plt

P_STYLE = {61: ('#86b6ef', 'o'), 89: ('#3987e5', 's'), 113: ('#1c5cab', 'D'), 149: ('#0d366b', '^')}
X_C, A_RISE = 3.9, 2300.0


def title(c):
    r = c['runs'][0]
    return f"p = {r['p']}, frac {r['frac']:g}, lam_v {r['lam_v']:g} [{c.get('campaign', '?')}]"


def per_p(A, outdir):
    for p in (61, 89, 149):
        ks = sorted([k for k, c in A.items() if c['runs'][0]['p'] == p and (c.get('campaign') == 'ZP' or
                     (c.get('campaign') == 'Z' and c['runs'][0]['lam_v'] == {61: 0.45, 89: 0.33, 149: 0.23}[p]))],
                    key=lambda k: A[k]['runs'][0]['frac'])
        if not any(A[k].get('campaign') == 'ZP' for k in ks): continue
        PZ.column_figure([(title(A[k]), A[k]) for k in ks], os.path.join(outdir, f'ZP_p{p}.png'),
                         f'p = {p}: training fraction (one block, gain 2 -> 1; Z = campaign Z, ZP = this campaign)', ce_logp=True)
        print(f'drew ZP_p{p}')


def appendix_figure(res, outdir):
    A = res['cells']; cols = []
    for p, a in sorted(res['appendix'].items(), key=lambda kv: int(kv[0])):
        c = A[a['cell']]; cols.append((title(c) + ('' if a['works'] else ' - below 4/5'), c))
    if cols:
        PZ.column_figure(cols, os.path.join(outdir, 'ZP_appendix.png'), 'One block across p (gain 2 -> 1; frac set per p)', ce_logp=True)
        print('drew ZP_appendix')


def timing_figure(res, outdir):
    A = res['cells']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    xs = np.linspace(X_C + 0.05, 7.6, 300)
    ax1.plot(xs, A_RISE / (xs - X_C), color='#555555', lw=1.0, ls=(0, (4, 2)), zorder=1, label='rule: 2300 / (x - 3.9)')
    ax1.axvline(X_C, color='#999999', lw=0.7, ls=':', zorder=0)
    seen = set()
    from matplotlib.lines import Line2D
    for k, c in A.items():
        for r in c['runs']:
            p = r['p']; col, mk = P_STYLE.get(p, ('#777777', 'o')); x = r['n_train'] / (p * math.log(p))
            seen.add(p)
            if r.get('rise'):
                ax1.scatter([x], [r['rise']['a']], s=22, marker=mk, color=col, edgecolor='white', lw=0.6, zorder=3)
                ratio = r['rise']['a'] / max(r['ce_min']['t'], 1)
                ax2.scatter([x], [ratio], s=24, marker=mk, zorder=3, lw=1.0,
                            color=col if r['pass_strict'] else 'white', edgecolor=col)
            else:
                ax1.scatter([x], [30000], s=22, marker=mk, color='white', edgecolor=col, lw=1.0, zorder=3)
    ax1.set_yscale('log'); ax1.set_ylim(250, 40000); ax1.set_xlim(3.5, 7.6)
    ax1.set_xlabel('x = n_train / (p ln p)'); ax1.set_ylabel('Start of the held-out rise\n(no-mirror accuracy crosses 0.1; updates)')
    ax1.set_title('(a) Rise start; open at the top: no rise by 30k updates')
    hs = [Line2D([], [], color='#555555', lw=1.0, ls=(0, (4, 2)))] + [Line2D([], [], ls='', marker=P_STYLE[p][1], color=P_STYLE[p][0],
          markeredgecolor='white', markersize=5.5) for p in sorted(seen) if p in P_STYLE]
    ax1.legend(hs, ['rule: 2300 / (x - 3.9)'] + [f'p = {p}' for p in sorted(seen) if p in P_STYLE], fontsize=6.8, loc='upper right')
    ax1.grid(alpha=0.25, lw=0.5)
    ax2.axhline(1.0, color='#555555', lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax2.set_yscale('log'); ax2.set_xlim(3.5, 7.6); ax2.set_xlabel('x = n_train / (p ln p)')
    ax2.set_ylabel('Rise start / CE dip'); ax2.grid(alpha=0.25, lw=0.5)
    ax2.set_title('(b) Filled: strict pass (both held-out definitions); open: fail')
    fig.suptitle('When the held-out rise starts (one block, gain 2 -> 1; campaigns Y, Z and ZP)', fontsize=9.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'ZP_timing.png'), dpi=160); plt.close(fig)
    print('drew ZP_timing')


def main(ana, outdir):
    res = json.load(open(ana)); os.makedirs(outdir, exist_ok=True)
    per_p(res['cells'], outdir); appendix_figure(res, outdir); timing_figure(res, outdir)


if __name__ == '__main__':
    main(*sys.argv[1:3])
