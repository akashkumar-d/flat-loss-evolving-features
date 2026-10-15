#!/usr/bin/env python3
import json, math, os, re, statistics as S, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import plot_flat_scales as PF

GREEN = "#009E73"
GROUPS = [
    ('Polar rule, campaign-T layout', [
        ('U_L1_lam_w0.02_eta_w0.1_lam_v0.2', '1 layer, $\\lambda_W$ 0.02, $\\lambda_V$ 0.2'),
        ('U_H1_lam_w0.01_eta_w0.1_lam_v3', '1 layer, $\\lambda_W$ 0.01, $\\lambda_V$ 3'),
        ('U_H1_lam_w0.01_eta_w0.1_lam_v10', '1 layer, $\\lambda_W$ 0.01, $\\lambda_V$ 10'),
        ('U_H2_lam_w0.01_eta_w0.1_lam_v1', '2 layers, $\\lambda_W$ 0.01, $\\lambda_V$ 1'),
        ('U_H2_lam_w0.01_eta_w0.1_lam_v2', '2 layers, $\\lambda_W$ 0.01, $\\lambda_V$ 2'),
        ('T_P1_f0.3_lam_w0.03_eta_w0.1_lam_v3', '1 layer, $\\lambda_W$ 0.03, $\\lambda_V$ 3 (T rerun)'),
        ('T_P1_f0.5_lam_w0.03_eta_w0.1_lam_v3', 'same, frac 0.5 (T rerun)'),
        ('T_P2_f0.3_lam_w0.03_eta_w0.1_lam_v0.2', '2 layers, $\\lambda_W$ 0.03, $\\lambda_V$ 0.2 (T rerun)')]),
    ('Polar rule, final RMS division', [
        ('U_N1_rms_lam_w0.01_eta_w0.1_lam_v0.25', '1 layer, $\\lambda_V$ 0.25'),
        ('U_N1_rms_lam_w0.01_eta_w0.1_lam_v0.5', '1 layer, $\\lambda_V$ 0.5'),
        ('U_N1_rms_lam_w0.01_eta_w0.1_lam_v1', '1 layer, $\\lambda_V$ 1'),
        ('U_N1_rms_lam_w0.01_eta_w0.03_lam_v0.5', '1 layer, $\\lambda_V$ 0.5, $\\eta_W$ 0.03'),
        ('U_N1_rms_lam_w0.01_eta_w0.03_lam_v1', '1 layer, $\\lambda_V$ 1, $\\eta_W$ 0.03'),
        ('U_N1_rms_lam_w0.02_eta_w0.1_lam_v0.5', '1 layer, $\\lambda_V$ 0.5, $\\lambda_W$ 0.02'),
        ('U_N1_rms_lam_w0.02_eta_w0.1_lam_v1', '1 layer, $\\lambda_V$ 1, $\\lambda_W$ 0.02'),
        ('U_N2_rms_lam_w0.01_eta_w0.1_lam_v0.5', '2 layers, $\\lambda_V$ 0.5'),
        ('U_N2_rms_lam_w0.01_eta_w0.1_lam_v1', '2 layers, $\\lambda_V$ 1')]),
    ('AdamW (lr 1e-3)', [
        ('U_A1_adamw_wd0.3', '1 layer, wd 0.3'), ('U_A1_adamw_wd1', '1 layer, wd 1'), ('U_A1_adamw_wd3', '1 layer, wd 3'),
        ('U_A2_adamw_wd1', '2 layers, wd 1'),
        ('U_A1_rms_adamw_wd0.3', '1 layer, RMS, wd 0.3'), ('U_A1_rms_adamw_wd1', '1 layer, RMS, wd 1'),
        ('U_A1_rms_adamw_wd3', '1 layer, RMS, wd 3'), ('U_A2_rms_adamw_wd1', '2 layers, RMS, wd 1')])]


def setting(rid): return re.sub(r'_s\d+$', '', rid)


def scoreboard(D, out):
    runs = {}
    for r in D['runs']: runs.setdefault(setting(r['id']), []).append(r)
    W = D['windows']; rows = []
    for g, items in GROUPS:
        rows.append(('group', g))
        for st, lab in items: rows.append(('row', st, lab))
    ncol = 1 + 2 * len(W)
    fig, ax = plt.subplots(figsize=(7.4, 0.25 * len(rows) + 1.25), layout='constrained')
    ax.set_xlim(-4.6, ncol); ax.set_ylim(len(rows), -1.35); ax.axis('off')
    green = np.array(to_rgb(GREEN)); white = np.ones(3)
    ax.text(0.5, -0.45, 'CE / log p\nover I', ha='center', va='bottom', fontsize=6.6)
    for j, kind in enumerate(('geometric', 'arithmetic')):
        x0 = 1 + j * len(W)
        ax.text(x0 + len(W) / 2, -1.2, f'{kind} average, window w (updates)', ha='center', va='bottom', fontsize=7.2)
        for i, w in enumerate(W): ax.text(x0 + i + 0.5, -0.45, (f'{w // 1000}k' if w >= 1000 else f'{w}'), ha='center', va='bottom', fontsize=6.6)
    for i, row in enumerate(rows):
        if row[0] == 'group':
            ax.text(-4.55, i + 0.62, row[1], ha='left', va='center', fontsize=7.2, fontweight='bold'); continue
        _, st, lab = row; rs = runs.get(st, [])
        ax.text(-4.45, i + 0.55, f'{lab}  ({len(rs)})', ha='left', va='center', fontsize=6.6)
        lv = [r['profile']['geo']['500']['level'] / r['log_p'] for r in rs if 'I' in r['profile']]
        ax.text(0.5, i + 0.55, (f'{S.median(lv):.2g}' if lv else 'no rise'), ha='center', va='center', fontsize=6.4)
        for j, kind in enumerate(('geo', 'arith')):
            for k, w in enumerate(W):
                ks = [r['key'][f'{kind}_{w}'] for r in rs]; n = len(ks); p = sum(x['pass'] for x in ks); p3 = sum(x['pass3'] for x in ks)
                frac = p / n if n else 0; col = white * (1 - 0.85 * frac) + green * 0.85 * frac if frac > 0 else np.array([0.955] * 3)
                x = 1 + j * len(W) + k
                ax.add_patch(plt.Rectangle((x + 0.04, i + 0.1), 0.92, 0.84, color=col, lw=0))
                tc = 'white' if frac > 0.6 else 'black'
                ax.text(x + 0.5, i + 0.55, f'{p}/{n}' + (f' ({p3})' if p3 else ''), ha='center', va='center', fontsize=6.2, color=tc)
    fig.savefig(os.path.join(out, 'U_scoreboard.pdf'), metadata={'Creator': 'plot_U.py', 'CreationDate': None})
    fig.savefig(os.path.join(out, 'U_scoreboard.png'), dpi=200); plt.close(fig)


def dynamics(res, D, out):
    runs = {}
    for r in D['runs']: runs.setdefault(setting(r['id']), []).append(r['id'])
    os.makedirs(os.path.join(out, 'settings'), exist_ok=True)
    for st, ids in sorted(runs.items()):
        ok = []
        for rid in sorted(ids):
            r = next(x for x in D['runs'] if x['id'] == rid)
            if 'I' in r['profile']: ok.append(rid)
        if ok: PF.main(res, os.path.join(out, 'settings'), tuple(ok), 'U_dyn_' + st)


if __name__ == '__main__':
    res, key, out = sys.argv[1:4]
    D = json.load(open(key)); os.makedirs(out, exist_ok=True)
    scoreboard(D, out)
    if '--no-dynamics' not in sys.argv: dynamics(res, D, out)
