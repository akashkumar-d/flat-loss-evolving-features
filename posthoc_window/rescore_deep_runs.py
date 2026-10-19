#!/usr/bin/env python3
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ovw, ovw2
import median_figs as MF
FIGS = True


def log_setting(fh, camp, name, R):
    s = ovw2.summary(R); s.update(camp=camp, setting=name,
                                 runs=[dict(id=r['id'], W2=r['W2'], g_te=round(r['g_te'], 4), g_rf=round(r['g_rf'], 4), ok2=bool(r['ok2']),
                                            ok_old=bool(r['ok_old']), fit2=r['fit2'], final=r['final'], share2=round(r['share2'], 4),
                                            refit=r['refit_kind']) for r in R])
    fh.write(json.dumps(s) + '\n'); fh.flush()


def sheet(R, title, out, stem, note=''):
    if FIGS: ovw2.sheet2(R, title, out, stem, note=note)


def grid(cols, out, stem, **kw):
    if FIGS: return ovw2.pgrid(cols, out, stem, **kw)
    return [dict(title=t.replace('\n', ' '), k=sum(r['ok2'] for r in R), n=len(R)) for t, R in cols]


def C(OUT):
    out = os.path.join(OUT, 'C'); os.makedirs(out, exist_ok=True); fh = open(os.path.join(out, 'settings.jsonl'), 'w')
    rec = lambda D: [f'C1_D{D}_s{i:03d}' for i in range(21, 41)] + [f'C2_D{D}_s{i:03d}' for i in range(1, 21)]
    fix = lambda D: [f'C2_D{D}_s{i:03d}:pilot' for i in range(1, 21)]
    r0 = {3: '0.50', 4: '0.35', 5: '0.35'}
    sets = []
    for D in (3, 4, 5):
        sets += [(f'Depth {D}\nrecipe', f'depth {D}, recipe', rec(D), f'C{2 * D - 3}_sheet_D{D}_recipe', f'Campaign C · depth {D}, recipe (C1 + C2), p = 31'),
                 (f'Depth {D}\nfixed r {r0[D]}', f'depth {D}, fixed r {r0[D]}', fix(D), f'C{2 * D - 2}_sheet_D{D}_fixed',
                  f'Campaign C · depth {D}, fixed rate r {r0[D]} (C2 paired arm), p = 31')]
    other = [('p = 23, depth 4\n(C3)', 'C3: depth 4, p = 23', [f'C3_D4p23_s{i:03d}' for i in range(41, 61)], 'C9_sheet_C3_p23', 'Campaign C3 · depth 4 at p = 23, recipe'),
             ('Width 512, depth 4\n(C4)', 'C4: depth 4, width 512', [f'C4_D4w512_s{i:03d}' for i in range(61, 65)], 'C10_sheet_C4_w512', 'Campaign C4 · depth 4 at width 512, recipe (x* 4.5)'),
             ('Normalized-gradient\nhidden, depth 3 (C5)', 'C5: depth 3, normalized-gradient hidden', [f'C5_D3ngd_s{i:03d}' for i in range(1, 21)],
              'C11_sheet_C5_ngd', 'Campaign C5 · depth 3, normalized-gradient hidden updates, recipe')]
    for group, stem in ((sets, 'C1_grid_recipe_vs_fixed'), (other, 'C2_grid_p23_w512_ngd')):
        cols = []
        for ct, name, ids, sstem, stitle in group:
            R = ovw2.load2(ids); cols.append((ct, R)); log_setting(fh, 'C', name, R)
            sheet(R, stitle, out, sstem); print('C', sstem, ovw2.summary(R)['k'], '/', len(R), flush=True)
        print('C grid', grid(cols, out, stem), flush=True)


def D(OUT):
    out = os.path.join(OUT, 'D'); os.makedirs(out, exist_ok=True); fh = open(os.path.join(out, 'settings.jsonl'), 'w')
    A = ovw.analysis('D'); rows = list(A.values()); groups = {}
    for r in rows: groups.setdefault((r['stage'], r['p']), []).append(r)
    order = [('H', p) for p in (23, 31, 47, 61, 97)] + [('HW', 97)] + [('D3', p) for p in (23, 47, 61, 97)] + [('D4', p) for p in (23, 47, 61, 97)]
    name = dict(H='one hidden layer (width 512), hidden-decay dial', HW='one hidden layer, widths 1024 and 2048',
                D3='depth 3 (width 256), last-layer rate dial', D4='depth 4 (width 256), last-layer rate dial')
    for n, (stage, p) in enumerate(order, 1):
        G = groups[(stage, p)]
        keyf = (lambda r: (r['width'], r['dial']['lam_w'])) if stage == 'HW' else (lambda r: r['dial']['lam_w']) if stage == 'H' else (lambda r: r['dial']['r'])
        cols = []
        for dv in sorted({keyf(r) for r in G}):
            ids = sorted([r['id'] for r in G if keyf(r) == dv], key=lambda s: int(s.rsplit('_s', 1)[1]))
            R = ovw2.load2(ids)
            lab = (f'w {dv[0]}, λ_W {dv[1]:g}' if stage == 'HW' else f'λ_W {dv:g}' if stage == 'H' else f'r {dv:g}')
            cols.append((lab, R)); log_setting(fh, 'D', f'{stage} p{p} {lab}', R)
        seeds = 'seeds 1 and 2' if max(len(R) for _, R in cols) > 1 else 'seed 1'
        st = grid(cols, out, f'D{n:02d}_{stage}_p{p}', note=f'Campaign D · {name[stage]} at p = {p} ({seeds})')
        print('D', stage, p, [(s['title'], s['k'], s['n']) for s in st], flush=True)


def E(OUT):
    out = os.path.join(OUT, 'E'); os.makedirs(out, exist_ok=True); fh = open(os.path.join(out, 'settings.jsonl'), 'w')
    A = ovw.analysis('E'); opt = dict(paper="the paper's learner", hybrid='hybrid (polar on attention and MLP, AdamW on embeddings)', adamw='AdamW')
    for n, o in enumerate(('paper', 'hybrid', 'adamw'), 1):
        cols = []
        for L in (1, 2):
            for rid in sorted(k for k in A if k.startswith(f'dev_L{L}_{o}_')):
                s, dv = rid.split('_')[3], rid.split('_')[4]
                R = ovw2.load2([rid]); lab = f"{L}L · ×{s[1]}\n{dv.replace('lw', 'λ ').replace('wd', 'wd ')}"
                cols.append((lab, R)); log_setting(fh, 'E', rid, R)
        st = grid(cols, out, f'E{n}_{o}', note=f'Campaign E · transformer at p = 31, {opt[o]} (one seed per column)', full=True)
        print('E', o, [(s['title'], s['k'], s['n']) for s in st], flush=True)


def F(OUT):
    out = os.path.join(OUT, 'F'); os.makedirs(out, exist_ok=True); fh = open(os.path.join(out, 'settings.jsonl'), 'w')
    spec = MF.specs()
    for fid in ('acrossp_1hl', 'acrossp_depth2', 'acrossp_depth3', 'acrossp_depth4', 'depths_p31_full'):
        cols_spec, rows, height = spec[fid]; cols = []
        for title, ids in cols_spec:
            if isinstance(ids, (str, tuple)): ids = ovw.c_prefix(ids)
            ids = [i for i in ids if any(ovw.store(k).has(i) for k in ('F', 'D', 'dev', 'C'))]
            R = ovw2.load2(ids); cols.append((title, R)); log_setting(fh, 'F-grid', f"{fid} | {title.replace(chr(10), ' ')}", R)
        st = grid(cols, out, fid, rows=rows, height=height); print('F', fid, [(s['title'], s['k'], s['n']) for s in st], flush=True)
    man = json.load(open(ovw.MANIFEST_F))
    desc = dict(H='1 hidden layer (width 512)', HW='1 hidden layer (width 2048)', D3='depth 3 (width 256)', D4='depth 4 (width 256)',
                D2='depth 2 (width 256)')
    order = {'H': 0, 'HW': 1, 'D2': 2, 'D3': 3, 'D4': 4}
    for n, s in enumerate(sorted(man['settings'], key=lambda s: (order[s['family']], s['p'])), 1):
        ids = s['prior'] + s['new']; R = ovw2.load2(ids)
        dial = ', '.join(f"{'λ_W' if k == 'lam_w' else k} {v:g}" for k, v in s['dial'].items())
        sheet(R, f"Campaign F · {desc[s['family']]}, p = {s['p']}, {dial}", out, f"F{n:02d}_sheet_{s['setting']}",
              note=f"Seeds: {', '.join(i.rsplit('_s', 1)[1].split('_')[0] for i in s['prior'])} from {s['source']}; "
                   f"{s['new'][0].rsplit('_s', 1)[1].split('_')[0]}–{s['new'][-1].rsplit('_s', 1)[1].split('_')[0]} new in campaign F.")
        log_setting(fh, 'F', s['setting'], R); print('F', s['setting'], ovw2.summary(R)['k'], '/', len(R), flush=True)


def G(OUT):
    out = os.path.join(OUT, 'G'); os.makedirs(out, exist_ok=True); fh = open(os.path.join(out, 'settings.jsonl'), 'w')
    man = json.load(open(ovw.MANIFEST_G))
    desc = dict(H='1 hidden layer (width 512)', D3='depth 3 (width 256)', D4='depth 4 (width 256)')
    for n, s in enumerate(man['settings'], 1):
        R = ovw2.load2(s['new'])
        dial = ', '.join(f"{'λ_W' if k == 'lam_w' else k} {v:g}" for k, v in s['dial'].items())
        sheet(R, f"Campaign G · {desc[s['family']]}, p = {s['p']}, {dial}", out, f"G{n:02d}_sheet_{s['setting']}",
              note=f"Seeds 1–10, new in campaign G: the p = 97 setting ({s['compare_p97']}) run unchanged at p = 101.")
        log_setting(fh, 'G', s['setting'], R); print('G', s['setting'], ovw2.summary(R)['k'], '/', len(R), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('campaigns', nargs='*', default=['C', 'D', 'F', 'G'])
    ap.add_argument('--out', required=True); ap.add_argument('--no-figs', action='store_true')
    a = ap.parse_args(); FIGS = not a.no_figs; ovw.configure()
    for g in a.campaigns: {'C': C, 'D': D, 'E': E, 'F': F, 'G': G}[g](a.out)
