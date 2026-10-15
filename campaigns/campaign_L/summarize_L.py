import json, sys, collections, statistics as S
R = json.load(open(sys.argv[1]))
runs = R['runs']; pap = R['paper_frac08']
def rng(v, fmt='{:.0f}', scale=1.0):
    v = [x for x in v if x is not None]
    if not v: return '-'
    a, b = min(v) / scale, max(v) / scale
    return fmt.format(a) if abs(a - b) < 1e-12 else fmt.format(a) + '-' + fmt.format(b)
def lead(x):
    r = x['refit']; return (r['t_heldout50'] - r['t_refit50']) if (r['t_heldout50'] is not None and r['t_refit50'] is not None) else None
cells = collections.OrderedDict()
for rid, r in runs.items():
    rule = r['dial']['lam_v_rule']; key = (r['depth'], r['p'], rule, r['dial']['lam_v'], r['dial']['lam_w'])
    cells.setdefault(key, []).append(r)
order = sorted(cells, key=lambda k: (k[0], k[1], {'paper': 0, 'sqrt': 1, 'linear': 2}.get(k[2], 3), k[4]))
print('| depth, p | setting (lam_v, lam_w) | endpoint 0.5 (0.8) | refit 0.5 (0.8) | pass3 0.5 | t10 at 0.5 (0.8), k | t90 at 0.5 (0.8), k | refit lead at 0.5 (0.8), k | CE peak / log p at 0.5 (0.8) | final held-out at 0.5 |')
print('|---|---|---|---|---|---|---|---|---|---|')
tot = collections.Counter()
for key in order:
    D, p, rule, lv, lw = key; rs = sorted(cells[key], key=lambda r: r['seed'])
    L = [r['L'] for r in rs]; K = [r['K'] for r in rs if 'K' in r]
    if rule == 'paper':
        K = pap.get(f'D{D}_p{p}', [])
    e_L = sum(bool(x['endpoint']['primary']) for x in L); f_L = sum(bool(x['refit']['ok']) for x in L); f3 = sum(bool(x['refit']['ok3']) for x in L)
    e_K = sum(bool(x['endpoint']['primary']) for x in K); f_K = sum(bool(x['refit']['ok']) for x in K)
    tot['n'] += len(L); tot['e'] += e_L; tot['f'] += f_L; tot['f3'] += f3; tot['nK'] += len(K); tot['eK'] += e_K; tot['fK'] += f_K
    t10 = rng([x['endpoint']['t10'] for x in L], '{:.0f}', 1000); t10k = rng([x['endpoint']['t10'] for x in K], '{:.0f}', 1000)
    nt90 = sum(x['endpoint']['t90'] is None for x in L)
    t90 = rng([x['endpoint']['t90'] for x in L], '{:.0f}', 1000) + (f' ({nt90} never)' if nt90 else ''); t90k = rng([x['endpoint']['t90'] for x in K], '{:.0f}', 1000)
    ld = rng([lead(x) for x in L], '{:.1f}', 1000); ldk = rng([lead(x) for x in K], '{:.1f}', 1000)
    ce = rng([x['endpoint']['CE_peak_over_logp'] for x in L], '{:.2f}'); cek = rng([x['endpoint']['CE_peak_over_logp'] for x in K], '{:.2f}')
    fin = rng([x['endpoint']['final_test'] for x in L], '{:.3f}')
    name = f"{'paper ' if rule == 'paper' else rule + ' '}{lv:g}, {lw:g}"
    print(f'| D{D} p{p} | {name} | {e_L}/{len(L)} ({e_K}/{len(K)}) | {f_L}/{len(L)} ({f_K}/{len(K)}) | {f3}/{len(L)} | {t10} ({t10k}) | {t90} ({t90k}) | {ld} ({ldk}) | {ce} ({cek}) | {fin} |')
print(); print('totals:', dict(tot))
why_e = collections.Counter(); why_f = collections.Counter()
for r in runs.values():
    for w in (r['L']['endpoint']['why_not'] or '').split(', '):
        if w: why_e[w] += 1
    why_f[','.join(r['L']['refit']['why']) or 'pass'] += 1
print('endpoint failing checks (0.5):', dict(why_e)); print('refit outcomes (0.5):', dict(why_f))
for key in order:
    rs = cells[key]; L = [r['L'] for r in rs]
    n99 = sum(x['endpoint']['final_test'] >= 0.99 for x in L); n0 = sum(x['endpoint']['t10'] is None for x in L)
    print(key, 'final>=0.99:', n99, '/', len(L), '| no held-out 0.1 ever:', n0, '| horizon', set(x['steps'] for x in L if x['endpoint']['final_test'] < 0.99))
