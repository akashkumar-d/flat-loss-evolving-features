#!/usr/bin/env python3
import json, os, sys
k = lambda t: f'{t / 1000:.1f}k'
def g(x): return f'{x:+.2f}'.replace('+', '') if abs(x) < 0.005 else f'{x:.2f}'
HEAD = ('| setting | pass (new) | of which refit only | old endpoint | window (median) | held-out gain: median (range) | '
        'refit gain: median (range) | refit | fit in window | final ≥ 0.99 |\n|---|---|---|---|---|---|---|---|---|---|')
def row(s):
    ro = sum(r['ok2'] and r['g_te'] < 0.2 for r in s['runs'])
    return (f"| {s['setting'].replace(' | ', ' · ')} | {s['k']}/{s['n']} | {ro} | {s['k_old']}/{s['n']} | {k(s['W'][0])}–{k(s['W'][1])} | {g(s['g_te'])} "
            f"({g(s['g_te_rng'][0])}–{g(s['g_te_rng'][1])}) | {g(s['g_rf'])} ({g(s['g_rf_rng'][0])}–{g(s['g_rf_rng'][1])}) | "
            f"{'/'.join(s['refit'])} | {s['fit']}/{s['n']} | {s['final_ok']}/{s['n']} |")
out = sys.argv[1]
for camp in (sys.argv[2:] or ['C', 'D', 'E', 'F']):
    fn = os.path.join(out, camp, 'settings.jsonl')
    if not os.path.exists(fn): continue
    S = [json.loads(l) for l in open(fn)]
    main = [s for s in S if s['camp'] == camp]; grid = [s for s in S if s['camp'] == 'F-grid']
    print(f'\n## {camp}\n'); print(HEAD)
    for s in main: print(row(s))
    tot = sum(s['n'] for s in main); kk = sum(s['k'] for s in main); ko = sum(s['k_old'] for s in main)
    print(f'\ntotal: {kk}/{tot} pass (old endpoint {ko}/{tot})')
    if grid:
        print(f'\n### {camp}: columns of the seed grids (figure · column; the same runs as the paper\'s median figures)\n'); print(HEAD)
        for s in grid: print(row(s))
