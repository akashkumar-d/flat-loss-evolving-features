#!/usr/bin/env python3
import argparse, json, math, sys


def walk(a, b, path, out):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        if a != b: out['mismatch'].append(f'{path}: {a!r} != {b!r}')
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if a != b: out['mismatch'].append(f'{path}: {a} != {b}')
            return
        out['floats'] += 1
        if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)): return
        rel = abs(a - b) / max(abs(a), abs(b))
        out['max_rel'] = max(out['max_rel'], rel); out['n_diff'] += 1
        if rel > out['rtol']: out['mismatch'].append(f'{path}: {a!r} vs {b!r} (relative {rel:.1e})')
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): out['mismatch'].append(f'{path}: keys differ: {sorted(set(a) ^ set(b))[:5]}')
        for k in sorted(set(a) & set(b)): walk(a[k], b[k], f'{path}/{k}', out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out['mismatch'].append(f'{path}: lengths {len(a)} != {len(b)}'); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f'{path}[{i}]', out)
        return
    out['mismatch'].append(f'{path}: types {type(a).__name__} != {type(b).__name__}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rebuilt'); ap.add_argument('committed'); ap.add_argument('--rtol', type=float, default=1e-9)
    a = ap.parse_args()
    A = json.load(open(a.rebuilt)); B = json.load(open(a.committed))
    out = dict(mismatch=[], floats=0, n_diff=0, max_rel=0.0, rtol=a.rtol)
    walk(A, B, '', out)
    same_bytes = open(a.rebuilt, 'rb').read() == open(a.committed, 'rb').read()
    print(f"{'byte-identical' if same_bytes else 'not byte-identical'}; {out['floats']} floats compared, {out['n_diff']} differ in the "
          f"last bits (max relative difference {out['max_rel']:.1e}); {len(out['mismatch'])} mismatch(es) beyond rtol {a.rtol:g}")
    for m in out['mismatch'][:20]: print('  ', m)
    sys.exit(1 if out['mismatch'] else 0)


if __name__ == '__main__':
    main()
