#!/usr/bin/env python3
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
import numpy as np
import scheme2 as S2


def first_rise(c, tol=1.03):
    i, m = 0, c[0]
    for t in range(1, len(c)):
        if c[t] < m: i, m = t, c[t]
        elif c[t] > tol * m: return i
    return len(c) - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("lines", nargs="+", choices=["H", "L1", "L2", "L3", "L4"])
    a = ap.parse_args()
    src = {"H": S2.P.H, "L1": S2.P.L1, "L2": S2.P.L2, "L3": S2.P.L3, "L4": S2.P.L4}
    report = {}
    for line in a.lines:
        groups = defaultdict(list)
        for f in sorted(glob.glob(str(src[line] / "*.npz"))):
            r = S2.onehl_run(f); t_c, c = r["ce"]
            row = dict(id=r["id"], used=r["start_rule"])
            starts = dict(dip=S2.dip_start(c), first=first_rise(c), zero=0)
            if r["start_rule"] == "knee": starts["knee"] = int(np.searchsorted(t_c, r["start"]))
            for name, st in starts.items():
                q = S2.score(dict(r), t_c, c, st)
                row[name] = dict(start=q["start"], W=q["W"], ok=q["primary"], g_te=round(q["g_te"], 4), g_rf=round(q["g_rf"], 4))
            groups[str(r["recipe"] if line in ("L1", "L3") else r["p"])].append(row)
        report[line] = groups
        for g, rows in groups.items():
            n = len(rows); names = [k for k in ("knee", "dip", "first", "zero") if k in rows[0]]
            msg = " ".join(f"{k}={sum(x[k]['ok'] for x in rows)}" for k in names)
            if "knee" in rows[0]:
                d = [x["dip"]["start"] - x["knee"]["start"] for x in rows]
                msg += (f"; dip vs knee: same verdict {sum(x['dip']['ok'] == x['knee']['ok'] for x in rows)}/{n}, same end "
                        f"{sum(x['dip']['W'][1] == x['knee']['W'][1] for x in rows)}/{n}, start offset {min(d)}..{max(d)}")
            print(line, g, f"n={n}", msg, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"check_starts_{'_'.join(a.lines)}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
