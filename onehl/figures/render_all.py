import json, sys
from pathlib import Path
import numpy as np
import hl1_figs as F

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
C = F.C
tables, heroes = {}, {}


def groups_by(R, key):
    by = {}
    for r in R:
        by.setdefault(r[key], []).append(r)
    ks = sorted(by, key=lambda k: (by[k][0]["p"], str(k)))
    for k in ks:
        by[k].sort(key=lambda r: r["id"])
    return ks, by


def table(name, ks, by, labels):
    rows = []
    for k, lab in zip(ks, labels):
        E = [F.endpoints(r) for r in by[k]]
        n = len(E)
        cnt = lambda f: sum(bool(e[f]) for e in E)
        med = lambda f: float(np.nanmedian([e[f] for e in E])) if np.isfinite([e[f] for e in E]).any() else None
        row = dict(group=lab, recipe=str(k), n=n, P10=cnt("P10"), P03=cnt("P03"), F02=cnt("F02"), AGOP02=cnt("AGOP02"),
                   rec90=cnt("rec90"), median_v_min=med("v_min"), median_dAH=med("dAH"), median_dAGOP=med("dAGOP"),
                   median_dnull=med("dnull"), window_rule=by[k][0]["window_rule"],
                   windows=sorted({(r["t0"], r["t1"]) for r in by[k]})[:3] if by[k][0]["window_rule"] == "fixed" else None)
        sc = [r.get("score") or {} for r in by[k]]
        if all("joint_PAG10_F02" in s for s in sc):
            row["strict_joint"] = sum(bool(s["joint_PAG10_F02"]) for s in sc)
        rows.append(row)
    tables[name] = rows


R = F.load_compact(D / "l1_confirm", "L1")
ks, by = groups_by(R, "recipe")
labs = [f"p = {by[k][0]['p']}, c = {by[k][0]['config']['c']:g}" for k in ks]
h = by["p17_candidate1"][0]
heroes["L1"] = dict(run=h["id"], rule="first confirmation pair of the p = 17 recipe", **F.hero(h, OUT, "L1_hero"))
F.cohort([by[k] for k in ks], OUT, "L1_cohort", col_titles=labs)
F.summary([by[k] for k in ks], OUT, "L1_summary", labs, strict_joint=True, short=[str(by[k][0]["p"]) for k in ks])
table("L1_confirmation", ks, by, labs)

R = F.load_compact(D / "l1_dev", "L1dev")
ks, by = groups_by(R, "recipe")
table("L1_development", ks, by, [str(k) for k in ks])

R = F.load_compact(D / "l2", "L2")
ks, by = groups_by(R, "p")
labs = [f"p = {k}" for k in ks]
rows2 = [("ta", "test_acc", C["test"], "held-out accuracy", 100), ("to", "AH", C["AH"], "hidden Fourier $A_H$", 1000),
         ("to", "agop_minus_null", C["agop"], "AGOP purity − null", 1000)]
F.cohort([by[k] for k in ks], OUT, "L2_cohort", col_titles=labs, rows=rows2)
F.summary([by[k] for k in ks], OUT, "L2_summary", labs, strict_joint=False, short=[str(k) for k in ks])
table("L2_c24_replication", ks, by, labs)

R = F.load_compact(D / "l3", "L3")
ks, by = groups_by(R, "recipe")
labs = [f"p = {by[k][0]['p']}, $\\lambda_W$ = {by[k][0]['config']['decay_W']:g}" for k in ks]
h = [r for r in R if r["id"] == "seed_p23_lw0.01_lv0.2_s0"][0]
heroes["L3"] = dict(run=h["id"], rule="first job of jobs_1hl_seed.json", **F.hero(h, OUT, "L3_hero"))
F.cohort([by[k] for k in ks], OUT, "L3_cohort", col_titles=labs)
F.summary([by[k] for k in ks], OUT, "L3_summary", labs, strict_joint=False, short=[str(by[k][0]["p"]) for k in ks])
table("L3_natural_plateau", ks, by, [f"p = {by[k][0]['p']}, lambda_W = {by[k][0]['config']['decay_W']:g}" for k in ks])

R = F.load_compact(D / "l4", "L4")
ks, by = groups_by(R, "recipe")
h = by[ks[0]][0]
heroes["L4"] = dict(run=h["id"], rule="first pair of the locked 40-pair list", **F.hero(h, OUT, "L4_hero", dloc="lower right"))
rows4 = [("ta", "test_acc", C["test"], "held-out accuracy", 500), ("to", "AH", C["AH"], "hidden Fourier $A_H$", 500),
         ("to", "agop", C["agop"], "AGOP top-4 purity", 500)]
F.cohort([by[k] for k in ks], OUT, "L4_cohort", col_titles=["quadratic, p = 13, width 64 (40 pairs)"], rows=rows4)
F.summary([by[k] for k in ks], OUT, "L4_summary", ["quadratic, p = 13"], strict_joint=False, short=["13"])
table("L4_quadratic", ks, by, ["quadratic p = 13, width 64"])

json.dump(dict(tables=tables, heroes=heroes), open(OUT / "endpoint_tables.json", "w"), indent=1, default=str)
print(json.dumps({k: [(r["group"], r["n"], r["P10"], r["F02"], r["AGOP02"], r["rec90"], r.get("strict_joint")) for r in v] for k, v in tables.items() if k != "L1_development"}, indent=0))
