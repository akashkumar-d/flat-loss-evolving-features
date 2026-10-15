import json, sys
from pathlib import Path
import numpy as np
import hl1_figs as F

src, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
R = sorted(F.load_compact(src, "H"), key=lambda r: r["id"])
assert len({r["recipe"] for r in R}) == 1 and all(r["window_rule"] == "fixed" for r in R)
h = [r for r in R if r["id"].endswith("_confirm01")][0]
eh = F.hero(h, OUT, "H_hero", logp=True, ce_scale="linear", feat="AH")
F.hero(h, OUT, "H_hero_withAGOP", logp=True, ce_scale="linear")
F.cohort([R], OUT, "H_cohort", ce_log=False, col_titles=[f"p = 31, 80% data, $\\lambda_W$ = 0.015, $\\lambda_V$ = 0.2 ({len(R)} fresh seeds)"])
keys = [("P10", "CE range\n≤ 10%"), ("CE01", "window CE\n≥ 0.1"), ("F06", "$\\Delta A_H$\n≥ 0.06"),
        ("AGOPnull02", "$\\Delta$AGOP − null\n≥ 0.02"), ("rec90", "held-out\n≥ 90%")]
E = F.summary([R], OUT, "H_summary", ["p = 31"], strict_joint=False, short=["31"], keys=keys, ref_dAH=0.06)[0]
for e, r in zip(E, R):
    e["joint"] = bool((r.get("score") or {}).get("joint_PAG10_F02"))
cnt = lambda k: sum(bool(e[k]) for e in E)
med = lambda k: float(np.nanmedian([e[k] for e in E]))
rng = lambda k: (float(np.nanmin([e[k] for e in E])), float(np.nanmax([e[k] for e in E])))
table = dict(n=len(E), window=[R[0]["t0"], R[0]["t1"]], config={k: R[0]["config"][k] for k in ("p", "c", "width", "train_fraction", "eta_W", "eta_V", "decay_W", "decay_V")},
             passes={k: cnt(k) for k in ("P10", "P03", "CE01", "F02", "F06", "AGOP02", "AGOPnull02", "rec90", "fit100", "joint")},
             median=dict(v_min=med("v_min"), ce_min=med("ce_min"), dAH=med("dAH"), dAGOP=med("dAGOP"), dnull=med("dnull")),
             range=dict(v_min=rng("v_min"), ce_min=rng("ce_min"), dAH=rng("dAH")),
             hero=dict(run=h["id"], rule="confirmation pair 1", **{k: (v if not isinstance(v, float) else round(v, 5)) for k, v in eh.items()}))
json.dump(table, open(OUT / "H_endpoints.json", "w"), indent=1)
print(json.dumps(table, indent=1))
