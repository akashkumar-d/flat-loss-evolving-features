import glob, hashlib, json, os, sys, time
import numpy as np
dev, out = sys.argv[1], sys.argv[2]
sel_path = os.path.join(dev, "out", "selection.json")
sel = json.load(open(sel_path))
tier = "strict" if sel.get("strict", {}).get("choice") else ("relaxed" if sel.get("relaxed", {}).get("choice") else None)
if tier is None:
    print("NO QUALIFYING RECIPE: no confirmation launched"); sys.exit(2)
ch = sel[tier]["choice"]
cfg = None
for f in sorted(glob.glob(os.path.join(dev, "manifests", "shard_*.json"))):
    for c in json.load(open(f))["configs"]:
        if c["config_id"] == ch["config_id"]:
            cfg = dict(c)
cfg["windows"] = [{"t0": ch["t0"], "t1": ch["t1"], "window_id": "hero_locked_dev_selected"}]
rng = np.random.default_rng(20260930)
pairs = [(int(a), int(b)) for a, b in rng.integers(0, 2**63, size=(20, 2), dtype=np.uint64)]
runs = [dict(config_id=cfg["config_id"], confirmation_pair=k + 1, init_seed=a, split_seed=b,
             run_id=f"{cfg['config_id']}_confirm{k+1:02d}") for k, (a, b) in enumerate(pairs)]
base = json.load(open(os.path.join(dev, "manifests", "shard_00.json")))
os.makedirs(os.path.join(out, "manifests"), exist_ok=True)
sizes = [8, 8, 4]; i = 0
for s, n in enumerate(sizes):
    m = dict(base, campaign_id=f"1hl-hero-confirm-20260929-v1-shard{s:02d}", configs=[cfg], runs=runs[i:i + n],
             evidence="fresh 20-seed confirmation of the dev-selected hero recipe and window (locked before launch)")
    json.dump(m, open(os.path.join(out, "manifests", f"shard_{s:02d}.json"), "w"), indent=1); i += n
lock = dict(locked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), selection_tier=tier, choice=ch, config=cfg,
            selection_json_sha256=hashlib.sha256(open(sel_path, "rb").read()).hexdigest(),
            seed_rule="numpy.random.default_rng(20260930).integers(0, 2**63, size=(20,2), dtype=uint64)", pairs=pairs,
            hero_rule="hero figure = confirmation pair 1", endpoints_to_report=["range<=10%", "window-min CE>=0.1", "dA_H>=0.06",
            "dAGOP-dnull>=0.02", "held-out<=20% at t0 and first 90% crossing in window", "100% train accuracy on window", "strict joint (score.json)"])
json.dump(lock, open(os.path.join(out, "CONFIRM_LOCK.json"), "w"), indent=1)
print("LOCKED", tier, ch)
