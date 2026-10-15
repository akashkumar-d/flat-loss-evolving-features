from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
ANCHORS = [(11, 24, 20000, 35000), (13, 24, 16000, 32000),
           (17, 24, 13000, 21000), (23, 40, 15000, 21000),
           (29, 80, 18000, 24000), (61, 256, 25000, 31000)]


def build(candidate=False):
    campaign = "1hl-protocol-v1-" + ("candidate" if candidate else "anchor") + "-dev-20260927"
    configs = []
    for p, c, t0, t1 in ANCHORS:
        base = dict(p=p, width=512, c=c, train_fraction=.8, activation="relu",
                    eta_W=.03, eta_V=.03, decay_W=.005, decay_V=.2,
                    margin_tolerance=1e-10, polar_cutoff=dict(mode="relative", value=1e-10),
                    windows=[dict(window_id="primary_fixed", t0=t0, t1=t1)])
        alternatives = [{}]
        if candidate:
            alternatives = ([dict(decay_W=.01), dict(decay_W=.015)] if p in (11,13) else
                            [dict(eta_V=.01), dict(c={17:28, 23:48, 29:96, 61:256}[p],
                                                  **(dict(decay_W=.01) if p == 61 else {}))])
        for index, overrides in enumerate(alternatives):
            config = dict(base, **overrides)
            config["config_id"] = f"p{p}_{'candidate'+str(index+1) if candidate else 'anchor'}"
            configs.append(config)
    seeds = np.random.SeedSequence([20260927, 9317, int(candidate)]).spawn(18)
    pair_seeds = [tuple(map(int, seed.generate_state(2))) for seed in seeds]
    runs = []
    for replicate in range(3):
        for config in configs:
            pi = [x[0] for x in ANCHORS].index(config["p"])
            init_seed, split_seed = pair_seeds[replicate * 6 + pi]
            runs.append(dict(run_id=f"{config['config_id']}_dev{replicate+1}",
                             config_id=config["config_id"], init_seed=init_seed,
                             split_seed=split_seed, development_pair=replicate+1))
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(ROOT.glob("*.py"))}
    return dict(campaign_id=campaign, evidence="prospective development; not confirmation",
                postprocess_after_each_run=True,
                updates=50000, limits=dict(workers=8, global_seconds=10800,
                                          per_arm_seconds=7200),
                observation=dict(cheap_every=100, full_every=500, checkpoint_every=500),
                configs=configs, runs=runs, source_hashes_at_preparation=sources,
                window_provenance="Chosen from historical cohorts before fresh seeds; same fixed window for candidate variants, no favorable window substitution.",
                split_design="uniform fixed-size subset without replacement; n=floor(0.8*p*p)",
                confirmation="Select and lock recipes after development; 40 fresh independent pairs per selected recipe, declared before two batches of20. Not launched by this manifest.",
                candidate_pairing="Candidate recipes within a modulus share three pairs; do not count arms as independent pairs.")


if __name__ == "__main__":
    for candidate in (False, True):
        path = ROOT / ("candidate_manifest.json" if candidate else "anchor_manifest.json")
        if path.exists():
            raise SystemExit(f"Refusing to overwrite frozen manifest: {path}")
        path.write_text(json.dumps(build(candidate), indent=2, allow_nan=False) + "\n")
        print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
