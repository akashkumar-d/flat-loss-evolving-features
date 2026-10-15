import datetime
import hashlib
import json
import sys

sa, sp = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
lock_a = json.load(open("CONFIRM_LOCK_p23.json"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
src = dict(lock_a["source_sha256"])
src["score_p23b.py"] = sha("score_p23b.py")
lock = dict(
    locked_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    purpose="Second fresh confirmation of the p = 23 recipe: held-out accuracy rises from <= 20% to >= 50% while the "
            "cycle-mean training CE does not decrease.",
    disclosure=("The first pre-registered confirmation (CONFIRM_LOCK_p23.json; no decrease up to the first 90% held-out "
                f"crossing) failed its >= 18/20 rule: joint {sa['counts']['joint']}/20, ND {sa['counts']['ND']}/20. "
                "Post hoc, on those 20 evaluation runs, the cycle-mean CE did not decrease up to the first 50% and 70% "
                f"held-out crossings (ND50 {sp['counts']['ND50']}/20, ND70 {sp['counts']['ND70']}/20; largest drawdown to the "
                f"50% crossing {100 * sp['ranges']['drawdown50'][2]:.4f}%), and held-out accuracy at the cycle-mean CE peak was "
                f"{100 * sp['ranges']['heldout_at_peak'][0]:.1f}-{100 * sp['ranges']['heldout_at_peak'][2]:.1f}%. This lock tests "
                "that post-hoc observation on 20 new pairs; none of the first cohort's runs enter this decision."),
    config=lock_a["config"],
    horizon_updates=lock_a["horizon_updates"],
    window=lock_a["window"],
    window_rule="unchanged from CONFIRM_LOCK_p23.json (selected on the 4 development runs only)",
    seed_rule="numpy.random.default_rng(20261002).integers(0, 2**63, size=(20,2), dtype=uint64); no overlap with the first cohort",
    pairs=[[r["init_seed"], r["split_seed"]] for r in json.load(open("manifests/p23_confirmB.json"))["runs"]],
    primary_endpoints=dict(
        P10=lock_a["primary_endpoints"]["P10"],
        ND50=("no loss decrease during the first half of recovery: T50 = first update > t0 with held-out accuracy >= 0.5 "
              "satisfies T50 <= t1 - 500; on [t0, T50] the two-update cycle-mean CE Lbar_t = (L_t + L_{t+1})/2 never falls "
              "more than 0.5% below its running maximum since t0, and Lbar(T50) >= Lbar(t0)"),
        FIT=lock_a["primary_endpoints"]["FIT"],
        H20=lock_a["primary_endpoints"]["H20"],
        F04=lock_a["primary_endpoints"]["F04"]),
    joint="P10 and ND50 and FIT and H20 and F04",
    decision_rule=("CONFIRMED if the joint endpoint holds in at least 18 of 20 runs (two-sided 95% Wilson lower bound >= 0.70); "
                   "otherwise reported as a failed confirmation. Either way the result is reported. A confirmed cohort is "
                   "offered to the authors as a Figure 2 candidate; it does not replace Figure 2 automatically."),
    secondary=["ND70 and ND90 (same definition with 70% / 90%)", "held-out accuracy at the cycle-mean CE peak on [t0, t1) >= 50%",
               "T90 within the 20k horizon", "dAGOP - dnull >= 0.02 (q=4, median of 8 frames)",
               "first refit 90% crossing (500-update grid)", "held-out accuracy at 20000", "window-minimum CE"],
    scoring="score_p23b.py <compact_dir> CONFIRM_LOCK_p23b.json <out.json>",
    illustration_rule="any single-run illustration uses confirmation pair 1",
    source_sha256=src,
    manifest_sha256=sha("manifests/p23_confirmB.json"),
    environment=lock_a["environment"])
json.dump(lock, open("CONFIRM_LOCK_p23b.json", "w"), indent=1)
print(lock["locked_utc"], sha("CONFIRM_LOCK_p23b.json"))
print(lock["disclosure"])
