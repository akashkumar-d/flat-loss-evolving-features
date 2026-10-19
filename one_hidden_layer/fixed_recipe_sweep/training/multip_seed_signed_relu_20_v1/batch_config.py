from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "multip_seed_signed_relu_v1"
MODULI = [11, 13, 17, 19, 23, 29]
REPLICATES = 20
STEPS = 40000
WORKERS = 3
ARM_SECONDS = 3600
TRAIN_SECONDS = 43200
DISK_RESERVE = 8 * 1024**3
THREAD_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]
ROSTER = [dict(name=f"p{p}_seed{rep + 1}", p=p, width=512, replicate=rep + 1,
               init_seed=2026091301 + 6 * rep + j, split_seed=2026191301 + 6 * rep + j,
               steps=STEPS, train_fraction=.8, initial_scale=24., eta_w=.03, eta_v=.03,
               decay_w=.005, decay_v=.2)
          for rep in range(REPLICATES) for j, p in enumerate(MODULI)]


def validate_roster(roster=ROSTER):
    assert len(roster) == len(MODULI) * REPLICATES == 120
    assert len({r["name"] for r in roster}) == 120
    assert len({r["init_seed"] for r in roster} | {r["split_seed"] for r in roster}) == 240
    for p in MODULI:
        assert sorted(r["replicate"] for r in roster if r["p"] == p) == list(range(1, 21))
    assert not ({r["init_seed"] for r in roster} & set(range(2026091201, 2026091210)))
    assert not ({r["split_seed"] for r in roster} & set(range(2026191201, 2026191210)))
    assert sum(r["steps"] for r in roster) == 4800000
    return True
