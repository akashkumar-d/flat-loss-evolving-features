import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
DATA = Path(os.environ.get("PAPER_DATA", CODE / "data")).expanduser().resolve()

for _p in (CODE / "deep_mlp" / "depths3to5_preregistered_p31", CODE / "deep_mlp" / "development_depths1to5", CODE / "posthoc_window", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ONEHL = DATA / "onehl"
H = ONEHL / "H"
L1 = ONEHL / "L1"
L1_DEV = ONEHL / "L1_dev"
L2 = ONEHL / "L2"
L3 = ONEHL / "L3"
L4 = ONEHL / "L4"
P23A = ONEHL / "p23A_compact.tar.gz"
P23B = ONEHL / "p23B_compact.tar.gz"

DEEP = DATA / "deep"
CAMPAIGN = {k: DEEP / f"campaign_{k}" for k in "CDEFGH"}
DEV_RUNS = DEEP / "development"
DEV_EXTRA = DEEP / "development_extra"

WINDOW_RESULTS = CODE / "posthoc_window" / "results"
CAMPAIGN_CODE = {"C": CODE / "deep_mlp" / "depths3to5_preregistered_p31", "D": CODE / "deep_mlp" / "across_primes_development", "F": CODE / "deep_mlp" / "across_primes_ten_seeds", "G": CODE / "deep_mlp" / "across_primes_p101", "H": CODE / "one_hidden_layer" / "theorem41_iid_init_p101"}
DEV_MANIFEST = CODE / "deep_mlp" / "development_depths1to5" / "RUN_MANIFEST.tsv"
EXPECTED = HERE.parent / "expected"
