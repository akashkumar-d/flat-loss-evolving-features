import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
DATA = Path(os.environ.get("PAPER_DATA", CODE / "data")).expanduser().resolve()

for _p in (CODE / "deep" / "campaign_C", CODE / "deep" / "development", CODE / "posthoc_window", HERE / "lib", HERE):
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
DEV_MANIFEST = CODE / "deep" / "development" / "RUN_MANIFEST.tsv"
EXPECTED = HERE / "expected"
