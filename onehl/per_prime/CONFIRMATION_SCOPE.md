# Locked plateau-first confirmation

This is a fresh, prospective 60-run confirmation of three configurations selected
from completed development data: p=17/c=24/head step=.01, p=23/c=48,
and p=29/c=96. There are 20 new initialization/split pairs per configuration.
The main endpoint is P10: per-run raw CE range/min <=10% over the unchanged
predeclared fixed window. Fourier change, AGOP gain, fit, and test recovery
remain separately reported. Existing strict grokking criteria are not redefined.

Training code and all scoring/observer definitions are unchanged. This is
width-512 dense ReLU, full-batch CE, fixed support-Polar first-layer descent and
normalized-head descent with fixed weight decay. There is no adaptive CE
balancing, imposed floor, head installation, or outcome-dependent stopping.

The 50k horizon and all 60 seeds/windows are locked before the first update.
All declared runs stay in the denominator, including failed/partial/unrun arms;
there are no outcome-based retries or replacements. Eight single-thread CPU
workers run under a 12h campaign cap and a 3h per-arm cap. These safety ceilings
are not estimates of runtime. No GPU request or machine change is required.

The only source patch in this staging directory is a presentation-label fix
in postprocess.py: evidence is read from the manifest instead of always saying
DEVELOPMENT. STAGING_VERIFICATION.json records byte identity for the learner,
runner, endpoints, observers and plotting, plus before/after postprocessor
hashes. SEED_AUDIT.json states the local audit scope; unseen remote or binary-only
seed declarations are not certified by that audit.

An optional separate 60-run plateau-control declaration for p=11/13/61 exists
locally but is not uploaded or launched here. It is not part of this denominator.
The manifest's declared_not_launched field is its immutable pre-launch status;
PROGRESS.json and the launch receipt are authoritative for execution status.
