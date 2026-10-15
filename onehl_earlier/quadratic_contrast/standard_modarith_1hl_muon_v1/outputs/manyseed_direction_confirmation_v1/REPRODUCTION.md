# Reproduce the 40-pair many-seed confirmation

The training protocol and all 40 model/split pairs were frozen before
execution. The manifest SHA-256 is

```text
10478d6b6303c0581137f5fc1c967a079cbdaa25e4d5d63b73463df6aabab7ec
```

From the repository root, verify the training lock:

```bash
sha256sum -c standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/RUN_SOURCE_LOCK.sha256
```

Run the four independent ten-pair stages, using a different output directory
for each stage:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=standard_modarith_1hl_muon_v1/src \
python standard_modarith_1hl_muon_v1/src/run_observer_manifest.py \
  --manifest standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/locked_manifest.json \
  --stage batch_1 \
  --output-dir standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/runs/batch_1
```

Repeat with `batch_2`, `batch_3`, and `batch_4`, changing both `--stage` and
the final output-directory component. The runner refuses to overwrite an
existing stage.

No seed may be replaced after seeing an outcome. An infrastructure-interrupted
run may be repeated only with its identical declared model/split pair, and the
attempt must be retained in the audit record.

After all four stages have written terminal `stage_summary.json` files, verify
the two independently frozen analysis implementations:

```bash
sha256sum -c standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/ANALYSIS_LOCK.sha256
sha256sum -c standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/ANALYSIS_SOURCE_LOCK.sha256
sha256sum -c standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/INDEPENDENT_AUDIT_LOCK.sha256
sha256sum -c standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/INDEPENDENT_AUDIT_SOURCE_LOCK.sha256
```

Create the locked aggregate and ordinary-time figures:

```bash
MPLCONFIGDIR=/tmp/manyseed_confirmation_mpl \
PYTHONPATH=standard_modarith_1hl_muon_v1/src \
python3 \
standard_modarith_1hl_muon_v1/src/analyze_manyseed_confirmation.py \
  --manifest standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/locked_manifest.json \
  --output-root standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/runs \
  --output-dir standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/aggregate_final
```

Run the clean-room endpoint audit into a separate result file:

```bash
python3 -B \
standard_modarith_1hl_muon_v1/src/audit_manyseed_confirmation_independent.py \
  --manifest standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/locked_manifest.json \
  --campaign-root standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1 \
  --runs-root standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/runs \
  --workspace-root "<folder that holds standard_modarith_1hl_muon_v1/>" \
  --output-json standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/aggregate_final/independent_manyseed_audit.json
```

Finally, generate the phase-aligned update-direction dashboard:

```bash
MPLCONFIGDIR=/tmp/manyseed_direction_mpl \
PYTHONPATH=standard_modarith_1hl_muon_v1/src \
python3 \
standard_modarith_1hl_muon_v1/src/plot_update_direction_sequence.py \
  --manifest standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/locked_manifest.json \
  --output-root standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/runs \
  --output-dir standard_modarith_1hl_muon_v1/outputs/manyseed_direction_confirmation_v1/aggregate_final/update_direction_all \
  --stage all
```
