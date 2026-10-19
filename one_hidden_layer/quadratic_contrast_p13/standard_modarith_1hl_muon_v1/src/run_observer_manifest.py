#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

from modarith_1hl import RunConfig, run_experiment
from run_experiment import plot_title, plot_trajectory


def load_manifest(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    required = {"schema_version", "common_config", "cells", "stages"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"manifest is missing required fields: {sorted(missing)}")
    return payload, hashlib.sha256(raw).hexdigest()


def resolve_stage(manifest: dict, stage_name: str) -> list[tuple[str, str, dict]]:
    try:
        stage = manifest["stages"][stage_name]
    except KeyError as error:
        choices = ", ".join(sorted(manifest["stages"]))
        raise ValueError(f"unknown stage {stage_name!r}; choose one of: {choices}") from error

    field_names = {field.name for field in fields(RunConfig)}
    tuple_fields = {"agop_ranks", "agop_ridge_alphas"}
    resolved = []
    for seed_pair in stage["seed_pairs"]:
        for cell_name in stage["cells"]:
            cell = manifest["cells"][cell_name]
            config = dict(manifest["common_config"])
            config.update({key: value for key, value in cell.items() if key != "role"})
            config.update(seed_pair)
            unknown = set(config) - field_names
            if unknown:
                raise ValueError(
                    f"cell {cell_name!r} contains fields unsupported by RunConfig: "
                    f"{sorted(unknown)}"
                )
            for name in tuple_fields & config.keys():
                config[name] = tuple(config[name])
            child_name = (
                f"{stage_name}_{cell_name}_seed{config['seed']}_split{config['split_seed']}"
            )
            resolved.append((child_name, cell_name, config))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--stage",
        required=True,
        help="run exactly one stage; the extension gate must be applied separately",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print resolved configs without creating output",
    )
    args = parser.parse_args()

    manifest, digest = load_manifest(args.manifest)
    resolved = resolve_stage(manifest, args.stage)
    preview = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": digest,
        "stage": args.stage,
        "run_count": len(resolved),
        "runs": [
            {"run_name": name, "cell": cell, "config": config}
            for name, cell, config in resolved
        ],
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True, allow_nan=False))
        return

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "resolved_stage.json").open("x", encoding="utf-8") as handle:
        json.dump(preview, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    records = []
    for index, (child_name, cell_name, config_values) in enumerate(resolved, start=1):
        print(f"[{index}/{len(resolved)}] {child_name}", flush=True)
        config = RunConfig(**config_values)
        child = args.output_dir / child_name
        result = run_experiment(config, child)
        plot_trajectory(result["trajectory"], child, plot_title(config))
        records.append(
            {
                "run_name": child_name,
                "cell": cell_name,
                "role": manifest["cells"][cell_name].get("role"),
                "config": asdict(config),
                "summary": result["summary"],
            }
        )

    output = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": digest,
        "stage": args.stage,
        "run_count": len(records),
        "runs": records,
    }
    with (args.output_dir / "stage_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
