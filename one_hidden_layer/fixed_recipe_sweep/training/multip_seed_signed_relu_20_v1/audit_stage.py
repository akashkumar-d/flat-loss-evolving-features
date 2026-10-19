import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

from batch_config import HERE, PRIOR, ROSTER, validate_roster
spec = importlib.util.spec_from_file_location("independent_prior_checker", PRIOR / "check_outputs.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.HERE = HERE


def main():
    validate_roster()
    start = time.monotonic()
    result_hash = checker.sha(HERE / "RESULTS.json")
    source_hash = checker.sha(__file__)
    root = json.loads((HERE / "RESULTS.json").read_text())
    checker.dump(HERE / "CHECK_LAUNCH.json", dict(source_sha256=source_hash,
        independent_checker_sha256=checker.sha(PRIOR / "check_outputs.py"),
        results_sha256=result_hash, work_seconds=1080, new_training_updates=0))
    audit = checker.Audit(start + 1080)
    reports, error, status = [], None, "failed"
    try:
        closed = json.loads((HERE / "CLOSED.json").read_text())
        launch = json.loads((HERE / "LAUNCH.json").read_text())
        assert closed["results_sha256"] == result_hash and closed["status"] == root["status"]
        assert root["roster"] == launch["roster"] == ROSTER
        assert len(root["runs"]) == root["declared_runs"] == 120
        assert not root["source_binding_errors"]
        assert root["source_sha256"] == launch["sources"]
        assert root["total_new_updates"] == sum(r["updates_completed"] or 0 for r in root["runs"])
        audit.bindings(root["source_sha256"], HERE)
        audit.bindings(root["output_sha256"], HERE)
        for cfg, record in zip(ROSTER, root["runs"]):
            audit.tick()
            report = checker.check_case(audit, cfg, record)
            reports.append(report)
            print(json.dumps(report), flush=True)
        assert sum(r["status"] == "complete" for r in reports) == root["completed_runs"]
        assert checker.sha(HERE / "RESULTS.json") == result_hash
        assert checker.sha(__file__) == source_hash
        status = "complete"
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    out = dict(status=status, scientific_status=root["status"], results_sha256=result_hash,
               source_sha256=source_hash, runs=reports, declared_roster_retained=120,
               checkpoints_checked=sum(r["checkpoints_checked"] for r in reports),
               crossed_tables_checked=sum(r["crossed_tables_checked"] for r in reports),
               scalar_comparisons=audit.scalar_comparisons, hash_bindings=audit.hash_bindings,
               maximum_absolute_discrepancy=audit.maximum_discrepancy, verified_hashes=audit.verified_hashes,
               elapsed_seconds=time.monotonic() - start, error=error,
               new_training_updates=0, interval_arithmetic=False)
    checker.dump(HERE / "CHECKS.json", out)
    print(json.dumps({key: out[key] for key in ["status", "checkpoints_checked", "scalar_comparisons", "error"]}), flush=True)
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
