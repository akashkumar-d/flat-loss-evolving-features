import json
import os
import sys
import time
import traceback

from batch_config import HERE, THREAD_VARS
from pipeline import dump, sha, supervise_stage


def main():
    assert all(os.environ.get(k) == "1" for k in THREAD_VARS)
    launch = json.loads((HERE / "PIPELINE_LAUNCH.json").read_text())
    original_status = json.loads((HERE / "PIPELINE_STATUS.json").read_text())
    result = json.loads((HERE / "RESULTS.json").read_text())
    closed = json.loads((HERE / "CLOSED.json").read_text())
    assert result["status"] == "complete" and result["completed_runs"] == 120
    assert result["total_new_updates"] == 4800000
    assert closed["results_sha256"] == sha(HERE / "RESULTS.json")
    assert all(sha(p) == h for p, h in launch["sources"].items())
    for name in ["CHECK_LAUNCH.json", "CHECKS.json", "POSTPROCESS_LAUNCH.json",
                 "FIGURES.json", "independent_audit.log", "plots_and_report.log"]:
        assert not (HERE / name).exists(), f"Already started: {name}"
    dump(HERE / "RECOVERY_LAUNCH.json", dict(
        started_unix=time.time(), pid=os.getpid(), python=sys.executable,
        source_sha256=sha(__file__), original_pipeline_status=original_status,
        original_pipeline_launch_sha256=sha(HERE / "PIPELINE_LAUNCH.json"),
        results_sha256=sha(HERE / "RESULTS.json"), new_training_updates=0,
        reason="All training closed; original supervisor absent, saved-data stages never launched.",
        process_absence_check="Read-only process-list check on September 13, 2026; original exec session absent.",
        no_automatic_retries=True))
    records = []
    start = time.monotonic()
    current_stage = "preflight"
    try:
        for current_stage, script, timeout in [
            ("independent_audit", "audit_stage.py", 1200),
            ("plots_and_report", "postprocess.py", 900),
        ]:
            assert all(sha(p) == h for p, h in launch["sources"].items())
            record = dict(stage=current_stage, script=script, status="running",
                          started_unix=time.time(), timeout_seconds=timeout)
            state = dict(status="running", recovery=True, training_status="complete",
                         completed_runs=120, current=record, completed_stages=records,
                         new_training_updates=0)
            dump(HERE / "RECOVERY_STATUS.json", state, exclusive=False)
            dump(HERE / "PIPELINE_STATUS.json", state, exclusive=False)
            print(json.dumps(record), flush=True)
            code = supervise_stage(HERE / script, timeout, HERE / f"{current_stage}.log")
            record.update(exit_code=code, finished_unix=time.time(),
                          status="complete" if code == 0 else "failed")
            records.append(record)
            assert code == 0, f"{current_stage} failed; no automatic retry"
            marker = "CHECKS.json" if current_stage == "independent_audit" else "FIGURES.json"
            assert json.loads((HERE / marker).read_text())["status"] == "complete"
        assert all(sha(p) == h for p, h in launch["sources"].items())
        out = dict(status="complete", recovery=True, training_status="complete",
                   completed_runs=120, declared_runs=120, completed_stages=records,
                   elapsed_seconds=time.monotonic() - start, new_training_updates=0,
                   checks_sha256=sha(HERE / "CHECKS.json"),
                   figures_sha256=sha(HERE / "FIGURES.json"),
                   note="Training completed earlier; only audit and plotting were continued. Original stale status preserved in RECOVERY_LAUNCH.json.")
        dump(HERE / "RECOVERY_CLOSED.json", out)
        dump(HERE / "RECOVERY_STATUS.json", out, exclusive=False)
        dump(HERE / "PIPELINE_STATUS.json", out, exclusive=False)
        print(json.dumps(out), flush=True)
    except BaseException as exc:
        out = dict(status="failed", recovery=True, stage=current_stage,
                   completed_stages=records, new_training_updates=0,
                   error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        dump(HERE / "RECOVERY_FAILED.json", out)
        dump(HERE / "RECOVERY_STATUS.json", out, exclusive=False)
        dump(HERE / "PIPELINE_STATUS.json", out, exclusive=False)
        raise


if __name__ == "__main__":
    main()
