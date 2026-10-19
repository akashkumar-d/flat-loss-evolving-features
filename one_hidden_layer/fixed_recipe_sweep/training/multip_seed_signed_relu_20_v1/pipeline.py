import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from batch_config import HERE, PRIOR, ROSTER, THREAD_VARS, DISK_RESERVE, validate_roster


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value, exclusive=True):
    path = Path(path)
    if exclusive:
        with path.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
    else:
        temp = path.with_name(path.name + ".tmp")
        with temp.open("w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
        os.replace(temp, path)


def supervise_stage(script, timeout, logfile):
    with Path(logfile).open("xb") as log:
        child = subprocess.Popen([sys.executable, "-B", str(script)], cwd=HERE,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
            raise


def main():
    validate_roster()
    assert all(os.environ.get(k) == "1" for k in THREAD_VARS)
    assert shutil.disk_usage(HERE).free >= DISK_RESERVE
    sources = {str(HERE / name): sha(HERE / name) for name in [
        "SCOPE.txt", "batch_config.py", "train_stage.py", "audit_stage.py", "pipeline.py",
        "postprocess.py", "test_batch.py", "test_postprocess.py", "REVIEW.md"]}
    sources.update({str(PRIOR / name): sha(PRIOR / name) for name in [
        "core.py", "run_sweep.py", "check_outputs.py", "plot_sweep.py"]})
    stages = [("training", "train_stage.py", 43800),
              ("independent_audit", "audit_stage.py", 1200),
              ("plots_and_report", "postprocess.py", 900)]
    start = time.monotonic()
    dump(HERE / "PIPELINE_LAUNCH.json", dict(roster=ROSTER, sources=sources,
        python=sys.executable, pid=os.getpid(), started_unix=time.time(),
        stage_timeouts_seconds={name: limit for name, _, limit in stages},
        declared_new_updates=4800000, all_new_seed_pairs=True, no_automatic_retries=True))
    records = []
    current_stage = "preflight"
    try:
        for current_stage, script, timeout in stages:
            assert all(sha(path) == expected for path, expected in sources.items()), "Pinned source changed"
            record = dict(stage=current_stage, status="running", script=script,
                          started_unix=time.time(), timeout_seconds=timeout,
                          log=str(HERE / f"{current_stage}.log"))
            dump(HERE / "PIPELINE_STATUS.json", dict(status="running", current=record,
                 completed_stages=records, elapsed_seconds=time.monotonic() - start), exclusive=False)
            print(json.dumps(record), flush=True)
            code = supervise_stage(HERE / script, timeout, HERE / f"{current_stage}.log")
            record = dict(record, exit_code=code, finished_unix=time.time(), status="complete" if code == 0 else "failed")
            records.append(record)
            assert code == 0, f"Stage {current_stage} failed with exit code {code}; no automatic restart"
            if current_stage == "training":
                r = json.loads((HERE / "RESULTS.json").read_text())
                c = json.loads((HERE / "CLOSED.json").read_text())
                assert c["results_sha256"] == sha(HERE / "RESULTS.json")
                assert len(r["runs"]) == 120
            if current_stage == "independent_audit":
                assert json.loads((HERE / "CHECKS.json").read_text())["status"] == "complete"
            if current_stage == "plots_and_report":
                assert json.loads((HERE / "FIGURES.json").read_text())["status"] == "complete"
                assert (HERE / "REPORT.md").is_file()
        training = json.loads((HERE / "RESULTS.json").read_text())
        out = dict(status="complete", training_status=training["status"], completed_stages=records,
                   completed_runs=training["completed_runs"], declared_runs=120,
                   elapsed_seconds=time.monotonic() - start,
                   note="Pipeline completion does not imply all scientific runs completed or grokked.")
        dump(HERE / "PIPELINE_STATUS.json", out, exclusive=False)
        dump(HERE / "PIPELINE_CLOSED.json", out)
        print(json.dumps(out), flush=True)
    except BaseException as exc:
        out = dict(status="failed", stage=current_stage, completed_stages=records,
                   elapsed_seconds=time.monotonic() - start,
                   error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()),
                   note="Stopped without an automatic retry; no missing scientific outcomes are fabricated.")
        dump(HERE / "PIPELINE_STATUS.json", out, exclusive=False)
        dump(HERE / "PIPELINE_FAILED.json", out)
        raise


if __name__ == "__main__":
    main()
