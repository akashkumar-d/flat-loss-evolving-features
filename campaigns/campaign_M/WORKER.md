# Campaign M: instructions for a worker session

You run one share of campaign M: training runs of deep ReLU networks (depths 3 and 4) on modular addition at p = 47 and 61, at several training fractions, with numpy (CPU only). Your worker ID (one of M1-M24) is in the message that started you. Below,
`MX` stands for that ID.

The code is written and tested. Your job:

- start it;
- keep this session active until the whole campaign is finished (about 9 hours);
- report problems.

Every command below is run from the repository root.

## Rules

- **Files.** Do not edit, move, delete or commit any file yourself. The worker commits and pushes its own results, only under
  `natural_plateau_shared/campaign_M/results/` and `natural_plateau_shared/campaign_M/status/`.
- **Git.** Stay on the branch this session was set up with. Do not switch branches, merge, rebase, open pull requests or push
  anything yourself.
- **Scope.** Do not start other experiments, use a GPU, pass `--cores`, or change any setting.
- **Secrets.** Never write tokens, passwords or credentials anywhere.

## Steps

1. **Branch and code.**
   - If your session instructions name a branch to develop on and you are not on it yet, create or check it out first. Stay on
     it from then on.
   - Check that `natural_plateau_shared/campaign_M/worker.py` exists. If it does not, run
     `git pull origin claude/focused-gauss-cy16t3`.
2. **Python packages.**
   - Check: `python3 -c "import numpy; print(numpy.__version__)"`. If that fails, run `pip install numpy` (if pip refuses because
     the environment is externally managed: `pip install --break-system-packages numpy`).
3. **Selftest** (a few minutes; give the Bash tool a timeout of 600000 ms; it also makes one small test push to your branch):

       python3 natural_plateau_shared/campaign_M/worker.py selftest --worker MX

   - The last line must start with `SELFTEST PASS`.
   - If it says `SELFTEST FAIL`, stop. Tell the user the `FAIL` lines and wait for instructions.
4. **Start and watch:**

       python3 natural_plateau_shared/campaign_M/worker.py watch --worker MX

   - **Timeout.** Give the Bash tool a timeout of 600000 ms (10 minutes). Each call blocks about 9 minutes and then prints a short
     status and the exact command to run next. If your tool cannot wait that long, add `--minutes 1.5`.
   - **Background worker.** The first call starts the worker in the background. Later calls restart it automatically if it has
     stopped.
   - **Repeat.** As soon as a call returns, run the command again. Keep going until the output ends with `ALL DONE`.
     - `ALL DONE` means the whole campaign is finished, not only this session's share.
     - When its own share is done, this session takes over runs of sessions that never started or stopped, and otherwise stands
       by. Keep calling watch while it stands by.
   - **Stay active.** The session must stay active the whole time. An idle session can be paused, and that stops the runs.
   - **Between calls.** Do nothing else and write no commentary. One short line now and then is plenty.
5. **When the output says `ALL DONE`:** reply with that last status block and stop.

## If something goes wrong

- **`PUSH FAILING`.** Keep watching: results are kept and pushed later, and `ALL DONE` waits until they are pushed. Tell the
  user the error line once, and again if it is still failing an hour later.
- **`ERROR ... exited with code`** for a run. The worker retries the run (up to 3 times). Tell the user if the same run fails
  three times.
- **A Python traceback from a command itself.** Tell the user and run the command once more. If it happens again, stop and wait
  for the user.
- **The session was paused or restarted.**
  - Same machine: go back to step 4. Each run resumes from its last checkpoint.
  - Fresh machine: redo steps 2 to 4. Unfinished runs start again; some may already have been taken over by other sessions.
- **Useful for questions:** `python3 natural_plateau_shared/campaign_M/worker.py status --all` shows the whole campaign across
  sessions.

## What the worker does (for your information)

- **Runs.** It runs your share of `manifest.json`, one process per run (one thread each), as many at once as the session's cores hold.
- **Results.** Every finished run is committed under `results/<run>/` and pushed to your branch, together with a status file.
  The status file is also pushed when runs start and every 30 minutes.
- **Other sessions.** Before starting a run, the worker skips any run that another session has finished or is running.
- **Idle.** When its own list is done, it takes over runs whose session never started or stopped, or from the back of a slow
  session's list. Otherwise it stands by until every run is finished.
