"""Run a phase of the plan with N parallel worker processes (each trial in its own subprocess,
1 thread each). Resumable: trials whose result.json exists are skipped.

    python -m experiments.sweep --phase screen --workers 2 [--datasets mqar fact] [--seeds 0]
                                [--shard 0/2] [--set train.steps=3000] [--dry]

``--shard i/n`` keeps every n-th trial starting at i, so two machines can split one phase.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backbone.config import load_run_config, run_name  # noqa: E402
from backbone.paths import DATA_ROOT, RUNS_ROOT  # noqa: E402
from experiments.plan import phase  # noqa: E402


def cmd_for(t, sets, runs, data):
    c = [sys.executable, "-m", "experiments.run", "--structure", t["structure"], "--variant", t["variant"],
         "--dataset", t["dataset"], "--balance", t["balance"], "--seed", str(t["seed"]), "--quiet",
         "--runs", str(runs), "--data", str(data)]
    for s in sets or []:
        c += ["--set", s]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--set", action="append")
    ap.add_argument("--runs", default=str(RUNS_ROOT))
    ap.add_argument("--data", default=str(DATA_ROOT))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="start no new trial after this many minutes (lets a time-limited job end cleanly)")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    trials = phase(a.phase, a.datasets, a.seeds)[i::n]
    todo = []
    for t in trials:
        cfg = load_run_config(t["structure"], t["variant"], {"dataset": t["dataset"], "routing.balance": t["balance"],
                                                             "train.seed": t["seed"]})
        if not (Path(a.runs) / run_name(cfg) / "result.json").exists():
            todo.append(t)
    print(f"phase {a.phase}: {len(trials)} trials, {len(todo)} to run", flush=True)
    if a.dry:
        for t in todo:
            print(" ".join(cmd_for(t, a.set, a.runs, a.data)))
        return
    t0 = time.time()
    done = [0]

    def work(t):
        if a.max_minutes is not None and time.time() - t0 > a.max_minutes * 60:
            return
        # another sweep process may have taken this trial since the list was made: claim it with a lock file
        cfg = load_run_config(t["structure"], t["variant"], {"dataset": t["dataset"], "routing.balance": t["balance"],
                                                             "train.seed": t["seed"]})
        rd = Path(a.runs) / run_name(cfg)
        rd.mkdir(parents=True, exist_ok=True)
        if (rd / "result.json").exists():
            return
        try:
            fd = os.open(rd / ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
        except FileExistsError:
            if time.time() - (rd / ".lock").stat().st_mtime < 3600:   # stale locks (killed jobs) expire after 1 h
                return
        r = subprocess.run(cmd_for(t, a.set, a.runs, a.data), cwd=ROOT, capture_output=True, text=True)
        (rd / ".lock").unlink(missing_ok=True)
        done[0] += 1
        tail = (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else ("FAILED " + r.stderr[-800:])
        print(f"[{done[0]}/{len(todo)} {time.time() - t0:.0f}s] {tail}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))


if __name__ == "__main__":
    main()
