"""Paper 3 -- full clean sweep driver.

Archives any existing results, then runs all four scenarios for the four
baselines once, and for BeamVNE at every (beam width, queue weight) in the grid.
Each cell group runs as its own python subprocess, because BeamVNE reads
BEAM_WIDTH and QUEUE_WEIGHT at import time and they cannot be changed within a
live process.

Run from the repository root:

    python run_all.py                 # full sweep, 50 episodes
    python run_all.py --episodes 2    # smoke test
    python run_all.py --dry-run       # print the plan and exit

Not subject to PowerShell execution policy.
"""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EVALUATOR = "evaluator_networked.py"

# (beam width, queue weight) grid for BeamVNE.
BEAM_WIDTHS = ["4", "8", "16"]
QUEUE_WEIGHTS = ["0.0", "0.5"]

BASELINES = "SoftIso,CloudQC,GreedyDfs,Random"


def plan():
    """The cell groups to run, in order. Baselines first so that the longest
    job surfaces failures early rather than after the cheap cells."""
    jobs = [("baselines (4 algos x 4 scenarios)",
             {"ALGOS": BASELINES, "BEAM_WIDTH": "16", "QUEUE_WEIGHT": "0.0"})]
    for b in BEAM_WIDTHS:
        for q in QUEUE_WEIGHTS:
            jobs.append((f"BeamVNE B{b} Q{q} (4 scenarios)",
                         {"ALGOS": "BeamVNE", "BEAM_WIDTH": b, "QUEUE_WEIGHT": q}))
    return jobs


def archive_previous(stamp):
    """Move existing results aside. Both the agg and rngcheck writers append, so
    a rerun into a live file would stack duplicate episode numbers."""
    targets = sorted(Path(".").glob("agg_*.jsonl")) + \
              sorted(Path(".").glob("rngcheck_*.csv")) + \
              sorted(Path(".").glob("sweep_manifest_*.csv"))
    if not targets:
        print("No previous result files to archive")
        return None
    dest = Path("archive") / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for t in targets:
        shutil.move(str(t), str(dest / t.name))
    print(f"Archived {len(targets)} previous result file(s) to {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=os.environ.get("EPISODES", "50"),
                    help="episodes per cell (default 50)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running anything")
    ap.add_argument("--no-archive", action="store_true",
                    help="do not move existing results aside (appends!)")
    args = ap.parse_args()

    jobs = plan()

    if args.dry_run:
        print(f"{len(jobs)} invocations, {args.episodes} episodes per cell:")
        for label, env in jobs:
            print(f"  {label}")
            print(f"      ALGOS={env['ALGOS']} BEAM_WIDTH={env['BEAM_WIDTH']} "
                  f"QUEUE_WEIGHT={env['QUEUE_WEIGHT']}")
        return 0

    if not Path(EVALUATOR).exists():
        print(f"ERROR: {EVALUATOR} not found -- run this from the repository root",
              file=sys.stderr)
        return 2

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not args.no_archive:
        archive_previous(stamp)

    log_path = Path(f"sweep_{stamp}.log")
    sweep_start = time.time()
    results = []

    for label, job_env in jobs:
        env = os.environ.copy()
        env.update(job_env)
        env["EPISODES"] = str(args.episodes)
        env["PYTHONIOENCODING"] = "utf-8"
        env["NAT_DEBUG"] = "0"
        env["DIRECTED"] = "0"          # undirected task graph, matching SoftIso

        print(f"\n{'=' * 70}\n=== {label}\n{'=' * 70}", flush=True)
        t0 = time.time()

        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n### {label}\n")
            log.flush()
            proc = subprocess.Popen(
                [sys.executable, "-u", EVALUATOR],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            proc.wait()

        mins = (time.time() - t0) / 60
        status = "ok" if proc.returncode == 0 else f"EXIT {proc.returncode}"
        results.append((label, mins, status))
        print(f"--- {label}: {status} in {mins:.1f} min", flush=True)

        if proc.returncode != 0:
            print(f"\nStopping: '{label}' exited {proc.returncode}. "
                  f"Earlier results are kept; see {log_path} and "
                  f"sweep_manifest_reset.csv for which cells completed.",
                  file=sys.stderr)
            break

    total = (time.time() - sweep_start) / 60
    print(f"\n{'=' * 70}")
    print(f"Sweep finished in {total:.1f} min")
    for label, mins, status in results:
        print(f"  {status:10s} {mins:7.1f} min  {label}")

    aggs = sorted(Path(".").glob("agg_*.jsonl"))
    print(f"\n{len(aggs)} agg file(s):")
    for a in aggs:
        print(f"  {a.name}  ({sum(1 for _ in a.open()):d} records)")
    print(f"\nLog: {log_path}")

    return 0 if all(s == "ok" for _, _, s in results) else 1


if __name__ == "__main__":
    sys.exit(main())
