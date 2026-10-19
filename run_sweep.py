"""Paper 3 -- sweep runner for the workflow-size and density experiments.

Set the flags in the CONFIG block below and run it directly:

    python run_sweep.py

No command-line arguments and no shell script. By default it runs all three
sweeps back to back, so there is nothing to edit between them.

Results land in sweeps/<name>/ as agg_*.jsonl, one file per policy per cell,
one JSON record per episode. Zip the whole sweeps/ folder when it finishes.

Each cell runs as its own subprocess, so one failure or timeout cannot take the
rest of the run with it, and the run continues from where it was if you stop it
and start it again (see RESUME). Cells run one at a time on purpose: decision
time is a reported metric, so two cells running at once would contaminate each
other's timings.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ===================== CONFIG =====================

# Which sweeps to run, in order. All three by default. Narrow it to a single
# name, e.g. WHICH = ["density"], to run just one.
WHICH = ["size", "softiso", "density"]

# Skip any cell whose output file already exists, so a stopped run can be
# restarted without repeating finished cells. Set False to force a rerun.
RESUME = True

# Episodes per cell. 20 is enough for the deterministic metrics; the paper's
# existing cells use 50. Raise if you have the time budget.
EPISODES = 20

# Seconds a single cell may take before it is abandoned and recorded as
# TIMEOUT. Applies to every cell except the SoftIso ones, which carry their own.
TIMEOUT = 10800

# Set True to list the cells and exit without running anything.
DRY_RUN = False

# ===================================================

ROOT = Path(__file__).resolve().parent
BASELINES = "SoftIso,CloudQC,GreedyDfs,Random"
BEAM = [(4, 0.5), (16, 0.5)]          # (beam width, queue weight)

# Workflow size. The mapping is injective, one circuit per QPU, so k cannot
# exceed the substrate size. SoftIso is excluded here and handled separately
# below, because its cost makes k >= 6 unrunnable.
SIZE_CELLS = [(20, k, 0.9, 500) for k in (4, 8, 12, 16)]

# Substrate density at the configuration the paper draws its conclusions from.
DENSITY_CELLS = [(20, 4, p, 500) for p in (0.3, 0.5, 0.7, 0.9)]

# SoftIso enumerates injective maps of k circuits into the substrate, so its
# work grows as the falling factorial 20!/(20-k)!:
#     k=4        116,280 maps        1x
#     k=5      1,860,480 maps       16x
#     k=6     27,907,200 maps      240x
#     k=8  5,079,110,400 maps   43,680x
# So k=6 is a few hours per episode and k=8 is out of reach entirely. We
# measure it where it is measurable and record a bound where it is not: the
# k=6 cell is expected to come back TIMEOUT, and that is the datapoint.
# (cell, episodes, timeout seconds)
# k=4 runs the same 20 episodes as the size sweep so that wait time, which
# accumulates across episodes, is comparable against the other policies at
# that k. k=5 and k=6 are for the growth and ceiling argument, where only
# decision time is read, so a shorter run is enough.
SOFTISO_CELLS = [((20, 4, 0.9, 500), 20, 7200),
                 ((20, 5, 0.9, 500),  5, 7200),
                 ((20, 6, 0.9, 500),  1, 3600)]


def build_jobs(which):
    if which == "softiso":
        return [(c, "SoftIso", {}, eps, tmo) for c, eps, tmo in SOFTISO_CELLS]

    cells = SIZE_CELLS if which == "size" else DENSITY_CELLS
    # SoftIso is run separately for the size sweep; it is affordable for the
    # density sweep because k stays at 4 there.
    base = BASELINES.replace("SoftIso,", "") if which == "size" else BASELINES

    jobs = []
    for cell in cells:
        jobs.append((cell, base, {}, EPISODES, TIMEOUT))
        for width, weight in BEAM:
            jobs.append((cell, "BeamVNE",
                         {"BEAM_WIDTH": str(width), "QUEUE_WEIGHT": str(weight)},
                         EPISODES, TIMEOUT))
    return jobs


def already_done(cell, algos, extra, outdir):
    """A cell counts as done if the first policy's output file is present."""
    n, k, rho, batch = cell
    first = algos.split(",")[0]
    width = extra.get("BEAM_WIDTH", "8")
    weight = extra.get("QUEUE_WEIGHT", "0")
    pattern = f"agg_{first}_*_{n}n_{k}t_{rho}p_{batch}b_B{width}_Q{weight}.jsonl"
    return any(outdir.glob(pattern))


def run_cell(cell, algos, extra, episodes, timeout, outdir):
    n, k, rho, batch = cell
    env = dict(os.environ)
    env["SCENARIOS"] = f"{n},{k},{rho},{batch}"
    env["ALGOS"] = algos
    env["EPISODES"] = str(episodes)
    # The evaluator reconfigures its stdout to UTF-8. Without this the child
    # may still pick the console code page for other streams, and on Windows
    # the parent would then have to guess.
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra)

    tag = f"{n}n_{k}t_{rho}p_{batch}b_{algos.split(',')[0]}"
    if extra:
        tag += f"_B{extra['BEAM_WIDTH']}_Q{extra['QUEUE_WEIGHT']}"

    start = time.time()
    try:
        # encoding is given explicitly: text=True alone decodes with the
        # locale codec, which on Windows is cp1252 and raises
        # UnicodeDecodeError on the evaluator's UTF-8 output. That exception
        # lands in subprocess's reader thread, where it cannot be caught here
        # and silently costs the captured log.
        proc = subprocess.run([sys.executable, "evaluator_networked.py"],
                              env=env, cwd=str(ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        if proc.returncode != 0:
            (outdir / f"FAILED_{tag}.log").write_text(
                (proc.stdout or "") + "\n===== stderr =====\n" + (proc.stderr or ""),
                encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        (outdir / f"TIMEOUT_{tag}.txt").write_text(
            f"no episode completed within {timeout} s\n", encoding="utf-8")

    elapsed = (time.time() - start) / 60.0

    moved = 0
    for pattern in ("agg_*.jsonl", "rngcheck_*.csv"):
        for f in ROOT.glob(pattern):
            shutil.move(str(f), str(outdir / f.name))
            moved += 1

    print(f"  [{status:9s}] {tag:36s} {elapsed:7.1f} min  "
          f"{moved} file(s) kept", flush=True)
    return status


def main():
    names = [WHICH] if isinstance(WHICH, str) else list(WHICH)
    bad = [w for w in names if w not in ("size", "softiso", "density")]
    if bad:
        print(f"WHICH may only contain size, softiso or density; got {bad}")
        return 2

    plan = []
    for name in names:
        outdir = ROOT / "sweeps" / name
        outdir.mkdir(parents=True, exist_ok=True)
        plan.append((name, outdir, build_jobs(name)))

    total = sum(len(j) for _, _, j in plan)
    print(f"plan: {' then '.join(names)}  ({total} cells, run one at a time)")
    for name, outdir, jobs in plan:
        print(f"\n  {name} -> {outdir}  ({len(jobs)} cells)")
        for cell, algos, extra, eps, tmo in jobs:
            print(f"    {cell}  {algos}  {extra}  {eps} ep  timeout {tmo} s")
    if DRY_RUN:
        print("\nDRY_RUN is True, nothing was run.")
        return 0

    # Both output kinds are opened in append mode by the evaluator, so anything
    # left in the working directory from an earlier run would be appended to
    # and silently mixed into this one. They are moved, never deleted: the
    # rngcheck files carry the substrate hashes the pairing analysis rests on.
    stale = list(ROOT.glob("agg_*.jsonl")) + list(ROOT.glob("rngcheck_*.csv"))
    if stale:
        staledir = ROOT / "sweeps" / "stale"
        staledir.mkdir(parents=True, exist_ok=True)
        print(f"\n{len(stale)} stale output file(s) in the working directory; "
              f"moving them to sweeps/stale/ before starting (not deleting)")
        for f in stale:
            target = staledir / f.name
            if target.exists():
                target = staledir / (f.stem + "_prev" + f.suffix)
            shutil.move(str(f), str(target))

    start = time.time()
    tally = {"ok": 0, "skipped": 0, "other": 0}
    for name, outdir, jobs in plan:
        print(f"\n===== {name} =====", flush=True)
        for cell, algos, extra, eps, tmo in jobs:
            if RESUME and already_done(cell, algos, extra, outdir):
                print(f"  [skipped  ] {cell} {algos.split(',')[0]} "
                      f"{extra} (output already present)", flush=True)
                tally["skipped"] += 1
                continue
            r = run_cell(cell, algos, extra, eps, tmo, outdir)
            tally["ok" if r == "ok" else "other"] += 1

    print(f"\nall sweeps finished in {(time.time()-start)/60:.1f} min; "
          f"{tally['ok']} ok, {tally['skipped']} skipped, "
          f"{tally['other']} not ok")
    if tally["other"]:
        print("  a TIMEOUT on the SoftIso k=6 cell is expected and is itself "
              "a datapoint; check FAILED_*.log for anything else")
    print("\nzip the sweeps/ folder and send it")
    return 0


if __name__ == "__main__":
    sys.exit(main())