"""Paper 3 -- sweep runner for the workflow-size and density experiments.

Set the flags in the CONFIG block below and run it directly:

    python run_sweep.py

No command-line arguments and no shell script. Run it three times, changing
WHICH each time: "size", then "softiso", then "density".

Results land in sweeps/<WHICH>/ as agg_*.jsonl, one file per policy per cell,
one JSON record per episode. Zip that whole sweeps/ folder when all three are
done.

Each cell runs as its own subprocess, so one failure or timeout cannot take the
rest of the sweep with it. Cells run one at a time on purpose: decision time is
a reported metric, so two cells running at once would contaminate each other's
timings.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ===================== CONFIG =====================

# Which sweep to run: "size", "softiso", or "density".
WHICH = "size"

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
SOFTISO_CELLS = [((20, 4, 0.9, 500), 10, 3600),
                 ((20, 5, 0.9, 500),  5, 7200),
                 ((20, 6, 0.9, 500),  1, 3600)]


def build_jobs():
    if WHICH == "softiso":
        return [(c, "SoftIso", {}, eps, tmo) for c, eps, tmo in SOFTISO_CELLS]

    cells = SIZE_CELLS if WHICH == "size" else DENSITY_CELLS
    # SoftIso is run separately for the size sweep; it is affordable for the
    # density sweep because k stays at 4 there.
    base = BASELINES.replace("SoftIso,", "") if WHICH == "size" else BASELINES

    jobs = []
    for cell in cells:
        jobs.append((cell, base, {}, EPISODES, TIMEOUT))
        for width, weight in BEAM:
            jobs.append((cell, "BeamVNE",
                         {"BEAM_WIDTH": str(width), "QUEUE_WEIGHT": str(weight)},
                         EPISODES, TIMEOUT))
    return jobs


def run_cell(cell, algos, extra, episodes, timeout, outdir):
    n, k, rho, batch = cell
    env = dict(os.environ)
    env["SCENARIOS"] = f"{n},{k},{rho},{batch}"
    env["ALGOS"] = algos
    env["EPISODES"] = str(episodes)
    env.update(extra)

    tag = f"{n}n_{k}t_{rho}p_{batch}b_{algos.split(',')[0]}"
    if extra:
        tag += f"_B{extra['BEAM_WIDTH']}_Q{extra['QUEUE_WEIGHT']}"

    start = time.time()
    try:
        proc = subprocess.run([sys.executable, "evaluator_networked.py"],
                              env=env, cwd=str(ROOT), capture_output=True,
                              text=True, timeout=timeout)
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
    for f in ROOT.glob("agg_*.jsonl"):
        shutil.move(str(f), str(outdir / f.name))
        moved += 1
    for f in ROOT.glob("rngcheck_*.csv"):
        try:
            f.unlink()
        except OSError:
            pass

    print(f"  [{status:9s}] {tag:36s} {elapsed:7.1f} min  {moved} file(s)",
          flush=True)
    return status


def main():
    if WHICH not in ("size", "softiso", "density"):
        print(f"WHICH must be size, softiso or density, not {WHICH!r}")
        return 2

    outdir = ROOT / "sweeps" / WHICH
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()

    print(f"{WHICH} sweep: {len(jobs)} cells -> {outdir}")
    for cell, algos, extra, eps, tmo in jobs:
        print(f"    {cell}  {algos}  {extra}  {eps} ep  timeout {tmo} s")
    if DRY_RUN:
        print("\nDRY_RUN is True, nothing was run.")
        return 0

    # Any agg_ files left in the working directory from an earlier run would be
    # appended to and silently mixed into this sweep.
    stale = list(ROOT.glob("agg_*.jsonl"))
    if stale:
        print(f"\n{len(stale)} stale agg_*.jsonl in the working directory; "
              f"moving them to sweeps/stale/ before starting")
        (ROOT / "sweeps" / "stale").mkdir(parents=True, exist_ok=True)
        for f in stale:
            shutil.move(str(f), str(ROOT / "sweeps" / "stale" / f.name))

    print()
    start = time.time()
    results = [run_cell(c, a, x, e, t, outdir) for c, a, x, e, t in jobs]
    ok = sum(1 for r in results if r == "ok")
    print(f"\n{WHICH} sweep finished in {(time.time()-start)/60:.1f} min; "
          f"{ok}/{len(results)} ok")
    if ok < len(results):
        print("  non-ok cells are expected for SoftIso at k=6; "
              "check FAILED_*.log for anything else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
