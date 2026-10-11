"""Analyse the availability-term probe written by BeamVNE's NAT_PROBE.

    python analyse_natprobe.py natprobe_lpmr.csv natprobe_splr.csv

Reports, per file, what fraction of real allocation decisions had a saturated
availability term, and how much the term varied on the decisions where it did
not saturate. The second number is what decides whether the non-saturated
decisions matter: if the spread there is orders of magnitude below the variation
in the other cost terms, the term is doing no useful work even when it is
technically not pinned.
"""

import statistics as st
import sys
from pathlib import Path


def analyse(path):
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.startswith("n_mapped"):
                continue
            n, mx, sig, sat = line.split(",")
            rows.append((int(n), float(mx), float(sig), int(sat)))

    if not rows:
        print(f"{path}: no rows")
        return

    total = len(rows)
    sat = sum(r[3] for r in rows)
    print(f"\n=== {path}")
    print(f"  allocations logged     : {total}")
    print(f"  availability saturated : {sat} ({100 * sat / total:.2f}%)")
    print(f"  not saturated          : {total - sat} "
          f"({100 * (total - sat) / total:.2f}%)")

    unsat = [r for r in rows if not r[3]]
    if not unsat:
        print("  the term was pinned at 1.0 on every logged decision")
        return

    mxs = [r[1] for r in unsat]
    sigs = [r[2] for r in unsat]
    print(f"  on the {len(unsat)} non-saturated decisions:")
    print(f"    max clock : min={min(mxs):.4f}  median={st.median(mxs):.4f}  "
          f"max={max(mxs):.4f}")
    print(f"    SIG value : min={min(sigs):.12f}  max={max(sigs):.12f}")
    print(f"    spread    : {max(sigs) - min(sigs):.3e}")
    print("    (for comparison, the runtime, error and communication terms each")
    print("     vary by order 1e-2 between candidate mappings, so a spread")
    print("     below about 1e-6 means the term still cannot decide anything)")


if __name__ == "__main__":
    args = sys.argv[1:] or ["natprobe.csv"]
    missing = [a for a in args if not Path(a).exists()]
    if missing:
        print("not found: " + ", ".join(missing))
        print("The CSV is written to the directory you run python from, and only")
        print("when NAT_PROBE is set in the environment for that run.")
        sys.exit(1)
    for a in args:
        analyse(a)
