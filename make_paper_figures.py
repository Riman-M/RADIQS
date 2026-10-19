"""Figures and tables for the workflow-size and substrate-density sweeps.

Set the flags in the CONFIG block below and run it directly:

    python make_paper_figures.py

No command-line arguments. It reads the sweep output written by run_sweep.py,
prints every derived number, and writes the figures and LaTeX tables.

Printing the numbers before plotting them is deliberate: a figure is checked
against the records, never read off the chart.

Inputs
    sweeps/size/agg_*.jsonl       workflow size, baselines and BeamVNE
    sweeps/softiso/agg_*.jsonl    workflow size, SoftIso at k = 4, 5, 6
    sweeps/density/agg_*.jsonl    substrate density, all policies

Outputs (into OUT_DIR)
    fig_size.pdf / .png           decision time and completion against k
    fig_density.pdf / .png        completion and decision time against rho
    table_size.tex                workflow-size table
    table_density.tex             substrate-density table

Requires matplotlib. Nothing else beyond the standard library.
"""

import glob
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== CONFIG =====================

ROOT = Path(__file__).resolve().parent     # repository root
OUT_DIR = ROOT / "figures"                 # where figures and tables are written

MAKE_FIGURES = True                        # False prints the numbers only
SAVE_PNG = True                            # PNG alongside PDF, for quick viewing

# The configurations the sweeps hold fixed. Only used for axis limits and
# captions; change them if run_sweep.py's cell lists change.
SIZE_RHO = 0.9                             # density held fixed in the size sweep
DENSITY_K = 4                              # workflow size held fixed in the density sweep
K_VALUES = [4, 5, 6, 8, 12, 16]            # every k that may appear
RHO_VALUES = [0.3, 0.5, 0.7, 0.9]
FULL_EPISODES = 20                         # cells with fewer are drawn hollow

# ===================================================

# One colour per policy, fixed, never cycled: a policy keeps its colour even
# when a chart shows a different subset. Checked for colour-vision separation;
# the three lighter hues fall below 3:1 against white, which is why every
# series also carries a direct label or a legend entry.
COLOUR = {"SoftIso": "#2a78d6", "BeamVNE": "#eb6834", "CloudQC": "#1baf7a",
          "GreedyDfs": "#eda100", "Random": "#e87ba4"}
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d4"

# Drawing order, and the line style that distinguishes the two beam widths.
SERIES = [("SoftIso", None, "SoftIso", "-"),
          ("CloudQC", None, "CloudQC", "-"),
          ("GreedyDfs", None, "GreedyDfs", "-"),
          ("Random", None, "Random", "-"),
          ("BeamVNE", 4, "BeamVNE $B$=4", "-"),
          ("BeamVNE", 16, "BeamVNE $B$=16", "--")]


def load(folder):
    """Read one sweep folder into {(k, rho, algo, beam_width): [records]}."""
    cells = {}
    for path in glob.glob(str(ROOT / folder / "agg_*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        if not records:
            continue
        first = records[0]
        beam = first.get("beam_width") if first["algo"] == "BeamVNE" else None
        cells[(first["task_per_set"], first["edge_prob"],
               first["algo"], beam)] = records
    return cells


def summarise(records):
    times = [r["decision_time"] for r in records]
    n = len(times)
    return {
        "n": n,
        "time": st.mean(times),
        "ci": 1.96 * st.stdev(times) / math.sqrt(n) if n > 1 else 0.0,
        "completion": 100 * (1 - st.mean(r["sla_count"] for r in records)),
        "comm": st.mean(r["communication_overhead"] for r in records),
        "wait": st.mean(r["wait_time"] for r in records) / 1e6,
        "fidelity": st.mean(r["fidelity"] for r in records),
        "qpus": st.mean(
            sum(1 for v in r["workload_distribution_time"].values() if v > 0)
            for r in records),
    }


def report(cells, var_index, var_name, title):
    """Print every cell, and return {(label, variable): summary}."""
    print(f"\n===== {title} =====")
    print(f"{'policy':20s} {var_name:>5s} {'n':>3s} {'time(s)':>9s} {'+-95%':>7s} "
          f"{'compl%':>7s} {'comm':>7s} {'wait(M)':>8s} {'fid':>7s} {'QPUs':>5s}")
    rows = {}
    for key in sorted(cells, key=lambda k: (k[var_index], k[2], k[3] or 0)):
        s = summarise(cells[key])
        label = key[2] if key[2] != "BeamVNE" else f"BeamVNE B={key[3]}"
        rows[(label, key[var_index])] = s
        print(f"{label:20s} {key[var_index]:>5} {s['n']:>3d} {s['time']:>9.3f} "
              f"{s['ci']:>7.3f} {s['completion']:>7.2f} {s['comm']:>7.1f} "
              f"{s['wait']:>8.1f} {s['fidelity']:>7.4f} {s['qpus']:>5.1f}")
    return rows


def report_growth(rows, var_name):
    print(f"\n----- decision-time growth with {var_name} -----")
    values = sorted({v for _, v in rows})
    for label in sorted({l for l, _ in rows}):
        present = [v for v in values if (label, v) in rows]
        if len(present) < 2:
            continue
        lo, hi = present[0], present[-1]
        print(f"  {label:20s} {var_name}={lo} to {hi}: "
              f"{rows[(label, hi)]['time'] / rows[(label, lo)]['time']:8.2f}x "
              f"({rows[(label, lo)]['time']:.3f} to {rows[(label, hi)]['time']:.3f} s)")
        for a, b in zip(present, present[1:]):
            print(f"      {var_name}={a} to {b}: "
                  f"{rows[(label, b)]['time'] / rows[(label, a)]['time']:6.2f}x")


def report_gap(rows):
    print("\n----- completion gap against SoftIso, by density -----")
    for rho in sorted({v for _, v in rows}):
        ref = rows.get(("SoftIso", rho))
        if not ref:
            continue
        for label in ("BeamVNE B=4", "BeamVNE B=16"):
            ours = rows.get((label, rho))
            if not ours:
                continue
            print(f"  rho={rho}  {label:14s} gap "
                  f"{ref['completion'] - ours['completion']:+6.2f} pp   "
                  f"speedup {ref['time'] / ours['time']:6.2f}x")


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def _setup():
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "figure.dpi": 200})


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / name}.pdf")


def figure_size(cells):
    """Decision time and completion against workflow size.

    Cells with fewer than FULL_EPISODES episodes are drawn with hollow
    markers, since SoftIso at k = 5 and 6 is too slow to run 20 times and the
    reader should be able to see which points those are.
    """
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for algo, beam, label, ls in SERIES:
        xs = [k for k in K_VALUES if (k, SIZE_RHO, algo, beam) in cells]
        if not xs:
            continue
        colour = COLOUR[algo]
        stats = [summarise(cells[(k, SIZE_RHO, algo, beam)]) for k in xs]

        ax[0].plot(xs, [s["time"] for s in stats], ls, color=colour, lw=2, zorder=3)
        full = [i for i, s in enumerate(stats) if s["n"] >= FULL_EPISODES]
        part = [i for i, s in enumerate(stats) if s["n"] < FULL_EPISODES]
        ax[0].plot([xs[i] for i in full], [stats[i]["time"] for i in full], "o",
                   color=colour, ms=5, mec="white", mew=0.8, zorder=4)
        ax[0].plot([xs[i] for i in part], [stats[i]["time"] for i in part], "o",
                   mfc="white", mec=colour, mew=1.4, ms=5, zorder=4)
        ax[0].annotate(label, (xs[-1], stats[-1]["time"]), xytext=(4, 0),
                       textcoords="offset points", fontsize=6.5,
                       color=colour, va="center")

        # The completion panel gets a legend rather than direct labels: at
        # k = 16 three series sit within two points of each other and the
        # labels collide.
        ax[1].plot(xs, [s["completion"] for s in stats], ls, color=colour, lw=2,
                   marker="o", ms=5, mec="white", mew=0.8, zorder=3, label=label)

    ax[1].legend(frameon=False, fontsize=6.5, loc="center left",
                 bbox_to_anchor=(0.0, 0.42), handlelength=1.6)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Decision time per episode (s)")
    ax[1].set_ylabel("Workflow completion (%)")
    ax[1].set_ylim(-4, 108)
    for a in ax:
        a.set_xlabel("Circuits per workflow $k$")
        a.set_xticks(K_VALUES)
        a.set_xlim(3, 18.5)
        _style(a)
    fig.tight_layout()
    _save(fig, "fig_size")


def figure_density(cells):
    """Completion and decision time against substrate density.

    The completion panel carries an inset: the three strongest policies sit
    within a percentage point of each other, and the distance between them is
    the whole question, so it cannot be left to a full-range axis.
    """
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for algo, beam, label, ls in SERIES:
        xs = [p for p in RHO_VALUES if (DENSITY_K, p, algo, beam) in cells]
        if not xs:
            continue
        colour = COLOUR[algo]
        stats = [summarise(cells[(DENSITY_K, p, algo, beam)]) for p in xs]
        ax[0].plot(xs, [s["completion"] for s in stats], ls, color=colour, lw=2,
                   marker="o", ms=5, mec="white", mew=0.8, zorder=3, label=label)
        ax[1].plot(xs, [s["time"] for s in stats], ls, color=colour, lw=2,
                   marker="o", ms=5, mec="white", mew=0.8, zorder=3)
        ax[1].annotate(label, (xs[-1], stats[-1]["time"]), xytext=(4, 0),
                       textcoords="offset points", fontsize=6.5,
                       color=colour, va="center")

    ax[0].set_ylabel("Workflow completion (%)")
    ax[0].set_ylim(-4, 108)

    inset = ax[0].inset_axes([0.44, 0.14, 0.52, 0.40], zorder=6)
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    for algo, beam, label, ls in SERIES:
        if algo not in ("SoftIso", "BeamVNE"):
            continue
        xs = [p for p in RHO_VALUES if (DENSITY_K, p, algo, beam) in cells]
        if not xs:
            continue
        stats = [summarise(cells[(DENSITY_K, p, algo, beam)]) for p in xs]
        inset.plot(xs, [s["completion"] for s in stats], ls, color=COLOUR[algo],
                   lw=1.6, marker="o", ms=3.5, mec="white", mew=0.6, zorder=3)
    inset.set_ylim(99.2, 100.15)
    inset.set_xticks(RHO_VALUES)
    inset.set_yticks([99.5, 100.0])
    inset.tick_params(labelsize=5.5)
    inset.grid(color=GRID, linewidth=0.5)
    inset.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        inset.spines[side].set_visible(True)
        inset.spines[side].set_color(GRID)
    inset.set_title("detail, 99.2 to 100", fontsize=5.5, color=MUTED, pad=2)

    ax[1].set_yscale("log")
    ax[1].set_ylabel("Decision time per episode (s)")
    for a in ax:
        a.set_xlabel(r"Substrate density $\varrho$")
        a.set_xticks(RHO_VALUES)
        _style(a)
    ax[0].set_xlim(0.26, 0.94)
    ax[1].set_xlim(0.26, 1.06)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.07),
               handlelength=2.4, columnspacing=1.4)
    fig.tight_layout()
    _save(fig, "fig_density")


def write_table(cells, values, var_index, var_header, caption, label, filename):
    lines = [r"\begin{table}[tb]", r"\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\small",
             r"\begin{tabular}{lrrrrr}", r"\hline",
             f"Policy & {var_header} & Time (s) & Compl.\\ (\\%) & Comm. & "
             r"Wait ($10^6$) \\", r"\hline"]
    for algo, beam, name, _ in SERIES:
        wrote = False
        for v in values:
            key = (v, SIZE_RHO, algo, beam) if var_index == 0 else (DENSITY_K, v, algo, beam)
            if key not in cells:
                continue
            s = summarise(cells[key])
            mark = "$^{*}$" if s["n"] < FULL_EPISODES else ""
            lines.append(f"{name}{mark} & {v} & {s['time']:.3f} & "
                         f"{s['completion']:.2f} & {s['comm']:.1f} & "
                         f"{s['wait']:.1f} \\\\")
            wrote = True
        if wrote:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / filename}")


def main():
    size = load("sweeps/size")
    size.update(load("sweeps/softiso"))
    density = load("sweeps/density")

    if not size and not density:
        print("No sweep output found. Run run_sweep.py first.")
        return 2

    size_rows = density_rows = {}
    if size:
        size_rows = report(size, 0, "k",
                           f"Workflow size, 20 QPUs, rho={SIZE_RHO}, batch 500")
        report_growth(size_rows, "k")
    if density:
        density_rows = report(density, 1, "rho",
                              f"Substrate density, 20 QPUs, k={DENSITY_K}, batch 500")
        report_gap(density_rows)

    if not MAKE_FIGURES:
        print("\nMAKE_FIGURES is False; no figures written.")
        return 0

    _setup()
    if size:
        figure_size(size)
        write_table(
            size, K_VALUES, 0, "$k$",
            f"Workflow size at 20 QPUs, $\\varrho = {SIZE_RHO}$, batch 500. "
            f"{FULL_EPISODES} episodes per cell except the two marked "
            "$^{*}$, which are SoftIso at 5 and 1 episodes; wait time "
            "accumulates across episodes and is therefore not comparable for "
            "those two rows.",
            "tab:size", "table_size.tex")
    if density:
        figure_density(density)
        write_table(
            density, RHO_VALUES, 1, r"$\varrho$",
            f"Substrate density at 20 QPUs, $k = {DENSITY_K}$, batch 500, "
            f"{FULL_EPISODES} episodes per cell.",
            "tab:density", "table_density.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
