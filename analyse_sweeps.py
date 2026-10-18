"""Paper 3 -- analysis and figures for the workflow-size and density sweeps.

Set the flags in the CONFIG block below and run it directly:

    python analyse_sweeps.py

No command-line arguments. It reads sweeps/size, sweeps/softiso and
sweeps/density, prints every derived number, and writes the figures to
figures_new/.

Every number is printed before it is plotted, so it can be checked against the
records rather than read off a chart.
"""
import glob
import json
import math
import statistics as st
from pathlib import Path

# ===================== CONFIG =====================

# Set False to print the numbers without writing figures.
MAKE_FIGURES = True

# ===================================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_new"

# Fixed entity -> colour, matching the existing figures. Validated with the
# dataviz palette checker: all checks pass; the sub-3:1 contrast of three of
# the hues is relieved by direct labels, which these figures carry.
COLOUR = {"SoftIso": "#2a78d6", "BeamVNE": "#eb6834", "CloudQC": "#1baf7a",
          "GreedyDfs": "#eda100", "Random": "#e87ba4"}
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d4"


def load(folder):
    cells = {}
    for p in glob.glob(str(ROOT / "sweeps" / folder / "agg_*.jsonl")):
        r = [json.loads(l) for l in open(p) if l.strip()]
        if not r:
            continue
        a = r[0]
        key = (a["n_nodes"], a["task_per_set"], a["edge_prob"], a["algo"],
               a.get("beam_width") if a["algo"] == "BeamVNE" else None,
               a.get("queue_weight") if a["algo"] == "BeamVNE" else None)
        cells[key] = r
    return cells


def summarise(r):
    dt = [x["decision_time"] for x in r]
    n = len(dt)
    ci = 1.96 * st.stdev(dt) / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "time": st.mean(dt), "ci": ci,
            "compl": 100 * (1 - st.mean(x["sla_count"] for x in r)),
            "comm": st.mean(x["communication_overhead"] for x in r),
            "wait": st.mean(x["wait_time"] for x in r) / 1e6,
            "fid": st.mean(x["fidelity"] for x in r)}


def label(algo, b, q):
    return algo if algo != "BeamVNE" else f"BeamVNE B={b}"


def report(cells, varname, varidx, title):
    print(f"\n===== {title} =====")
    print(f"{'':22s} {varname:>6s} {'n':>3s} {'time(s)':>9s} {'+-95%':>8s} "
          f"{'compl%':>7s} {'comm':>7s} {'wait(M)':>8s} {'fid':>7s}")
    rows = {}
    for key in sorted(cells, key=lambda k: (k[3], k[4] or 0, k[varidx])):
        n, t, p, algo, b, q = key
        s = summarise(cells[key])
        rows[(label(algo, b, q), key[varidx])] = s
        print(f"{label(algo,b,q):22s} {key[varidx]:>6} {s['n']:>3d} "
              f"{s['time']:>9.3f} {s['ci']:>8.3f} {s['compl']:>7.2f} "
              f"{s['comm']:>7.1f} {s['wait']:>8.1f} {s['fid']:>7.4f}")
    return rows


def growth(rows, who, lo, hi):
    a, b = rows.get((who, lo)), rows.get((who, hi))
    if a and b and a["time"]:
        return b["time"] / a["time"]
    return None


def main():
    size = load("size")
    soft = load("softiso")
    dens = load("density")
    size.update(soft)

    srows = report(size, "k", 1, "Workflow size: 20 QPUs, rho=0.9, batch 500") if size else {}
    drows = report(dens, "rho", 2, "Substrate density: 20 QPUs, k=4, batch 500") if dens else {}

    if srows:
        print("\n-- decision-time growth across workflow size")
        ks = sorted({k for _, k in srows})
        for who in sorted({w for w, _ in srows}):
            present = [k for k in ks if (who, k) in srows]
            if len(present) >= 2:
                g = growth(srows, who, present[0], present[-1])
                print(f"   {who:22s} k={present[0]} -> k={present[-1]}: "
                      f"{g:6.2f}x   ({srows[(who,present[0])]['time']:.3f} -> "
                      f"{srows[(who,present[-1])]['time']:.3f} s)")

    if drows:
        print("\n-- completion gap, SoftIso minus BeamVNE, by density")
        for rho in sorted({r for _, r in drows}):
            s = drows.get(("SoftIso", rho))
            for who in ["BeamVNE B=4", "BeamVNE B=16"]:
                v = drows.get((who, rho))
                if s and v:
                    print(f"   rho={rho}  {who:14s} gap {s['compl']-v['compl']:+6.2f} pp"
                          f"   (SoftIso {s['compl']:.2f}, ours {v['compl']:.2f})")

    if MAKE_FIGURES and (srows or drows):
        make_figures(srows, drows)
    return 0


def make_figures(srows, drows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "figure.dpi": 200})

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_axisbelow(True)

    if srows:
        ks = sorted({k for _, k in srows})
        whos = sorted({w for w, _ in srows})
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
        for ax, metric, ylab, logy in [(axes[0], "time", "Decision time per episode (s)", True),
                                       (axes[1], "compl", "Workflow completion (%)", False)]:
            for who in whos:
                c = COLOUR["BeamVNE"] if who.startswith("BeamVNE") else COLOUR[who]
                ls = "--" if who.endswith("B=16") else "-"
                xs = [k for k in ks if (who, k) in srows]
                ys = [srows[(who, k)][metric] for k in xs]
                if not xs:
                    continue
                ax.plot(xs, ys, ls, color=c, lw=2, marker="o", ms=5,
                        mec="white", mew=0.8, label=who, zorder=3)
                ax.annotate(who, (xs[-1], ys[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=6.5,
                            color=c, va="center")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Circuits per workflow $k$")
            ax.set_ylabel(ylab)
            ax.set_xticks(ks)
            style(ax)
        axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / "fig8_size.pdf", bbox_inches="tight")
        fig.savefig(OUT / "fig8_size.png", bbox_inches="tight")
        print(f"\nwrote {OUT/'fig8_size.pdf'}")

    if drows:
        rhos = sorted({r for _, r in drows})
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        for who in sorted({w for w, _ in drows}):
            c = COLOUR["BeamVNE"] if who.startswith("BeamVNE") else COLOUR[who]
            ls = "--" if who.endswith("B=16") else "-"
            xs = [r for r in rhos if (who, r) in drows]
            ys = [drows[(who, r)]["compl"] for r in xs]
            if not xs:
                continue
            ax.plot(xs, ys, ls, color=c, lw=2, marker="o", ms=5,
                    mec="white", mew=0.8, label=who, zorder=3)
        ax.set_xlabel(r"Substrate density $\varrho$")
        ax.set_ylabel("Workflow completion (%)")
        ax.set_xticks(rhos)
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        style(ax)
        fig.tight_layout()
        fig.savefig(OUT / "fig9_density.pdf", bbox_inches="tight")
        fig.savefig(OUT / "fig9_density.png", bbox_inches="tight")
        print(f"wrote {OUT/'fig9_density.pdf'}")


if __name__ == "__main__":
    raise SystemExit(main())
