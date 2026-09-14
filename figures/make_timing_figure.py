"""Figure: construction cost.

Re-plots the campaign's fixed-configuration timing study from
``results_computational/timing_mean.csv`` for use in the paper. 

Two panels:

(a) mean construction time against training-fold size, log-log, markers only.
    The campaign figure joins the datasets with lines, which produces a
    sawtooth because the matched budget is not monotone in ``n``; the points
    alone carry the separation into a fast group and a slow group.
(b) SPINE against GLVQ, one point per dataset, with the line of equality.
    GLVQ descends the same margin over isolated prototypes, so this is the
    comparison the subsection is read for, and points below the line are
    datasets where SPINE is the faster of the two.

Colour marks SPINE and GLVQ, the two methods being compared; every other
method is grey.

Outputs ``timing_paper.pdf`` and ``timing_paper.png``.
"""

import csv
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = ("/mnt/user-data/uploads/Mapper-Construction-Private/"
       "results_computational/timing_mean.csv")

BLUE, VERM, GRN = "#0072B2", "#D55E00", "#009E73"
PURP, ORNG = "#CC79A7", "#E69F00"   # the other two methods Table 4 reports
INK, MID, LIGHT, GRID = "#3F3F46", "#6B7280", "#9AA1AA", "#E6E7EA"

#: label, colour, marker -- panel (a), in legend order.
#: Coloured iff the method appears in Table~\ref{tab:timing}; the two the
#: table omits stay grey, so figure and table carry the same roster.
SERIES = [
    ("SPINE", BLUE, "o"),
    ("GLVQ", VERM, "s"),
    ("GNG", GRN, "^"),
    ("LVQ3", PURP, "*"),
    ("RSP3", ORNG, "v"),
    ("KMeans", MID, "X"),
    ("SPOT", LIGHT, "d"),
]

PRETTY = {"eeg-eye-state": "EEG Eye State", "wdbc": "WDBC"}
SHORT = {"eeg-eye-state": "EEG"}          # panel (b) label, kept short to avoid collisions


def load(path=SRC):
    t = collections.defaultdict(dict)
    n = {}
    for r in csv.DictReader(open(path)):
        t[r["method"]][r["dataset"]] = float(r["cpu_seconds"])
        n[r["dataset"]] = float(r["n_train"])
    return t, n


def main():
    t, n = load()
    datasets = sorted(n, key=lambda d: (n[d], d))   # stable at the Ring/Twonorm tie

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.linewidth": 0.6,
        "axes.titlesize": 9, "figure.dpi": 200,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.3))

    # (a) cost against training-fold size, with a log-log power-law fit.
    # The points are 17 heterogeneous datasets, so they are NOT joined --
    # a fitted exponent is the honest summary of "how it scales", and the
    # residual spread is the per-class-vertex effect the prose describes.
    import numpy as np
    xs = np.array([n[d] for d in datasets])
    lx = np.log10(xs)
    grid = np.linspace(lx.min(), lx.max(), 50)
    for label, colour, marker in SERIES:
        big = colour not in (MID, LIGHT)
        ys = np.array([t[label][d] for d in datasets])
        leg = label
        if big:
            b, a = np.polyfit(lx, np.log10(ys), 1)
            ax.plot(10 ** grid, 10 ** (a + b * grid), color=colour,
                    linewidth=1.2, alpha=0.85, zorder=3)
            leg = "{0} ($b$={1:.2f})".format(label, b)
        ax.scatter(xs, ys, s=26 if big else 15, marker=marker, color=colour,
                   linewidths=0.5, edgecolors="white",
                   zorder=5 if big else 2, label=leg)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training-fold size $n$")
    ax.set_ylabel("Mean construction time (CPU s)")
    ax.set_title("(a) Cost and scaling against training-fold size")
    ax.legend(ncols=2, frameon=False, fontsize=7, loc="upper left",
              handletextpad=0.3, columnspacing=1.0, labelspacing=0.3)

    # (b) ratio against training-fold size, which is what the median hides
    xs = [n[d] for d in datasets]
    bx.axhline(1.0, color=INK, linewidth=0.9, zorder=2)
    for label, colour, marker in (("GLVQ", VERM, "s"), ("GNG", GRN, "^")):
        bx.scatter(xs, [t["SPINE"][d] / t[label][d] for d in datasets],
                   s=26, marker=marker, color=colour, linewidths=0.5,
                   edgecolors="white", zorder=4, label="vs " + label)
    for d in datasets:                  # name the datasets SPINE loses on
        y = t["SPINE"][d] / t["GNG"][d]
        if y > 1.4:
            bx.annotate(SHORT.get(d, PRETTY.get(d, d.capitalize())), (n[d], y),
                        textcoords="offset points", xytext=(-5, 2),
                        fontsize=6.6, color=INK, ha="right", va="bottom")
    # axes-fraction x, data y: pinned either side of the unit line
    yt = bx.get_yaxis_transform()
    bx.text(0.985, 1.06, "SPINE slower", transform=yt, ha="right",
            va="bottom", fontsize=7.2, color=MID)
    bx.text(0.985, 0.94, "SPINE faster", transform=yt, ha="right",
            va="top", fontsize=7.2, color=MID)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("Training-fold size $n$")
    bx.set_ylabel("SPINE time / competitor time")
    bx.set_title("(b) Cost ratio against training-fold size")
    bx.legend(frameon=False, fontsize=7.2, loc="upper left",
              handletextpad=0.3, labelspacing=0.3)

    for a in (ax, bx):
        a.grid(True, which="major", color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        a.tick_params(colors=INK, length=3)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            a.spines[side].set_color("#C9CDD3")

    fig.tight_layout(pad=0.4, w_pad=1.6)
    fig.savefig("timing_paper.pdf", bbox_inches="tight")
    fig.savefig("timing_paper.png", bbox_inches="tight", dpi=320)

    slower = [d for d in datasets if t["SPINE"][d] > t["GLVQ"][d]]
    ratios = sorted(t["SPINE"][d] / t["GLVQ"][d] for d in datasets)
    print("datasets:", len(datasets))
    print("SPINE slower than GLVQ on:", slower)
    print("median ratio %.3f  min %.3f  max %.3f"
          % (ratios[len(ratios) // 2], ratios[0], ratios[-1]))


if __name__ == "__main__":
    main()
