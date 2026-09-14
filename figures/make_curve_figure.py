"""Figure: mean 1-NN accuracy against the prototype budget.

Re-plots the aggregate budget curve from the campaign's own
``results_curve/curve_mean.csv`` for use in the paper. 

Two departures from the campaign figure, both deliberate:

* Colour marks the two SPINE readouts only.  The figure exists to show the
  gap between them, and every comparator is context.  The campaign figure
  gives a third hue to K-Means as "the strongest budget-matched baseline",
  which does not hold: GLVQ leads it at 0.5% and LVQ3 leads it at every
  other grid point.
* No title inside the axes; the caption carries it.

Full is omitted from the plot, as in the campaign figure.  It is still built,
scored and present in the CSV.

Outputs ``curve_mean_paper.pdf`` and ``curve_mean_paper.png``.
"""

import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/mnt/user-data/uploads/Mapper-Construction-Private/results_curve/curve_mean.csv"

BLUE, VERM = "#0072B2", "#D55E00"
INK, MID, LIGHT = "#3F3F46", "#6B7280", "#9AA1AA"

#: label, colour, marker, linestyle -- swept series, in legend order.
SERIES = [
    ("SPINE+Graph", BLUE, "o", "-"),
    ("SPINE+1NN", VERM, "s", "-"),
    ("LVQ3", INK, "*", ":"),
    ("KMeans", INK, "^", "--"),
    ("GNG", MID, "P", "--"),
    ("GLVQ", MID, "v", "-."),
    ("SPOT", LIGHT, "d", "-."),
    ("Random", LIGHT, "X", ":"),
]
#: label, colour, linestyle -- flat references at their own operating point.
REFERENCES = [("RSP3", INK, (0, (6, 3)))]


def load(path=SRC):
    """Unweighted mean over datasets, per method and nominal budget fraction."""
    rows = list(csv.DictReader(open(path)))
    fracs = sorted({float(r["budget_fraction_nominal"]) for r in rows})
    by = {}
    for r in rows:
        key = (r["method"], float(r["budget_fraction_nominal"]))
        by.setdefault(key, []).append(float(r["acc_1nn"]))
    curves = {}
    for label, *_ in SERIES + [(l, c, s) for l, c, s in REFERENCES]:
        curves[label] = [st.mean(by[(label, f)]) for f in fracs]
    return fracs, curves


def main():
    fracs, curves = load()
    x = range(len(fracs))

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.linewidth": 0.6,
        "figure.dpi": 200,
    })
    fig, ax = plt.subplots(figsize=(5.4, 3.5))

    for label, colour, marker, ls in SERIES:
        lw = 1.7 if colour in (BLUE, VERM) else 1.1
        z = 4 if colour in (BLUE, VERM) else 2
        ax.plot(x, curves[label], color=colour, marker=marker, linestyle=ls,
                linewidth=lw, markersize=4.6, markeredgecolor="white",
                markeredgewidth=0.5, label=label, zorder=z)
    for label, colour, ls in REFERENCES:
        ax.axhline(st.mean(curves[label]), color=colour, linestyle=ls,
                   linewidth=1.0, zorder=1,
                   label="{0} (own budget)".format(label))

    ax.set_xticks(list(x))
    ax.set_xticklabels(["{0:g}\\%".format(f * 100).replace("\\", "")
                        for f in fracs])
    ax.set_xlabel("Prototype budget (fraction of the training fold)",
                  color=INK)
    ax.set_ylabel("Mean 1-NN accuracy", color=INK)
    ax.grid(True, color="#E6E7EA", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#C9CDD3")

    ax.legend(ncols=2, frameon=False, fontsize=7.2, loc="lower right",
              handlelength=2.6, columnspacing=1.4, labelspacing=0.35)

    fig.tight_layout(pad=0.3)
    fig.savefig("curve_mean_paper.pdf", bbox_inches="tight")
    fig.savefig("curve_mean_paper.png", bbox_inches="tight", dpi=320)

    print("fractions:", fracs)
    for label, *_ in SERIES:
        print(f"  {label:<12}" + "".join(f"{v:>8.4f}" for v in curves[label]))
    for label, _, _ in REFERENCES:
        print(f"  {label:<12}{st.mean(curves[label]):>8.4f} (flat)")


if __name__ == "__main__":
    main()
