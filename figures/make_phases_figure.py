"""Figure: the SPINE phases on a toy problem.

Class 0 is a noisy ring, class 1 a blob at its centre, so class 0's nerve
carries a 1-cycle and the figure shows that cycle surviving every phase.

Outputs ``spine_phases.pdf`` and ``spine_phases.png``.
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

import spine.model as model_mod
from spine.model import SPINE, SkeletonClassifier
from spine.topology import betti

SEED = 3
N_RING, N_BLOB = 400, 200
BUDGET = 30

CLS_COL = ["#0072B2", "#D55E00"]          # Okabe-Ito blue / vermillion
PT_COL = ["#9CC7E4", "#F0B08A"]
GHOST = "#C9CDD3"
INK = "#1F2328"


def toy(seed=SEED):
    """Ring plus central blob, both slightly elongated and rotated.

    Standardising a class equalises its coordinate variances, so a circular
    ring has no leading within-class direction and the PC1 lens is not
    defined.  Mild ellipticity gives the lens something to find.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 2.0 * np.pi, N_RING)
    r = 1.0 + rng.normal(0.0, 0.14, N_RING)
    ring = np.c_[1.45 * r * np.cos(t), 1.0 * r * np.sin(t)]
    blob = rng.multivariate_normal([0.0, 0.0], [[0.165, 0.066],
                                                [0.066, 0.090]], N_BLOB)
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    X = np.vstack([ring, blob]) @ R.T
    y = np.r_[np.zeros(N_RING, int), np.ones(N_BLOB, int)]
    return StandardScaler().fit_transform(X), y


def toy_moons(seed=SEED):
    """Two interleaving moons, rotated so within-class PC1 is well defined.

    Each moon is symmetric about a coordinate axis, so its z-scored
    correlation matrix is the identity and the lens is undefined; a rotation
    gives each class a genuine leading direction.
    """
    X, y = make_moons(n_samples=N_RING + N_BLOB, noise=0.11,
                      random_state=seed)
    th = np.deg2rad(40.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return StandardScaler().fit_transform(X @ R.T), y


DATASETS = {"ring": (toy, ("ring class", "blob class"), "spine_phases"),
            "moons": (toy_moons, ("moon A", "moon B"),
                      "spine_phases_moons")}


def run_with_snapshots(X, y):
    """Call ``SPINE.select`` unmodified, recording each phase's output."""
    snaps = {"phase0": [], "phase1": [], "phase2b": [], "grow": []}
    p0, p1, p2a, p2b, prune = (model_mod.phase0, model_mod.phase1,
                               model_mod.phase2a, model_mod.phase2b,
                               model_mod.prune_to_budget)

    def w0(*a, **k):
        out = p0(*a, **k)
        snaps["phase0"].append((out[0].copy(), list(out[1])))
        return out

    def w1(*a, **k):
        out = p1(*a, **k)
        snaps["phase1"].append(out.copy())
        return out

    def w2a(*a, **k):
        sk, info = p2a(*a, **k)
        snaps["phase2a"] = [(V.copy(), list(E)) for V, E in sk]
        snaps["admission"] = info
        return sk, info

    def w2b(*a, **k):
        sk = p2b(*a, **k)
        snaps["phase2b"].append([(V.copy(), list(E)) for V, E in sk])
        return sk

    def wg(*a, **k):
        V, E, nf = p1grow(*a, **k)
        snaps["grow"].append((V.copy(), list(E)))
        return V, E, nf

    p1grow = model_mod.grow_class
    model_mod.phase0, model_mod.phase1 = w0, w1
    model_mod.phase2a, model_mod.phase2b = w2a, w2b
    model_mod.grow_class = wg

    def wp(*a, **k):
        sk, nf = prune(*a, **k)
        snaps["phase4"] = [(V.copy(), list(E)) for V, E in sk]
        return sk, nf

    model_mod.prune_to_budget = wp
    try:
        est = SPINE()
        Xp, yp, _ = est.select(X, y, total_count=BUDGET, random_state=SEED)
    finally:
        (model_mod.phase0, model_mod.phase1, model_mod.phase2a,
         model_mod.phase2b, model_mod.grow_class,
         model_mod.prune_to_budget) = p0, p1, p2a, p2b, p1grow, prune
    return est, Xp, yp, snaps


def draw_points(ax, X, y, alpha=1.0, s=4.0):
    for c in (0, 1):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=s, linewidths=0,
                   color=PT_COL[c], alpha=alpha, zorder=1)


def draw_skeletons(ax, skeletons, removed=None, vs=16, lw=1.5):
    if removed:
        for c, E in removed.items():
            V = skeletons[c][0]
            for i, j in E:
                ax.plot([V[i, 0], V[j, 0]], [V[i, 1], V[j, 1]], color=GHOST,
                        linewidth=1.1, linestyle=(0, (2.2, 1.6)), zorder=2)
    for c, (V, E) in enumerate(skeletons):
        for i, j in E:
            ax.plot([V[i, 0], V[j, 0]], [V[i, 1], V[j, 1]], color=CLS_COL[c],
                    linewidth=lw, solid_capstyle="round", zorder=3)
        ax.scatter(V[:, 0], V[:, 1], s=vs, color=CLS_COL[c], zorder=4,
                   edgecolors="white", linewidths=0.5)


def betti_label(ax, skeletons):
    """One Betti pair per class, in that class's colour."""
    for c, (V, E) in enumerate(skeletons):
        ax.text(0.06 + 0.88 * c, 0.015,
                r"$\beta=({0},{1})$".format(*betti(len(V), E)),
                transform=ax.transAxes, ha="left" if c == 0 else "right",
                va="bottom", fontsize=6.4, color=CLS_COL[c])


def main(which="ring"):
    gen, labels, stem = DATASETS[which]
    X, y = gen()
    est, Xp, yp, snaps = run_with_snapshots(X, y)

    sk0 = [(V, E) for V, E in snaps["phase0"]]
    sk1 = [(snaps["phase1"][c], snaps["phase0"][c][1]) for c in (0, 1)]
    sk2a = snaps["phase2a"]
    sk2b = snaps["phase2b"][0]
    sk_grown = snaps["phase2b"][-2]
    sk_final = snaps["phase2b"][-1]
    removed = {c: [e for e in sk1[c][1] if e not in set(sk2a[c][1])]
               for c in (0, 1)}

    check = np.vstack([V for V, _ in sk_final])
    assert np.allclose(np.sort(check, axis=0), np.sort(Xp, axis=0)), \
        "final panel does not match what select() returned"

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 7.6,
        "axes.linewidth": 0.6, "figure.dpi": 200,
    })
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.5))
    ax = axes.ravel()

    draw_points(ax[0], X, y, s=5.0)
    ax[0].set_title("(1) Training data")

    for k, (title, sk, rem) in enumerate([
            (r"(2) Phase 0: Mapper", sk0, None),
            (r"(3) Phase 1: Annealing", sk1, None),
            (r"(4) Phase 2a: Edge Screening", sk2a, removed),
            (r"(5) Phase 2b: Fit", sk2b, None),
            (r"(6) Phase 3: Growth", sk_grown, None),
            (r"(7) Phase 4: Prune and Refit", sk_final, None)], start=1):
        draw_points(ax[k], X, y, alpha=0.55, s=3.2)
        draw_skeletons(ax[k], sk, removed=rem)
        ax[k].set_title(title)
        betti_label(ax[k], sk)

    lim = float(np.abs(X).max()) * 1.12
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 320),
                         np.linspace(-lim, lim, 320))
    grid = np.c_[gx.ravel(), gy.ravel()]
    pred = SkeletonClassifier(sk_final, est.classes_).predict(grid)
    ax[7].contourf(gx, gy, pred.reshape(gx.shape), levels=[-0.5, 0.5, 1.5],
                   colors=[PT_COL[0], PT_COL[1]], alpha=0.32, zorder=0)
    draw_skeletons(ax[7], sk_final, vs=14, lw=1.4)
    ax[7].set_title("(8) Decision regions")

    for a in ax:
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            a.spines[side].set_color("#D4D7DC")

    handles = [Line2D([], [], color=CLS_COL[c], marker="o", markersize=3.4,
                      linewidth=1.4, markeredgecolor="white",
                      markeredgewidth=0.4, label=lab)
               for c, lab in enumerate(labels)]
    if any(removed.values()):
        handles.append(Line2D([], [], color=GHOST, linewidth=1.1,
                              linestyle=(0, (2.2, 1.6)),
                              label="edge deleted in Phase 2a"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.015))

    fig.tight_layout(pad=0.4, w_pad=1.1, h_pad=2.6, rect=(0, 0.045, 1, 1))
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=320)

    print("budget", BUDGET, "emitted", len(Xp))
    print("per-class vertices", [len(V) for V, _ in sk_final])
    for name, sk in (("nerve", sk0), ("frozen", sk2a), ("final", sk_final)):
        print(name, [betti(len(V), E) for V, E in sk])
    print("admission", snaps["admission"])
    print("provenance keys", sorted(est.spine_params_))
    print("forced", est.spine_params_["forced_removals"],
          "repr-growth", est.spine_params_["growth_representation_fallbacks"],
          "retention", est.spine_params_["topology_retention_nerve_to_frozen"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ring")
