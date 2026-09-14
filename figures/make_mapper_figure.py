"""Figure: a worked Mapper example on a noisy annulus.

Outputs ``mapper_example.pdf`` and ``mapper_example.png``.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from spine import cover as cover_mod
from spine.lens import apply_lens
from spine.phase0 import phase0, min_cluster_size, _cluster_slab, _reassign_noise
from spine.topology import betti, normalize_edges

SEED = 7
N = 350

DATA_GREY = "#AEB4BC"
GHOST_GREY = "#DDE0E4"
RULE = "#9AA1AA"
INK = "#1F2328"
RAMP = plt.cm.Blues
STEPS = [0.42, 0.56, 0.70, 0.84, 0.97]


def annulus(n, seed=SEED):
    """A noisy elliptical annulus, rotated so within-class PC1 is well defined.

    Standardising a class makes both coordinates unit variance, so an
    axis-aligned ring has an identity correlation matrix and no leading
    direction.  Elongating and rotating the ring gives the lens a real
    direction to find.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    r = 1.0 + rng.normal(0.0, 0.06, n)
    X = np.c_[2.0 * r * np.cos(t), 1.0 * r * np.sin(t)]
    th = np.deg2rad(35.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return X @ R.T


def lens_affine(X, f):
    """Recover ``g, c`` with ``f(x) = x . g + c``; the lens is affine in ``x``."""
    A = np.c_[X, np.ones(len(X))]
    sol, *_ = np.linalg.lstsq(A, f, rcond=None)
    g, c = sol[:-1], sol[-1]
    assert np.allclose(A @ sol, f, atol=1e-8), "lens is not affine in x"
    return g, c


def rebuild(X, n_total):
    """Phase 0's own loop, retaining per-cluster membership for drawing."""
    mcs = min_cluster_size(n_total)
    n_int = cover_mod.n_intervals(len(X))
    f, _ = apply_lens(X, kind="pc1")
    cover, placement, f_eff = cover_mod.build_cover(f, n_int, mcs)

    V, groups, slab_of = [], [], []
    for s, idx in enumerate(cover_mod.slab_members(f_eff, cover)):
        if len(idx) < max(3, mcs):
            continue
        labels, core = _reassign_noise(X[idx], _cluster_slab(X[idx], mcs))
        for lab in np.unique(labels):
            in_cluster = labels == lab
            core_rows, all_rows = idx[in_cluster & core], idx[in_cluster]
            V.append(X[core_rows if len(core_rows) else all_rows].mean(axis=0))
            groups.append(all_rows)
            slab_of.append(s)
    sets = [set(int(r) for r in grp) for grp in groups]
    E = normalize_edges([(i, j) for i in range(len(V))
                         for j in range(i + 1, len(V)) if sets[i] & sets[j]])
    return np.asarray(V), E, groups, slab_of, f, cover, placement


def main():
    X = annulus(N)
    V_ref, E_ref, prov = phase0(X, N)
    V, E, groups, slab_of, f, cover, placement = rebuild(X, N)
    assert np.allclose(V, V_ref) and E == E_ref, "figure diverged from phase0"
    b0, b1 = betti(len(V), E)
    g, c = lens_affine(X, f)

    # Display basis: vertical axis along the lens direction.
    u = g / np.linalg.norm(g)
    if u[1] < 0:
        u = -u
        g, c = -g, -c
    w = np.array([-u[1], u[0]])
    B = np.c_[w, u]
    P, Q = X @ B, V @ B
    ctr = P.mean(axis=0)
    P, Q = P - ctr, Q - ctr
    cover_disp = [((lo - c) / np.linalg.norm(g), (hi - c) / np.linalg.norm(g))
                  for lo, hi in cover]
    band = [(lo - ctr[1], hi - ctr[1]) for lo, hi in cover_disp]
    fvals = P[:, 1]
    slab_col = [RAMP(s) for s in STEPS[:len(cover)]]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 7.2,
        "axes.linewidth": 0.6, "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5))
    xlim, ylim = 1.95, 2.45

    # (a) data coloured by the lens
    ax = axes[0]
    ax.scatter(P[:, 0], P[:, 1], c=fvals, cmap=RAMP,
               vmin=fvals.min() - 0.55 * np.ptp(fvals), vmax=fvals.max(),
               s=9, linewidths=0.25, edgecolors="white", zorder=2)
    ax.annotate("", xy=(-0.86 * xlim, 0.66 * ylim), xytext=(-0.86 * xlim, -0.66 * ylim),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.8,
                                shrinkA=0, shrinkB=0), zorder=5)
    ax.text(-0.86 * xlim, 0.72 * ylim, r"$f$", color=INK, fontsize=8.5,
            ha="center", va="bottom")
    ax.set_title(r"(1) Apply Lens Function, $f$")

    # (b) the overlapping cover
    ax = axes[1]
    ax.scatter(P[:, 0], P[:, 1], color=DATA_GREY, s=7, linewidths=0, zorder=2)
    for k, (lo, hi) in enumerate(band):
        ax.add_patch(Rectangle((-xlim, lo), 2 * xlim, hi - lo,
                               facecolor=to_rgba(slab_col[k], 0.22),
                               edgecolor=to_rgba(slab_col[k], 0.85),
                               linewidth=0.5, zorder=1))
        ax.text(0.93 * xlim, 0.5 * (lo + hi), rf"$U_{{{k + 1}}}$", fontsize=6.6,
                color=INK, ha="right", va="center", zorder=4)
    lo1, hi0 = band[4][0], band[3][1]
    ax.annotate("overlap", xy=(-0.42 * xlim, 0.5 * (lo1 + hi0)),
                xytext=(-0.93 * xlim, 0.88 * ylim), fontsize=6.6, color=INK,
                ha="left", va="top", zorder=5,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.55,
                                shrinkA=2, shrinkB=1))
    ax.set_title(r"(2) Overlapping Cover $\mathcal{U}$")

    # (c) cluster each preimage
    ax = axes[2]
    for rows, s in zip(groups, slab_of):
        ax.scatter(P[rows, 0], P[rows, 1], color=slab_col[s], s=9,
                   linewidths=0.25, edgecolors="white", zorder=2)
    ax.scatter(Q[:, 0], Q[:, 1], marker="o", s=24, facecolor="white",
               edgecolors=INK, linewidths=0.9, zorder=4)
    ax.set_title(r"(3) Cluster Each Preimage")

    # (d) the nerve
    ax = axes[3]
    ax.scatter(P[:, 0], P[:, 1], color=GHOST_GREY, s=5, linewidths=0, zorder=1)
    for i, j in E:
        ax.plot([Q[i, 0], Q[j, 0]], [Q[i, 1], Q[j, 1]], color=RULE,
                linewidth=1.5, solid_capstyle="round", zorder=2)
    ax.scatter(Q[:, 0], Q[:, 1], c=[slab_col[s] for s in slab_of],
               s=[26 + 1.6 * len(rows) for rows in groups],
               edgecolors=INK, linewidths=0.8, zorder=3)
    ax.set_title("(4) Add Edges Based on Overlap")

    for ax in axes:
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_color("#D4D7DC")

    fig.tight_layout(pad=0.35, w_pad=0.7)
    fig.savefig("mapper_example.pdf", bbox_inches="tight")
    fig.savefig("mapper_example.png", bbox_inches="tight", dpi=320)
    print(f"n={N} slabs={len(cover)} placement={placement} "
          f"|V|={len(V)} |E|={len(E)} betti=({b0},{b1}) "
          f"k_per_slab={[s['k'] for s in prov['slabs']]}")


if __name__ == "__main__":
    main()
