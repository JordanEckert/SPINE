"""Phase 0 -- structure """

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN

from . import cover as cover_mod
from .lens import apply_lens
from .topology import normalize_edges

#: Section 4.3 global constants.  Not tuned, not adapted per slab.
MIN_SAMPLES = 5
CLUSTER_SELECTION_METHOD = "eom"
ALLOW_SINGLE_CLUSTER = True
CLUSTER_SELECTION_EPSILON = 0.0


def min_cluster_size(n_total):
    """``clip(0.005 * n, 15, 100)`` -- global, mildly sublinear in total n."""
    return int(np.clip(0.005 * float(n_total), 15, 100))


def _cluster_slab(X_slab, mcs):
    """HDBSCAN on one slab at the pinned constants; returns raw labels.

    ``min_cluster_size`` is lowered to the slab population only when the
    slab is smaller than the global value, since HDBSCAN cannot be asked
    for a cluster larger than the data it is given.  That is a library
    constraint, not a per-slab retuning: the global value is the ceiling
    everywhere and slabs at least that large all see the same number.
    """
    n = len(X_slab)
    effective = int(min(mcs, max(2, n - 1)))
    model = HDBSCAN(
        min_cluster_size=effective,
        min_samples=int(min(MIN_SAMPLES, n)),
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        allow_single_cluster=ALLOW_SINGLE_CLUSTER,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        copy=True,          # never mutate the caller's slab in place
    )
    return model.fit_predict(X_slab)


def _reassign_noise(X_slab, labels):
    """Send noise points to their nearest in-slab cluster (EDGES ONLY).

    Returns ``(labels_for_edges, is_core)`` where ``is_core`` marks the
    points HDBSCAN itself assigned.  Vertex positions use ``is_core``;
    membership (and hence edges) uses ``labels_for_edges``.  If HDBSCAN
    called the whole slab noise, the slab is treated as one cluster --
    the honest reading of "no density structure resolvable here", and the
    coarse direction, which is the safe one by the third organising
    principle.
    """
    core = labels >= 0
    if not core.any():
        return np.zeros(len(labels), dtype=int), np.ones(len(labels), bool)
    out = labels.copy()
    if (~core).any():
        core_idx = np.where(core)[0]
        nearest = cdist(X_slab[~core], X_slab[core]).argmin(axis=1)
        out[~core] = labels[core_idx[nearest]]
    return out, core


def phase0(Xc, n_total, gain=cover_mod.DEFAULT_GAIN, lens="pc1",
           n_neighbors=10, lens_scaling="standardize"):
    """Build one class's nerve.

    Returns ``(V, E, provenance)``.  ``V`` is ``(m, d)`` cluster
    centroids, ``E`` a canonical edge list, and ``provenance`` a dict
    recording the lens, the slab count, the placement actually used, the
    per-slab populations and cluster counts, and the noise fraction --
    everything needed to say afterwards what cover a fold ran under.
    """
    Xc = np.asarray(Xc, dtype=float)
    n_c = len(Xc)
    if n_c == 0:
        raise ValueError("phase0 received an empty class")

    mcs = min_cluster_size(n_total)
    n_int = cover_mod.n_intervals(n_c)
    f, lens_info = apply_lens(Xc, kind=lens, n_neighbors=n_neighbors,
                              scaling=lens_scaling)
    constant_lens = bool(np.ptp(f) <= 0.0)
    # ``f_eff`` is the coordinate the cover lives in: the lens itself
    # under uniform placement, its rank reparametrisation under the
    # quantile fallback.  Membership must be tested against it, not
    # against the raw lens (see cover.py's module docstring).
    cover, placement, f_eff = cover_mod.build_cover(
        f, n_int, mcs, gain=gain)
    members_of_slab = cover_mod.slab_members(f_eff, cover)

    V, members, per_slab = [], [], []
    n_noise = 0
    for idx in members_of_slab:
        if len(idx) < max(3, mcs):
            # Too small to hold a resolvable cluster at the global scale.
            per_slab.append({"n": int(len(idx)), "k": 0})
            continue
        raw = _cluster_slab(Xc[idx], mcs)
        labels, core = _reassign_noise(Xc[idx], raw)
        n_noise += int((~core).sum())
        uniq = np.unique(labels)
        per_slab.append({"n": int(len(idx)), "k": int(len(uniq))})
        for lab in uniq:
            in_cluster = labels == lab
            core_rows = idx[in_cluster & core]
            all_rows = idx[in_cluster]
            # Vertex at the centroid of the cluster's own (non-noise)
            # members; reassigned noise contributes membership, and hence
            # edges, but never position.
            seed = core_rows if len(core_rows) else all_rows
            V.append(Xc[seed].mean(axis=0))
            members.append(set(int(r) for r in all_rows))

    provenance = {
        "lens": lens,
        "n_intervals": int(n_int),
        "placement": placement,
        "min_cluster_size": int(mcs),
        "min_samples": int(MIN_SAMPLES),
        "gain": float(gain),
        "slabs": per_slab,
        "noise_reassigned": int(n_noise),
        "n_class": int(n_c),
        "constant_lens": constant_lens,
    }
    provenance.update(lens_info)

    if not V:
        # Every slab was under-populated: the class supports no cover at
        # this resolution.  One vertex at the class centroid is the
        # coarsest honest answer and is recorded as such.
        provenance["degenerate"] = "no_slab_reached_min_cluster_size"
        return Xc.mean(axis=0)[None, :], [], provenance

    # Nerve: two clusters are adjacent when they share at least one
    # point.  Clusters within a slab partition it, so a shared point
    # necessarily comes from an overlap between slabs; the all-pairs scan
    # is therefore identical to the adjacent-slab rule of section 4.4 and
    # needs no assumption about which slabs can meet.
    E = [(i, j)
         for i in range(len(V))
         for j in range(i + 1, len(V))
         if members[i] & members[j]]
    provenance["n_vertices"] = len(V)
    provenance["n_edges"] = len(E)
    return np.asarray(V, dtype=float), normalize_edges(E), provenance
