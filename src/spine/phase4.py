"""Phase 4 -- prune to budget """

import numpy as np
from scipy.spatial.distance import cdist

from .geometry import (other_class_extrema, risk_given_others,
                       segment_distances, skeleton_distance)
from .topology import (as_normalized, degrees,
                       is_homotopy_preserving_removal, neighbours,
                       normalize_edges, remove_vertex)

#: How many nearest OBJECTS per point the incremental pruner keeps.  A
#: homotopy-preserving removal deletes at most three columns of the
#: object-distance matrix (the vertex itself plus its <= 2 incident
#: edges), so four candidates always contain one survivor.
TOP_K = 4


def _object_distances(X, V, E):
    """``(n, |V| + |E|)`` distance from each point to each OBJECT of ``|S|``.

    Columns ``0 .. |V|-1`` are the vertices and the rest are the segments,
    in edge-list order.  ``skeleton_distance`` is the row-wise minimum of
    this matrix; keeping the matrix itself is what lets the pruner price a
    removal without rebuilding it.
    """
    V = np.asarray(V, dtype=float)
    vsq = cdist(X, V, "sqeuclidean")
    dv = np.sqrt(np.maximum(vsq, 0.0))
    if not len(E):
        return dv
    D, _ = segment_distances(X, V, E, vertex_sqdist=vsq)
    return np.hstack([dv, D])


def _top_objects(D):
    """The ``TOP_K`` nearest object columns per row, sorted by distance."""
    n, K = D.shape
    k = min(TOP_K, K)
    part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(D, part, axis=1)
    order = np.argsort(vals, axis=1, kind="stable")
    return (np.take_along_axis(part, order, axis=1),
            np.take_along_axis(vals, order, axis=1))


def _distance_excluding(cols, vals, excluded):
    """Row-wise nearest object whose column is not in ``excluded``.

    Exact whenever ``len(excluded) < cols.shape[1]``, which the caller
    guarantees by using this path only for removals that delete at most
    ``TOP_K - 1`` columns.
    """
    banned = np.isin(cols, np.asarray(list(excluded), dtype=cols.dtype))
    first = np.argmax(~banned, axis=1)
    return vals[np.arange(len(vals)), first]


def candidate_pool(V, E):
    """Vertices Phase 4 may remove, and whether the choice is forced.

    Returns ``(pool, forced)``.  The preferred pool is the set of
    homotopy-preserving removals.  When it is empty the budget still has
    to be met, so Appendix B's degree <= 2 rule is the fallback and every
    vertex is the last resort -- and ``forced`` says so, which is what
    keeps Phase 5's assertion honest instead of vacuous.
    """
    n = len(V)
    E = as_normalized(E)
    nb = neighbours(n, E)
    safe = [i for i in range(n)
            if is_homotopy_preserving_removal(n, E, i, nb=nb)]
    if safe:
        return safe, False
    deg = degrees(n, E)
    return ([i for i in range(n) if deg[i] <= 2] or list(range(n))), True


def _class_distance_without(V, E, Xv, i, cols, vals, nb):
    """``dist(., |S_c \\ {i}|)`` without rebuilding the whole skeleton.

    Deleting vertex ``i`` removes its own column and the columns of its
    incident edges, and -- when it had degree 2 -- introduces exactly one
    new segment joining its neighbours.  Everything else is untouched, so
    the new class distance is the nearest surviving object, compared
    against that one new segment.  Exact, not an approximation.

    Returns ``None`` when the removal deletes too many columns for the
    cached ``TOP_K``, and the caller falls back to a full rebuild.
    """
    excluded = {int(i)}
    m = len(V)
    for k, (a, b) in enumerate(E):
        if a == i or b == i:
            excluded.add(m + k)
    if len(excluded) >= cols.shape[1]:
        return None
    rest = _distance_excluding(cols, vals, excluded)
    ends = sorted(nb[i])
    if len(ends) == 2:
        # Only the two endpoints matter, so this is a distance to a
        # 2-vertex skeleton.  Passing the full ``V`` here would make every
        # candidate cost O(n |V|) through the internal cdist and put the
        # whole pruning phase back where the cached columns took it from.
        new, _ = segment_distances(Xv, V[[ends[0], ends[1]]], [(0, 1)])
        rest = np.minimum(rest, new[:, 0])
    return rest


def prune_class_once(V, E, c, Xv, yv, other_min, other_arg):
    """Remove one vertex of class ``c``.  Returns ``(candidate, forced)``.

    Only class ``c``'s distances change, so the other classes enter
    through ``other_min`` / ``other_arg``, computed once by the caller --
    the same exact substitution Phase 2a uses, tie-breaking included.
    Within the class, each candidate is priced from a cached
    object-distance matrix rather than by rebuilding the skeleton, so one
    pruning step costs ``O(|V| n_val)`` instead of ``O(|V|^2 n_val)``.
    """
    V = np.asarray(V, dtype=float)
    E = as_normalized(E)
    if len(V) <= 1:
        return None, False
    pool, forced = candidate_pool(V, E)

    D = _object_distances(Xv, V, E)
    cols, vals = _top_objects(D)
    nb = neighbours(len(V), E)

    best_i, best_risk = None, np.inf
    for i in pool:
        d_c = _class_distance_without(V, E, Xv, i, cols, vals, nb)
        if d_c is None:                            # too many columns gone
            cand_V, cand_E = remove_vertex(V, E, i)
            d_c = skeleton_distance(Xv, cand_V, cand_E)
        r = risk_given_others(d_c, other_min, other_arg, c, yv)
        if r < best_risk:                          # first minimum wins
            best_risk, best_i = r, i
    if best_i is None:
        return None, forced
    return remove_vertex(V, E, best_i), forced


def prune_to_budget(skeletons, targets, Xv, yv):
    """Prune every class down to its target vertex count.

    ``targets[c]`` is the exact number of vertices class ``c`` must end
    with.  Returns ``(skeletons, n_forced)``.
    """
    skeletons = [(np.asarray(V, dtype=float), normalize_edges(E))
                 for (V, E) in skeletons]
    yv = np.asarray(yv)
    n_forced = 0
    for c, target in enumerate(targets):
        if len(skeletons[c][0]) <= int(target):
            continue
        # Fixed for the whole of class c's pruning: no other class moves.
        other_min, other_arg = other_class_extrema(Xv, skeletons, c)
        while len(skeletons[c][0]) > int(target):
            candidate, forced = prune_class_once(
                skeletons[c][0], skeletons[c][1], c, Xv, yv,
                other_min, other_arg)
            if candidate is None:
                break
            skeletons[c] = candidate
            n_forced += int(forced)
    return skeletons, n_forced
