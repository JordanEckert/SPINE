"""Distance primitives for an embedded 1-complex (SPINE, section 1).

A skeleton ``S_c = (V_c, E_c)`` has vertices ``V_c`` (an ``(m, d)`` array)
and edges ``E_c`` (a list of ``(i, j)`` index pairs, ``i < j``).  Its
geometric realisation ``|S_c|`` is the union of the vertices and the
closed segments of its edges, and the prediction rule of section 1 is

    yhat(x) = argmin_c dist(x, |S_c|).

Everything downstream (edge admission, discriminative fitting, growth,
pruning) is expressed through the two routines here, so their exactness
is the exactness of the method.

Segment distance
----------------
For a segment ``[u, v]`` with ``w = v - u``:

    t = clip( <x - u, w> / <w, w> , 0, 1 )
    p = u + t w
    d = || x - p ||

Vectorised form.  Writing ``num = <x - u, w>`` and ``ww = <w, w>``, the
raw (unclipped) parameter is ``t_raw = num / ww`` and

    ||x - p||^2 = || (x-u) - t w ||^2
                = ||x-u||^2 - 2 t num + t^2 ww.

Rather than evaluate that quadratic (which cancels badly when ``t`` is
clipped), the three branches are taken directly:

    t_raw <= 0  ->  ||x - u||^2                  (clipped to u)
    t_raw >= 1  ->  ||x - v||^2                  (clipped to v)
    otherwise   ->  ||x - u||^2 - num^2 / ww     (interior foot)

The two endpoint terms are *gathered* from a single
``cdist(X, V, 'sqeuclidean')`` -- one exact pairwise computation per
skeleton, no ``||x||^2 - 2<x,u> + ||u||^2`` expansion anywhere -- so the
only floating-point subtraction of comparable magnitudes is the interior
branch, where the result is bounded below by 0 and clipped there.

Argmin convention
-----------------
``skeleton_argmin`` reports which object won.  Ties resolve exactly as
the reference implementation of Appendix B does: the vertex minimum is
established first and a segment displaces it only on a STRICT
improvement, and among segments the lowest edge index wins.  Tie-breaking
is fixed rather than incidental because Phase 2a compares risks that
differ by a single validation point.
"""

import numpy as np
from scipy.spatial.distance import cdist

#: Squared length below which an edge is treated as degenerate (its two
#: endpoints coincident).  Such an edge contributes the distance to its
#: first endpoint, which is the limit of the segment distance as the
#: edge collapses, so the primitive stays continuous.
DEGENERATE_WW = 1e-12

#: Sentinel used in the ``win_edge`` output when a VERTEX, not a
#: segment, is the nearest object.
NO_EDGE = -1


def edge_arrays(edges):
    """Split an edge list into two index arrays ``(Ei, Ej)``."""
    if len(edges) == 0:
        return (np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp))
    arr = np.asarray(edges, dtype=np.intp)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "edges must be a sequence of (i, j) pairs, got shape {0}"
            .format(arr.shape))
    return arr[:, 0].copy(), arr[:, 1].copy()


def segment_distances(X, V, edges, vertex_sqdist=None):
    """Distance from every row of ``X`` to every segment of ``edges``.

    Returns ``(D, T)``, both ``(n, |E|)``: ``D[a, k]`` is the Euclidean
    distance from ``X[a]`` to segment ``k`` and ``T[a, k]`` its clipped
    projection parameter.  ``vertex_sqdist`` may supply a precomputed
    ``cdist(X, V, 'sqeuclidean')`` to avoid recomputing it.
    """
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    Ei, Ej = edge_arrays(edges)
    n, m = len(X), len(Ei)
    if m == 0:
        return np.zeros((n, 0)), np.zeros((n, 0))

    if vertex_sqdist is None:
        vertex_sqdist = cdist(X, V, "sqeuclidean")

    U = V[Ei]                                     # (m, d) segment starts
    W = V[Ej] - U                                 # (m, d) segment vectors
    ww = np.einsum("kd,kd->k", W, W)              # (m,)
    degenerate = ww < DEGENERATE_WW
    ww_safe = np.where(degenerate, 1.0, ww)

    # num[a, k] = <X[a] - U[k], W[k]>
    num = X @ W.T - np.einsum("kd,kd->k", U, W)[None, :]
    t_raw = num / ww_safe[None, :]
    t_raw[:, degenerate] = 0.0

    XU2 = vertex_sqdist[:, Ei]                    # ||x - u||^2
    XV2 = vertex_sqdist[:, Ej]                    # ||x - v||^2
    interior = XU2 - (num * num) / ww_safe[None, :]
    d2 = np.where(t_raw <= 0.0, XU2,
                  np.where(t_raw >= 1.0, XV2, interior))
    np.maximum(d2, 0.0, out=d2)                   # guard the interior branch
    return np.sqrt(d2), np.clip(t_raw, 0.0, 1.0)


def skeleton_distance(X, V, edges, vertex_sqdist=None):
    """Min distance from every row of ``X`` to ``|S| = V union segments``."""
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    if len(V) == 0:
        raise ValueError("a skeleton must have at least one vertex")
    if vertex_sqdist is None:
        vertex_sqdist = cdist(X, V, "sqeuclidean")
    best = np.sqrt(np.maximum(vertex_sqdist.min(axis=1), 0.0))
    if len(edges):
        D, _ = segment_distances(X, V, edges, vertex_sqdist=vertex_sqdist)
        np.minimum(best, D.min(axis=1), out=best)
    return best


def skeleton_argmin(X, V, edges, vertex_sqdist=None):
    """Nearest object of ``|S|`` for every row of ``X``.

    Returns ``(best, win_edge, win_t, win_vertex)``:

    * ``best``       -- the distance;
    * ``win_edge``   -- winning edge index, or ``NO_EDGE`` if a vertex won;
    * ``win_t``      -- projection parameter on the winning segment (0 if
                        a vertex won);
    * ``win_vertex`` -- index of the nearest VERTEX, always defined (it is
                        the fallback object and is what the update rule
                        uses when ``win_edge == NO_EDGE``).
    """
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    if len(V) == 0:
        raise ValueError("a skeleton must have at least one vertex")
    if vertex_sqdist is None:
        vertex_sqdist = cdist(X, V, "sqeuclidean")

    n = len(X)
    win_vertex = vertex_sqdist.argmin(axis=1)
    best = np.sqrt(np.maximum(
        vertex_sqdist[np.arange(n), win_vertex], 0.0))
    win_edge = np.full(n, NO_EDGE, dtype=np.intp)
    win_t = np.zeros(n)

    if len(edges):
        D, T = segment_distances(X, V, edges, vertex_sqdist=vertex_sqdist)
        k = D.argmin(axis=1)                       # first minimum wins
        dk = D[np.arange(n), k]
        # STRICT: a segment displaces the vertex minimum only if it beats
        # it outright, matching Appendix B's ``m = d < best``.
        better = dk < best
        best = np.where(better, dk, best)
        win_edge = np.where(better, k, win_edge)
        win_t = np.where(better, T[np.arange(n), k], win_t)
    return best, win_edge, win_t, win_vertex


def class_distances(X, skeletons):
    """``(n, C)`` matrix of distances from ``X`` to each class skeleton."""
    return np.stack([skeleton_distance(X, V, E) for (V, E) in skeletons],
                    axis=1)


def predict(X, skeletons):
    """Skeleton decision rule: ``argmin_c dist(x, |S_c|)``.

    Ties resolve to the lowest class index, matching ``np.argmin`` on the
    stacked distance matrix.  Returns positional class indices, not
    labels; the estimator maps them back through ``classes_``.
    """
    return class_distances(X, skeletons).argmin(axis=1)


def risk(X, y, skeletons):
    """0-1 risk of the skeleton rule (``y`` as positional class indices)."""
    return float((predict(X, skeletons) != np.asarray(y)).mean())


# --------------------------------------------------------------------
# One-class-at-a-time evaluation.
# --------------------------------------------------------------------

def other_class_extrema(X, skeletons, c):
    """``(min, argmin)`` over every class except ``c``, in original indices."""
    keep = [k for k in range(len(skeletons)) if k != c]
    if not keep:
        raise ValueError("a margin needs at least two classes")
    sub = np.stack(
        [skeleton_distance(X, skeletons[k][0], skeletons[k][1])
         for k in keep], axis=1)
    j = sub.argmin(axis=1)
    return sub[np.arange(len(sub)), j], np.asarray(keep, dtype=int)[j]


def predicted_is_class(d_c, other_min, other_arg, c):
    """Would ``argmin`` pick class ``c``, given the other classes' best?

    Class ``c`` takes a tie only when its index is the smaller one, which
    is exactly what ``np.argmin`` on the stacked distance matrix does.
    """
    return (d_c < other_min) | ((d_c == other_min) & (c < other_arg))


def risk_given_others(d_c, other_min, other_arg, c, y):
    """0-1 risk when only class ``c``'s distances change."""
    pred = np.where(predicted_is_class(d_c, other_min, other_arg, c),
                    c, other_arg)
    return float((pred != np.asarray(y)).mean())
