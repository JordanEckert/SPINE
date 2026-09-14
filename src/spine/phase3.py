"""Phase 3 -- growth by a homotopy-preserving grammar """

import numpy as np

from .geometry import NO_EDGE, skeleton_argmin
from .topology import as_normalized, degrees, normalize_edges

#: Fraction of attributed errors that must clamp at one endpoint before
#: the site is treated as an endpoint pile-up rather than an interior one.
CLAMP_MAJORITY = 0.7
#: Tolerance for calling a projection parameter clamped.
CLAMP_TOL = 1e-6


def _attribute(X_sel, V, E):
    """Group the selected rows by the nearest object of ``(V, E)``.

    Returns ``{("e", k) | ("v", i): row indices}`` together with the
    per-row projection parameters, so the caller can decide bisect vs
    extend from where the errors actually project.
    """
    if len(X_sel) == 0:
        return {}, np.zeros(0)
    _, win_e, win_t, win_v = skeleton_argmin(X_sel, V, E)
    groups = {}
    for r in range(len(X_sel)):
        key = (("e", int(win_e[r])) if win_e[r] != NO_EDGE
               else ("v", int(win_v[r])))
        groups.setdefault(key, []).append(r)
    return groups, win_t


def _grow_once(V, E, X_site, groups, win_t):
    """Apply one grammar operation at the highest-mass site.

    ``X_site`` are the points that were attributed; ``groups`` maps a site
    to their row indices.  Returns ``(V, E)`` or ``None`` when no site is
    available.
    """
    if not groups:
        return None
    # Highest error mass; ties resolve to the lexicographically smallest
    # site key, so the choice is deterministic across runs.
    key = max(sorted(groups), key=lambda kk: len(groups[kk]))
    rows = groups[key]
    w = X_site[rows].mean(axis=0)
    kind, idx = key

    if kind == "v":
        # EXTEND from a vertex that was itself the nearest object.
        V = np.vstack([V, w])
        return V, normalize_edges(list(E) + [(idx, len(V) - 1)])

    a, b = E[idx]
    ts = np.asarray(win_t)[rows]
    clamped = (ts <= CLAMP_TOL) | (ts >= 1.0 - CLAMP_TOL)
    if clamped.mean() > CLAMP_MAJORITY:
        end = a if (ts < 0.5).mean() > 0.5 else b
        if degrees(len(V), as_normalized(E))[end] <= 1:
            # EXTEND from a leaf the errors pile up at.
            V = np.vstack([V, w])
            return V, normalize_edges(list(E) + [(end, len(V) - 1)])
    # BISECT: subdivision is a homeomorphism.
    V = np.vstack([V, w])
    nv = len(V) - 1
    keep = [e for e in normalize_edges(E) if e != (a, b)]
    return V, normalize_edges(keep + [(a, nv), (nv, b)])


def _representation_site(Xc_train, V, E):
    """Fallback site: the object carrying the most squared distortion."""
    if len(Xc_train) == 0:
        return None, None, None
    best, win_e, win_t, win_v = skeleton_argmin(Xc_train, V, E)
    groups = {}
    for r in range(len(Xc_train)):
        key = (("e", int(win_e[r])) if win_e[r] != NO_EDGE
               else ("v", int(win_v[r])))
        groups.setdefault(key, []).append(r)
    if not groups:
        return None, None, None
    mass = {k: float((best[rows] ** 2).sum()) for k, rows in groups.items()}
    key = max(sorted(groups), key=lambda kk: mass[kk])
    return {key: groups[key]}, win_t, key


def grow_class(V, E, Xv_c, errors_fn, Xc_train, n_add):
    """Add ``n_add`` vertices to one class skeleton.

    ``Xv_c`` holds that class's VALIDATION points and ``errors_fn(V, E)``
    returns the boolean mask of the ones the current model gets wrong --
    a callable rather than a fixed mask because the attribution must be
    recomputed after every addition, exactly as Appendix B's ``grow``
    re-predicts inside its loop.  ``Xc_train`` holds the class's training
    points and is consulted only by the no-error fallback.

    Returns ``(V, E, n_fallback)``.
    """
    V = np.array(V, dtype=float, copy=True)
    E = normalize_edges(E)
    n_fallback = 0
    for _ in range(int(n_add)):
        if len(Xv_c):
            bad = Xv_c[errors_fn(V, E)]
        else:
            bad = Xv_c
        groups, win_t = _attribute(bad, V, E)
        site_points = bad
        if not groups:
            groups, win_t, _ = _representation_site(Xc_train, V, E)
            site_points = Xc_train
            if not groups:
                break
            n_fallback += 1
        grown = _grow_once(V, E, site_points, groups, win_t)
        if grown is None:
            break
        V, E = grown
    return V, E, n_fallback
