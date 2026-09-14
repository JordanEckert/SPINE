"""Phase 2 -- edge admission (2a) and discriminative fitting (2b). """

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

from .geometry import (NO_EDGE, risk_given_others, segment_distances,
                       skeleton_argmin, skeleton_distance)
from .topology import normalize_edges

ADMISSION_CRITERIA = ("risk", "support", "support_then_risk")
BOUNDARY_MODES = ("both", "negative_only", "off")
STEP_RULES = ("scale_free", "clipped", "raw")

#: Cap on a single vertex displacement under ``step_rule="clipped"``, in
#: DATA units, so its meaning depends on the harness's scaler: under
#: standardisation 0.05 is a twentieth of a standard deviation, under
#: min-max a twentieth of each feature's range.  Only ``step_rule
#: = "clipped"`` reads it; the default rule is bounded without a cap.
DEFAULT_MAX_STEP = 0.05


# --------------------------------------------------------------- 2a: risk

def _class_distance_parts(X, V, edges):
    """Vertex-minimum column and full segment-distance matrix for one class."""
    V = np.asarray(V, dtype=float)
    vsq = cdist(X, V, "sqeuclidean")
    dv = np.sqrt(np.maximum(vsq.min(axis=1), 0.0))
    if len(edges):
        D, _ = segment_distances(X, V, edges, vertex_sqdist=vsq)
    else:
        D = np.zeros((len(X), 0))
    return dv, D


def _combined(dv, D, alive):
    """Class distance using only the alive segments (plus the vertices)."""
    if D.shape[1] == 0 or not alive.any():
        return dv.copy()
    sub = np.where(alive[None, :], D, np.inf)
    return np.minimum(dv, sub.min(axis=1))


def _others(dist_matrix, c):
    """Min and argmin over every class except ``c`` (lowest index on ties)."""
    C = dist_matrix.shape[1]
    keep = [k for k in range(C) if k != c]
    sub = dist_matrix[:, keep]
    j = sub.argmin(axis=1)
    return sub[np.arange(len(sub)), j], np.asarray(keep, dtype=int)[j]


def admit_edges_by_risk(skeletons, Xv, yv):
    """Strict-improvement edge removal on the validation split.

    Returns ``(skeletons, n_removed)``.  Edges are visited longest-first
    (least supported first) within each class, and classes in index
    order, exactly as Appendix B does.
    """
    skeletons = [(np.asarray(V, dtype=float), normalize_edges(E))
                 for (V, E) in skeletons]
    C = len(skeletons)
    yv = np.asarray(yv)

    parts = [_class_distance_parts(Xv, V, E) for (V, E) in skeletons]
    alive = [np.ones(len(E), dtype=bool) for (_, E) in skeletons]
    current = np.stack(
        [_combined(dv, D, a) for (dv, D), a in zip(parts, alive)], axis=1)
    base = float((current.argmin(axis=1) != yv).mean())
    n_removed = 0

    for c, (V, E) in enumerate(skeletons):
        if not E:
            continue
        dv, D = parts[c]
        other_min, other_arg = _others(current, c)
        lengths = np.array([np.linalg.norm(V[i] - V[j]) for (i, j) in E])
        order = np.argsort(-lengths, kind="stable")   # longest first
        for k in order:
            if not alive[c][k]:
                continue
            trial = alive[c].copy()
            trial[k] = False
            d_c = _combined(dv, D, trial)
            r = risk_given_others(d_c, other_min, other_arg, c, yv)
            if r < base:                              # STRICT
                base = r
                alive[c] = trial
                n_removed += 1
        skeletons[c] = (V, [E[k] for k in range(len(E)) if alive[c][k]])
        current[:, c] = _combined(dv, D, alive[c])
    return skeletons, n_removed


# ------------------------------------------------------------ 2a: support

def support_rho(Xc, V):
    """``rho`` = median distance from a class point to its nearest vertex."""
    Xc = np.asarray(Xc, dtype=float)
    V = np.asarray(V, dtype=float)
    if len(Xc) == 0 or len(V) == 0:
        return 0.0
    return float(np.median(np.sqrt(np.maximum(
        cdist(Xc, V, "sqeuclidean").min(axis=1), 0.0))))


def edge_is_supported(Xc, V, edge, rho):
    """Appendix C's gap test for one edge.

    Admit when no gap in the supported projection parameters (with the
    endpoints 0 and 1 always counted as supported) exceeds
    ``rho / ||v - u||``.  A zero-length edge is supported by definition:
    it spans no distance in which to be unsupported.
    """
    i, j = edge
    u, v = V[i], V[j]
    length = float(np.linalg.norm(v - u))
    if length <= 0.0:
        return True
    if rho <= 0.0:
        return False
    D, T = segment_distances(Xc, V, [edge])
    t = np.sort(T[D[:, 0] <= rho, 0])
    knots = np.concatenate(([0.0], t, [1.0]))
    return bool(np.max(np.diff(knots)) <= rho / length)


def admit_edges_by_support(skeletons, X_by_class):
    """Remove unsupported edges class by class.  Returns ``(skels, n)``."""
    out, n_removed = [], 0
    for c, (V, E) in enumerate(skeletons):
        V = np.asarray(V, dtype=float)
        E = normalize_edges(E)
        Xc = X_by_class[c]
        rho = support_rho(Xc, V)
        keep = [e for e in E if edge_is_supported(Xc, V, e, rho)]
        n_removed += len(E) - len(keep)
        out.append((V, keep))
    return out, n_removed


def phase2a(skeletons, Xv, yv, X_by_class, criterion="risk"):
    """Edge admission.  Returns ``(skeletons, info)``.

    ``info`` records how many edges each pass removed, so a run can say
    afterwards which criterion did the work.
    """
    if criterion not in ADMISSION_CRITERIA:
        raise ValueError(
            "unknown admission criterion {0!r}; expected one of {1}"
            .format(criterion, ADMISSION_CRITERIA))
    info = {"criterion": criterion, "removed_support": 0, "removed_risk": 0}
    if criterion in ("support", "support_then_risk"):
        skeletons, n = admit_edges_by_support(skeletons, X_by_class)
        info["removed_support"] = int(n)
    if criterion in ("risk", "support_then_risk"):
        skeletons, n = admit_edges_by_risk(skeletons, Xv, yv)
        info["removed_risk"] = int(n)
    return skeletons, info


# ------------------------------------------------------- 2b: discriminative

def _forward(X, skeletons):
    """Nearest object in every class skeleton for every row of ``X``."""
    per_class = [skeleton_argmin(X, V, E) for (V, E) in skeletons]
    dist = np.stack([p[0] for p in per_class], axis=1)
    return per_class, dist


def _movable_vertices(neg_class, neg_edge, neg_vertex, skeletons):
    """Which vertices section 6.2 permits Phase 2b to move.

    A boundary OBJECT is one that is the nearest wrong-class object for at
    least one training point -- read straight off the ``d-`` winners.  A
    movable VERTEX is one incident to a boundary segment, or a boundary
    vertex itself (the isolated-vertex case, which section 6.2's phrase
    "objects that are somewhere the nearest wrong-class object" covers and
    which its segment-only wording does not name).  Returned as one
    boolean mask per class, so the test in the inner loop is an array
    lookup rather than a scan over the boundary set.
    """
    movable = [np.zeros(len(V), dtype=bool) for (V, _) in skeletons]
    for c, k, v in zip(neg_class, neg_edge, neg_vertex):
        c, k, v = int(c), int(k), int(v)
        if k >= 0:
            a, b = skeletons[c][1][k]
            movable[c][a] = True
            movable[c][b] = True
        else:
            movable[c][v] = True
    return movable


def phase2b(skeletons, X, y, epochs=25, lr=0.05, sigma=3.0,
            batch_size=256, boundary_restriction="both",
            step_rule="scale_free", max_step=DEFAULT_MAX_STEP, rng=None):
    """Margin-driven vertex placement at fixed topology.

    ``y`` holds positional class indices.  Topology is never touched
    here; only vertex coordinates move, so the invariant holds trivially.

    ``step_rule`` selects how the exact gradient is turned into a step;
    see the module docstring for the measurement that motivates the
    default.  All three descend the SAME loss in the SAME direction and
    differ only in the per-sample step length.
    """
    if boundary_restriction not in BOUNDARY_MODES:
        raise ValueError(
            "unknown boundary_restriction {0!r}; expected one of {1}"
            .format(boundary_restriction, BOUNDARY_MODES))
    if step_rule not in STEP_RULES:
        raise ValueError(
            "unknown step_rule {0!r}; expected one of {1}"
            .format(step_rule, STEP_RULES))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    skeletons = [(np.array(V, dtype=float, copy=True), normalize_edges(E))
                 for (V, E) in skeletons]
    C = len(skeletons)
    n = len(X)
    if n == 0 or epochs <= 0:
        return skeletons
    if rng is None:
        rng = np.random.default_rng(0)
    batch_size = max(1, int(batch_size))

    # Bootstrap the boundary set with one forward pass; afterwards it is
    # rebuilt from the d- winners each epoch observes, at no extra cost.
    per_class, dist = _forward(X, skeletons)
    masked = dist.copy()
    masked[np.arange(n), y] = np.inf
    neg_c = masked.argmin(axis=1)
    movable = _movable_vertices(
        neg_c,
        [per_class[c][1][i] for i, c in enumerate(neg_c)],
        [per_class[c][3][i] for i, c in enumerate(neg_c)],
        skeletons)

    for _ in range(epochs):
        order = rng.permutation(n)
        seen_c, seen_k, seen_v = [], [], []
        for start in range(0, n, batch_size):
            rows = order[start:start + batch_size]
            Xb = X[rows]
            yb = y[rows]
            per_class, dist = _forward(Xb, skeletons)
            masked = dist.copy()
            masked[np.arange(len(rows)), yb] = np.inf
            nc = masked.argmin(axis=1)

            for b in range(len(rows)):
                cp, cn = int(yb[b]), int(nc[b])
                dp = float(per_class[cp][0][b])
                dn = float(per_class[cn][0][b])
                kp, tp, vp = (int(per_class[cp][1][b]),
                              float(per_class[cp][2][b]),
                              int(per_class[cp][3][b]))
                kn, tn, vn = (int(per_class[cn][1][b]),
                              float(per_class[cn][2][b]),
                              int(per_class[cn][3][b]))
                seen_c.append(cn)
                seen_k.append(kn)
                seen_v.append(vn)

                s = dp + dn
                if s < 1e-12:
                    continue
                mu = (dp - dn) / s
                z = expit(sigma * mu)
                phi_prime = sigma * z * (1.0 - z)

                for (cls, k, t, iv, coef, own) in (
                        (cp, kp, tp, vp, 2.0 * dn / (s * s), dp),
                        (cn, kn, tn, vn, -2.0 * dp / (s * s), dn)):
                    restricted = (
                        boundary_restriction == "both"
                        or (boundary_restriction == "negative_only"
                            and coef < 0.0))
                    V, E = skeletons[cls]
                    if k >= 0:
                        a, bb = E[k]
                        p = V[a] + t * (V[bb] - V[a])
                        wa, wb = 1.0 - t, t
                    else:
                        a = bb = iv
                        p = V[a]
                        wa, wb = 1.0, 0.0
                    r = Xb[b] - p
                    nr = float(np.sqrt(r @ r))
                    if nr < 1e-12:
                        continue
                    gd = -r / nr                  # dd/dp direction
                    step = lr * phi_prime * coef
                    if step_rule == "scale_free":
                        # Precondition by the moved object's own distance.
                        # d(d^2/2)/dv = d * dd/dv, so this is the exact
                        # gradient taken through the SQUARED distance --
                        # the same descent direction (a positive multiple)
                        # with the 1/d factor cancelled.  The surviving
                        # coefficient is 2 d+ d- / s^2, bounded by 1/2.
                        step *= own
                    elif step_rule == "clipped" and abs(step) > max_step:
                        step = float(np.sign(step)) * max_step
                    # The restriction is on VERTICES, so each endpoint is
                    # gated separately: a boundary segment with one
                    # interior endpoint moves only its boundary end.
                    if not restricted or movable[cls][a]:
                        V[a] -= step * wa * gd
                    if bb != a and (not restricted or movable[cls][bb]):
                        V[bb] -= step * wb * gd
        movable = _movable_vertices(seen_c, seen_k, seen_v, skeletons)
    return skeletons


def class_risk_parts(X, skeletons):
    """Convenience: full class-distance matrix (used by Phases 3 and 4)."""
    return np.stack([skeleton_distance(X, V, E) for (V, E) in skeletons],
                    axis=1)


__all__ = [
    "ADMISSION_CRITERIA", "BOUNDARY_MODES", "admit_edges_by_risk",
    "admit_edges_by_support", "edge_is_supported", "phase2a", "phase2b",
    "support_rho", "class_risk_parts", "NO_EDGE",
]
