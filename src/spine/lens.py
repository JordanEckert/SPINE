"""Phase 0 lenses (section 4.1).

The lens is chosen for what it DESCRIBES, not for what it predicts:
a Reeb graph is a property of a space together with a function on it, so
picking the lens is picking which Reeb graph the construction intends to
recover.  Supervision therefore stays out of the lens and enters only
downstream, in Phases 2a onward.  

Internal standardisation before the lens
----------------------------------------
The harness scales every feature with a scaler fit on the training
fold -- standardisation by default, min-max or median/IQR on request.
Whichever is chosen, it is a per-feature affine map fixed by GLOBAL
statistics, and within a class it is arbitrary: under min-max, a feature
whose global range is mostly between-class separation is compressed
relative to one whose range is within-class, so PC1 can end up pointing
along whichever feature the global range happened to treat most kindly.
Section 4.1 wants PC1 to be "the direction of maximal variation within
that class", and under any global rescaling it is instead the direction
of maximal variation in units the class did not choose.

``standardize`` therefore z-scores the class's own points before the lens
is computed.  

Features with zero within-class variance are left at zero rather than
divided by zero -- they carry no within-class information and cannot
contribute to a direction of maximal within-class variation.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.spatial.distance import cdist

LENSES = ("pc1", "eccentricity", "fiedler")
LENS_SCALINGS = ("standardize", "none")


def standardize(Xc):
    """Z-score a class's points, leaving zero-variance columns at zero."""
    Xc = np.asarray(Xc, dtype=float)
    if len(Xc) < 2:
        return Xc - Xc.mean(axis=0) if len(Xc) else Xc
    Z = Xc - Xc.mean(axis=0)
    sd = Z.std(axis=0)
    good = sd > 0
    out = np.zeros_like(Z)
    out[:, good] = Z[:, good] / sd[good]
    return out


def _fix_sign(f):
    """Deterministic sign convention: the entry of largest magnitude is +.

    Ties (exact magnitude equality) resolve to the earliest such entry,
    so the rule depends on the data alone and not on solver internals.
    """
    f = np.asarray(f, dtype=float)
    if f.size == 0:
        return f
    k = int(np.argmax(np.abs(f)))
    return -f if f[k] < 0 else f


def lens_pc1(Xc):
    """Within-class PC1: projection onto the top right singular vector."""
    Xc = np.asarray(Xc, dtype=float)
    Z = Xc - Xc.mean(axis=0)
    if len(Z) < 2:
        return np.zeros(len(Z))
    # full_matrices=False gives Vt with min(n, d) rows; row 0 is PC1.
    Vt = np.linalg.svd(Z, full_matrices=False)[2]
    return _fix_sign(Z @ Vt[0])


def lens_eccentricity(Xc):
    """Mean distance from each point to the rest of its own class.

    The point's own zero distance is excluded, so the value is a genuine
    mean over the remaining ``n_c - 1`` points rather than one shrunk by
    a spurious zero.
    """
    Xc = np.asarray(Xc, dtype=float)
    n = len(Xc)
    if n < 2:
        return np.zeros(n)
    D = cdist(Xc, Xc)
    return _fix_sign(D.sum(axis=1) / float(n - 1))


def lens_fiedler(Xc, n_neighbors=10):
    """Fiedler vector of the class's symmetrised kNN-graph Laplacian.

    ``n_neighbors`` is the one genuine hyperparameter among the three
    lenses; section 4.1 accepts it because a spurious edge perturbs one
    entry of a matrix whose relevant eigenvector is a global average,
    which is local and attenuated -- unlike a geodesic construction,
    where one short-circuit corrupts every pairwise distance at once.

    The graph is UNWEIGHTED, which is what section 4.1 specifies ("the
    graph Laplacian on the class's kNN graph") and which keeps
    ``n_neighbors`` the single hyperparameter the section sanctions.  An
    edge-weighting kernel would add a bandwidth rule that section 4.1
    does not license, and it would make the connectivity repair below
    non-neutral: repair edges would have to be given some weight, and any
    weight at the top of the distribution places an artificially strong
    link exactly where the Fiedler vector looks for the minimal-conductance
    cut.  Unweighted, a repair edge is indistinguishable from a genuine
    one and biases nothing.

    A disconnected kNN graph has a degenerate Fiedler vector (the second
    eigenvalue is also 0 and the eigenvector is an arbitrary indicator of
    components), so the components are joined first, each through its
    single closest pair of points.  That is a deterministic repair, and
    it is reported rather than silent: the second return value is the
    number of components before repair, which ``phase0`` records.
    """
    Xc = np.asarray(Xc, dtype=float)
    n = len(Xc)
    if n < 3:
        return np.zeros(n), 1
    k = int(min(max(1, n_neighbors), n - 1))

    D = cdist(Xc, Xc)
    # Mask the diagonal rather than dropping the first sorted column: with
    # duplicate points a coincident row of lower index sorts ahead of the
    # point itself, and slicing [1:] would then make the point its own
    # neighbour and lose a genuine one.
    masked = D.copy()
    np.fill_diagonal(masked, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = order.reshape(-1)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)                            # symmetrise

    n_comp, labels = connected_components(A, directed=False)
    if n_comp > 1:
        # Deterministic repair: absorb each remaining component into the
        # growing merged set through its single closest pair of points.
        A = A.tolil()
        merged = labels == labels[0]
        for _ in range(n_comp - 1):
            left = np.where(merged)[0]
            right = np.where(~merged)[0]
            sub = D[np.ix_(left, right)]
            i, j = np.unravel_index(int(np.argmin(sub)), sub.shape)
            a, b = int(left[i]), int(right[j])
            A[a, b] = A[b, a] = 1.0
            merged |= labels == labels[b]
        A = A.tocsr()

    L = laplacian(A, normed=True).toarray()
    L = 0.5 * (L + L.T)                           # exact symmetry for eigh
    vals, vecs = np.linalg.eigh(L)
    return _fix_sign(vecs[:, 1]), int(n_comp)


def apply_lens(Xc, kind="pc1", n_neighbors=10, scaling="standardize"):
    """Dispatch to the requested lens.

    ``scaling`` selects the coordinates the lens is computed in; see the
    module docstring.  It affects the lens ONLY -- the returned values are
    filter values, and every position downstream is computed from the
    caller's own coordinates.

    Returns ``(f, info)``.  ``info`` carries anything the lens had to
    decide that the caller should be able to see afterwards: the scaling
    used, and for Fiedler the number of connected components its kNN
    graph had before repair.
    """
    if scaling not in LENS_SCALINGS:
        raise ValueError(
            "unknown lens scaling {0!r}; expected one of {1}"
            .format(scaling, LENS_SCALINGS))
    Z = standardize(Xc) if scaling == "standardize" else np.asarray(
        Xc, dtype=float)
    info = {"lens_scaling": scaling}
    if kind == "pc1":
        return lens_pc1(Z), info
    if kind == "eccentricity":
        return lens_eccentricity(Z), info
    if kind == "fiedler":
        f, n_comp = lens_fiedler(Z, n_neighbors=n_neighbors)
        info["knn_components"] = int(n_comp)
        return f, info
    raise ValueError(
        "unknown lens {0!r}; expected one of {1}".format(kind, LENSES))
