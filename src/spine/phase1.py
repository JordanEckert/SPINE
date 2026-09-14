"""Phase 1 -- annealed representation """

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from .topology import normalize_edges


def hop_distances(n_vertices, edges):
    """All-pairs hop distance; disconnected pairs get ``n_vertices``.

    A finite stand-in for infinity is required because the update weight
    is ``exp(-d/sigma)``: with ``d = inf`` the weight is exactly 0, which
    is the intended behaviour, but a finite ceiling keeps the arithmetic
    free of ``inf`` and makes the weight for a far-but-connected vertex
    and an unreachable one differ continuously.  ``n_vertices`` exceeds
    any realisable hop distance in a simple graph on that many vertices.
    """
    n = int(n_vertices)
    edges = normalize_edges(edges)
    if n == 1:
        return np.zeros((1, 1))
    if not edges:
        H = np.full((n, n), float(n))
        np.fill_diagonal(H, 0.0)
        return H
    arr = np.asarray(edges, dtype=np.intp)
    data = np.ones(len(arr))
    adj = coo_matrix((data, (arr[:, 0], arr[:, 1])), shape=(n, n))
    H = shortest_path(adj, method="D", directed=False, unweighted=True)
    H[~np.isfinite(H)] = float(n)
    return H


def phase1(Xc, V, edges, T=30, sigma_0=2.0, sigma_f=0.05,
           eps_0=0.4, eps_f=0.02, rng=None):
    """Anneal ``V`` onto ``Xc`` using the skeleton as the lattice."""
    Xc = np.asarray(Xc, dtype=float)
    V = np.array(V, dtype=float, copy=True)
    m = len(V)
    if m == 0:
        raise ValueError("phase1 received an empty skeleton")
    if rng is None:
        rng = np.random.default_rng(0)
    if m == 1:
        # One vertex: the winner is always itself and the annealed update
        # is plain competitive learning, whose fixed point is the mean.
        return Xc.mean(axis=0)[None, :]
    if len(Xc) == 0:
        return V

    H = hop_distances(m, edges)
    denom = float(max(1, T - 1))
    for epoch in range(T):
        frac = epoch / denom
        sigma = sigma_0 * (sigma_f / sigma_0) ** frac
        eps = eps_0 * (eps_f / eps_0) ** frac
        weights = np.exp(-H / sigma)              # (m, m), row = winner
        for x in Xc[rng.permutation(len(Xc))]:
            diff = x - V                          # (m, d)
            b = int(np.argmin(np.einsum("md,md->m", diff, diff)))
            V += (eps * weights[b])[:, None] * diff
    return V
