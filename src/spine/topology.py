"""Betti numbers and the homotopy bookkeeping

A skeleton is a graph -- a 1-dimensional complex -- so its homotopy type
is determined by two integers:

* ``beta_0`` = the number of connected components (isolated vertices
  included), and
* ``beta_1`` = the cyclomatic number ``|E| - |V| + beta_0``.

Section 10's invariant is that every operation after Phase 2a leaves
both unchanged, and this module supplies the primitives the phases use
to keep that true and the check Phase 5 uses to prove it:

* :func:`betti` -- the measurement;
* :func:`degrees` -- vertex degrees, which decide what Phase 4 may remove;
* :func:`is_homotopy_preserving_removal` -- the guard Appendix B omits.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def normalize_edges(edges):
    """Canonical edge list: sorted pairs, de-duplicated, no self-loops.

    The skeleton is a SIMPLE graph.  Phases 3 and 4 create edges by
    subdivision and pendant attachment, which can emit an unsorted pair,
    and canonicalising here means ``(i, j)`` and ``(j, i)`` can never be
    counted as two edges by ``betti`` -- which would corrupt ``beta_1``.
    """
    seen = []
    have = set()
    for (a, b) in edges:
        a, b = int(a), int(b)
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in have:
            continue
        have.add(key)
        seen.append(key)
    return seen


def _is_normalized(edges):
    """Cheap check that an edge list is already canonical."""
    have = set()
    for e in edges:
        if type(e) is not tuple or len(e) != 2:
            return False
        a, b = e
        if type(a) is not int or type(b) is not int or a >= b or e in have:
            return False
        have.add(e)
    return True


def as_normalized(edges):
    """``normalize_edges`` with a fast path for already-canonical input.

    The phase code canonicalises at every boundary, so the vast majority
    of calls are re-normalising a list that is already canonical; Phase 4
    alone made tens of thousands of them per fold.
    """
    return edges if _is_normalized(edges) else normalize_edges(edges)


def betti(n_vertices, edges):
    """Return ``(beta_0, beta_1)`` of the graph ``(n_vertices, edges)``."""
    n = int(n_vertices)
    if n <= 0:
        raise ValueError("a skeleton must have at least one vertex")
    edges = as_normalized(edges)
    if not edges:
        return n, 0
    arr = np.asarray(edges, dtype=np.intp)
    data = np.ones(len(arr), dtype=np.int8)
    adj = coo_matrix((data, (arr[:, 0], arr[:, 1])), shape=(n, n))
    beta_0 = int(connected_components(adj, directed=False)[0])
    beta_1 = int(len(arr) - n + beta_0)
    return beta_0, beta_1


def betti_per_class(skeletons):
    """``[(beta_0, beta_1), ...]`` for a list of ``(V, E)`` skeletons."""
    return [betti(len(V), E) for (V, E) in skeletons]


def degrees(n_vertices, edges):
    """Degree of every vertex (simple graph, so each edge adds 1 to both)."""
    deg = np.zeros(int(n_vertices), dtype=int)
    for (a, b) in as_normalized(edges):
        deg[a] += 1
        deg[b] += 1
    return deg


def neighbours(n_vertices, edges):
    """Adjacency as a list of sets."""
    nb = [set() for _ in range(int(n_vertices))]
    for (a, b) in as_normalized(edges):
        nb[a].add(b)
        nb[b].add(a)
    return nb


def is_homotopy_preserving_removal(n_vertices, edges, v, nb=None):
    """May vertex ``v`` be removed without changing ``(beta_0, beta_1)``?

    Exactly three cases preserve homotopy type, and they are precisely
    the inverses of the Phase 3 grammar:

    * degree 1 -- un-extend a leaf: ``|V|`` and ``|E|`` both fall by one,
      the component survives through the neighbour, both Betti numbers
      hold;
    * degree 2 with two DISTINCT, NON-ADJACENT neighbours -- un-subdivide:
      two edges out, one in, so ``|E|`` and ``|V|`` both fall by one;
    * nothing else.

    Degree 0 lowers ``beta_0``; degree 2 with already-adjacent neighbours
    lowers ``beta_1``; degree >= 3 disconnects or worse.

    ``nb`` may supply a precomputed adjacency (as from :func:`neighbours`)
    when the caller is testing many vertices of the same graph.
    """
    if nb is None:
        nb = neighbours(n_vertices, edges)
    nv = nb[v]
    if len(nv) == 1:
        return True
    if len(nv) == 2:
        a, b = tuple(nv)
        return b not in nb[a]
    return False


def remove_vertex(V, edges, v):
    """Delete vertex ``v``, reconnecting its neighbours if it had degree 2.

    Returns ``(V_new, edges_new)`` with vertex indices renumbered.  The
    caller is responsible for checking
    :func:`is_homotopy_preserving_removal` first if the invariant is to
    hold; this routine performs the deletion it is asked for.
    """
    V = np.asarray(V, dtype=float)
    n = len(V)
    edges = as_normalized(edges)
    nb = sorted(neighbours(n, edges)[v])
    keep = [(a, b) for (a, b) in edges if v != a and v != b]
    if len(nb) == 2:
        keep.append((nb[0], nb[1]))          # un-subdivide
    remap = {old: new for new, old in
             enumerate(j for j in range(n) if j != v)}
    new_edges = normalize_edges(
        [(remap[a], remap[b]) for (a, b) in keep])
    return np.delete(V, v, axis=0), new_edges
