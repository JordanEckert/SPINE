"""Betti numbers, the removal guard, and the section-10 invariant.

The two cases that matter most are the ones Appendix B's ``prune_one``
gets wrong: removing an isolated vertex (which lowers ``beta_0``) and
un-subdividing a degree-2 vertex whose neighbours are already adjacent
(which lowers ``beta_1``).  Both are asserted here directly, because a
guard that is never exercised is a guard that is not there.
"""

import numpy as np
import pytest

from spine.topology import (betti, degrees, is_homotopy_preserving_removal,
                            normalize_edges, remove_vertex)


def test_betti_of_a_path_tree_and_cycle():
    assert betti(4, [(0, 1), (1, 2), (2, 3)]) == (1, 0)      # path
    assert betti(4, [(0, 1), (0, 2), (0, 3)]) == (1, 0)      # star
    assert betti(4, [(0, 1), (1, 2), (2, 3), (0, 3)]) == (1, 1)   # 4-cycle
    assert betti(3, [(0, 1), (1, 2), (0, 2)]) == (1, 1)      # triangle


def test_betti_counts_isolated_vertices_as_components():
    assert betti(5, [(0, 1)]) == (4, 0)
    assert betti(3, []) == (3, 0)


def test_theta_graph_has_two_independent_cycles():
    # Two vertices joined by three internally disjoint paths.
    edges = [(0, 1), (1, 2), (0, 3), (3, 2), (0, 4), (4, 2)]
    assert betti(5, edges) == (1, 2)


def test_normalize_edges_is_canonical():
    assert normalize_edges([(3, 1), (1, 3), (2, 2), (0, 1)]) == \
        [(1, 3), (0, 1)]


def test_degrees_match_the_edge_list():
    assert list(degrees(4, [(0, 1), (1, 2), (1, 3)])) == [1, 3, 1, 1]


def test_isolated_vertex_removal_is_refused():
    # Removing vertex 2 would drop beta_0 from 2 to 1.
    assert betti(3, [(0, 1)]) == (2, 0)
    assert not is_homotopy_preserving_removal(3, [(0, 1)], 2)


def test_degree_two_with_adjacent_neighbours_is_refused():
    # Triangle: un-subdividing vertex 2 would need edge (0,1), which
    # exists, so beta_1 would fall from 1 to 0.
    tri = [(0, 1), (1, 2), (0, 2)]
    assert betti(3, tri) == (1, 1)
    assert not is_homotopy_preserving_removal(3, tri, 2)
    V = np.zeros((3, 2))
    Vn, En = remove_vertex(V, tri, 2)
    assert betti(len(Vn), En) == (1, 0)          # the damage, if forced


def test_degree_two_with_distinct_non_adjacent_neighbours_is_allowed():
    path = [(0, 1), (1, 2)]
    assert is_homotopy_preserving_removal(3, path, 1)
    V = np.array([[0.0], [1.0], [2.0]])
    Vn, En = remove_vertex(V, path, 1)
    assert betti(len(Vn), En) == betti(3, path) == (1, 0)
    assert En == [(0, 1)]


def test_leaf_removal_preserves_both_betti_numbers():
    star = [(0, 1), (0, 2), (0, 3)]
    assert is_homotopy_preserving_removal(4, star, 3)
    V = np.zeros((4, 2))
    Vn, En = remove_vertex(V, star, 3)
    assert betti(len(Vn), En) == betti(4, star)


def test_degree_three_removal_is_refused():
    star = [(0, 1), (0, 2), (0, 3)]
    assert not is_homotopy_preserving_removal(4, star, 0)


@pytest.mark.parametrize("seed", range(20))
def test_allowed_removals_never_change_betti(seed):
    """The guard is sound: every removal it permits preserves homotopy."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 12))
    pairs = set()
    for _ in range(int(rng.integers(2, 3 * n))):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            pairs.add((int(min(i, j)), int(max(i, j))))
    E = normalize_edges(pairs)
    before = betti(n, E)
    V = rng.normal(size=(n, 2))
    for v in range(n):
        if is_homotopy_preserving_removal(n, E, v):
            Vn, En = remove_vertex(V, E, v)
            assert betti(len(Vn), En) == before, (
                "removal of {0} changed {1} -> {2}".format(
                    v, before, betti(len(Vn), En)))
