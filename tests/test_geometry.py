"""Geometry primitives against a brute-force reference.

The vectorised segment distance is the load-bearing routine of the whole
method -- prediction, admission, fitting, growth and pruning all read
through it -- so it is checked against a transparently correct
implementation of the section-1 formula rather than against itself.
"""

import numpy as np
import pytest

from spine.geometry import (NO_EDGE, predict, segment_distances,
                            skeleton_argmin, skeleton_distance)


def brute_segment(x, u, v):
    """The section-1 primitive, written out literally."""
    w = v - u
    ww = float(w @ w)
    t = 0.0 if ww < 1e-12 else float(np.clip((x - u) @ w / ww, 0.0, 1.0))
    p = u + t * w
    return float(np.linalg.norm(x - p)), t


def random_case(rng, n=40, m=9, d=4, n_edges=12):
    X = rng.normal(size=(n, d))
    V = rng.normal(size=(m, d))
    pairs = set()
    while len(pairs) < n_edges:
        i, j = sorted(rng.choice(m, size=2, replace=False))
        pairs.add((int(i), int(j)))
    return X, V, sorted(pairs)


@pytest.mark.parametrize("seed", range(8))
def test_segment_distances_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    X, V, E = random_case(rng)
    D, T = segment_distances(X, V, E)
    for a in range(len(X)):
        for k, (i, j) in enumerate(E):
            d_ref, t_ref = brute_segment(X[a], V[i], V[j])
            assert D[a, k] == pytest.approx(d_ref, abs=1e-10)
            assert T[a, k] == pytest.approx(t_ref, abs=1e-10)


def test_degenerate_edge_is_the_endpoint_distance():
    V = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    X = np.array([[3.0, 4.0]])
    D, T = segment_distances(X, V, [(0, 1), (0, 2)])
    assert D[0, 0] == pytest.approx(5.0)
    assert T[0, 0] == 0.0


def test_skeleton_distance_is_min_over_vertices_and_segments():
    rng = np.random.default_rng(3)
    X, V, E = random_case(rng)
    D, _ = segment_distances(X, V, E)
    dv = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)
    expected = np.minimum(dv.min(axis=1), D.min(axis=1))
    assert np.allclose(skeleton_distance(X, V, E), expected, atol=1e-10)


def test_argmin_reports_a_consistent_winner():
    rng = np.random.default_rng(11)
    X, V, E = random_case(rng)
    best, we, wt, wv = skeleton_argmin(X, V, E)
    assert np.allclose(best, skeleton_distance(X, V, E), atol=1e-10)
    for a in range(len(X)):
        if we[a] == NO_EDGE:
            assert best[a] == pytest.approx(
                float(np.linalg.norm(X[a] - V[wv[a]])), abs=1e-10)
        else:
            i, j = E[we[a]]
            d_ref, t_ref = brute_segment(X[a], V[i], V[j])
            assert best[a] == pytest.approx(d_ref, abs=1e-10)
            assert wt[a] == pytest.approx(t_ref, abs=1e-10)


def test_vertex_wins_ties_against_a_segment():
    # The point projects exactly onto vertex 0, so the vertex distance and
    # the segment distance are equal; the convention keeps the vertex.
    V = np.array([[0.0, 0.0], [1.0, 0.0]])
    X = np.array([[0.0, 1.0]])
    best, we, _, wv = skeleton_argmin(X, V, [(0, 1)])
    assert best[0] == pytest.approx(1.0)
    assert we[0] == NO_EDGE
    assert wv[0] == 0


def test_empty_edge_list_reduces_to_nearest_vertex():
    rng = np.random.default_rng(5)
    X, V, _ = random_case(rng)
    dv = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)
    assert np.allclose(skeleton_distance(X, V, []), dv.min(axis=1))


def test_predict_breaks_ties_toward_the_lower_class_index():
    a = (np.array([[0.0]]), [])
    b = (np.array([[2.0]]), [])
    X = np.array([[1.0]])                     # equidistant from both
    assert predict(X, [a, b])[0] == 0
    assert predict(X, [b, a])[0] == 0
