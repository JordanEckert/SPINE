"""SPINE against this repository's prototype contract, on real data.

sklearn's bundled iris and wine are real measured datasets and stand in
for KEEL data in offline unit tests only; the campaign runner never
touches them.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine

from harness.base import validate_selection
from harness.config import make_scaler
from spine import SPINE


def _iris():
    # Scaled by the protocol's own scaler, not a hard-coded one: config.py
    # documents that Phase 2b misbehaves under min-max, so a fixture that
    # pinned min-max would test SPINE under conditions the campaign has
    # deliberately moved away from.
    d = load_iris()
    return make_scaler().fit_transform(d.data), d.target


def _wine():
    d = load_wine()
    return make_scaler().fit_transform(d.data), d.target


def test_contract():
    X, y = _iris()
    m = SPINE()
    Xp, yp, idx = m.select(X, y, total_count=15, random_state=0)
    validate_selection(m, X, y, np.asarray(Xp, float), np.asarray(yp),
                       np.asarray(idx))


@pytest.mark.parametrize("budget", [6, 15, 31, 60])
def test_emits_exactly_the_injected_budget(budget):
    X, y = _wine()
    Xp, yp, idx = SPINE().select(X, y, total_count=budget, random_state=1)
    assert len(Xp) == len(yp) == len(idx) == budget


def test_per_class_counts_follow_the_harness_apportionment():
    from harness.base import allocate_per_class
    X, y = _wine()
    _, yp, _ = SPINE().select(X, y, total_count=31, random_state=2)
    got = dict(zip(*np.unique(yp, return_counts=True)))
    want = allocate_per_class(y, 31)
    assert {int(k): int(v) for k, v in got.items()} == \
        {int(k): int(v) for k, v in want.items()}


def test_is_deterministic_given_the_seed():
    X, y = _iris()
    a = SPINE().select(X, y, total_count=15, random_state=7)
    b = SPINE().select(X, y, total_count=15, random_state=7)
    for u, v in zip(a, b):
        assert np.array_equal(np.asarray(u), np.asarray(v))


def test_requires_a_random_state_and_a_budget():
    X, y = _iris()
    with pytest.raises(ValueError, match="random_state"):
        SPINE().select(X, y, total_count=15)
    with pytest.raises(ValueError, match="total_count"):
        SPINE().select(X, y, random_state=0)


def test_topology_is_frozen_at_phase_2a_and_survives_to_convergence():
    X, y = _wine()
    m = SPINE()
    m.select(X, y, total_count=30, random_state=3)
    p = m.spine_params_
    if p["forced_removals"] == 0:
        assert p["betti_final"] == p["betti_frozen"], (
            "section 10 invariant broke with no forced removal")
        assert p["soundness_bound_holds"]
    assert p["invariant_held"] == (p["betti_final"] == p["betti_frozen"])
    # Section 9.2's reportable result is nerve vs post-2a, recorded as a
    # named field rather than left for a reader to derive.
    r = p["topology_retention_nerve_to_frozen"]
    assert r["beta1_frozen"] <= r["beta1_nerve"]
    assert r["beta0_frozen"] >= r["beta0_nerve"]


def test_admission_can_only_lose_topology_never_invent_it():
    """Section 10: beta_1(final) <= beta_1(nerve), beta_0(final) >= beta_0."""
    X, y = _wine()
    m = SPINE()
    m.select(X, y, total_count=30, random_state=4)
    for (b0n, b1n), (b0f, b1f) in zip(m.spine_params_["betti_nerve"],
                                      m.spine_params_["betti_final"]):
        assert b1f <= b1n
        assert b0f >= b0n


def test_use_edges_false_keeps_the_vertices_and_drops_the_segments():
    """Rung 1 of the ladder: same fit, segments removed at prediction."""
    X, y = _iris()
    full = SPINE(use_edges=True)
    bare = SPINE(use_edges=False)
    Xa, ya, _ = full.select(X, y, total_count=15, random_state=5)
    Xb, yb, _ = bare.select(X, y, total_count=15, random_state=5)
    assert np.array_equal(Xa, Xb)
    assert np.array_equal(ya, yb)
    assert bare.estimator_.n_segments == 0
    assert full.estimator_.n_segments >= 0


def test_estimator_predicts_labels_not_positions():
    X, y = _wine()
    m = SPINE()
    m.select(X, y, total_count=21, random_state=6)
    pred = m.estimator_.predict(X)
    assert set(np.unique(pred)).issubset(set(np.unique(y)))
    assert pred.dtype == y.dtype or set(np.unique(pred)) <= set(np.unique(y))


def test_margin_is_bounded_and_non_positive_for_the_winner():
    X, y = _iris()
    m = SPINE()
    m.select(X, y, total_count=15, random_state=8)
    mu = m.estimator_.decision_margin(X)
    assert np.all(mu >= -1.0) and np.all(mu <= 0.0)


def test_exemplars_index_the_training_fold():
    X, y = _iris()
    m = SPINE()
    Xp, yp, _ = m.select(X, y, total_count=15, random_state=9)
    assert len(m.exemplars_) == len(Xp)
    assert m.exemplars_.min() >= 0 and m.exemplars_.max() < len(X)
    # An exemplar carries its vertex's class BECAUSE the search is
    # restricted to that class's own training points.  An unrestricted
    # nearest-point search does not have this property on overlapping
    # data (measured: 3/40 wrong-class vertices on KEEL segment, 16-25/40
    # on banana), and section 9.3's contrastive explanation needs it.
    assert np.array_equal(y[m.exemplars_], yp)


def test_single_class_fold_is_refused():
    X, _ = _iris()
    with pytest.raises(ValueError, match="single-class"):
        SPINE().select(X, np.zeros(len(X), dtype=int),
                       total_count=5, random_state=0)
