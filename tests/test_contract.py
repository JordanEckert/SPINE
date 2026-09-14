"""Output-contract tests for every method, on real (bundled) data.

sklearn's bundled iris and wine are real measured datasets; they stand
in for KEEL data in offline unit tests only -- the experiment scripts
themselves never touch them.  They are also small, which matters for
SPINE: its construction is the most expensive in the roster, so the
contract is pinned on data a test suite can afford to run repeatedly.

The scaler is the protocol's own (``config.make_scaler``) rather than a
hard-coded one, so these tests exercise the preprocessing the campaign
actually applies.  That is not cosmetic for SPINE: its Phase 2b is a
GLVQ-type gradient update, whose behaviour depends on the scaling (see
the SCALING note in config.py).
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine

from baselines import (GLVQ, GNG, LVQ3, RSP3, FullSet, KMeansPerClass,
                       RandomSubsample, SPOTGreedy)
from harness.base import validate_selection
from harness.config import make_scaler
from spine import SPINE


def _data():
    d = load_iris()
    X = make_scaler().fit_transform(d.data)
    return X, d.target


METHODS = [
    ("Full", FullSet(), {}),
    ("Random", RandomSubsample(), {"random_state": 0}),
    ("KMeans", KMeansPerClass(), {"random_state": 0}),
    ("RSP3", RSP3(), {}),
    ("LVQ3", LVQ3(), {"random_state": 0}),
    ("SPOT", SPOTGreedy(), {"total_count": 15}),
    # The method under test and the two new budget-matched comparators.
    # All three are generators and all three are stochastic, so each
    # needs both an injected budget and an injected seed -- the same two
    # things the runner hands them per fold.
    ("SPINE", SPINE(), {"total_count": 15, "random_state": 0}),
    ("GLVQ", GLVQ(), {"total_count": 15, "random_state": 0}),
    ("GNG", GNG(), {"total_count": 15, "random_state": 0}),
]

#: The budget-matched roster: every one of these is handed a total by the
#: runner and must emit exactly that total, whatever it is.
BUDGET_MATCHED = [
    ("SPINE", SPINE()),
    ("GLVQ", GLVQ()),
    ("GNG", GNG()),
    ("SPOT", SPOTGreedy()),
]


@pytest.mark.parametrize("method,params",
                         [(m, p) for _, m, p in METHODS],
                         ids=[i for i, _, _ in METHODS])
def test_contract(method, params):
    X, y = _data()
    X_proto, y_proto, indices = method.select(X, y, **params)
    X_proto = np.asarray(X_proto, dtype=float)
    validate_selection(method, X, y, X_proto, np.asarray(y_proto),
                       np.asarray(indices))


@pytest.mark.parametrize("method,params",
                         [(RSP3(), {}),
                          (SPOTGreedy(), {"total_count": 15})],
                         ids=["RSP3", "SPOT"])
def test_deterministic_repeat(method, params):
    # The parameter-free, deterministic methods: no seed enters, so a
    # repeat must reproduce bit-for-bit.  SPINE, GLVQ and GNG are
    # stochastic and declare it (``deterministic = False``); their
    # reproducibility is a property of the SEED and is pinned separately.
    X, y = _data()
    a = method.select(X, y, **params)
    b = method.select(X, y, **params)
    for u, v in zip(a, b):
        assert np.array_equal(np.asarray(u), np.asarray(v))


@pytest.mark.parametrize("name,method",
                         [(n, m) for n, m in
                          (("SPINE", SPINE()), ("GLVQ", GLVQ()),
                           ("GNG", GNG()))],
                         ids=["SPINE", "GLVQ", "GNG"])
def test_seeded_methods_reproduce_under_the_same_seed(name, method):
    # The campaign derives one seed per (dataset, fold) and states that a
    # rerun of a single fold reproduces it bit-for-bit.  That claim is
    # only true if the stochastic methods are functions of their injected
    # seed alone, which is what this pins -- for SPINE above all, whose
    # RNG threads through the validation split, Phase 1's annealing and
    # Phase 2b's batching.
    X, y = _data()
    a = method.select(X, y, total_count=15, random_state=7)
    b = method.select(X, y, total_count=15, random_state=7)
    for u, v in zip(a, b):
        assert np.array_equal(np.asarray(u), np.asarray(v)), name


@pytest.mark.parametrize("name,method", BUDGET_MATCHED,
                         ids=[n for n, _ in BUDGET_MATCHED])
def test_budget_matched_methods_emit_exactly_the_injected_budget(name,
                                                                 method):
    # These methods are budget-matched, not budget-choosing: whatever
    # total the runner injects is what they emit, on any dataset.  The
    # smallest budget tested is 3 on a 3-class dataset, i.e. one
    # prototype per class -- the floor the per-class apportionment
    # guarantees, and the point at which a method that quietly rounds its
    # own count would give itself away.
    d = load_wine()
    X = make_scaler().fit_transform(d.data)
    for budget in (3, 20, 47):
        params = {"total_count": budget}
        if not method.deterministic:
            params["random_state"] = 0
        Xp, yp, idx = method.select(X, d.target, **params)
        assert len(Xp) == len(yp) == len(idx) == budget, (name, budget)


def test_spine_exposes_the_skeleton_it_fitted():
    # SPINE's two campaign rows come from ONE fit: the vertices are
    # scored like any prototype set, and ``estimator_`` is the same
    # fitted model under its own decision rule.  The runner and the
    # budget-curve script both depend on that attribute existing after
    # ``select``, so the contract is pinned here rather than only
    # implicitly through those scripts.
    X, y = _data()
    spine = SPINE()
    Xp, _yp, _idx = spine.select(X, y, total_count=15, random_state=0)
    pred = spine.estimator_.predict(X)
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)) <= set(np.unique(y))
    assert spine.estimator_.n_vertices == len(Xp)
    # and the construction provenance the campaign records per fold
    assert isinstance(spine.spine_params_, dict)
