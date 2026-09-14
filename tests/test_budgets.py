"""Budget-matching rules: per-class counts, total allocation, SPOT."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from baselines import LVQ3, KMeansPerClass, RandomSubsample, SPOTGreedy
from harness.base import allocate_per_class


def _data():
    d = load_iris()
    return MinMaxScaler().fit_transform(d.data), d.target


def test_allocate_exact_total_and_floor():
    y = np.array([0] * 60 + [1] * 30 + [2] * 10)
    counts = allocate_per_class(y, 10)
    assert sum(counts.values()) == 10
    assert all(v >= 1 for v in counts.values())
    # proportionality: class 0 gets the most
    assert counts[0] > counts[1] > 0


def test_allocate_caps_at_class_size():
    y = np.array([0] * 3 + [1] * 97)
    counts = allocate_per_class(y, 50)
    assert counts[0] <= 3
    assert sum(counts.values()) == 50


def test_allocate_floor_wins_below_n_classes():
    y = np.array([0] * 5 + [1] * 5 + [2] * 5)
    counts = allocate_per_class(y, 2)  # < n_classes
    assert all(v == 1 for v in counts.values())  # documented floor


def test_random_and_kmeans_honor_class_counts():
    X, y = _data()
    cc = {0: 4, 1: 7, 2: 2}
    for method in (RandomSubsample(), KMeansPerClass()):
        Xp, yp, _ = method.select(X, y, class_counts=cc, random_state=1)
        got = {c: int(k) for c, k in
               zip(*np.unique(yp, return_counts=True))}
        assert got == cc, method.name


def test_class_counts_missing_class_is_hard_error():
    X, y = _data()
    with pytest.raises(KeyError):
        RandomSubsample().select(X, y, class_counts={0: 1},
                                 random_state=0)


def test_lvq3_budget_matched():
    X, y = _data()
    # LVQ3 is budget-matched: given the anchor's total count it returns
    # exactly that many prototypes, apportioned across classes.
    Xp, yp, _ = LVQ3().select(X, y, total_count=12, random_state=3)
    assert len(Xp) == 12
    assert set(np.unique(yp)) == {0, 1, 2}  # >=1 per class survives


def test_spot_budget_and_selection_semantics():
    X, y = _data()
    Xp, yp, idx = SPOTGreedy().select(X, y, total_count=9)
    assert len(Xp) == 9
    assert np.array_equal(Xp, X[idx]) and np.array_equal(yp, y[idx])


def test_spot_requires_budget():
    X, y = _data()
    with pytest.raises(ValueError):
        SPOTGreedy().select(X, y)
