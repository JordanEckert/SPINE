"""End-to-end fold pipeline on real bundled data (offline smoke)."""

import json

import numpy as np
import pytest
from sklearn.datasets import load_iris

from baselines.lvq3 import LVQ3
from harness.base import allocate_per_class
from harness.config import BUDGET_ANCHOR, make_scaler
from harness.runner import (SPINE_GRAPH, SPINE_POINTS, fold_seed,
                            run_fold, tune_by_grid)

SMALL_LVQ3_GRID = {"window": [0.2, 0.3], "epsilon": [0.1, 0.3]}

#: Every method that is held to the anchor's total, by the names the
#: runner writes.  SPINE appears twice because one fit produces two rows.
MATCHED_METHODS = (SPINE_POINTS, SPINE_GRAPH, "KMeans", "Random", "LVQ3",
                   "SPOT", "GLVQ", "GNG")

#: The only method whose hyperparameters the campaign selects on the
#: data.  SPINE, GLVQ and GNG are all run at declared settings, so a
#: nonzero tuning bill anywhere else would mean an undeclared search.
TUNED_METHODS = ("LVQ3",)


def _split():
    d = load_iris()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(d.data))
    tr, te = idx[:120], idx[120:]
    return d.data[tr], d.target[tr], d.data[te], d.target[te]


def test_fold_seed_deterministic_and_distinct():
    assert fold_seed("wine", 0) == fold_seed("wine", 0)
    assert fold_seed("wine", 0) != fold_seed("wine", 1)
    assert fold_seed("wine", 0) != fold_seed("magic", 0)


def test_tune_by_grid_returns_grid_member():
    Xtr, ytr, _, _ = _split()
    best, scores, cpu = tune_by_grid(
        lambda combo: LVQ3(**combo), SMALL_LVQ3_GRID, Xtr, ytr, seed=1)
    assert set(best) == {"window", "epsilon"}
    assert best["window"] in (0.2, 0.3)
    assert len(scores) == 4 and cpu >= 0.0


def test_run_fold_full_protocol_rows():
    Xtr, ytr, Xte, yte = _split()
    rows = run_fold("iris-test", Xtr, ytr, Xte, yte, fold_id=0,
                    classifiers=("1nn",))
    by_method = {r["method"]: r for r in rows}
    assert set(by_method) == {"Full", "RSP3", SPINE_POINTS, SPINE_GRAPH,
                              "KMeans", "Random", "LVQ3", "SPOT",
                              "GLVQ", "GNG"}
    # one row per method name: SPINE's two rows are two names, not a
    # duplicated one, so nothing downstream has to de-duplicate.
    assert len(rows) == len(by_method)

    # RSP3 is the budget anchor: every budget-matched method emits its
    # count exactly, so the whole matched block shares one reduction rate.
    budget = by_method["RSP3"]["n_proto"]
    for m in MATCHED_METHODS:
        assert by_method[m]["n_proto"] == budget, m
        assert by_method[m]["reduction"] == by_method["RSP3"]["reduction"], m

    assert by_method["Full"]["n_proto"] == len(Xtr)
    assert by_method["Full"]["reduction"] == 0.0
    # the outer fold count is recorded on every row (10-fold here)
    assert all(r["n_outer_folds"] == 10 for r in rows)
    for r in rows:
        assert 0.0 <= r["acc_1nn"] <= 1.0
        assert r["cpu_select"] >= 0.0
    # LVQ3 is the one tuned method left, and it records its inner-CV
    # search cost separately from its construction cost.
    lvq3 = by_method["LVQ3"]
    assert lvq3["cpu_tuning"] > 0.0
    assert "lvq3_params" in lvq3
    # Everything else is untuned by construction -- the anchor, and the
    # three methods run at declared settings.
    for m in ("RSP3", SPINE_POINTS, SPINE_GRAPH, "GLVQ", "GNG"):
        assert by_method[m]["cpu_tuning"] == 0.0, m


def test_spine_emits_two_rows_from_one_fit():
    # The SPINE+1NN / SPINE+Graph pair exists to isolate the
    # hypothesis-class change: the two rows must be the SAME fitted model
    # scored two ways, never two fits.  Two fits would differ by the
    # RNG's consumption as well as by the decision rule, and the paired
    # difference would stop meaning what the ablation says it means.
    # One fit is observable from the outside as: identical prototype
    # counts, identical construction cost, identical provenance, and an
    # accuracy column that is generally NOT identical because only the
    # decision rule changed.
    Xtr, ytr, Xte, yte = _split()
    rows = run_fold("iris-test", Xtr, ytr, Xte, yte, fold_id=0,
                    classifiers=("1nn",))
    by_method = {r["method"]: r for r in rows}
    points, graph = by_method[SPINE_POINTS], by_method[SPINE_GRAPH]

    assert points["n_proto"] == graph["n_proto"]
    assert points["cpu_select"] == graph["cpu_select"]
    assert points["reduction"] == graph["reduction"]
    # the construction provenance is written on both rows, identically
    assert points["spine_params"] == graph["spine_params"]
    prov = json.loads(points["spine_params"])
    assert "betti_nerve" in prov and "betti_frozen" in prov
    # the graph row carries exactly one accuracy column: the skeleton
    # rule is not a classifier fitted to a prototype set, so the extra
    # classifiers (when requested) do not apply to it
    assert {k for k in graph if k.startswith("acc_")} == {"acc_1nn"}


def test_per_class_budgets_are_proportional_not_a_method_s():
    # The per-class split must be an apportionment of the TRAINING class
    # frequencies.  Deriving it from a method in the table (the anchor's
    # prototype labels being the obvious shortcut) would let that method
    # hand its own class allocation to its comparators; this pins the
    # actual vector, not just its sum, so that substitution cannot pass
    # silently.
    Xtr, ytr, Xte, yte = _split()
    rows = run_fold("iris-test", Xtr, ytr, Xte, yte, fold_id=0,
                    classifiers=("1nn",))
    by_method = {r["method"]: r for r in rows}
    total = by_method["RSP3"]["n_proto"]

    expected = allocate_per_class(ytr, total)
    expected_json = {str(c): int(k) for c, k in expected.items()}
    for m in ("KMeans", "Random"):
        assert json.loads(by_method[m]["class_counts"]) == expected_json, m
        assert by_method[m]["n_proto"] == total, m

    # ... and it is NOT the anchor's own prototype histogram, which on
    # this fold genuinely differs (guards the substitution directly).
    # SPINE cannot serve as the contrast here: it consumes the harness's
    # allocate_per_class itself, so its histogram IS the proportional
    # split by construction, and comparing against it would be vacuous.
    assert _anchor_class_histogram(Xtr, ytr) != expected_json


def _anchor_class_histogram(Xtr, ytr):
    # Scaled exactly as the runner scales it, so this is the histogram
    # the anchor really produced inside that fold and not one from
    # differently preprocessed data.
    from baselines.rsp3 import RSP3
    X = make_scaler().fit_transform(Xtr)
    _V, Vy, _i = RSP3().select(X, ytr)
    c, k = np.unique(Vy, return_counts=True)
    return {str(a): int(b) for a, b in zip(c, k)}


def test_missing_anchor_is_hard_error():
    Xtr, ytr, Xte, yte = _split()
    with pytest.raises(ValueError, match="requires"):
        run_fold("iris-test", Xtr, ytr, Xte, yte, fold_id=0,
                 methods=("full", "spine"), classifiers=("1nn",))
    # named rather than spelled out, so the check follows the declared
    # anchor if it ever moves
    assert BUDGET_ANCHOR == "rsp3"
