"""Budget-curve script (fraction axis): protocol invariants, pinned.

The curve shares the campaign's scientific-integrity requirements -- the
per-class split is a proportional apportionment of the TRAINING class
frequencies and never a competing method's prototype histogram; every
swept method attains the common budget exactly; SPINE's two series come
from one fit -- and adds its own: the fraction grid is realised with an
explicit class-count floor whose consequences are recorded rather than
hidden, coincident budgets are built once and flagged, reference levels
are built once per fold, and the on-disk shards recombine to exactly
what was in memory.  These tests execute the curve path end to end on a
small real dataset and pin each of those on its own.

Small real data is not only for speed: SPINE is the most expensive
construction in the roster and the curve refits it at every distinct
budget, so a fixture large enough to be interesting would make these
tests too slow to run on every change, which is when they are worth
having.  The end-to-end fixture is computed once per session.
"""

import csv
import functools
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
from sklearn.datasets import load_wine

from harness.base import allocate_per_class
from harness.config import (BUDGET_CURVE_FRACTIONS, LVQ3_TUNING_GRID,
                            make_scaler)
from harness.runner import fold_seed, grid_combinations

_SPEC = importlib.util.spec_from_file_location(
    "run_curve",
    os.path.join(os.path.dirname(__file__), os.pardir, "scripts",
                 "run_curve.py"))
run_curve = importlib.util.module_from_spec(_SPEC)
sys.modules["run_curve"] = run_curve
_SPEC.loader.exec_module(run_curve)

#: Budget-matched series, by the names the curve writes.  SPINE is two
#: of them: one fit per budget, scored under both decision rules.
SWEPT = (run_curve.SPINE_POINTS, run_curve.SPINE_GRAPH, "KMeans",
         "Random", "LVQ3", "SPOT", "GLVQ", "GNG")
REFERENCE = ("Full", "RSP3")
#: Reference levels computed and written to the CSVs but deliberately
#: left off the figures (2026-08-29).  Listing them here keeps the
#: coverage test below meaningful: a new reference method still has to
#: be classified as drawn or explicitly undrawn.
UNDRAWN_REFERENCE = ("Full",)

#: Grid-point bookkeeping: the columns that legitimately differ between
#: a build's rows and their ``shared_build`` copies at a later point.
POINT_COLUMNS = ("budget_fraction_nominal", "budget", "budget_fraction",
                 "budget_floored", "shared_build")
CPU_COLUMNS = ("cpu_select", "cpu_tuning", "cpu_total")

#: A three-point grid on the 140-row fixture fold: 2.8 -> 3 (equal to
#: the class count, so NOT floored), 7 and 14 prototypes.
GRID = (0.02, 0.05, 0.10)
#: A grid whose first two points fall under the class count on the
#: fixture: 0.7 -> 1 and 1.4 -> 1 both floor to 3, and 2.8 -> 3 meets it.
FLOOR_GRID = (0.005, 0.01, 0.02)


def _fold(permutation_seed=0):
    d = load_wine()
    rng = np.random.default_rng(permutation_seed)
    idx = rng.permutation(len(d.data))
    tr, te = idx[:140], idx[140:]
    return d.data[tr], d.target[tr], d.data[te], d.target[te]


@functools.lru_cache(maxsize=None)
def _cached_rows(fractions, fold_id=0, permutation_seed=0):
    Xtr, ytr, Xte, yte = _fold(permutation_seed)
    return tuple(run_curve.run_fold_curve(
        "wine-test", Xtr, ytr, Xte, yte, fold_id=fold_id,
        fractions=fractions))


def _rows(fractions=GRID, fold_id=0, permutation_seed=0):
    # Fresh dicts every time, so a test cannot mutate the shared fixture.
    return [dict(r) for r in _cached_rows(fractions, fold_id,
                                          permutation_seed)]


def _by_point(rows):
    out = {}
    for r in rows:
        out.setdefault(r["budget_fraction_nominal"], {})[r["method"]] = r
    return out


# ------------------------------------------------------------ the grid

def test_default_grid_is_the_frozen_config_grid():
    assert BUDGET_CURVE_FRACTIONS == (0.005, 0.01, 0.02, 0.05, 0.10, 0.25)
    assert run_curve.validate_fractions(BUDGET_CURVE_FRACTIONS) == \
        BUDGET_CURVE_FRACTIONS
    # the script's default realisation uses exactly that grid
    got = run_curve.budgets_for(18000, 26)
    assert [rho for rho, _b, _f in got] == list(BUDGET_CURVE_FRACTIONS)
    assert [b for _rho, b, _f in got] == [90, 180, 360, 900, 1800, 4500]
    assert not any(f for _rho, _b, f in got)


def test_budgets_for_realises_the_floor_and_records_it():
    got = run_curve.budgets_for(140, 3, (0.005, 0.01, 0.02, 0.05))
    assert got == ((0.005, 3, True), (0.01, 3, True), (0.02, 3, False),
                   (0.05, 7, False))
    budgets = [b for _rho, b, _f in got]
    assert budgets == sorted(budgets)               # non-decreasing
    assert all(b >= 3 for b in budgets)             # never below a class each
    # the realised fraction is what the row will carry; it exceeds the
    # nominal one where the floor bound and only there (2.8 -> 3 is
    # rounding, not the floor, and 3/140 > 0.02 by rounding alone, so
    # the rounding case is excluded from the identity)
    for rho, b, floored in got:
        if floored:
            assert b / 140.0 > rho
        else:
            assert abs(b - rho * 140.0) <= 0.5


def test_rounding_is_half_up_not_bankers():
    # 0.05 * 130 = 6.5: Python's round() gives 6 (half to even); the
    # budget rule says 7.
    assert run_curve.budgets_for(130, 2, (0.05,)) == ((0.05, 7, False),)
    # 0.005 * 100 = 0.5 -> 1, then the floor lifts it to the class count
    assert run_curve.budgets_for(100, 2, (0.005,)) == ((0.005, 2, True),)


@pytest.mark.parametrize("bad, message", [
    ((), "empty"),
    ((0.005, float("nan"), 0.25), "finite"),
    ((0.0, 0.1), "in \\(0, 1\\]"),
    ((0.1, 1.5), "in \\(0, 1\\]"),
    ((-0.1, 0.1), "in \\(0, 1\\]"),
    ((0.25, 0.10), "ascending"),
    ((0.10, 0.10), "ascending"),
])
def test_fraction_grid_declaration_errors_are_loud(bad, message):
    with pytest.raises(ValueError, match=message):
        run_curve.validate_fractions(bad)
    with pytest.raises(ValueError, match=message):
        run_curve.budgets_for(1000, 3, bad)


def test_unknown_fold_is_a_hard_error():
    assert run_curve.validate_folds(None, 10, "x") is None
    assert run_curve.validate_folds([9, 0, 0], 10, "x") == {0, 9}
    with pytest.raises(ValueError, match="asked for \\[10\\]"):
        run_curve.validate_folds({0, 10}, 10, "x")
    with pytest.raises(ValueError, match="asked for \\[-1\\]"):
        run_curve.validate_folds([-1], 10, "x")


# ------------------------------------------------ the sweep on one fold

def test_curve_per_class_split_is_proportional_not_a_method_s():
    rows = _rows()
    Xtr, ytr, _Xte, _yte = _fold()
    by_point = _by_point(rows)
    assert by_point, "the curve produced no rows"

    # The contrast is the ANCHOR's prototype histogram: a real method in
    # the table, on the same fold, scaled the way the curve scales it.
    # SPINE cannot play that role -- it calls the harness's own
    # allocate_per_class, so its histogram IS the proportional split by
    # construction and the comparison would be vacuous.
    from baselines.rsp3 import RSP3
    _V, Vy, _i = RSP3().select(make_scaler().fit_transform(Xtr), ytr)
    anchor_hist = dict(zip(*[[int(a) for a in arr]
                             for arr in np.unique(Vy, return_counts=True)]))

    differs_somewhere = False
    for _rho, at in by_point.items():
        budget = at["KMeans"]["budget"]
        expected = {str(k): int(v)
                    for k, v in allocate_per_class(ytr, budget).items()}
        assert sum(expected.values()) == budget
        # K-Means and Random are handed the split and record it verbatim
        for name in ("KMeans", "Random"):
            assert json.loads(at[name]["class_counts"]) == expected
            assert at[name]["n_proto"] == budget
        if anchor_hist != {int(k): v for k, v in expected.items()}:
            differs_somewhere = True
    # A hard assertion rather than a skip: if the proportional split
    # happened to equal the anchor's histogram at every budget, this test
    # would be passing without testing anything.
    assert differs_somewhere, (
        "the proportional split equals the anchor's prototype histogram "
        "at every budget on this fixture, so the test has no teeth")


def test_curve_sweeps_every_method_to_the_declared_budget():
    rows = _rows()
    for r in rows:
        if r["method"] in SWEPT:
            assert r["swept"] is True, r["method"]
            assert r["n_proto"] == r["budget"], (r["method"], r["budget"])
        else:
            assert r["method"] in REFERENCE
            assert r["swept"] is False, r["method"]
    # every series -- swept and reference -- is present at every point
    by_point = _by_point(rows)
    assert set(by_point) == set(GRID)
    for rho, at in by_point.items():
        assert set(at) == set(SWEPT) | set(REFERENCE), rho
        # and the point's budget columns are consistent on every row
        budgets = {r["budget"] for r in at.values()}
        assert len(budgets) == 1
        for r in at.values():
            assert r["budget_fraction"] == r["budget"] / 140.0
            assert r["budget_fraction_nominal"] == rho


def test_curve_spine_rows_come_from_one_fit_per_distinct_budget(monkeypatch):
    # The campaign's convention, reproduced on the curve: SPINE+1NN and
    # SPINE+Graph are one fitted model scored two ways.  Fitting twice
    # per budget would make the two series differ by the RNG's
    # consumption as well as by the decision rule.
    calls = []
    original = run_curve.SPINE.select

    def counting_select(self, X, y, **params):
        calls.append(params.get("total_count"))
        return original(self, X, y, **params)

    monkeypatch.setattr(run_curve.SPINE, "select", counting_select)
    Xtr, ytr, Xte, yte = _fold()
    rows = run_curve.run_fold_curve("wine-test", Xtr, ytr, Xte, yte,
                                    fold_id=0, fractions=GRID)
    distinct_budgets = sorted({r["budget"] for r in rows if r["swept"]})
    assert sorted(calls) == distinct_budgets == [3, 7, 14]

    for _rho, at in _by_point(rows).items():
        points, graph = at[run_curve.SPINE_POINTS], at[run_curve.SPINE_GRAPH]
        assert points["n_proto"] == graph["n_proto"] == points["budget"]
        # one construction: one cost, one provenance, written on both
        assert points["cpu_select"] == graph["cpu_select"]
        assert points["reduction"] == graph["reduction"]
        assert points["spine_params"] == graph["spine_params"]
        assert 0.0 <= graph[run_curve.CURVE_METRIC] <= 1.0


def test_curve_scores_only_1nn():
    rows = _rows()
    acc_columns = {k for r in rows for k in r if k.startswith("acc_")}
    assert acc_columns == {"acc_1nn"}
    assert run_curve.CURVE_METRIC == "acc_1nn"


def test_floored_points_share_one_build_and_say_so(monkeypatch):
    calls = []
    original = run_curve.SPINE.select

    def counting_select(self, X, y, **params):
        calls.append(params.get("total_count"))
        return original(self, X, y, **params)

    monkeypatch.setattr(run_curve.SPINE, "select", counting_select)
    Xtr, ytr, Xte, yte = _fold()
    rows = run_curve.run_fold_curve("wine-test", Xtr, ytr, Xte, yte,
                                    fold_id=0, fractions=FLOOR_GRID)
    by_point = _by_point(rows)
    assert set(by_point) == set(FLOOR_GRID)      # the grid stays complete

    # every point realises the same budget -- the class count -- and
    # SPINE (like every swept method) was built exactly once for it
    assert {r["budget"] for r in rows} == {3}
    assert calls == [3]
    assert {r["budget_fraction"] for r in rows} == {3 / 140.0}

    expected = {0.005: (True, False), 0.01: (True, True),
                0.02: (False, True)}       # (budget_floored, shared_build)
    for rho, (floored, shared) in expected.items():
        swept = [r for r in by_point[rho].values() if r["swept"]]
        assert len(swept) == len(SWEPT)
        assert {r["budget_floored"] for r in swept} == {floored}, rho
        assert {r["shared_build"] for r in swept} == {shared}, rho
        refs = [r for r in by_point[rho].values() if not r["swept"]]
        assert {r["budget_floored"] for r in refs} == {False}
        # reference repeats are copies of the fold's one build, and say so
        assert {r["shared_build"] for r in refs} == {rho != FLOOR_GRID[0]}

    # the shared rows are copies of the first point's build: identical
    # in every column except the grid-point bookkeeping
    first = by_point[0.005]
    for rho in (0.01, 0.02):
        for name in SWEPT:
            a, b = first[name], by_point[rho][name]
            for col in a:
                if col not in POINT_COLUMNS:
                    assert a[col] == b[col], (rho, name, col)


def test_reference_levels_are_built_once_and_constant(monkeypatch):
    from baselines.rsp3 import RSP3
    calls = []
    original = RSP3.select

    def counting_select(self, X, y, **params):
        calls.append(len(X))
        return original(self, X, y, **params)

    monkeypatch.setattr(RSP3, "select", counting_select)
    Xtr, ytr, Xte, yte = _fold()
    rows = run_curve.run_fold_curve("wine-test", Xtr, ytr, Xte, yte,
                                    fold_id=0, fractions=GRID)
    # RSP3 runs ONCE per fold as a reference; the LVQ3 tuning and the
    # per-budget sweeps never touch it
    assert calls == [140]
    by_point = _by_point(rows)
    for name in REFERENCE:
        versions = [by_point[rho][name] for rho in GRID]
        for v in versions[1:]:
            for col in versions[0]:
                if col not in POINT_COLUMNS:
                    assert v[col] == versions[0][col], (name, col)
        # its budget columns denote the grid point, not its own count
        assert [v["budget"] for v in versions] == [3, 7, 14]
        assert len({v["n_proto"] for v in versions}) == 1
        assert [v["shared_build"] for v in versions] == [False, True, True]
        assert all(v["budget_floored"] is False for v in versions)


def test_every_row_carries_the_fold_seed_and_a_rerun_is_bit_identical():
    rows = _rows()
    assert {r["seed"] for r in rows} == {fold_seed("wine-test", 0)}
    assert {r["fold"] for r in rows} == {0}
    Xtr, ytr, Xte, yte = _fold()
    again = run_curve.run_fold_curve("wine-test", Xtr, ytr, Xte, yte,
                                     fold_id=0, fractions=GRID)
    assert len(again) == len(rows)
    for a, b in zip(rows, again):
        for col in a:
            if col not in CPU_COLUMNS:
                assert a[col] == b[col], col


def test_provenance_columns_are_recorded_where_they_apply():
    rows = _rows()
    combos = list(grid_combinations(LVQ3_TUNING_GRID))
    for r in rows:
        if r["method"] == "LVQ3":
            assert json.loads(r["lvq3_params"]) in combos
            assert r["cpu_tuning"] > 0.0
        else:
            assert "lvq3_params" not in r
            assert r["cpu_tuning"] == 0.0
        if r["method"] in ("KMeans", "Random"):
            assert json.loads(r["class_counts"])
        else:
            assert "class_counts" not in r
        if r["method"] in (run_curve.SPINE_POINTS, run_curve.SPINE_GRAPH):
            assert isinstance(json.loads(r["spine_params"]), dict)
        else:
            assert "spine_params" not in r


# --------------------------------------------- aggregation and shards

def test_aggregate_is_keyed_by_nominal_fraction_and_averages_folds():
    rows = _rows(fold_id=0) + _rows(fold_id=1, permutation_seed=1)
    agg = run_curve.aggregate(rows)
    keys = {(a["dataset"], a["budget_fraction_nominal"], a["method"])
            for a in agg}
    assert len(keys) == len(agg) == len(GRID) * (len(SWEPT) + len(REFERENCE))
    for a in agg:
        sub = [r for r in rows
               if r["budget_fraction_nominal"] == a["budget_fraction_nominal"]
               and r["method"] == a["method"]]
        assert a["n_folds"] == len(sub) == 2
        assert a["acc_1nn"] == float(np.mean([r["acc_1nn"] for r in sub]))
        assert a["budget"] == float(np.mean([r["budget"] for r in sub]))
        assert a["budget_fraction"] == float(np.mean(
            [r["budget_fraction"] for r in sub]))
        assert a["swept"] == sub[0]["swept"]
        assert a["budget_floored"] == any(r["budget_floored"] for r in sub)
        assert a["shared_build"] == any(r["shared_build"] for r in sub)


def test_shards_round_trip_combine_and_refuse_overlap(tmp_path):
    rows0 = _rows(fold_id=0)
    rows1 = _rows(fold_id=1, permutation_seed=1)
    out = str(tmp_path)
    run_curve.write_csv(rows0, run_curve.shard_path(out, "wine-test", {0}))
    run_curve.write_csv(rows1, run_curve.shard_path(out, "wine-test", {1}))

    back, paths = run_curve.read_shards(out)
    assert len(paths) == 2 and len(back) == len(rows0) + len(rows1)
    # typed exactly as in memory: every non-string column round-trips
    for r_mem, r_disk in zip(rows0 + rows1, back):
        for col, val in r_mem.items():
            assert r_disk[col] == val, col

    fold_path, mean_path, figures, _paths = run_curve.combine(out)
    assert os.path.exists(fold_path) and os.path.exists(mean_path)
    with open(mean_path, newline="") as fh:
        mean_rows = list(csv.DictReader(fh))
    assert len(mean_rows) == len(GRID) * (len(SWEPT) + len(REFERENCE))
    assert {r["n_folds"] for r in mean_rows} == {"2"}
    # the anchor tick is the fold-mean of RSP3's OWN fraction of the fold,
    # attached by combine from the non-copy reference rows
    agg = run_curve.aggregate(back)
    anchors = [r["n_proto"] / r["n_train"] for r in back
               if r["method"] == "RSP3" and not r["shared_build"]]
    assert len(anchors) == 2
    for a in agg:
        assert "anchor_fraction" not in a
    with open(fold_path, newline="") as fh:
        assert "anchor_fraction" not in next(csv.reader(fh))
    assert sorted(os.path.basename(f) for f in figures) == \
        ["curve_mean.png", "curve_wine-test.png"]
    for f in figures:
        assert os.path.exists(f) and os.path.exists(f[:-4] + ".pdf")

    # a re-run shard of a fold already on disk would double-count it
    run_curve.write_csv(rows0, run_curve.shard_path(out, "wine-test", {0, 1}))
    with pytest.raises(ValueError, match="overlapping shards"):
        run_curve.read_shards(out)


def test_pdf_figures_are_byte_reproducible(tmp_path):
    out = str(tmp_path)
    run_curve.write_csv(_rows(), run_curve.shard_path(out, "wine-test", None))

    def digests():
        run_curve.combine(out)
        out_digests = {}
        for name in ("curve_wine-test.pdf", "curve_mean.pdf"):
            with open(os.path.join(out, name), "rb") as fh:
                out_digests[name] = hashlib.sha256(fh.read()).hexdigest()
        return out_digests

    assert digests() == digests()


def test_write_csv_refuses_an_empty_run(tmp_path):
    with pytest.raises(SystemExit, match="no rows"):
        run_curve.write_csv([], str(tmp_path / "curve_folds.csv"))


def test_aggregate_figure_refuses_an_incomplete_block(tmp_path):
    agg = run_curve.aggregate(_rows())
    # drop one cell: the mean over datasets would silently shift
    agg = [a for a in agg if not (a["method"] == "GNG"
                                  and a["budget_fraction_nominal"] == 0.05)]
    with pytest.raises(ValueError, match="incomplete"):
        run_curve.plot_aggregate(agg, str(tmp_path / "curve_mean"))


def test_graph_row_is_scored_by_the_skeleton_rule(monkeypatch):
    # On a small fixture the two SPINE readouts can coincide numerically,
    # so the MECHANISM is pinned: the graph row's metric must come from
    # the skeleton rule's accuracy_score call, and the points row must
    # not.  Deleting the re-score line leaves SPINE+Graph carrying the
    # vertices' 1-NN score -- silently identical series.
    sentinel = 0.123456789
    monkeypatch.setattr(run_curve, "accuracy_score",
                        lambda y_true, y_pred: sentinel)
    Xtr, ytr, Xte, yte = _fold()
    rows = run_curve.run_fold_curve("wine-test", Xtr, ytr, Xte, yte,
                                    fold_id=0, fractions=(0.05,))
    at = _by_point(rows)[0.05]
    assert at[run_curve.SPINE_GRAPH][run_curve.CURVE_METRIC] == sentinel
    assert at[run_curve.SPINE_POINTS][run_curve.CURVE_METRIC] != sentinel
    # and no other acc_ column ever reached the graph row
    assert {k for k in at[run_curve.SPINE_GRAPH] if k.startswith("acc_")} \
        == {run_curve.CURVE_METRIC}


def test_lvq3_is_tuned_at_each_distinct_budget(monkeypatch):
    seen = []
    original = run_curve.tune_by_grid

    def recording(make_method, grid, X, y, seed, **kw):
        seen.append(kw["select_params"]["total_count"])
        return original(make_method, grid, X, y, seed, **kw)

    monkeypatch.setattr(run_curve, "tune_by_grid", recording)
    Xtr, ytr, Xte, yte = _fold()
    rows = run_curve.run_fold_curve("wine-test", Xtr, ytr, Xte, yte,
                                    fold_id=0, fractions=GRID)
    assert seen == sorted({r["budget"] for r in rows if r["swept"]})
    assert seen == [3, 7, 14]


class _ShortGNG:
    """A stand-in for GNG that returns one prototype fewer than asked."""
    name = "GNG"
    is_generator = True

    def select(self, X, y, total_count=None, random_state=None):
        from harness.base import SYNTHETIC_INDEX
        k = int(total_count) - 1
        return (np.asarray(X[:k], dtype=float), np.asarray(y[:k]),
                np.full(k, SYNTHETIC_INDEX))


class _LyingGNG:
    """A stand-in that claims selection indices its rows do not match."""
    name = "GNG"
    is_generator = False

    def select(self, X, y, total_count=None, random_state=None):
        k = int(total_count)
        return (np.asarray(X[:k], dtype=float) + 1.0, np.asarray(y[:k]),
                np.arange(k))


def test_a_shortfall_is_a_hard_error(monkeypatch):
    monkeypatch.setattr(run_curve, "GNG", _ShortGNG)
    Xtr, ytr, Xte, yte = _fold()
    with pytest.raises(AssertionError, match="returned 13 prototypes for a "
                                             "budget of 14"):
        run_curve.run_fold_curve("wine-test", Xtr, ytr, Xte, yte,
                                 fold_id=0, fractions=(0.10,))


def test_a_contract_violation_is_a_hard_error(monkeypatch):
    monkeypatch.setattr(run_curve, "GNG", _LyingGNG)
    Xtr, ytr, Xte, yte = _fold()
    with pytest.raises(AssertionError, match="do not equal X_train"):
        run_curve.run_fold_curve("wine-test", Xtr, ytr, Xte, yte,
                                 fold_id=0, fractions=(0.10,))


def test_shards_from_two_fraction_grids_are_refused(tmp_path):
    # The same fold under two declared grids shares no (fold, nominal
    # fraction) pair, but it is still the same fold twice.
    out = str(tmp_path)
    rows = _rows()
    run_curve.write_csv(rows, run_curve.shard_path(out, "wine-test", {0}))
    remapped = []
    for r in rows:
        r = dict(r)
        r["budget_fraction_nominal"] = r["budget_fraction_nominal"] * 0.9
        remapped.append(r)
    run_curve.write_csv(remapped, os.path.join(
        out, "curve_folds__wine-test__folds_0alt.csv"))
    with pytest.raises(ValueError, match="overlapping shards"):
        run_curve.read_shards(out)


def test_combined_files_are_never_read_back_as_shards(tmp_path):
    out = str(tmp_path)
    run_curve.write_csv(_rows(), run_curve.shard_path(out, "wine-test", {0}))
    run_curve.combine(out)
    _rows_back, paths = run_curve.read_shards(out)
    assert [os.path.basename(p) for p in paths] == \
        ["curve_folds__wine-test__folds_0.csv"]
    # and the combined files carry a sorted header and no helper column
    for name in ("curve_folds.csv", "curve_mean.csv"):
        with open(os.path.join(out, name), newline="") as fh:
            header = next(csv.reader(fh))
        assert header == sorted(header)
        assert "anchor_fraction" not in header and "n_train_mean" not in header


def test_aggregate_figure_skips_unequal_fold_coverage(tmp_path):
    out = str(tmp_path)
    run_curve.write_csv(_rows(fold_id=0),
                        run_curve.shard_path(out, "wine-test", {0}))
    run_curve.write_csv(_rows(fold_id=1, permutation_seed=1),
                        run_curve.shard_path(out, "wine-test", {1}))
    other = [dict(r, dataset="other-test") for r in _rows(fold_id=0)]
    run_curve.write_csv(other, run_curve.shard_path(out, "other-test", {0}))
    # equal coverage first, so a stale aggregate exists on disk
    run_curve.combine(str(tmp_path))
    assert os.path.exists(os.path.join(out, "curve_mean.png")) is False
    # (wine-test has 2 folds, other-test 1: skipped, and nothing left)
    _fold_path, _mean_path, figures, _paths = run_curve.combine(out)
    assert sorted(os.path.basename(f) for f in figures) == \
        ["curve_other-test.png", "curve_wine-test.png"]
    assert not os.path.exists(os.path.join(out, "curve_mean.pdf"))


def test_aggregate_flags_are_any_over_folds():
    floored = _rows(fractions=FLOOR_GRID)              # floored, shared
    plain = [dict(r, fold=1) for r in _rows(fractions=FLOOR_GRID)]
    for r in plain:                                    # pretend fold 1 was not
        r["budget_floored"] = False
        r["shared_build"] = False
    agg = run_curve.aggregate(floored + plain)
    swept = [a for a in agg if a["swept"]
             and a["budget_fraction_nominal"] == FLOOR_GRID[0]]
    assert swept and all(a["n_folds"] == 2 for a in swept)
    assert all(a["budget_floored"] is True for a in swept)
    at_second = [a for a in agg if a["swept"]
                 and a["budget_fraction_nominal"] == FLOOR_GRID[1]]
    assert all(a["shared_build"] is True for a in at_second)


def test_combine_hands_the_anchor_fraction_to_the_figure(
        tmp_path, monkeypatch):
    out = str(tmp_path)
    rows = _rows()
    run_curve.write_csv(rows, run_curve.shard_path(out, "wine-test", {0}))
    captured = {}
    original = run_curve.plot_curve

    def capture(agg, dataset, out_base, **kw):
        captured[dataset] = [r for r in agg if r["dataset"] == dataset]
        return original(agg, dataset, out_base, **kw)

    monkeypatch.setattr(run_curve, "plot_curve", capture)
    run_curve.combine(out)
    expected = [r["n_proto"] / r["n_train"] for r in rows
                if r["method"] == "RSP3" and not r["shared_build"]]
    assert len(expected) == 1
    got = {r["anchor_fraction"] for r in captured["wine-test"]}
    assert got == {expected[0]}
    # the anchor's own fraction, not the grid point's
    assert expected[0] not in {r["budget_fraction"]
                               for r in rows if r["swept"]}


def test_foreign_output_in_out_dir_is_refused(tmp_path):
    out = str(tmp_path)
    with open(os.path.join(out, "curve_folds.csv"), "w") as fh:
        fh.write("acc_1nn,budget,budget_per_class,dataset\n")
    with pytest.raises(SystemExit, match="different instrument"):
        run_curve.combine(out)


def test_plot_series_covers_every_swept_method():
    # A method swept but missing from PLOT_SERIES would be computed,
    # written to the CSV, and then silently left off the figure.
    plotted = {label for label, _c, _m, _ls in run_curve.PLOT_SERIES}
    assert plotted == set(SWEPT)
    referenced = {label for label, _c, _ls in run_curve.REFERENCE_SERIES}
    assert referenced == set(REFERENCE) - set(UNDRAWN_REFERENCE)
    assert set(UNDRAWN_REFERENCE) <= set(REFERENCE)
    assert run_curve.ANCHOR in referenced


def test_default_grid_is_log_spaced_and_reaches_below_the_old_cap():
    # Log-spaced (every step at least doubles, at most 2.5x), and the
    # bottom sits an order of magnitude below the old per-class grid's
    # 10% cap so the severe-compression regime is actually measured.
    fr = BUDGET_CURVE_FRACTIONS
    ratios = [b / a for a, b in zip(fr, fr[1:])]
    assert all(2.0 <= r <= 2.5 for r in ratios), ratios
    assert fr[0] <= 0.01 and fr[-1] == 0.25
    assert math.isclose(math.log10(fr[-1] / fr[0]), math.log10(50))
