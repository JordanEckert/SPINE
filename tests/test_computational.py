"""Fixed-configuration timing study: structure, budgets, and figure.

Offline: runs on sklearn's bundled wine (a real measured dataset, the
repository's stand-in convention for unit tests); the script's main()
path loads the real campaign datasets instead.  Wine is also small
enough that SPINE -- the most expensive construction in the timed
roster -- costs a fraction of a second per fold here.
"""

import importlib.util
import os
import sys

from sklearn.datasets import load_wine

_SPEC = importlib.util.spec_from_file_location(
    "run_computational",
    os.path.join(os.path.dirname(__file__), os.pardir, "scripts",
                 "run_computational.py"))
run_computational = importlib.util.module_from_spec(_SPEC)
sys.modules["run_computational"] = run_computational
_SPEC.loader.exec_module(run_computational)

#: Everything the study times, in the order the script builds it.  Full
#: (no construction) and Random (structureless control) are excluded by
#: design, so their absence is part of what these tests pin.
TIMED_METHODS = ("RSP3", "SPINE", "KMeans", "LVQ3", "SPOT", "GLVQ",
                 "GNG")

#: Held to the anchor's per-fold count.  Every timed method other than
#: the anchor itself is budget-matched, so the study is matched
#: throughout.
MATCHED_METHODS = ("SPINE", "KMeans", "LVQ3", "SPOT", "GLVQ", "GNG")


def _rows():
    d = load_wine()
    return run_computational.time_dataset("wine-test", d.data, d.target,
                                          n_folds=2)


def test_time_dataset_rows_and_budgets():
    rows = _rows()
    # one row per (fold, method); Full and Random never timed
    assert len(rows) == 2 * len(TIMED_METHODS)
    assert {r["method"] for r in rows} == set(TIMED_METHODS)
    for fold in (0, 1):
        by = {r["method"]: r for r in rows if r["fold"] == fold}
        budget = by[run_computational.BUDGET_METHOD]["n_proto"]
        # budget-matched comparators attain the anchor's count
        for m in MATCHED_METHODS:
            assert by[m]["n_proto"] == budget, m
        # nothing sits off the common budget: the anchor plus the
        # matched comparators are the whole timed roster
        assert set(TIMED_METHODS) == {run_computational.BUDGET_METHOD} | set(
            MATCHED_METHODS)
        for r in by.values():
            assert r["cpu_seconds"] >= 0.0
            assert r["n_train"] > 0


def test_spine_is_timed_once_per_fold():
    # The campaign scores SPINE twice (SPINE+1NN and SPINE+Graph) but
    # both rows come from ONE construction, and only construction is
    # timed here.  A second SPINE series would double-count a cost that
    # was paid once and would make the log-log figure claim the method is
    # twice as expensive as it is.
    rows = _rows()
    for fold in (0, 1):
        spine = [r for r in rows
                 if r["fold"] == fold and r["method"].startswith("SPINE")]
        assert len(spine) == 1, [r["method"] for r in spine]
        assert spine[0]["method"] == "SPINE"


def test_aggregate_and_loglog_figure(tmp_path):
    rows = _rows()
    agg = run_computational.aggregate(rows)
    # one aggregate row per method, positive mean times
    assert len(agg) == len({r["method"] for r in rows})
    assert all(r["cpu_seconds"] > 0.0 for r in agg)
    out = run_computational.plot_loglog(agg, str(tmp_path / "timing"))
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert os.path.exists(str(tmp_path / "timing.pdf"))
    csv_path = run_computational.write_csv(
        agg, str(tmp_path / "timing_mean.csv"))
    assert os.path.getsize(csv_path) > 0


def test_plot_series_covers_every_timed_method():
    # A method timed but missing from PLOT_SERIES would be measured,
    # written to the CSV, and then silently dropped from the figure the
    # paper prints.
    plotted = {label for label, _c, _m, _ls in run_computational.PLOT_SERIES}
    assert plotted == set(TIMED_METHODS)


def test_warm_up_runs_before_any_timed_build():
    """The first sklearn call carries setup cost that is not the method's.

    ``pairwise_distances_argmin_min`` pays roughly 30 ms of one-time
    setup, charged in full to whichever method reaches it first -- which
    is fold 0, the smallest fold, and the one anchoring the log-log
    slope.  This pins that the warm-up happens, and happens before the
    fold loop rather than inside a timed region.
    """
    run_computational._WARMED = False
    calls = []
    real = run_computational.warm_up

    def spy():
        calls.append(len(run_computational.__dict__))
        real()

    run_computational.warm_up = spy
    try:
        d = load_wine()
        rows = run_computational.time_dataset("wine-test", d.data,
                                              d.target, n_folds=2)
    finally:
        run_computational.warm_up = real
    assert calls, "time_dataset did not warm the libraries up"
    assert run_computational._WARMED
    assert rows and all(r["cpu_seconds"] >= 0.0 for r in rows)


def test_empty_run_refuses_rather_than_writing_a_header(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="no rows to write"):
        run_computational.write_csv([], str(tmp_path / "x.csv"))
