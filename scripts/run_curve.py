#!/usr/bin/env python3
"""Budget curve: accuracy against prototype budget, budget = fraction of
the training fold.

The main campaign (``run_experiments.py``) fixes ONE operating point --
the budget anchor's own per-fold prototype count -- and answers "who
wins at RSP3's budget".  

This script sweeps the budget instead.  The axis is REDUCTION SEVERITY:
for each declared fraction ``rho`` in ``config.BUDGET_CURVE_FRACTIONS``
(a 1-2-5 grid, approximately log-spaced, 0.5% to 25%) the budget on a
training fold of ``n_train`` rows with ``n_classes`` classes is

    budget = max(round(rho * n_train), n_classes)      (round half up)

identical for every swept method at that grid point, so accuracy is
compared at a common reduction rate rather than at a method's
convenience, and the same grid point means the same compression on
every dataset.  

Outputs, written to ``--out-dir`` (kept out of ``results/`` so the
campaign's completeness checks never see them):

* one per-(dataset, shard) CSV, ``curve_folds__<dataset>__folds_<shard>.csv``,
  written as soon as that dataset finishes, so a failure on a later
  dataset loses nothing and a campaign can be sharded across
  invocations with ``--datasets`` / ``--folds``;
* the combined ``curve_folds.csv`` and the per-(dataset, nominal
  fraction, method) fold-mean ``curve_mean.csv``, regenerated at the end
  of every invocation from EVERY shard CSV present in the directory
  (never from this invocation alone, so shards from earlier invocations
  are combined rather than clobbered; overlapping shards are a hard
  error);
* one accuracy-against-realised-fraction figure per dataset present, and
  ONE aggregate figure of the mean over datasets against the nominal
  fraction.  The aggregate is an UNWEIGHTED mean over the datasets on
  disk, drawn only when every dataset rests on the same number of
  folds (a partially sharded campaign would otherwise average a 10-fold
  mean with a 2-fold one; the figure is then skipped with a notice and
  any stale copy removed, while the CSVs are still written).  Its
  left-hand cells on wine and sonar are ``shared_build`` copies (the
  floor): a flat segment there is one measurement plotted twice, not a
  plateau -- recoverable from ``curve_mean.csv``.  PDFs are written
  without a creation date so they are byte-reproducible.

No formal statistics are attached to the curve: it is a graphical
instrument, by decision.

Example::

    python scripts/run_curve.py --datasets wine segment
    python scripts/run_curve.py --datasets magic --folds 0 1 2 3 4
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from baselines.glvq import GLVQ
from baselines.gng import GNG
from baselines.lvq3 import LVQ3
from baselines.reference import FullSet, KMeansPerClass, RandomSubsample
from baselines.rsp3 import RSP3
from baselines.spotgreedy import SPOTGreedy
from harness.base import allocate_per_class, validate_selection
from harness.config import (BASE_SEED, BUDGET_CURVE_FRACTIONS, DATASETS,
                            LVQ3_TUNING_GRID, N_OUTER_FOLDS, NORMALIZE,
                            make_scaler)
from harness.datasets import load_dataset, load_keel_folds
from harness.metrics import evaluate_prototypes, reduction_rate
from harness.runner import (SPINE_GRAPH, SPINE_POINTS, _warm_up_libraries,
                            fold_seed, tune_by_grid)
from spine import SPINE

#: The one scored classifier.  Not an option: see the module docstring.
CURVE_METRIC = "acc_1nn"

#: The budget anchor of the main campaign, marked on every per-dataset
#: figure at its realised fraction.
ANCHOR = "RSP3"

#: Methods run once per fold at their own operating point and repeated
#: at every grid point as a reference level (never budget-matched).
REFERENCE_METHODS = (("Full", FullSet()), ("RSP3", RSP3()))

#: Plot series: label, hex, marker, linestyle.  SPINE's two decision
#: rules carry the categorical hues, since the gap between them is what
#: the figure is read for; K-Means keeps the third hue as the strongest
#: budget-matched baseline, and the remaining swept baselines are the
#: ink grays, separated by marker and dash rather than by colour.  The
#: unswept reference levels are drawn as flat dashed lines; Full is
#: deliberately NOT among them (2026-08-29).  It is still built, scored
#: and written to both CSVs -- only the figure omits it.
PLOT_SERIES = (
    # named from the runner's constants, so a rename of either row
    # cannot silently drop a series off the figure
    (SPINE_GRAPH, "#2a78d6", "o", "-"),
    (SPINE_POINTS, "#eb6834", "s", "-"),
    ("KMeans", "#1baf7a", "^", "-"),
    ("GLVQ", "#52514e", "v", "--"),
    ("GNG", "#898781", "P", "--"),
    ("LVQ3", "#52514e", "*", ":"),
    ("SPOT", "#898781", "d", "-."),
    ("Random", "#898781", "X", ":"),
)
REFERENCE_SERIES = (
    ("RSP3", "#52514e", "--"),
)


def _round_half_up(x):
    """Round to the nearest integer, halves up (not banker's rounding).

    Python's ``round`` rounds halves to even, so 0.25 * 4770 = 1192.5
    would become 1192 and 0.05 * 130 = 6.5 would become 6; the budget
    rule is stated as ordinary rounding and is implemented as such.
    """
    return int(math.floor(x + 0.5))


def validate_fractions(fractions):
    """The declared grid must be usable before any compute is spent.

    Non-empty; every value finite with ``0 < rho <= 1``; strictly
    ascending (a repeated or unsorted declaration is a declaration
    error, not a grid point to realise twice).  Returns the grid as a
    tuple of floats.
    """
    fr = tuple(float(f) for f in fractions)
    if not fr:
        raise ValueError("the budget-curve fraction grid is empty")
    for f in fr:
        if not math.isfinite(f) or not 0.0 < f <= 1.0:
            raise ValueError(
                "budget-curve fractions must be finite and in (0, 1]; "
                "got {0!r}".format(fr))
    if any(b <= a for a, b in zip(fr, fr[1:])):
        raise ValueError(
            "budget-curve fractions must be strictly ascending; got "
            "{0!r}".format(fr))
    return fr


def budgets_for(n_train, n_classes, fractions=BUDGET_CURVE_FRACTIONS):
    """The declared fraction grid realised on one training fold.

    Returns ``((rho, budget, floored), ...)`` in declared order, with
    ``budget = max(round(rho * n_train), n_classes)`` and ``floored``
    True where the class-count floor bound.  Budgets are non-decreasing
    in ``rho``; repeats (several fractions realising one budget) are
    kept, so the caller can build once and label every grid point.
    """
    fr = validate_fractions(fractions)
    n_train = int(n_train)
    n_classes = int(n_classes)
    if n_train < 1 or n_classes < 1:
        raise ValueError("n_train and n_classes must be positive")
    out = []
    for rho in fr:
        raw = _round_half_up(rho * n_train)
        floored = raw < n_classes
        out.append((rho, max(raw, n_classes), floored))
    return tuple(out)


def validate_folds(folds, k, dataset_name):
    """A requested fold id the dataset does not have is a hard error."""
    if folds is None:
        return None
    wanted = sorted({int(f) for f in folds})
    bad = [f for f in wanted if f < 0 or f >= k]
    if bad:
        raise ValueError(
            "{0} has folds 0..{1}; --folds asked for {2}".format(
                dataset_name, k - 1, bad))
    return set(wanted)


def run_fold_curve(dataset_name, X_train, y_train, X_test, y_test,
                   fold_id, n_outer_folds=N_OUTER_FOLDS,
                   normalize=NORMALIZE, fractions=BUDGET_CURVE_FRACTIONS):
    """Every method at every declared fraction on one outer fold."""
    seed = fold_seed(dataset_name, fold_id)
    _warm_up_libraries()

    if normalize:
        scaler = make_scaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr = np.array(X_train, dtype=float)
        X_te = np.array(X_test, dtype=float)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    n_train = len(X_tr)
    n_classes = len(np.unique(y_train))
    grid = budgets_for(n_train, n_classes, fractions)

    def _point_columns(point, swept):
        rho, budget, floored = point
        return {
            "budget_fraction_nominal": rho,
            "budget": budget,
            "budget_fraction": float(budget) / float(n_train),
            "budget_floored": bool(floored) if swept else False,
            "shared_build": False,
        }

    def _row(method, X_proto, y_proto, cpu, point, swept, extra=None,
             classifiers=("1nn",)):
        row = {
            "dataset": dataset_name, "fold": fold_id,
            "n_outer_folds": n_outer_folds, "method": method,
            "seed": seed, "n_train": n_train, "n_test": len(X_te),
            "swept": swept, "n_proto": len(X_proto),
            "reduction": reduction_rate(len(X_proto), n_train),
            "cpu_select": cpu, "cpu_tuning": 0.0,
        }
        row.update(_point_columns(point, swept))
        row.update(evaluate_prototypes(
            X_proto, y_proto, X_te, y_test, classifiers=classifiers,
            random_state=seed))
        if extra:
            row.update(extra)
        row["cpu_total"] = row["cpu_select"] + row["cpu_tuning"]
        return row

    def _timed(method, **params):
        # The campaign's _timed_select, reproduced: process-CPU time of
        # select() only, then the shared contract checks.
        t0 = time.process_time()
        X_proto, y_proto, indices = method.select(X_tr, y_train, **params)
        cpu = time.process_time() - t0
        X_proto = np.asarray(X_proto, dtype=float)
        y_proto = np.asarray(y_proto)
        validate_selection(method, X_tr, y_train, X_proto, y_proto,
                           indices)
        return X_proto, y_proto, cpu

    def _check_budget(label, n_proto, budget):
        if n_proto != budget:
            raise AssertionError(
                "{0} on {1} fold {2} returned {3} prototypes for a budget "
                "of {4}; the curve compares methods at a common budget and "
                "a shortfall must be decided on, not absorbed".format(
                    label, dataset_name, fold_id, n_proto, budget))

    def _sweep(point):
        """Every swept method built at this grid point's budget."""
        _rho, budget, _floored = point
        out = []

        # Proportional apportionment of this budget over the training
        # class frequencies -- from the data, not from any method.
        class_counts = allocate_per_class(y_train, budget)
        counts_json = json.dumps({str(k): int(v)
                                  for k, v in class_counts.items()},
                                 sort_keys=True)

        # SPINE: ONE fit per budget, TWO rows -- the campaign runner's
        # convention reproduced here, so a curve point and a campaign
        # row for the same method mean the same thing.
        spine = SPINE()
        Xp, yp, cpu = _timed(spine, total_count=budget, random_state=seed)
        _check_budget(SPINE_POINTS, len(Xp), budget)
        provenance = {"spine_params": json.dumps(
            spine.spine_params_, sort_keys=True, default=str)}
        out.append(_row(SPINE_POINTS, Xp, yp, cpu, point, True,
                        extra=provenance))
        # The same fitted model under the skeleton rule.  Built WITHOUT
        # _row's classifier pass, as the campaign runner builds it: a
        # skeleton is not a prototype set, and a 1-NN score on its
        # vertices would be the other row's number.  Leaving the column
        # absent until the skeleton rule fills it also means a slip here
        # produces a MISSING column rather than a silently wrong one.
        graph_row = _row(SPINE_GRAPH, Xp, yp, cpu, point, True,
                         extra=provenance, classifiers=())
        graph_row[CURVE_METRIC] = float(accuracy_score(
            y_test, spine.estimator_.predict(X_te)))
        out.append(graph_row)

        # LVQ3: re-tuned at this budget; the choice and its bill recorded.
        best_combo, _scores, cpu_tuning = tune_by_grid(
            lambda combo: LVQ3(**combo), LVQ3_TUNING_GRID, X_tr, y_train,
            seed, select_params={"total_count": budget})
        Xp, yp, cpu = _timed(LVQ3(**best_combo), total_count=budget,
                             random_state=seed)
        _check_budget("LVQ3", len(Xp), budget)
        out.append(_row("LVQ3", Xp, yp, cpu, point, True, extra={
            "cpu_tuning": cpu_tuning,
            "lvq3_params": json.dumps(best_combo, sort_keys=True)}))

        for label, method, params, extra in (
                ("KMeans", KMeansPerClass(),
                 {"class_counts": class_counts, "random_state": seed},
                 {"class_counts": counts_json}),
                ("Random", RandomSubsample(),
                 {"class_counts": class_counts, "random_state": seed},
                 {"class_counts": counts_json}),
                ("SPOT", SPOTGreedy(), {"total_count": budget}, None),
                # GLVQ and GNG are budget-matched and stochastic, so both
                # take the fold's derived seed exactly as LVQ3 does; both
                # are untuned, at the settings their own literature
                # declares, so nothing about them is re-selected per
                # budget the way LVQ3's grid is.
                ("GLVQ", GLVQ(),
                 {"total_count": budget, "random_state": seed}, None),
                ("GNG", GNG(),
                 {"total_count": budget, "random_state": seed}, None)):
            Xp, yp, cpu = _timed(method, **params)
            _check_budget(label, len(Xp), budget)
            out.append(_row(label, Xp, yp, cpu, point, True, extra=extra))
        return out

    # Reference levels: built and scored ONCE per fold.  Their rows are
    # repeated at every grid point below with that point's bookkeeping.
    reference_rows = []
    for label, method in REFERENCE_METHODS:
        Xp, yp, cpu = _timed(method)
        reference_rows.append(_row(label, Xp, yp, cpu, grid[0], False))

    rows = []
    built = {}   # budget -> the swept rows constructed at that budget
    for i, point in enumerate(grid):
        _rho, budget, _floored = point
        for base in reference_rows:
            row = dict(base)
            row.update(_point_columns(point, False))
            row["shared_build"] = i > 0     # a copy of the fold's one build
            rows.append(row)
        if budget in built:
            for base in built[budget]:
                row = dict(base)
                row.update(_point_columns(point, True))
                row["shared_build"] = True
                rows.append(row)
            continue
        swept_rows = _sweep(point)
        built[budget] = swept_rows
        rows.extend(swept_rows)
    return rows


def outer_folds_for(dataset_name, data_dir, download=True):
    """The dataset's outer folds and their count, as the campaign has them."""
    spec = DATASETS[dataset_name]
    if spec["source"] == "keel" and not spec.get("own_folds"):
        outer_folds, k, _labels = load_keel_folds(
            dataset_name, data_dir, download=download)
        if k != N_OUTER_FOLDS:
            print("note: {0} uses KEEL's {1}-fold partition for the outer "
                  "CV (recorded as n_outer_folds in the results)".format(
                      dataset_name, k))
    else:
        X, y, _labels = load_dataset(dataset_name, data_dir,
                                     download=download)
        k = N_OUTER_FOLDS
        skf = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=BASE_SEED)
        outer_folds = [(X[tr], y[tr], X[te], y[te])
                       for tr, te in skf.split(X, y)]
    return outer_folds, k


def run_dataset_curve(dataset_name, data_dir, folds=None, download=True,
                      normalize=NORMALIZE, fractions=BUDGET_CURVE_FRACTIONS):
    """The budget curve on one dataset; returns its result rows."""
    outer_folds, k = outer_folds_for(dataset_name, data_dir, download)
    wanted = validate_folds(folds, k, dataset_name)
    rows = []
    for fold_id, (X_tr, y_tr, X_te, y_te) in enumerate(outer_folds):
        if wanted is not None and fold_id not in wanted:
            continue
        rows.extend(run_fold_curve(
            dataset_name, X_tr, y_tr, X_te, y_te, fold_id,
            n_outer_folds=k, normalize=normalize, fractions=fractions))
    return rows


def aggregate(rows, metric=CURVE_METRIC):
    """Fold means per (dataset, nominal fraction, method).

    ``budget`` and ``budget_fraction`` (realised) are averaged over the
    folds too, since the floor and rounding can realise slightly
    different budgets on folds of different sizes; ``budget_floored``
    and ``shared_build`` are True if any fold's row was.
    """
    groups = {}
    for r in rows:
        groups.setdefault(
            (r["dataset"], r["budget_fraction_nominal"], r["method"]),
            []).append(r)
    out = []
    for (dataset, rho, method), sub in sorted(groups.items()):
        out.append({
            "dataset": dataset, "budget_fraction_nominal": rho,
            "method": method, "swept": sub[0]["swept"],
            "n_folds": len(sub),
            "budget": float(np.mean([r["budget"] for r in sub])),
            "budget_fraction": float(np.mean([r["budget_fraction"]
                                              for r in sub])),
            "budget_floored": any(r["budget_floored"] for r in sub),
            "shared_build": any(r["shared_build"] for r in sub),
            "n_proto": float(np.mean([r["n_proto"] for r in sub])),
            "reduction": float(np.mean([r["reduction"] for r in sub])),
            metric: float(np.mean([r[metric] for r in sub])),
        })
    return out


# ---------------------------------------------------------------- figures

class UnequalCoverage(ValueError):
    """The datasets on disk do not rest on the same number of folds."""


def _percent_ticks(ax, fractions, extra=()):
    """Major ticks at the declared fractions, labelled as percentages.

    A log axis over 0.5%-25% labels only the decades (1%, 10%); the
    declared grid points are the positions a reader wants to read off.
    The view is widened to hold every declared fraction and every
    ``extra`` position (the anchor), so a floored dataset -- whose first
    markers sit to the RIGHT of the 0.5% and 1% ticks, at the floor's
    realised fraction -- shows the displacement rather than hiding the
    ticks off-axis.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    fr = sorted(set(fractions))
    ax.xaxis.set_major_locator(FixedLocator(fr))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _pos: ("{0:g}%".format(100.0 * v))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    lo, hi = ax.get_xlim()
    positions = list(fr) + [float(e) for e in extra]
    ax.set_xlim(min(lo, min(positions) / 1.25), max(hi, max(positions) * 1.25))


def _style_axes(ax, title, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.set_title(title, color="#0b0b0b")
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    # ``ax.spines`` are matplotlib's axis borders; the loop variable is
    # named for them rather than ``spine``, which in this module means
    # the method under test.
    for border in ax.spines.values():
        border.set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")


def _save(fig, out_base):
    """PNG + PDF; the PDF without a creation date so it is reproducible."""
    fig.savefig(out_base + ".png", dpi=200)
    fig.savefig(out_base + ".pdf", metadata={"CreationDate": None})


def plot_curve(agg, dataset, out_base, metric=CURVE_METRIC):
    """One accuracy-against-realised-fraction figure for one dataset.

    The anchor tick is the fold-mean of the anchor's own ``n_proto /
    n_train``; ``combine`` attaches it to the aggregated rows as
    ``anchor_fraction``.  Without it (rows aggregated directly from
    memory) the tick is simply omitted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = [r for r in agg if r["dataset"] == dataset]
    if not sub:
        raise ValueError("no aggregated rows for {0!r}".format(dataset))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, color, marker, linestyle in PLOT_SERIES:
        pts = sorted((r["budget_fraction"], r[metric]) for r in sub
                     if r["method"] == label)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, marker=marker, linestyle=linestyle,
                linewidth=1.6, markersize=6, markeredgecolor="white",
                markeredgewidth=0.5, label=label)
    for label, color, linestyle in REFERENCE_SERIES:
        vals = [r[metric] for r in sub if r["method"] == label]
        if not vals:
            continue
        ax.axhline(float(np.mean(vals)), color=color, linestyle=linestyle,
                   linewidth=1.2, label="{0} (own budget)".format(label))
    # The campaign's operating point: the anchor's own realised fraction.
    anchor = [r["anchor_fraction"] for r in sub
              if r["method"] == ANCHOR and "anchor_fraction" in r]
    if anchor:
        at = float(anchor[0])
        ax.axvline(at, color="#52514e", linestyle="-", linewidth=0.9,
                   alpha=0.7, label="{0} anchor ({1:.1f}% of fold)".format(
                       ANCHOR, 100.0 * at))
    _style_axes(ax, "Accuracy against Budget - {0}".format(dataset),
                "Prototype Budget (Realised Fraction of the Training Fold)",
                "Mean 1-NN Accuracy")
    _percent_ticks(ax, [r["budget_fraction_nominal"] for r in sub],
                   extra=anchor[:1])
    ax.legend(ncols=2, frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)
    return out_base + ".png"


def plot_aggregate(agg, out_base, metric=CURVE_METRIC):
    """One figure: mean over datasets of the fold-mean accuracy, against
    the NOMINAL fraction (complete blocks by construction).

    Refuses an incomplete block: a dataset missing a grid point for a
    method would silently bias the mean, and by construction none can
    be missing, so a gap is a data problem to be looked at.  Refuses
    UNEQUAL FOLD COVERAGE for the same reason (``UnequalCoverage``): a
    partially sharded campaign would otherwise average a 10-fold mean
    with a 2-fold one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in agg})
    fractions = sorted({r["budget_fraction_nominal"] for r in agg})
    methods = sorted({r["method"] for r in agg})
    cells = {(r["dataset"], r["budget_fraction_nominal"], r["method"]): r
             for r in agg}
    for d in datasets:
        for f in fractions:
            for m in methods:
                if (d, f, m) not in cells:
                    raise ValueError(
                        "aggregate block incomplete: {0} has no row for "
                        "{1} at nominal fraction {2}".format(d, m, f))
    coverage = {d: sorted({r["n_folds"] for r in agg if r["dataset"] == d})
                for d in datasets}
    if len({tuple(v) for v in coverage.values()}) != 1 or \
            any(len(v) != 1 for v in coverage.values()):
        raise UnequalCoverage(
            "aggregate figure not drawn: datasets rest on different fold "
            "counts {0}; finish the sharded run first".format(coverage))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, color, marker, linestyle in PLOT_SERIES:
        if label not in methods:
            continue
        ys = [float(np.mean([cells[(d, f, label)][metric]
                             for d in datasets])) for f in fractions]
        ax.plot(fractions, ys, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.6, markersize=6,
                markeredgecolor="white", markeredgewidth=0.5, label=label)
    for label, color, linestyle in REFERENCE_SERIES:
        if label not in methods:
            continue
        val = float(np.mean([cells[(d, fractions[0], label)][metric]
                             for d in datasets]))
        ax.axhline(val, color=color, linestyle=linestyle, linewidth=1.2,
                   label="{0} (own budget)".format(label))
    _style_axes(ax, "Accuracy against Budget - Average over All Datasets",
                "Prototype Budget (Nominal Fraction of the Training Fold)",
                "Mean 1-NN Accuracy (Unweighted over Datasets)")
    _percent_ticks(ax, fractions)
    ax.legend(ncols=2, frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)
    return out_base + ".png"


# ------------------------------------------------------------ CSV plumbing

def write_csv(rows, path):
    """Write the rows; an empty set is a loud stop, not a header file."""
    if not rows:
        raise SystemExit(
            "run_curve: no rows to write to {0} -- the run produced "
            "nothing (was --datasets or --folds given an empty "
            "selection?)".format(path))
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


_INT_COLUMNS = ("fold", "n_outer_folds", "seed", "n_train", "n_test",
                "budget", "n_proto")
_FLOAT_COLUMNS = ("budget_fraction_nominal", "budget_fraction",
                  "reduction", "cpu_select", "cpu_tuning", "cpu_total",
                  CURVE_METRIC)
_BOOL_COLUMNS = ("swept", "budget_floored", "shared_build")


def _typed(row):
    """Restore the types ``write_csv`` flattened, so a shard read back
    from disk aggregates as the rows did in memory.  (Sparse provenance
    columns come back as empty strings on rows that never carried them;
    nothing downstream reads those.)"""
    out = dict(row)
    for k in _INT_COLUMNS:
        if k in out and out[k] != "":
            out[k] = int(out[k])
    for k in _FLOAT_COLUMNS:
        if k in out and out[k] != "":
            out[k] = float(out[k])
    for k in _BOOL_COLUMNS:
        if k in out:
            if out[k] not in ("True", "False"):
                raise ValueError(
                    "column {0} holds {1!r}, not True/False".format(
                        k, out[k]))
            out[k] = out[k] == "True"
    return out


def shard_path(out_dir, dataset_name, folds):
    shard = ("all" if folds is None
             else "-".join(str(f) for f in sorted(folds)))
    return os.path.join(out_dir, "curve_folds__{0}__folds_{1}.csv".format(
        dataset_name, shard))


def read_shards(out_dir):
    """Every per-dataset shard CSV in ``out_dir``, typed and checked.

    Two shards carrying the same (dataset, fold, method) -- e.g. an
    ``all`` shard and a re-run fold shard for one dataset, or the same
    fold run under two different declared grids -- would double-count
    that fold in every mean; refused with both file names.  (The triple
    legitimately repeats WITHIN one shard, once per grid point.)
    """
    paths = sorted(glob.glob(os.path.join(out_dir, "curve_folds__*.csv")))
    rows, seen = [], {}
    for path in paths:
        with open(path, newline="") as fh:
            for raw in csv.DictReader(fh):
                row = _typed(raw)
                key = (row["dataset"], row["fold"], row["method"])
                if seen.setdefault(key, path) != path:
                    raise ValueError(
                        "overlapping shards: {0} appears in both {1} and "
                        "{2}".format(key, seen[key], path))
                rows.append(row)
    return rows, paths


def _refuse_foreign_output(out_dir):
    """A ``curve_folds.csv`` from another instrument in ``out_dir`` would
    be overwritten while its per-dataset figures stayed behind, leaving
    a directory that mixes two protocols under one set of names."""
    path = os.path.join(out_dir, "curve_folds.csv")
    if not os.path.exists(path):
        return
    with open(path, newline="") as fh:
        header = next(csv.reader(fh), [])
    if "budget_fraction_nominal" not in header:
        raise SystemExit(
            "run_curve: {0} was written by a different instrument (no "
            "budget_fraction_nominal column); move that output away "
            "before writing here".format(path))


def combine(out_dir):
    """Regenerate the combined CSVs and every figure from the shards."""
    _refuse_foreign_output(out_dir)
    rows, paths = read_shards(out_dir)
    if not rows:
        raise SystemExit(
            "run_curve: no shard CSVs found in {0}".format(out_dir))
    agg = aggregate(rows)
    # the anchor's own fraction of the fold, averaged over folds, for the
    # per-dataset figures' tick
    anchor = {}
    for r in rows:
        # the fold's one anchor build is the reference row that is not a
        # copy (shared_build is False only at the first grid point)
        if r["method"] == ANCHOR and not r["shared_build"]:
            anchor.setdefault(r["dataset"], []).append(
                r["n_proto"] / float(r["n_train"]))
    for r in agg:
        if r["dataset"] in anchor:
            r["anchor_fraction"] = float(np.mean(anchor[r["dataset"]]))
    fold_path = write_csv(rows, os.path.join(out_dir, "curve_folds.csv"))
    mean_rows = [{k: v for k, v in r.items() if k != "anchor_fraction"}
                 for r in agg]
    mean_path = write_csv(mean_rows, os.path.join(out_dir, "curve_mean.csv"))
    figures = []
    for name in sorted({r["dataset"] for r in rows}):
        figures.append(plot_curve(
            agg, name, os.path.join(out_dir, "curve_{0}".format(name))))
    aggregate_base = os.path.join(out_dir, "curve_mean")
    try:
        figures.append(plot_aggregate(agg, aggregate_base))
    except UnequalCoverage as exc:
        # Not drawn, and no stale copy left to be mistaken for current.
        for ext in (".png", ".pdf"):
            if os.path.exists(aggregate_base + ext):
                os.remove(aggregate_base + ext)
        print("run_curve: {0}".format(exc))
    return fold_path, mean_path, figures, paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results_curve",
                    help="output directory (separate from results/ so "
                         "the campaign's completeness checks never see "
                         "curve CSVs)")
    ap.add_argument("--datasets", nargs="*", default=sorted(DATASETS),
                    choices=sorted(DATASETS))
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="outer-fold ids to run (default: all); an id the "
                         "dataset does not have is an error")
    ap.add_argument("--no-download", action="store_true",
                    help="require raw data to already exist locally")
    args = ap.parse_args()

    validate_fractions(BUDGET_CURVE_FRACTIONS)
    if not args.datasets:
        raise SystemExit("run_curve: --datasets selected nothing")
    if args.folds is not None and not args.folds:
        raise SystemExit("run_curve: --folds given without any fold id")
    folds = set(args.folds) if args.folds else None
    os.makedirs(args.out_dir, exist_ok=True)
    _refuse_foreign_output(args.out_dir)

    # Fold ids are checked against every requested dataset BEFORE any
    # compute, so a typo cannot surface hours in.
    for name in args.datasets:
        _folds, k = outer_folds_for(name, args.data_dir,
                                    download=not args.no_download)
        validate_folds(folds, k, name)

    for name in args.datasets:
        t0 = time.time()
        rows = run_dataset_curve(
            name, args.data_dir, folds=folds,
            download=not args.no_download)
        path = write_csv(rows, shard_path(args.out_dir, name, folds))
        print("{0}: {1} rows in {2:.1f}s wall -> {3}".format(
            name, len(rows), time.time() - t0, path))

    try:
        fold_path, mean_path, figures, paths = combine(args.out_dir)
    except ValueError as exc:      # overlapping shards: the CSVs are safe
        raise SystemExit("run_curve: {0}".format(exc))
    print("combined {0} shard file(s)".format(len(paths)))
    print("per-fold rows -> {0}".format(fold_path))
    print("fold means    -> {0}".format(mean_path))
    for fig in figures:
        print("figure        -> {0} (+ .pdf)".format(fig))


if __name__ == "__main__":
    main()
