#!/usr/bin/env python3
"""Fixed-configuration computational study: construction-time profiles.

This script times PROTOTYPE CONSTRUCTION alone, at one declared 
configuration per method, under a seeded stratified 5-fold outer CV per dataset.  

SPINE, the method under test, is timed at its declared defaults. It appears 
as a SINGLE series even though the campaign scores it two ways, because both
scored rows come from one fit -- the skeleton decision rule changes what
prediction costs, not what construction costs, and prediction is not
timed here.

This is a MATCHED-BUDGET timing study throughout: every
series constructs the same number of prototypes on a given fold, so
their runtime differences are the algorithms' and not the output sizes'.
RSP3 is trivially matched -- its own count IS the budget.  ``n_proto``
is recorded on every row so the condition is auditable rather than
assumed.  Because the budget is RSP3's count and that grows with the
fold size, the matched series conflate "cost grows with n" with "cost
grows with the budget"; the recorded ``n_proto`` column is what lets a
reader separate them.

One asymmetry is stated rather than corrected: SPINE reserves a
stratified validation split, so it constructs from 75% of the training
fold while every comparator sees all of it.  Correcting for that would
mean handing SPINE a bigger fold than the protocol gives it, which would
no longer be the method whose accuracy is reported; the recorded
``n_train`` is the whole fold for every series, and the split is a
property of the method being timed.

Timing is process-CPU seconds of ``select()`` only, the repository's
timing convention; normalization sits outside the timer and no
classifier runs at all.  BLAS/OpenMP threads are capped at
``--blas-threads`` (default 1) before numpy is imported, because
``process_time`` sums CPU across threads and the methods thread to
different degrees; the setting is recorded on every row.

Outputs, written to ``--out-dir`` (kept separate from ``results/`` so
the accuracy campaign's completeness checks never see them): a per-fold
CSV, a per-(dataset, method) mean CSV, and a log-log figure of mean
construction time against training-fold size, one series per method.

Example::

    python scripts/run_computational.py --datasets wine banana letter
"""

import argparse
import csv
import os
import sys
import time

#: BLAS/OpenMP threads per process.  ``time.process_time`` sums CPU
#: across threads, so an uncapped run inflates every timing by roughly
#: the thread count -- and by a DIFFERENT factor per method, since the
#: methods thread to different degrees (K-Means heavily, the sequential
#: constructions such as LVQ3 barely).
#: This is the study that produces the paper's construction-cost figure,
#: so an uncapped run would plot parallelism rather than algorithms.
#: Same five variables and the same default of 1 as
#: run_experiments_parallel.py.
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS")
DEFAULT_BLAS_THREADS = 1


def _blas_threads_from_argv(argv):
    """Read --blas-threads before argparse can run.

    The caps must be in the environment BEFORE numpy -- and therefore
    the BLAS backend -- is imported, and numpy is imported at module
    scope here (unlike run_experiments_parallel.py, which imports it
    only inside its workers and can set the caps in main()).  So the
    flag is scanned off ``sys.argv`` directly; ``main`` re-parses it
    with argparse and asserts the two agree.
    """
    for i, arg in enumerate(argv):
        if arg == "--blas-threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith("--blas-threads="):
            return int(arg.split("=", 1)[1])
    return DEFAULT_BLAS_THREADS


BLAS_THREADS = _blas_threads_from_argv(sys.argv[1:])
if BLAS_THREADS < 1:
    raise SystemExit("--blas-threads must be >= 1")
for _var in BLAS_THREAD_VARS:
    os.environ[_var] = str(BLAS_THREADS)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np
from sklearn.model_selection import StratifiedKFold

from baselines.glvq import GLVQ
from baselines.gng import GNG
from baselines.lvq3 import LVQ3
from baselines.reference import KMeansPerClass
from baselines.rsp3 import RSP3
from baselines.spotgreedy import SPOTGreedy
from harness.config import (BASE_SEED, DATASETS,
                            LVQ3_TUNING_GRID, NORMALIZE)
from harness.config import make_scaler
from harness.base import allocate_per_class
from harness.datasets import load_dataset
from harness.runner import fold_seed, grid_combinations
from spine import SPINE

#: Outer timing folds (5-fold, seeded stratified; the accuracy
#: campaign's 10-fold KEEL partitions are not needed here because no
#: accuracy is recorded).
N_TIMING_FOLDS = 5

#: The method whose per-fold prototype count sets the budgets of the
#: budget-matched comparators -- the campaign's anchor, reproduced.
#: Parameter-free, so it needs no fixed configuration of its own.
BUDGET_METHOD = "RSP3"

#: LVQ3 placement hyperparameters: the first declared grid combination
#: (the timing study tunes nothing).
LVQ3_FIXED = next(grid_combinations(LVQ3_TUNING_GRID))

#: Plot series: label, hex, marker, linestyle.  SPINE carries the
#: primary categorical hue as the method under test, and GLVQ -- the
#: comparator that shares its discriminative objective -- carries the
#: second, so the pair a reader compares first is the pair that differs
#: in colour.  The remaining baselines are the context layer in the two
#: ink grays, distinguished by marker and dash.
PLOT_SERIES = (
    ("SPINE", "#2a78d6", "o", "-"),
    ("GLVQ", "#eb6834", "s", "-"),
    ("RSP3", "#52514e", "v", "--"),
    ("KMeans", "#52514e", "X", ":"),
    ("LVQ3", "#898781", "*", ":"),
    ("SPOT", "#52514e", "d", "-."),
    ("GNG", "#898781", "^", "-"),
)


#: Set by :func:`warm_up`; read by the test that pins it ran.
_WARMED = False


def warm_up():
    """Pay every library's first-call cost BEFORE any timed region.

    ``sklearn.metrics.pairwise_distances_argmin_min`` carries roughly
    30 ms of one-time setup that has nothing to do with any method's
    algorithm, and whichever method reaches it first in a process is
    charged the whole of it.  The cost is paid once per PROCESS, so it
    lands entirely on fold 0 of the first dataset timed, and being a
    fixed ~30 ms it is a far larger fraction of a small-n build than of a
    large-n one -- which biases the log-log slope the figure is read for,
    not merely its intercept.  Measured on wine before this existed, the
    first timed method's 5-fold mean was inflated by roughly a tenth, and
    SPOT's by about 14%.

    The fix is to touch every library the study uses, on tiny arrays,
    outside every timer, once per process.  SPINE is called too rather
    than only the sklearn primitives, because its Phase 0 pulls in SciPy
    and the Mapper construction whose import-and-first-call costs would
    otherwise land on the first timed fold as well.
    """
    global _WARMED
    if _WARMED:
        return
    from sklearn.cluster import KMeans
    from sklearn.metrics import (pairwise_distances,
                                 pairwise_distances_argmin_min)
    from scipy.cluster.hierarchy import linkage

    Xw = np.linspace(0.0, 1.0, 24).reshape(12, 2)
    yw = np.array([0] * 6 + [1] * 6)
    pairwise_distances_argmin_min(Xw, Xw[:3], metric="sqeuclidean")
    pairwise_distances(Xw, Xw, metric="euclidean")
    linkage(Xw, method="ward")
    KMeans(n_clusters=2, n_init=1, random_state=0).fit(Xw)
    SPINE().select(Xw, yw, total_count=4, random_state=0)
    _WARMED = True


def _timed_select(method_or_gen, X_tr, y_tr, **params):
    """One construction under the process-CPU timer."""
    t0 = time.process_time()
    X_proto, y_proto, _indices = method_or_gen.select(X_tr, y_tr, **params)
    cpu = time.process_time() - t0
    return np.asarray(X_proto), np.asarray(y_proto), cpu


def time_dataset(name, X, y, normalize=NORMALIZE,
                 n_folds=N_TIMING_FOLDS):
    """Time every method's construction on one dataset.

    Returns one row dict per (fold, method): dataset, fold, method,
    n_train, n_proto, cpu_seconds.  Any method failure propagates --
    a dataset a method cannot run on is a protocol decision, not a
    row to skip silently.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    warm_up()
    rows = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=BASE_SEED)
    for fold_id, (tr, _te) in enumerate(skf.split(X, y)):
        seed = fold_seed(name, fold_id)
        if normalize:
            X_tr = make_scaler().fit_transform(X[tr])
        else:
            X_tr = np.array(X[tr], dtype=float)
        y_tr = y[tr]

        def _record(method_label, n_proto, cpu):
            rows.append({
                "dataset": name, "fold": fold_id,
                "method": method_label, "n_train": len(X_tr),
                "n_proto": int(n_proto),
                "cpu_seconds": float(cpu),
                "blas_threads": BLAS_THREADS,
            })

        # The anchor first: its own count is the matched total, exactly
        # as in the campaign.
        Xp, _yp, cpu = _timed_select(RSP3(), X_tr, y_tr)
        _record(BUDGET_METHOD, len(Xp), cpu)
        total_count = int(len(Xp))

        # Proportional apportionment of that total over the training
        # class frequencies -- from the data, not from any method.
        class_counts = allocate_per_class(y_tr, total_count)

        # SPINE is timed ONCE per fold even though the campaign scores it
        # twice: both scored rows come from this one construction, and
        # the second differs only in how the fitted model predicts.
        comparators = (
            ("SPINE", SPINE(),
             {"total_count": total_count, "random_state": seed}),
            ("KMeans", KMeansPerClass(),
             {"class_counts": class_counts, "random_state": seed}),
            ("LVQ3", LVQ3(**LVQ3_FIXED),
             {"total_count": total_count, "random_state": seed}),
            ("SPOT", SPOTGreedy(), {"total_count": total_count}),
            ("GLVQ", GLVQ(),
             {"total_count": total_count, "random_state": seed}),
            ("GNG", GNG(),
             {"total_count": total_count, "random_state": seed}),
        )
        for label, method, params in comparators:
            Xp, _yp, cpu = _timed_select(method, X_tr, y_tr, **params)
            _record(label, len(Xp), cpu)
    return rows


def aggregate(rows):
    """Mean over folds per (dataset, method).

    Returns row dicts: dataset, method, n_train (mean), n_proto
    (mean), cpu_seconds (mean over the timing folds).
    """
    keys = sorted({(r["dataset"], r["method"]) for r in rows})
    out = []
    for dataset, method in keys:
        sub = [r for r in rows
               if r["dataset"] == dataset and r["method"] == method]
        out.append({
            "dataset": dataset, "method": method,
            "n_train": float(np.mean([r["n_train"] for r in sub])),
            "n_proto": float(np.mean([r["n_proto"] for r in sub])),
            "cpu_seconds": float(np.mean([r["cpu_seconds"]
                                          for r in sub])),
            "blas_threads": BLAS_THREADS,
        })
    return out


def plot_loglog(agg, out_base):
    """One log-log figure: mean construction time vs training size.

    ``agg`` is the :func:`aggregate` output over any number of
    datasets; every method becomes one series ordered by training
    size.  Writes ``<out_base>.png`` and ``<out_base>.pdf``; returns
    the PNG path.  A non-positive mean time is a hard error (a log
    axis cannot show it, and it would mean the timer resolved nothing).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for r in agg:
        if not r["cpu_seconds"] > 0.0:
            raise ValueError(
                "non-positive mean construction time for {0!r} on "
                "{1!r}: a log-log axis cannot display it".format(
                    r["method"], r["dataset"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, color, marker, linestyle in PLOT_SERIES:
        pts = sorted(((r["n_train"], r["cpu_seconds"]) for r in agg
                      if r["method"] == label))
        if not pts:
            continue
        xs, ys = zip(*pts)
        # Every series is at the anchor's per-fold budget, so the
        # legend needs no operating-point qualifier.
        ax.plot(xs, ys, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.6, markersize=6,
                markeredgecolor="white", markeredgewidth=0.5,
                label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training-fold size $n$ (rows)", color="#52514e")
    ax.set_ylabel("Mean construction time (process-CPU s)",
                  color="#52514e")
    ax.set_title("Prototype construction time at fixed configurations",
                 color="#0b0b0b")
    # The matched-budget condition is a property of the whole study, so
    # it belongs on the figure rather than only in the caption.
    ax.annotate(
        "budget matched to {0}'s prototype count per fold; "
        "{1} BLAS thread(s)".format(BUDGET_METHOD, BLAS_THREADS),
        xy=(0.0, 1.02), xycoords="axes fraction", fontsize=8,
        color="#52514e")
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    # ``ax.spines`` are matplotlib's axis borders; the loop variable is
    # named for them rather than ``spine``, which in this module now
    # means the method under test.
    for border in ax.spines.values():
        border.set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(ncols=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=200)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    return out_base + ".png"


def write_csv(rows, path):
    """Write the rows; an empty set is a loud stop, not a header file.

    Reaching here with nothing to write means the run selected no
    datasets, and the compute has already been spent.  A header-only CSV
    would be indistinguishable from a finished study whose methods all
    produced nothing, so this refuses instead -- and ``main`` checks the
    dataset list up front so the usual way of getting here costs no time
    at all.
    """
    if not rows:
        raise SystemExit(
            "run_computational: no rows to write to {0} -- the run "
            "produced no timed builds (was --datasets given an empty "
            "list?)".format(path))
    fieldnames = list(rows[0])
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results_computational",
                    help="output directory (separate from results/ so "
                         "the accuracy campaign's completeness checks "
                         "never see timing CSVs)")
    ap.add_argument("--datasets", nargs="*", default=sorted(DATASETS),
                    choices=sorted(DATASETS),
                    help="datasets to time (default: all)")
    ap.add_argument("--blas-threads", type=int,
                    default=DEFAULT_BLAS_THREADS,
                    help="BLAS/OpenMP threads per process (default 1): "
                         "process_time sums CPU across threads, so an "
                         "uncapped run inflates timings by roughly the "
                         "thread count, unequally across methods. "
                         "Applied at import time, before numpy loads.")
    ap.add_argument("--no-download", action="store_true",
                    help="require raw data to already exist locally")
    args = ap.parse_args()
    if args.blas_threads != BLAS_THREADS:
        # Only reachable if the argv scan and argparse disagree, which
        # would mean the caps in os.environ are not the ones reported.
        raise SystemExit(
            "--blas-threads parsed as {0} but {1} was applied before "
            "numpy was imported; pass it as '--blas-threads N'".format(
                args.blas_threads, BLAS_THREADS))

    if not args.datasets:
        raise SystemExit(
            "run_computational: --datasets selected nothing; there is "
            "no timing study to run")

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []
    for name in args.datasets:
        X, y, _labels = load_dataset(name, args.data_dir,
                                     download=not args.no_download)
        t0 = time.time()
        rows = time_dataset(name, X, y)
        all_rows.extend(rows)
        print("{0}: {1} timed builds in {2:.1f}s wall".format(
            name, len(rows), time.time() - t0))

    agg = aggregate(all_rows)
    fold_path = write_csv(all_rows,
                          os.path.join(args.out_dir, "timing_folds.csv"))
    mean_path = write_csv(agg,
                          os.path.join(args.out_dir, "timing_mean.csv"))
    fig_path = plot_loglog(agg,
                           os.path.join(args.out_dir, "timing_loglog"))
    print("per-fold timings -> {0}".format(fold_path))
    print("fold means       -> {0}".format(mean_path))
    print("log-log figure   -> {0} (+ .pdf)".format(fig_path))


if __name__ == "__main__":
    main()
