"""Experiment runner.

Per (dataset, outer fold):

1. Scale features with config.make_scaler (fit on the training fold
   only),
   unless ``normalize=False`` -- the raw-features ablation, which is an
   API-level switch rather than a script.
2. Run the budget anchor RSP3 first.  RSP3 is parameter-free and
   deterministic, so its prototype count is a property of the training
   fold alone -- no tuning, no seed, nothing selected on the data.
   That count is the TOTAL budget of every budget-matched method.
3. Split that total across classes by PROPORTIONAL apportionment of
   the training-fold class frequencies (``base.allocate_per_class``),
   and hand the result to the per-class comparators.  The split is a
   property of the DATASET, not of any method in the table: deriving
   it from a competing method would let that method's idea of a good
   class allocation flow into its comparators, which is neither what a
   practitioner would do nor a comparison anyone should have to defend.
4. Run every method: NOP (full set), SPINE (the method under test),
   K-Means and stratified Random at the proportional per-class counts,
   and LVQ3, SPOTGreedy, GLVQ and GNG at the anchor's total.  LVQ3's window/epsilon are selected by inner
   5-fold CV at the matched budget.

   SPINE emits TWO rows from ONE fit.  ``SPINE+1NN`` is its vertices
   scored by sklearn's 1-NN, exactly like any other prototype set --
   which is rung 1 of the ablation ladder, the model with its segments
   removed at prediction time.  ``SPINE+Graph`` is the same fitted model
   under its own decision rule, ``argmin_c dist(x, |S_c|)``.  The two
   differ only in the decision rule; the method name is what
   disambiguates the ``acc_*`` column, which is why they are separate
   methods rather than separate columns.  ``cpu_select`` is the cost of
   building the model and is written identically on both rows, since one
   fit produced them and any split would be arbitrary. SPINE records its 
   construction provenance -- the lens and cover used per class, 
   Betti numbers at the nerve, at the Phase 2a freeze and at convergence, 
   the topology-retention pair, forced removals and the
   count of representation-driven growth steps -- in a ``spine_params``
   JSON column, so a finished campaign can say what every fold built.
5. Evaluate every method's prototype set with the requested
   classifiers on the test fold; record accuracy, reduction rate, and
   preprocessing CPU time.


Failure policy: any method error aborts the run loudly. In particular a
budget-matched method that cannot attain the anchor's budget (a
per-class count exceeding the class's training size) raises rather than
silently under-filling.
"""

import csv
import itertools
import json
import os
import time
import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from baselines.glvq import GLVQ
from baselines.gng import GNG
from baselines.lvq3 import LVQ3
from baselines.reference import FullSet, KMeansPerClass, RandomSubsample
from baselines.rsp3 import RSP3
from baselines.spotgreedy import SPOTGreedy
from spine.model import SPINE

from .base import allocate_per_class, validate_selection
from .config import (BASE_SEED, BUDGET_ANCHOR, DATASETS, LVQ3_TUNING_GRID,
                     MAIN_METHODS, N_INNER_FOLDS, N_OUTER_FOLDS,
                     PRIMARY_CLASSIFIER, make_scaler)
from .datasets import load_dataset, load_keel_folds
from .metrics import evaluate_prototypes, reduction_rate


#: The two rows SPINE emits per fold, from one fitted model.
SPINE_POINTS = "SPINE+1NN"
SPINE_GRAPH = "SPINE+Graph"


def fold_seed(dataset_name, fold_id):
    """Deterministic per-(dataset, fold) seed, stable across runs.

    Derived from BASE_SEED and the dataset name via a fixed FNV-1a hash
    (not Python's salted ``hash``), so a rerun of any single fold
    reproduces the campaign bit-for-bit.
    """
    h = 2166136261
    for byte in dataset_name.encode("utf-8"):
        h = ((h ^ byte) * 16777619) % (1 << 32)
    return (BASE_SEED + h + 1000003 * fold_id) % (1 << 31)


def grid_combinations(grid):
    """Explicit-order combinations of a param grid (declared tie-break)."""
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def tune_by_grid(make_method, grid, X_train, y_train, seed,
                 n_inner=N_INNER_FOLDS, select_params=None):
    """Inner-CV tuning of LVQ3 on one training fold.

    ``make_method(combo)`` returns a fresh selector configured with the
    grid combination ``combo``; ``select_params`` are extra keyword
    arguments passed to every ``select`` call (e.g. the matched
    ``total_count`` for a budget-matched method, so tuning runs at the
    same budget as the final build).  Returns ``(best_combo, scores,
    cpu_tuning)``.  Criterion: highest mean 1-NN accuracy over the inner
    validation splits; ties resolve to the first combination in declared
    grid order.

    Degenerate-data rule (defined, loud in the log row): if the smallest
    class has fewer than 2 members no stratified split exists; the first
    grid combination is used without selection and ``scores`` is empty.
    """
    select_params = select_params or {}
    combos = list(grid_combinations(grid))
    if not combos:
        raise ValueError("empty tuning grid")

    _, counts = np.unique(y_train, return_counts=True)
    n_splits = int(min(n_inner, counts.min()))

    t0 = time.process_time()
    if n_splits < 2:
        return combos[0], {}, time.process_time() - t0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=seed)
    splits = list(skf.split(X_train, y_train))

    scores = {}
    best_combo, best_score = combos[0], -np.inf
    for combo in combos:
        accs = []
        for tr, va in splits:
            method = make_method(combo)
            X_proto, y_proto, _ = method.select(
                X_train[tr], y_train[tr], random_state=seed,
                **select_params)
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X_proto, y_proto)
            accs.append(float(clf.score(X_train[va], y_train[va])))
        mean_acc = float(np.mean(accs))
        scores[json.dumps(combo, sort_keys=True)] = mean_acc
        # Strict '>' keeps the FIRST maximum in declared grid order.
        if mean_acc > best_score:
            best_score, best_combo = mean_acc, combo
    cpu_tuning = time.process_time() - t0
    return best_combo, scores, cpu_tuning


_LIBRARIES_WARM = False


def _warm_up_libraries():
    """Pay sklearn's first-call setup before anything is timed."""
    global _LIBRARIES_WARM
    if _LIBRARIES_WARM:
        return
    from sklearn.metrics import (pairwise_distances,
                                 pairwise_distances_argmin_min)
    Xw = np.linspace(0.0, 1.0, 24).reshape(12, 2)
    pairwise_distances_argmin_min(Xw, Xw[:3], metric="sqeuclidean")
    pairwise_distances(Xw, Xw, metric="euclidean")
    _LIBRARIES_WARM = True


def _timed_select(method, X_train, y_train, **params):
    """Run one ``select`` under process-CPU timing + contract checks."""
    t0 = time.process_time()
    X_proto, y_proto, indices = method.select(X_train, y_train, **params)
    cpu = time.process_time() - t0
    X_proto = np.asarray(X_proto, dtype=float)
    y_proto = np.asarray(y_proto)
    validate_selection(method, X_train, y_train, X_proto, y_proto,
                       indices)
    return X_proto, y_proto, cpu


def run_fold(dataset_name, X_train, y_train, X_test, y_test, fold_id,
             methods=MAIN_METHODS, classifiers=("1nn",),
             n_outer_folds=N_OUTER_FOLDS, normalize=True):
    """Run every method on one outer fold; return one result row each."""
    seed = fold_seed(dataset_name, fold_id)

    # Imports are all at module scope, but a first CALL is not free:
    # sklearn's pairwise_distances_argmin_min carries ~31 ms of one-time
    # setup and the anchor is the first method to reach it, so fold 0 of a
    # shard charged that to whichever method happened to run first.  The
    # cpu_* columns are reference-only, but a systematically inflated
    # fold 0 is still misleading, so the cost is paid outside every
    # timer.
    _warm_up_libraries()

    if normalize:
        scaler = make_scaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        # raw features, no scaling; copy (like the scaled branch does) so
        # the two paths are symmetric and never alias the shared fold array
        X_tr = np.array(X_train, dtype=float)
        X_te = np.array(X_test, dtype=float)

    if BUDGET_ANCHOR not in methods:
        raise ValueError(
            "the protocol requires {0!r} in every run: its output is the "
            "total budget of every budget-matched method".format(
                BUDGET_ANCHOR))

    rows = []

    # ---- the anchor first: RSP3's own count is the total budget ------
    # RSP3 is parameter-free and deterministic, so this number depends
    # on the training fold alone -- nothing is tuned and no seed enters.
    Xp_anchor, yp_anchor, cpu_anchor = _timed_select(RSP3(), X_tr, y_train)
    total_count = int(len(Xp_anchor))

    # ---- the per-class split: PROPORTIONAL, from the data alone ------
    # A largest-remainder apportionment of the total over the training
    # class frequencies, with the >=1-per-class floor and the class-size
    # cap.  It is deliberately not read off any method's prototype
    # labels: a practitioner meeting a new dataset keeps its class
    # proportions, and letting the method under test dictate a
    # comparator's class allocation would hand it an advantage no
    # reviewer should have to take on trust.
    class_counts = allocate_per_class(y_train, total_count)
    # Recorded on the two rows it governs, so the allocation a result
    # was produced under is auditable from the CSV alone.
    _counts_json = json.dumps({str(c): int(k)
                               for c, k in class_counts.items()},
                              sort_keys=True)

    def _row(method_name, X_proto, y_proto, cpu, extra=None,
             classifiers=classifiers):
        row = {
            "dataset": dataset_name,
            "fold": fold_id,
            "n_outer_folds": n_outer_folds,
            "method": method_name,
            "seed": seed,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "n_proto": len(X_proto),
            "reduction": reduction_rate(len(X_proto), len(X_tr)),
            "cpu_select": cpu,
            "cpu_tuning": 0.0,
        }
        row.update(evaluate_prototypes(
            X_proto, y_proto, X_te, y_test, classifiers=classifiers,
            random_state=seed))
        if extra:
            row.update(extra)
        row["cpu_total"] = row["cpu_select"] + row["cpu_tuning"]
        return row

    def _check_budget(method_name, n_proto):
        """Budget-matched methods must attain the anchor's budget exactly.

        A shortfall (a per-class count exceeding that class's training
        size, capped inside the method) would silently break the
        reduction-rate equality the budget-matched comparison assumes --
        surfaced as a hard error to decide on, never absorbed.
        """
        if n_proto != total_count:
            raise AssertionError(
                "{0} attained {1} prototypes but the anchor-matched "
                "budget is {2} (dataset={3}, fold={4}); a per-class "
                "count exceeds a class's training size -- this needs a "
                "protocol decision, not a silent cap"
                .format(method_name, n_proto, total_count,
                        dataset_name, fold_id))

    for name in methods:
        if name == "rsp3":
            # Already built above: it is the anchor, not a re-run.
            rows.append(_row("RSP3", Xp_anchor, yp_anchor, cpu_anchor))
        elif name == "spine":
            spine = SPINE()
            Xp, yp, cpu = _timed_select(
                spine, X_tr, y_train, total_count=total_count,
                random_state=seed)
            _check_budget("SPINE", len(Xp))
            provenance = {"spine_params": json.dumps(
                spine.spine_params_, sort_keys=True, default=str)}
            # Row 1: the vertices, scored like any other prototype set.
            rows.append(_row(SPINE_POINTS, Xp, yp, cpu, extra=provenance))
            # Row 2: the SAME fitted model under the skeleton rule.  Built
            # WITHOUT _row's classifier pass: a skeleton is not a prototype
            # set, so fitting SVM or MLP to its vertices and labelling the
            # result "SPINE+Graph" would report a number the row does not
            # mean -- and _row would spend a full SVM+MLP fit producing it
            # only for the values to be discarded.
            graph_row = _row(SPINE_GRAPH, Xp, yp, cpu, extra=provenance,
                             classifiers=())
            if PRIMARY_CLASSIFIER in classifiers:
                graph_row["acc_" + PRIMARY_CLASSIFIER] = float(
                    accuracy_score(y_test,
                                   spine.estimator_.predict(X_te)))
            else:
                # The run asked only for classifiers that cannot be
                # applied to a skeleton.  Emitting an acc_1nn nobody asked
                # for would invent a column; emitting nothing lets the
                # completeness check in harness.stats refuse the
                # campaign, which is the correct loud failure.
                raise ValueError(
                    "SPINE+Graph is scored by the skeleton decision rule, "
                    "which is the {0!r} column; a run requesting only {1} "
                    "cannot score it. Include {0!r} in --classifiers."
                    .format(PRIMARY_CLASSIFIER, list(classifiers)))
            rows.append(graph_row)
        elif name == "full":
            Xp, yp, cpu = _timed_select(FullSet(), X_tr, y_train)
            rows.append(_row("Full", Xp, yp, cpu))
        elif name == "glvq":
            Xp, yp, cpu = _timed_select(
                GLVQ(), X_tr, y_train, total_count=total_count,
                random_state=seed)
            _check_budget("GLVQ", len(Xp))
            rows.append(_row("GLVQ", Xp, yp, cpu))
        elif name == "gng":
            Xp, yp, cpu = _timed_select(
                GNG(), X_tr, y_train, total_count=total_count,
                random_state=seed)
            _check_budget("GNG", len(Xp))
            rows.append(_row("GNG", Xp, yp, cpu))
        elif name == "kmeans":
            Xp, yp, cpu = _timed_select(
                KMeansPerClass(), X_tr, y_train,
                class_counts=class_counts, random_state=seed)
            _check_budget("KMeans", len(Xp))
            rows.append(_row("KMeans", Xp, yp, cpu,
                             extra={"class_counts": _counts_json}))
        elif name == "random":
            Xp, yp, cpu = _timed_select(
                RandomSubsample(), X_tr, y_train,
                class_counts=class_counts, random_state=seed)
            _check_budget("Random", len(Xp))
            rows.append(_row("Random", Xp, yp, cpu,
                             extra={"class_counts": _counts_json}))
        elif name == "lvq3":
            best_combo, _scores, lvq_tuning = tune_by_grid(
                lambda combo: LVQ3(**combo), LVQ3_TUNING_GRID,
                X_tr, y_train, seed,
                select_params={"total_count": total_count})
            Xp, yp, cpu = _timed_select(
                LVQ3(**best_combo), X_tr, y_train,
                total_count=total_count, random_state=seed)
            _check_budget("LVQ3", len(Xp))
            rows.append(_row(
                "LVQ3", Xp, yp, cpu,
                extra={"cpu_tuning": lvq_tuning,
                       "lvq3_params": json.dumps(best_combo,
                                                 sort_keys=True)}))
        elif name == "spotgreedy":
            Xp, yp, cpu = _timed_select(
                SPOTGreedy(), X_tr, y_train, total_count=total_count)
            _check_budget("SPOT", len(Xp))
            rows.append(_row("SPOT", Xp, yp, cpu))
        else:
            raise ValueError("unknown method {0!r}".format(name))
    return rows


def run_dataset(dataset_name, data_dir, results_dir,
                methods=MAIN_METHODS, classifiers=("1nn",),
                folds=None, download=True, normalize=True):
    """Run the evaluation on one dataset; write one CSV of result rows.

    Outer folds come from KEEL's published partitions for KEEL datasets
    (10-fold where available, else 5-fold); the one non-KEEL dataset
    (EEG Eye State) is split by a seeded stratified k-fold at the target
    fold count, matching KEEL's stratified-partition convention.
    ``folds`` restricts to a subset of outer-fold ids (for sharding a
    campaign across machines); the output file name records the shard.
    """
    spec = DATASETS[dataset_name]
    if spec["source"] == "keel" and not spec.get("own_folds"):
        outer_folds, k, _labels = load_keel_folds(
            dataset_name, data_dir, download=download)
        if k != N_OUTER_FOLDS:
            warnings.warn(
                "{0}: KEEL {1}-fold partition unavailable; using {2}-fold "
                "outer CV (recorded as n_outer_folds in the results)"
                .format(dataset_name, N_OUTER_FOLDS, k))
    else:
        X, y, _labels = load_dataset(dataset_name, data_dir,
                                     download=download)
        k = N_OUTER_FOLDS
        skf = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=BASE_SEED)
        outer_folds = [(X[tr], y[tr], X[te], y[te])
                       for tr, te in skf.split(X, y)]

    all_rows = []
    for fold_id, (X_tr, y_tr, X_te, y_te) in enumerate(outer_folds):
        if folds is not None and fold_id not in folds:
            continue
        all_rows.extend(run_fold(
            dataset_name, X_tr, y_tr, X_te, y_te, fold_id,
            methods=methods, classifiers=classifiers, n_outer_folds=k,
            normalize=normalize))

    os.makedirs(results_dir, exist_ok=True)
    shard = ("all" if folds is None
             else "-".join(str(f) for f in sorted(folds)))
    out_path = os.path.join(
        results_dir, "{0}__folds_{1}.csv".format(dataset_name, shard))
    fieldnames = sorted({k for row in all_rows for k in row})
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    return out_path, all_rows
