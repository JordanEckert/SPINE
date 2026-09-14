#!/usr/bin/env python3
"""Optional parallel campaign driver: the experiments sharded over (dataset, fold).

One worker process per (dataset, outer fold). Each shard is exactly one
``run_dataset(..., folds={fold})`` call and writes the same per-shard
CSV that a manual ``run_experiments.py --datasets D --folds f`` run
would, so the outputs compose with ``run_stats.py`` unchanged (its
duplicate-shard refusal still arbitrates a mixed campaign). Shard CSVs
that already exist are skipped, which makes an interrupted campaign
resumable by re-invoking the same command.

Data is prefetched sequentially in the parent before any worker starts
(concurrent downloads of the same archive would race); workers then run
strictly offline. Per-dataset fold counts come from the prefetch, so
datasets with only a 5-fold KEEL partition get exactly their real
shards and no empty CSVs. BLAS/OpenMP threading is capped per worker
(default 1 thread) so J processes never fan out into J x T threads;
shards are dispatched largest dataset first to shorten the makespan.

A failed shard does not stop the others: every failure is reported at
the end and the exit status is nonzero. No partial CSV is left behind
for a failed shard (the runner writes its CSV only after the fold
completes), so a plain rerun retries exactly the failures.

Examples::

    python scripts/run_experiments_parallel.py --jobs 8
    python scripts/run_experiments_parallel.py --datasets letter magic \
        --jobs 4 --classifiers 1nn svm mlp
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

# Constants only -- keep every numpy-importing module out of the parent
# until AFTER the BLAS thread caps are in the environment (main()).
from harness.config import (DATASETS, MAIN_METHODS,
                            N_OUTER_FOLDS, NORMALIZE)


def _prefetch(dataset_names, data_dir, download):
    """Download/verify every dataset once, sequentially, in the parent.

    Returns ``{dataset: [outer fold ids]}`` from each dataset's real
    partition (KEEL's published 10- or 5-fold count where applicable).
    Any download or shape failure aborts here, before workers start.
    """
    from harness.datasets import load_dataset, load_keel_folds

    plan = {}
    for name in dataset_names:
        spec = DATASETS[name]
        if spec["source"] == "keel" and not spec.get("own_folds"):
            _folds, k, _labels = load_keel_folds(name, data_dir,
                                                 download=download)
            plan[name] = list(range(k))
        else:
            load_dataset(name, data_dir, download=download)
            plan[name] = list(range(N_OUTER_FOLDS))
    return plan


def _run_shard(name, fold_id, data_dir, results_dir, methods, classifiers):
    """One (dataset, fold) shard in a worker process."""
    from harness.runner import run_dataset

    t0 = time.time()
    out_path, rows = run_dataset(
        name, data_dir, results_dir, methods=methods,
        classifiers=classifiers, folds={fold_id}, download=False,
        normalize=NORMALIZE)
    return out_path, len(rows), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results_parallel_run")
    ap.add_argument("--datasets", nargs="*", default=sorted(DATASETS),
                    choices=sorted(DATASETS),
                    help="datasets to run (default: all)")
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="outer-fold ids to run (default: every fold "
                         "the dataset's partition actually has)")
    ap.add_argument("--methods", nargs="*", default=list(MAIN_METHODS),
                    choices=list(MAIN_METHODS),
                    help="methods to run ('rsp3' is always required: "
                         "its per-fold prototype count is the budget "
                         "every budget-matched method is held to, so a "
                         "run without it has no budget at all; 'spine' "
                         "is the method under test and emits two rows, "
                         "SPINE+1NN and SPINE+Graph, from one fit)")
    ap.add_argument("--classifiers", nargs="*", default=["1nn"],
                    choices=["1nn", "svm", "mlp"])
    ap.add_argument("--jobs", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="worker processes (default: cpu count - 1)")
    ap.add_argument("--blas-threads", type=int, default=1,
                    help="BLAS/OpenMP threads per worker (default 1: "
                         "process parallelism owns the cores)")
    ap.add_argument("--no-download", action="store_true",
                    help="require raw data to already exist locally")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run shards whose CSV already exists "
                         "(default: skip them, i.e. resume)")
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.blas_threads < 1:
        raise SystemExit("--blas-threads must be >= 1")

    # Thread caps go into the environment BEFORE any numpy import in
    # this process (the prefetch below) and are inherited by the
    # spawned workers.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.blas_threads)

    plan = _prefetch(args.datasets, args.data_dir,
                     download=not args.no_download)

    requested = set(args.folds) if args.folds is not None else None
    units = []
    for name in args.datasets:
        for fold_id in plan[name]:
            if requested is not None and fold_id not in requested:
                continue
            units.append((name, fold_id))
    # Largest datasets first: the long shards start immediately instead
    # of trailing the campaign.
    units.sort(key=lambda u: (-DATASETS[u[0]]["n"], u[0], u[1]))

    os.makedirs(args.results_dir, exist_ok=True)
    todo, skipped = [], []
    for name, fold_id in units:
        out_csv = os.path.join(
            args.results_dir, "{0}__folds_{1}.csv".format(name, fold_id))
        if os.path.exists(out_csv) and not args.overwrite:
            skipped.append((name, fold_id))
        else:
            todo.append((name, fold_id))

    print("{0} shard(s) total; {1} already on disk (skipped); {2} to "
          "run on {3} worker(s), {4} BLAS thread(s) each".format(
              len(units), len(skipped), len(todo), args.jobs,
              args.blas_threads))
    if not todo:
        print("nothing to do")
        return

    t_start = time.time()
    failures = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs,
                             mp_context=ctx) as pool:
        futures = {
            pool.submit(_run_shard, name, fold_id, args.data_dir,
                        args.results_dir, tuple(args.methods),
                        tuple(args.classifiers)): (name, fold_id)
            for name, fold_id in todo
        }
        n_done = 0
        for fut in as_completed(futures):
            name, fold_id = futures[fut]
            n_done += 1
            err = fut.exception()
            if err is not None:
                failures.append((name, fold_id, err))
                print("[{0}/{1}] FAILED {2} fold {3}: {4!r}".format(
                    n_done, len(todo), name, fold_id, err))
            else:
                out_path, n_rows, wall = fut.result()
                print("[{0}/{1}] {2} fold {3}: {4} rows in {5:.1f}s "
                      "-> {6}".format(n_done, len(todo), name, fold_id,
                                      n_rows, wall, out_path))

    print("campaign wall time: {0:.1f}s".format(time.time() - t_start))
    if failures:
        lines = "\n".join(
            "  {0} fold {1}: {2!r}".format(n, f, e)
            for n, f, e in failures)
        raise SystemExit(
            "{0} shard(s) failed (rerun the same command to retry "
            "exactly these):\n{1}".format(len(failures), lines))


if __name__ == "__main__":
    main()
