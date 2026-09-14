#!/usr/bin/env python3
"""Run the experimental campaign.

One invocation runs one or more datasets sequentially; a campaign is
parallelised by sharding invocations across (dataset, fold) with
``--datasets`` and ``--folds`` (each shard writes its own CSV, and
``scripts/run_stats.py`` refuses duplicated shards).

Examples
--------
Full run on one dataset (all 10 folds, 1-NN only)::

    python scripts/run_experiments.py --datasets wine

One fold shard with the generalization classifiers::

    python scripts/run_experiments.py --datasets magic --folds 0 1 \
        --classifiers 1nn svm mlp
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from harness.config import DATASETS, MAIN_METHODS, NORMALIZE
from harness.runner import run_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results_run")
    ap.add_argument("--datasets", nargs="*", default=sorted(DATASETS),
                    choices=sorted(DATASETS),
                    help="datasets to run (default: all)")
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="outer-fold ids to run (default: all 10)")
    ap.add_argument("--methods", nargs="*", default=list(MAIN_METHODS),
                    choices=list(MAIN_METHODS),
                    help="methods to run ('rsp3' is always required: "
                         "its per-fold prototype count is the budget "
                         "every budget-matched method is held to, so a "
                         "run without it has no budget at all; 'spine' "
                         "is the method under test and emits two rows, "
                         "SPINE+1NN and SPINE+Graph, from one fit)")
    ap.add_argument("--classifiers", nargs="*", default=["1nn"],
                    choices=["1nn", "svm", "mlp"],
                    help="classifier set (svm/mlp = generalization "
                         "section)")
    ap.add_argument("--no-download", action="store_true",
                    help="require raw data to already exist locally")
    args = ap.parse_args()

    for name in args.datasets:
        out_path, rows = run_dataset(
            name, args.data_dir, args.results_dir,
            methods=tuple(args.methods),
            classifiers=tuple(args.classifiers),
            folds=(set(args.folds) if args.folds else None),
            download=not args.no_download, normalize=NORMALIZE)
        print("{0}: {1} rows -> {2}".format(name, len(rows), out_path))


if __name__ == "__main__":
    main()
