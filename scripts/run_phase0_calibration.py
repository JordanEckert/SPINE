#!/usr/bin/env python3
"""Phase 0 alone on the evaluation suite: the sigma_0 calibration.

Section 2.4 sets the Phase 1 neighbourhood width from the size of the
nerves Phase 0 builds: applied to every dataset in the suite, the median
vertex count per class is six, so the neural-gas rule 0.2 m gives 1.2
and sigma_0 = 2 is taken above it.  This script reproduces that number
without running Phases 1-4.

Each class-fold sees exactly the input Phase 0 saw in the campaign: the
same outer folds as ``harness.runner.run_dataset``, the scaler fit on
the training fold, and SPINE's own seeded stratified split (the first
draw from the fold's rng, as in ``SPINE.select``), with Phase 0 applied
per class to the fitting portion only.  No test partition and no
accuracy enter, so the result is a structural calibration, not a fit.

The per-class-fold table is written beside nothing else: keep it out of
``results_run/``, whose every ``*.csv`` is loaded as a runner shard.

Example
-------
::

    python scripts/run_phase0_calibration.py --out results_phase0/phase0_vertex_counts.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np
from sklearn.model_selection import StratifiedKFold

from harness.config import BASE_SEED, DATASETS, N_OUTER_FOLDS, make_scaler
from harness.datasets import load_dataset, load_keel_folds
from harness.runner import fold_seed
from spine.model import SPINE
from spine.phase0 import phase0

FIELDS = ["dataset", "fold", "cls", "n_class", "n_intervals",
          "min_cluster_size", "placement", "n_vertices", "n_edges",
          "degenerate", "noise_reassigned"]


def outer_folds(name, data_dir, download=True):
    """The campaign's outer folds, chosen as ``run_dataset`` chooses them."""
    spec = DATASETS[name]
    if spec["source"] == "keel" and not spec.get("own_folds"):
        folds, _k, _labels = load_keel_folds(name, data_dir,
                                             download=download)
        return folds
    X, y, _labels = load_dataset(name, data_dir, download=download)
    skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                          random_state=BASE_SEED)
    return [(X[tr], y[tr], X[te], y[te]) for tr, te in skf.split(X, y)]


def phase0_rows(name, fold_id, X_train, y_train):
    """One row per class: Phase 0 on that class's fitting portion."""
    spine = SPINE()
    X = make_scaler().fit_transform(X_train)
    classes = np.unique(y_train)
    y_pos = np.searchsorted(classes, y_train)
    rng = np.random.default_rng(fold_seed(name, fold_id))
    train_idx, _val_idx = spine._split(X, y_pos, rng)
    Xt, yt = X[train_idx], y_pos[train_idx]

    rows = []
    for c in range(len(classes)):
        V, E, prov = phase0(Xt[yt == c], len(Xt), gain=spine.gain,
                            lens=spine.lens,
                            n_neighbors=spine.n_neighbors,
                            lens_scaling=spine.lens_scaling)
        rows.append({
            "dataset": name,
            "fold": fold_id,
            "cls": c,
            "n_class": prov["n_class"],
            "n_intervals": prov["n_intervals"],
            "min_cluster_size": prov["min_cluster_size"],
            "placement": prov["placement"],
            "n_vertices": len(V),
            "n_edges": len(E),
            "degenerate": "degenerate" in prov,
            "noise_reassigned": prov["noise_reassigned"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=os.path.join(
        "results_phase0", "phase0_vertex_counts.csv"))
    ap.add_argument("--datasets", nargs="*", default=sorted(DATASETS),
                    choices=sorted(DATASETS),
                    help="datasets to run (default: all)")
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="outer-fold ids to run (default: all 10)")
    ap.add_argument("--no-download", action="store_true",
                    help="require raw data to already exist locally")
    args = ap.parse_args()

    rows = []
    for name in args.datasets:
        folds = outer_folds(name, args.data_dir,
                            download=not args.no_download)
        for fold_id, (X_tr, y_tr, _X_te, _y_te) in enumerate(folds):
            if args.folds is not None and fold_id not in args.folds:
                continue
            rows.extend(phase0_rows(name, fold_id, X_tr, y_tr))
        print("{0}: done".format(name), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row["n_vertices"])
    print("\nPhase 0 vertices per class (median over class-folds):")
    for name in sorted(by_dataset):
        print("  {0:<15} {1:>5}".format(
            name, float(np.median(by_dataset[name]))))
    counts = [row["n_vertices"] for row in rows]
    print("\nmedian vertex count per class: {0} over {1} class-folds "
          "(mean {2:.2f}, {3} degenerate)".format(
              float(np.median(counts)), len(counts), np.mean(counts),
              sum(row["degenerate"] for row in rows)))
    print("wrote {0}".format(args.out))


if __name__ == "__main__":
    main()
