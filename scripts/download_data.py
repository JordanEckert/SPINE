#!/usr/bin/env python3
"""Download and verify the full dataset roster (hard-fail).

Every dataset is downloaded from its declared source, parsed, 
and verified against the manifest in ``src/harness/config.py``.  
Any failure stops the script with the offending dataset named.

Prints the dataset table for the manuscript, including the imbalance
ratio computed from the loaded labels.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np

from harness.config import DATASETS, N_OUTER_FOLDS
from harness.datasets import (imbalance_ratio, load_dataset,
                              load_keel_folds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--datasets", nargs="*", default=sorted(DATASETS),
                    help="subset of dataset names (default: all)")
    args = ap.parse_args()

    header = "{0:<20}{1:>9}{2:>7}{3:>9}{4:>8}{5:>8}".format(
        "dataset", "n", "d", "classes", "IR", "folds")
    print(header)
    print("-" * len(header))
    for name in args.datasets:
        spec = DATASETS[name]
        if spec["source"] == "keel" and not spec.get("own_folds"):
            partitions, k, labels = load_keel_folds(
                name, args.data_dir, download=True)
            X_tr, y_tr, X_te, y_te = partitions[0]
            X = np.vstack([X_tr, X_te])
            y = np.concatenate([y_tr, y_te])
            foldnote = str(k)
        else:
            X, y, labels = load_dataset(name, args.data_dir, download=True)
            foldnote = "{0}*".format(N_OUTER_FOLDS)
        print("{0:<20}{1:>9}{2:>7}{3:>9}{4:>8.1f}{5:>8}".format(
            name, len(X), X.shape[1], len(labels),
            imbalance_ratio(y), foldnote))
    print("\nAll datasets downloaded and verified against the manifest.")
    print("folds: KEEL partition count; * = own seeded stratified k-fold.")


if __name__ == "__main__":
    main()
