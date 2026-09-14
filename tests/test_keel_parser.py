"""KEEL .dat parser and fold-partition loader tests."""

import os
import zipfile

import numpy as np
import pytest

from harness.config import DATASETS
from harness.datasets import (load_keel_folds, parse_keel_dat,
                              _to_float)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "banana_head.dat")


def test_parse_real_banana_excerpt():
    with open(FIXTURE) as fh:
        text = fh.read()
    X_raw, y_raw, names, nominal = parse_keel_dat(text)
    assert names == ["At1", "At2"]
    assert nominal == [False, False]
    X = _to_float(X_raw, names)
    assert X.shape == (5, 2)
    # first real data row of KEEL banana: 1.14, -0.114, class -1.0
    assert np.isclose(X[0, 0], 1.14) and np.isclose(X[0, 1], -0.114)
    assert y_raw[0] == "-1.0"
    assert set(y_raw) == {"-1.0", "1.0"}


def test_row_arity_mismatch_is_hard_error():
    with open(FIXTURE) as fh:
        text = fh.read()
    with pytest.raises(ValueError):
        parse_keel_dat(text + "\n1.0,2.0,3.0,4.0")


def _keel_text(X, y):
    """Minimal KEEL .dat text for a small numeric, 3-class fixture."""
    lines = ["@relation testds",
             "@attribute a1 real [0.0,1.0]",
             "@attribute a2 real [0.0,1.0]",
             "@attribute Class{0,1,2}",
             "@inputs a1, a2", "@outputs Class", "@data"]
    for xr, yr in zip(X, y):
        lines.append("{0:.6f},{1:.6f},{2}".format(xr[0], xr[1], int(yr)))
    return "\n".join(lines) + "\n"


def _write_partition(path, name, n, drop_last_test=False):
    rng = np.random.default_rng(0)
    X = rng.random((n, 2))
    y = np.array([i % 3 for i in range(n)])
    with zipfile.ZipFile(path, "w") as zf:
        for k in range(1, 11):
            te = [i for i in range(n) if i % 10 == (k - 1)]
            tr = [i for i in range(n) if i % 10 != (k - 1)]
            zf.writestr("{0}-10-{1}tra.dat".format(name, k),
                        _keel_text(X[tr], y[tr]))
            if not (drop_last_test and k == 10):
                zf.writestr("{0}-10-{1}tst.dat".format(name, k),
                            _keel_text(X[te], y[te]))


def test_load_keel_folds_assembles_partitions(tmp_path, monkeypatch):
    _write_partition(tmp_path / "testds-10-fold.zip", "testds", 30)
    monkeypatch.setitem(
        DATASETS, "testds",
        {"source": "keel", "keel_name": "testds",
         "n": 30, "d": 2, "n_classes": 3, "verified": True})
    folds, k, labels = load_keel_folds("testds", str(tmp_path),
                                       download=False)
    assert k == 10 and len(folds) == 10 and len(labels) == 3
    total_test = 0
    for X_tr, y_tr, X_te, y_te in folds:
        assert X_tr.shape[1] == 2
        assert len(X_tr) + len(X_te) == 30  # each fold tiles the set
        assert set(np.unique(y_te)).issubset({0, 1, 2})
        total_test += len(X_te)
    assert total_test == 30  # the ten test folds partition the data


def test_load_keel_folds_short_partition_is_hard_error(tmp_path,
                                                       monkeypatch):
    _write_partition(tmp_path / "shortds-10-fold.zip", "shortds", 30,
                     drop_last_test=True)
    monkeypatch.setitem(
        DATASETS, "shortds",
        {"source": "keel", "keel_name": "shortds",
         "n": 30, "d": 2, "n_classes": 3, "verified": True})
    with pytest.raises(IOError):
        load_keel_folds("shortds", str(tmp_path), download=False)


def test_load_keel_folds_manifest_mismatch_is_hard_error(tmp_path,
                                                         monkeypatch):
    _write_partition(tmp_path / "wrongds-10-fold.zip", "wrongds", 30)
    # manifest declares 40 rows; the partitions reconstruct 30 -> fail
    monkeypatch.setitem(
        DATASETS, "wrongds",
        {"source": "keel", "keel_name": "wrongds",
         "n": 40, "d": 2, "n_classes": 3, "verified": True})
    with pytest.raises(AssertionError):
        load_keel_folds("wrongds", str(tmp_path), download=False)


def test_load_keel_folds_union_label_encoding(tmp_path, monkeypatch):
    # Class '2' is rare (3 rows), so it is absent from most test folds;
    # the union encoding must still give it a stable code everywhere.
    n = 33
    rng = np.random.default_rng(2)
    X = rng.random((n, 2))
    y = np.array([0, 1] * 15 + [2, 2, 2])
    with zipfile.ZipFile(tmp_path / "rareds-10-fold.zip", "w") as zf:
        for k in range(1, 11):
            te = [i for i in range(n) if i % 10 == (k - 1)]
            tr = [i for i in range(n) if i % 10 != (k - 1)]
            zf.writestr("rareds-10-{0}tra.dat".format(k),
                        _keel_text(X[tr], y[tr]))
            zf.writestr("rareds-10-{0}tst.dat".format(k),
                        _keel_text(X[te], y[te]))
    monkeypatch.setitem(
        DATASETS, "rareds",
        {"source": "keel", "keel_name": "rareds",
         "n": 33, "d": 2, "n_classes": 3, "verified": True})
    folds, k, labels = load_keel_folds("rareds", str(tmp_path),
                                       download=False)
    assert list(labels) == ["0", "1", "2"]  # sorted union of all folds
    # a fold whose TEST set lacks class 2 but whose TRAIN has it must
    # still code that class as 2 (the union-encoding guarantee).
    saw_absent_test_present_train = False
    for X_tr, y_tr, X_te, y_te in folds:
        assert set(y_tr.tolist()) | set(y_te.tolist()) <= {0, 1, 2}
        if 2 not in set(y_te.tolist()) and 2 in set(y_tr.tolist()):
            saw_absent_test_present_train = True
    assert saw_absent_test_present_train
