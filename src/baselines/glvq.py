"""GLVQ -- Generalized Learning Vector Quantization (via ``sklvq``).

Sato and Yamada's GLVQ (NIPS 1996): the codebook is trained by gradient
descent on the relative-distance margin

    mu(x) = (d+ - d-) / (d+ + d-)

pushed through a monotone squashing function, where ``d+`` is the
distance to the nearest SAME-class prototype and ``d-`` to the nearest
different-class one.  

Implementation
--------------
Delegated to ``sklvq`` (https://github.com/rickvanveen/sklvq), a
scikit-learn-compatible LVQ package, rather than reimplemented here.
``sklvq.GLVQ`` is used at its own defaults for the loss and solver; the
only injected settings are the ones this protocol requires:

* ``prototype_n_per_class`` -- the harness's proportional apportionment of
  the anchor's total budget, so GLVQ is budget-matched exactly like every
  other method in the roster;
* ``random_state`` -- the fold's derived seed, since prototype
  initialisation and the stochastic solver both depend on it.
"""

import numpy as np

from harness.base import (SYNTHETIC_INDEX, PrototypeSelectorBase,
                          allocate_per_class)


def _install_shim():
    """Restore ``_validate_data`` for sklvq on scikit-learn >= 1.7."""
    from sklvq.models._base import LVQBaseClass
    if hasattr(LVQBaseClass, "_validate_data"):
        return
    from sklearn.utils.validation import validate_data

    def _validate_data_compat(self, X, y=None, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return validate_data(self, X=X, y=y, **kwargs)

    LVQBaseClass._validate_data = _validate_data_compat


class GLVQ(PrototypeSelectorBase):
    """Generalized LVQ (generation, stochastic), budget-matched."""

    is_generator = True
    deterministic = False
    order_dependent = True

    def __init__(self, frac=0.1):
        #: Used only when the runner injects no ``total_count``; the
        #: campaign always injects one.
        self.frac = frac

    def _counts_in_sklvq_order(self, y, total_count):
        """Per-class prototype counts, ordered as ``sklvq`` expects them."""
        classes = np.unique(y)
        if total_count is None:
            counts = {c: max(1, int(round(self.frac * int((y == c).sum()))))
                      for c in classes}
        else:
            counts = allocate_per_class(y, int(total_count))
        ordered = [int(counts[c]) for c in classes]
        for c, k in zip(classes, ordered):
            n_c = int((y == c).sum())
            if k > n_c:
                raise AssertionError(
                    "GLVQ: class {0!r} is allotted {1} prototypes but has "
                    "only {2} training points; capping would silently break "
                    "the budget-matched comparison".format(c, k, n_c))
        return classes, ordered

    def select(self, X_train, y_train, **params):
        _install_shim()
        from sklvq import GLVQ as _SkGLVQ

        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train)
        random_state = params.get("random_state")
        if random_state is None:
            raise ValueError("GLVQ requires a random_state in params")

        classes, counts = self._counts_in_sklvq_order(
            y, params.get("total_count"))

        # sklvq requires an ndarray here and rejects a plain list.
        model = _SkGLVQ(prototype_n_per_class=np.asarray(counts, dtype=int),
                        random_state=int(random_state))
        model.fit(X, y)

        P = np.asarray(model.prototypes_, dtype=float)
        Py = np.asarray(model.prototypes_labels_)
        # sklvq labels prototypes by POSITION into its own classes_; map
        # back to the caller's label values so the contract check on
        # y_proto compares like with like.
        if Py.dtype.kind in "iu" and not np.array_equal(
                np.unique(Py), np.unique(y)):
            Py = np.asarray(model.classes_)[Py]

        if len(P) != sum(counts):
            raise AssertionError(
                "GLVQ emitted {0} prototypes but was asked for {1}"
                .format(len(P), sum(counts)))
        return P, Py, np.full(len(P), SYNTHETIC_INDEX)

    @property
    def name(self):
        return "GLVQ"
