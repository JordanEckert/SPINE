"""Reference and naive baselines (Python-native).

* :class:`FullSet`         -- the whole training fold (reduction 0;
  the ceiling reference).
* :class:`RandomSubsample` -- stratified random selection (the floor
  reference); a *selection* method, so it returns real row indices.
* :class:`KMeansPerClass`  -- per-class k-means centroids (the naive
  carve-and-average baseline); a *generation* method.

Count-based methods use the per-class budget
``k_c = max(1, round(frac * N_c))`` with ``frac`` fixed a priori in code
(accuracy-blind) -- these are *not* tuned on the data.
"""

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from harness.base import SYNTHETIC_INDEX, PrototypeSelectorBase


def _per_class_count(n_c, frac):
    """Budget for a class of size ``n_c``: ``max(1, round(frac*n_c))``."""
    return max(1, int(round(frac * n_c)))


class FullSet(PrototypeSelectorBase):
    """Return the entire training fold unchanged (reduction rate 0)."""

    is_generator = False
    deterministic = True

    def select(self, X_train, y_train, **params):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        indices = np.arange(len(X_train))
        return X_train.copy(), y_train.copy(), indices

    @property
    def name(self):
        return "Full"


class RandomSubsample(PrototypeSelectorBase):
    """Stratified random subsample.

    Budget: ``class_counts`` (a ``{class: k}`` dict, injected by the
    runner as a proportional apportionment of the matched total over
    the training class frequencies) when provided; otherwise the legacy
    ``frac``-of-each-class rule.
    """

    is_generator = False
    deterministic = False  # stochastic -> multiple seeds

    def __init__(self, frac=0.1):
        self.frac = frac

    def select(self, X_train, y_train, **params):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        frac = params.get("frac", self.frac)
        class_counts = params.get("class_counts")  # budget-matching
        random_state = params.get("random_state")
        if random_state is None:
            raise ValueError("RandomSubsample requires a random_state in params")
        rng = np.random.default_rng(random_state)

        chosen = []
        for c in np.unique(y_train):
            idx_c = np.where(y_train == c)[0]
            if class_counts is not None:
                if c not in class_counts:
                    raise KeyError(
                        "class_counts is missing class {0!r}".format(c))
                k = int(class_counts[c])
            else:
                k = _per_class_count(len(idx_c), frac)
            k = min(k, len(idx_c))
            chosen.append(rng.choice(idx_c, size=k, replace=False))
        indices = np.sort(np.concatenate(chosen))
        return X_train[indices].copy(), y_train[indices].copy(), indices

    @property
    def name(self):
        return "Random"


class KMeansPerClass(PrototypeSelectorBase):
    """Per-class k-means centroids.

    Budget: ``class_counts`` (a ``{class: k}`` dict, injected by the
    runner as a proportional apportionment of the matched total over
    the training class frequencies) when provided; otherwise the legacy
    ``k_c = max(1, round(frac*N_c))`` rule.
    """

    is_generator = True
    deterministic = False  # k-means init is stochastic -> multiple seeds

    def __init__(self, frac=0.1):
        self.frac = frac

    def select(self, X_train, y_train, **params):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        frac = params.get("frac", self.frac)
        class_counts = params.get("class_counts")  # budget-matching
        random_state = params.get("random_state")
        if random_state is None:
            raise ValueError("KMeansPerClass requires a random_state in params")

        protos, labels = [], []
        for c in np.unique(y_train):
            X_c = X_train[y_train == c]
            if class_counts is not None:
                if c not in class_counts:
                    raise KeyError(
                        "class_counts is missing class {0!r}".format(c))
                k = int(class_counts[c])
            else:
                k = _per_class_count(len(X_c), frac)
            k = min(k, len(X_c))
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            with warnings.catch_warnings():
                # Duplicate points in a class can yield fewer than k
                # clusters; silence only this expected warning.
                warnings.simplefilter("ignore", ConvergenceWarning)
                km.fit(X_c)
            protos.append(km.cluster_centers_)
            labels.append(np.full(k, c))

        X_proto = np.vstack(protos)
        y_proto = np.concatenate(labels)
        indices = np.full(len(X_proto), SYNTHETIC_INDEX)
        return X_proto, y_proto, indices

    @property
    def name(self):
        return "KMeans"
