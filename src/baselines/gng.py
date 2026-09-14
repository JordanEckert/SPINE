"""Growing Neural Gas as a budget-matched prototype generator.

Fritzke's GNG (NIPS 1995) grows a graph of units that adapts to the
topology of the data it is shown.  It is unsupervised, so -- exactly as
for K-Means in this study -- it is run PER CLASS on that class's points
alone, and each class's units are labelled with that class.  The units
are synthetic points, so the output carries the generation sentinel.

Budget matching
---------------
Insertion stops at the class's quota and the network is topped up to it
after fitting (edge ageing can prune units below the target), so the
emitted count equals the injected budget exactly, like every other
budget-matched method here.
"""

import numpy as np

from harness.base import (SYNTHETIC_INDEX, PrototypeSelectorBase,
                          allocate_per_class)

from .vendor.growing_neural_gas import DEFAULTS, GrowingNeuralGas


class GNG(PrototypeSelectorBase):
    """Growing Neural Gas per class (generation, stochastic)."""

    is_generator = True
    deterministic = False
    order_dependent = True

    def __init__(self, frac=0.1, params=None, min_passes=1):
        self.frac = frac
        self.params = dict(DEFAULTS) if params is None else dict(params)
        self.min_passes = min_passes

    def _passes_for(self, n_points, target):
        """Passes needed for the insertion schedule to reach ``target``.

        One unit is inserted every ``l`` presentations and the network
        starts with two, so reaching ``target`` needs ``(target - 2) * l``
        presentations, i.e. that many divided by the class size, rounded
        up.  Derived from the schedule rather than chosen.
        """
        l = int(self.params["l"])
        needed = max(0, int(target) - 2) * l
        return max(int(self.min_passes),
                   int(np.ceil(needed / max(1, int(n_points)))))

    def select(self, X_train, y_train, **params):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train)
        random_state = params.get("random_state")
        if random_state is None:
            raise ValueError("GNG requires a random_state in params")
        rng = np.random.default_rng(random_state)

        total_count = params.get("total_count")
        classes = np.unique(y)
        if total_count is None:
            counts = {c: max(1, int(round(self.frac * int((y == c).sum()))))
                      for c in classes}
        else:
            counts = allocate_per_class(y, int(total_count))

        P, Py = [], []
        for c in classes:
            rows = np.where(y == c)[0]
            k = int(counts[c])
            if k > len(rows):
                raise AssertionError(
                    "GNG: class {0!r} is allotted {1} prototypes but has "
                    "only {2} training points; capping would silently break "
                    "the budget-matched comparison".format(c, k, len(rows)))
            Xc = X[rows]
            if k == 1 or len(Xc) < 2:
                # GNG needs two seed units; one prototype is the class
                # mean, which is what a single-unit quantiser gives.
                P.append(Xc.mean(axis=0)[None, :])
                Py.append(np.full(1, c))
                continue

            child = np.random.default_rng(int(rng.integers(0, 2 ** 31 - 1)))
            gng = GrowingNeuralGas(Xc, rng=child)
            gng.fit_network(passes=self._passes_for(len(Xc), k),
                            max_nodes=k, **self.params)
            gng.top_up(k, self.params["a"])
            V = gng.vectors()
            if len(V) != k:
                raise AssertionError(
                    "GNG: class {0!r} produced {1} units for a quota of {2}"
                    .format(c, len(V), k))
            P.append(V)
            Py.append(np.full(k, c))

        P = np.vstack(P)
        Py = np.concatenate(Py).astype(classes.dtype)
        return P, Py, np.full(len(P), SYNTHETIC_INDEX)

    @property
    def name(self):
        return "GNG"
