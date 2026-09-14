"""LVQ3 -- Learning Vector Quantization, third variant (Python port).

Kohonen's LVQ3 (Kohonen 1990; the positioning-adjustment family in the
Triguero et al. 2012 PG taxonomy). Included as the classical
generation-by-adjustment baseline: prototypes are initialised as real
training points and then MOVED, so the output is synthetic
(generation semantics, sentinel indices).

Algorithm:

1. Initialise a per-class codebook of real training points (>=1 per
   class). LVQ3 is budget-matched: when the runner injects
   ``total_count`` (the anchor's total prototype count), it is apportioned
   across classes by largest-remainder allocation; otherwise a
   ``frac``-of-each-class codebook (``k_c = max(1, round(frac * N_c))``)
   is used.
2. For each of ``presentations = epochs * n_train`` steps, draw a
   training sample x (with replacement, seeded rng) and find its two
   nearest prototypes m_i, m_j:
   - both correct class: both move toward x by ``epsilon * alpha_t``
     (the LVQ3 stabilising term);
   - one correct, one wrong, and x inside the **window**
     (min(d_i/d_j, d_j/d_i) > (1-w)/(1+w)): the correct one moves
     toward x, the wrong one away, by ``alpha_t``;
   - otherwise: no update.
   ``alpha_t = alpha * (1 - t/T)`` decays linearly to zero.

The codebook size (the matched budget), the learning rate (alpha = 0.1),
and the number of epochs (30 passes over the training fold) are fixed;
the window width w and the stabilising factor epsilon are selected per
training fold by stratified inner 5-fold CV on 1-NN accuracy.
"""

import numpy as np

from harness.base import (SYNTHETIC_INDEX,
                          PrototypeSelectorBase,
                          allocate_per_class)


class LVQ3(PrototypeSelectorBase):
    """LVQ3 positioning adjustment (generation, stochastic)."""

    is_generator = True
    deterministic = False

    def __init__(self, frac=0.1, epochs=30, alpha=0.1,
                 window=0.2, epsilon=0.1, batch_size=64):
        self.frac = frac
        self.epochs = epochs
        self.alpha = alpha
        self.window = window
        self.epsilon = epsilon
        self.batch_size = batch_size

    @property
    def param_grid(self):
        """Inner-CV tuning grid: window and epsilon (codebook size fixed)."""
        from harness.config import LVQ3_TUNING_GRID
        return dict(LVQ3_TUNING_GRID)

    def select(self, X_train, y_train, **params):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train)
        random_state = params.get("random_state")
        if random_state is None:
            raise ValueError("LVQ3 requires a random_state in params")
        rng = np.random.default_rng(random_state)

        # 1. Stratified initialisation (>=1 per class).  Budget: when the
        # runner injects ``total_count`` (the anchor's total count --
        # the budget-matching rule), it is apportioned across classes by
        # largest-remainder allocation; otherwise the frac-per-class rule
        # applies.
        total_count = params.get("total_count")
        counts = (allocate_per_class(y, int(total_count))
                  if total_count is not None else None)
        proto_idx = []
        for c in np.unique(y):
            rows = np.where(y == c)[0]
            k = counts[c] if counts is not None \
                else max(1, int(round(self.frac * len(rows))))
            k = min(k, len(rows))
            proto_idx.append(rng.choice(rows, size=k, replace=False))
        proto_idx = np.concatenate(proto_idx)
        P = X[proto_idx].copy()
        Py = y[proto_idx].copy()
        m = len(P)

        # 2. Presentations with linearly decaying alpha: ``epochs``
        # passes over the training fold.
        T = self.epochs * len(X)
        window_bound = (1.0 - self.window) / (1.0 + self.window)
        t = 0
        while t < T:
            batch = min(self.batch_size, T - t)
            sample_rows = rng.integers(0, len(X), size=batch)
            Xb = X[sample_rows]
            # Two nearest prototypes against the batch-start snapshot.
            d2 = ((Xb[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
            near2 = np.argpartition(d2, 1, axis=1)[:, :2]
            # order the pair by distance
            swap = (d2[np.arange(batch), near2[:, 0]]
                    > d2[np.arange(batch), near2[:, 1]])
            near2[swap] = near2[swap][:, ::-1]

            for b in range(batch):
                alpha_t = self.alpha * (1.0 - (t + b) / T)
                x, cls = Xb[b], y[sample_rows[b]]
                i, j = int(near2[b, 0]), int(near2[b, 1])
                di = float(((x - P[i]) ** 2).sum()) ** 0.5
                dj = float(((x - P[j]) ** 2).sum()) ** 0.5
                ci, cj = Py[i] == cls, Py[j] == cls
                if ci and cj:
                    P[i] += self.epsilon * alpha_t * (x - P[i])
                    P[j] += self.epsilon * alpha_t * (x - P[j])
                elif ci != cj:
                    ratio = min(di / dj, dj / di) if di > 0 and dj > 0 \
                        else 0.0
                    if ratio > window_bound:
                        good, bad = (i, j) if ci else (j, i)
                        P[good] += alpha_t * (x - P[good])
                        P[bad] -= alpha_t * (x - P[bad])
            t += batch

        indices = np.full(m, SYNTHETIC_INDEX)
        return P, Py, indices

    @property
    def name(self):
        return "LVQ3"
