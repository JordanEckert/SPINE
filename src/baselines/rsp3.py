"""RSP3 -- Reduction by Space Partitioning, variant 3 (Python port).

Sanchez, *High training set size reduction by space partitioning and
prototype abstraction* (Pattern Recognition 37(7), 2004). RSP3 is the
parameter-free member of the RSP family and a canonical
"carve-and-average" generation baseline: Ougiaroglou & Evangelidis's RHC
paper (2014) uses it as a primary comparator, which supplies published
accuracy/reduction numbers on five of this study's datasets
(letter, magic, penbased/pen-digits, phoneme, texture, satimage/landsat)
for validation anchoring.

Algorithm (parameter-free, deterministic):

1. Start with the whole training fold as one subset (FIFO queue, as in
   the published description -- output is order-independent; the queue
   is presentational).
2. Dequeue a subset. If it is class-homogeneous, store its centroid
   (mean) as a synthetic prototype labelled with that class.
3. Otherwise find the subset's **diameter pair** (p, q) -- the two
   mutually farthest members -- and split: every member joins the side
   of the nearer endpoint. Enqueue both sides.
4. Repeat until the queue is empty.
"""

import numpy as np

from harness.base import SYNTHETIC_INDEX, PrototypeSelectorBase

#: Row-chunk size for the diameter scan (memory bound: chunk x |B|).
_CHUNK = 512


def _diameter_pair(X):
    """Indices (i, j), i < j, of the farthest pair; ties -> smallest.

    Chunked O(n^2) scan; returns (0, 0) for a single point.
    """
    n = len(X)
    if n < 2:
        return 0, 0
    best_d2, best_i, best_j = -1.0, 0, 0
    sq = (X ** 2).sum(axis=1)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        # Squared distances of rows [start:stop] against all rows.
        d2 = (sq[start:stop, None] + sq[None, :]
              - 2.0 * (X[start:stop] @ X.T))
        # Only the upper triangle matters; mask j <= i to keep the
        # lexicographic tie rule exact.
        rows = np.arange(start, stop)[:, None]
        cols = np.arange(n)[None, :]
        d2 = np.where(cols > rows, d2, -np.inf)
        flat = int(np.argmax(d2))
        r, c = divmod(flat, n)
        val = float(d2[r, c])
        if val > best_d2:
            best_d2, best_i, best_j = val, start + r, c
    return best_i, best_j


class RSP3(PrototypeSelectorBase):
    """Reduction by Space Partitioning v3 (generation, deterministic)."""

    is_generator = True
    deterministic = True

    def select(self, X_train, y_train, **params):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)

        protos, labels = [], []
        queue = [(np.arange(len(X_train)),)]

        while queue:
            (idx,) = queue.pop(0)
            Xc, yc = X_train[idx], y_train[idx]
            classes = np.unique(yc)

            if len(classes) == 1:
                protos.append(Xc.mean(axis=0))
                labels.append(classes[0])
                continue

            i, j = _diameter_pair(Xc)
            if i == j or not (((Xc[i] - Xc[j]) ** 2).sum() > 0.0):
                # Degenerate: coincident points, mixed labels -- emit
                # per-class centroids (defined no-progress rule).
                for c in classes:
                    protos.append(Xc[yc == c].mean(axis=0))
                    labels.append(c)
                continue

            d_p = ((Xc - Xc[i]) ** 2).sum(axis=1)
            d_q = ((Xc - Xc[j]) ** 2).sum(axis=1)
            side_p = d_p <= d_q  # ties -> p's side (deterministic)
            queue.append((idx[side_p],))
            queue.append((idx[~side_p],))

        X_proto = np.vstack(protos)
        y_proto = np.asarray(labels)
        indices = np.full(len(X_proto), SYNTHETIC_INDEX)
        return X_proto, y_proto, indices

    @property
    def name(self):
        return "RSP3"
