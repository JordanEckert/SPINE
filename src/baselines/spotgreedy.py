"""SPOTGreedy -- prototype SELECTION via optimal transport.

Gurumoorthy, Jawanpuria & Mishra, "SPOT: A framework for selection of
prototypes using optimal transport", ECML PKDD 2021 (arXiv:2103.10159).

Implementation provenance
-------------------------
The greedy routine below is a *reimplementation* of the authors'
SPOTgreedy algorithm, not a call into their release.  It was written
because the reference translation shipped in ``interpret-core``
(``interpret.utils.SPOT_GreedySubsetSelection``, a NumPy port of
https://github.com/royparijat/SPOT) cannot run this study's largest
folds: it materialises the full ``n x n`` cost matrix AND two more
``n x n`` temporaries on every one of the ``m`` iterations, so on
magic's 17,118-row training fold it needs roughly 7 GB resident and
does O(m * n^2) work -- hours per fold at the budgets this protocol
uses.

The selection is the exhaustive greedy's, exactly.  The SPOT objective

    F(S) = sum_x  mu_x * min(M, min_{y in S} C[y, x]),     M = 1e6

is monotone and submodular in ``S`` (it is the facility-location form),
so the marginal gain of a candidate is non-increasing as ``S`` grows.
That is exactly the condition under which *lazy* greedy evaluation
(Minoux 1978; the CELF construction of Leskovec et al. 2007) returns
the same element the exhaustive argmax would, at a fraction of the
evaluations: a candidate popped from the priority queue whose gain is
current and no worse than every other candidate's stored upper bound is
provably the true maximiser.
"""

import heapq

import numpy as np
from sklearn.metrics import pairwise_distances

from harness.base import PrototypeSelectorBase

#: Source rows scored per pass of the initial sweep.  Bounds the
#: temporaries at ``_BLOCK x n`` instead of ``n x n``.
_BLOCK = 2048

#: Significant digits retained in a queue key.  Mathematically equal
#: gains that differ only in floating-point noise snap to the same key,
#: so the declared lowest-index tie-break decides them rather than
#: rounding; see the module docstring.
_KEY_DIGITS = 12


#: Below this decimal exponent the scale factor ``10 ** (_KEY_DIGITS -
#: 1 - exp)`` would overflow to inf and the snap would return nan, which
#: a heap silently mis-orders (every comparison with nan is False).  A
#: gain this small is indistinguishable from no gain at the resolution
#: the key needs, so it is snapped to zero instead.
_MIN_EXP = -290


def _snap(g):
    """Round gains to ``_KEY_DIGITS`` significant digits.

    Significant digits rather than decimal places, because the gains
    span magnitudes.  Non-finite input passes through unchanged (an
    infinite or undefined gain is not something rounding can fix, and
    silently turning it into a number would hide the fault); gains below
    ``10 ** _MIN_EXP`` snap to zero rather than overflowing the scale
    factor.  Accepts a scalar or an array and preserves the shape.
    """
    g = np.asarray(g, dtype=float)
    out = np.array(g, dtype=float, copy=True)
    nz = np.isfinite(g) & (g != 0.0)
    if nz.any():
        vals = g[nz]
        exp = np.floor(np.log10(np.abs(vals)))
        big = exp >= _MIN_EXP
        snapped = np.zeros(vals.shape, dtype=float)
        if big.any():
            scale = 10.0 ** (_KEY_DIGITS - 1 - exp[big])
            snapped[big] = np.round(vals[big] * scale) / scale
        out[nz] = snapped
    return out


#: The authors' empty-set convention: before anything is selected every
#: target point is charged this cost.  Kept literal so the objective --
#: and therefore the selection -- matches the reference exactly.
_EMPTY_SET_COST = 1e6


def spot_greedy(C, target_marginal, m):
    """Lazy-greedy SPOTgreedy: indices of the ``m`` selected sources.

    ``C`` is the ``n_source x n_target`` cost matrix and
    ``target_marginal`` the target histogram (normalised internally, as
    the reference does).  Returns the selected source indices in
    selection order -- the same sequence the exhaustive greedy returns,
    with mathematically tied gains resolved to the lowest index (see the
    module docstring on key snapping).
    """
    C = np.asarray(C, dtype=float)
    mu = np.asarray(target_marginal, dtype=float).ravel()
    n_source, n_target = C.shape
    if len(mu) != n_target:
        raise ValueError(
            "target_marginal has length {0}, expected {1}".format(
                len(mu), n_target))
    if (mu < 0.0).any():
        # Non-negativity is what makes the objective submodular, and
        # submodularity is the whole licence for lazy evaluation: with a
        # negative weight a stale key is no longer an upper bound and
        # the queue can return a non-greedy element.  Rejected rather
        # than silently answered wrongly.
        raise ValueError(
            "target_marginal must be non-negative (submodularity, and "
            "hence the lazy-greedy equivalence, depends on it)")
    total = mu.sum()
    if not total > 0:
        raise ValueError("target_marginal must have positive mass")
    mu = mu / total
    m = int(m)
    if not 1 <= m <= n_source:
        raise ValueError(
            "m must lie in [1, n_source]={0}, got {1}".format(n_source, m))

    curr_min = np.full(n_target, _EMPTY_SET_COST)

    def _blocked_gains(cm):
        """Snapped gains against ``cm``, blocked to bound temporaries."""
        g = np.empty(n_source)
        for start in range(0, n_source, _BLOCK):
            stop = min(start + _BLOCK, n_source)
            block = cm - C[start:stop]
            np.maximum(block, 0.0, out=block)
            g[start:stop] = block @ mu
        return _snap(g)

    # ---- iteration 1: decided on the UN-ANCHORED score ---------------
    # Before anything is selected every target sits at the empty-set cost
    # M, so the gain of candidate y is M - (C[y] @ mu) whenever the costs
    # are below M.  That is an O(1e6) number carrying an O(1) signal, and
    # no key of that magnitude can resolve better than ulp(1e6) = 1.2e-10
    # however it is snapped.  So the first prototype is chosen by a
    # direct argmin over the snapped distance part -- never anchored,
    # never put on the heap -- which is the same ordering at the full
    # ~1e-12 relative resolution.  np.argmin takes the lowest index on
    # ties, the declared rule.  If any cost exceeds M the clamp in the
    # general formula bites and the shortcut is unavailable, so the
    # anchored gains are used for the first pick too.
    anchored = bool((C > _EMPTY_SET_COST).any())
    if anchored:
        first = int(np.argmax(_blocked_gains(curr_min)))
    else:
        scores = np.empty(n_source)
        for start in range(0, n_source, _BLOCK):
            stop = min(start + _BLOCK, n_source)
            scores[start:stop] = C[start:stop] @ mu
        first = int(np.argmin(_snap(scores)))

    chosen = [first]
    np.minimum(curr_min, C[first], out=curr_min)

    # ---- the heap is seeded AFTER the first selection -----------------
    # curr_min is now O(1), so 12 significant digits really is ~1e-12
    # relative and the anchored-resolution problem cannot recur.
    gains = _blocked_gains(curr_min)

    # Max-heap keyed by (-gain, index): popping the smallest tuple takes
    # the largest gain and, among equal gains, the lowest index.
    heap = [(-gains[i], i) for i in range(n_source) if i != first]
    heapq.heapify(heap)

    taken = np.zeros(n_source, dtype=bool)
    taken[first] = True
    stale = np.zeros(n_source, dtype=bool)
    while len(chosen) < m:
        while True:
            neg_gain, cand = heapq.heappop(heap)
            if taken[cand]:
                continue
            if stale[cand]:
                # Re-price against the CURRENT curr_min and re-queue.
                # Submodularity makes the stored key an upper bound, so
                # this loop terminates with a key that is both current
                # and minimal -- the true argmax.
                d = curr_min - C[cand]
                np.maximum(d, 0.0, out=d)
                stale[cand] = False
                heapq.heappush(
                    heap, (-float(_snap(d @ mu)), cand))
                continue
            break
        taken[cand] = True
        chosen.append(cand)
        np.minimum(curr_min, C[cand], out=curr_min)
        # Every surviving key is now an upper bound, not a value.
        stale[:] = True
    return np.asarray(chosen, dtype=int)


class SPOTGreedy(PrototypeSelectorBase):
    """Global SPOTgreedy selection at an injected total budget."""

    is_generator = False
    #: The cost matrix and the greedy argmax are deterministic; argmax
    #: ties resolve to the lowest index, fixed by row order.
    deterministic = True

    def __init__(self, total_count=None):
        self.total_count = total_count

    def select(self, X_train, y_train, **params):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train)
        m = params.get("total_count", self.total_count)
        if m is None:
            raise ValueError(
                "SPOTGreedy requires total_count (the anchor-matched "
                "budget); it has no native budget rule")
        m = int(m)
        if not 1 <= m <= len(X):
            raise ValueError(
                "total_count must be in [1, n_train]={0}, got {1}".format(
                    len(X), m))
        if m == len(X):
            # Full budget: selection is the identity (the greedy loop
            # would select every row anyway); defined shortcut.
            indices = np.arange(len(X))
            return X.copy(), y.copy(), indices

        # Canonical SPOT usage (InterpretML example): source = target =
        # the training fold, Euclidean cost, uniform target marginal.
        C = pairwise_distances(X, X, metric="euclidean")
        target_marginal = np.full(len(X), 1.0 / len(X))
        indices = spot_greedy(C, target_marginal, m)
        if len(indices) != m or len(np.unique(indices)) != m:
            raise AssertionError(
                "spot_greedy returned {0} indices ({1} unique), expected "
                "{2} distinct".format(
                    len(indices), len(np.unique(indices)), m))
        return X[indices].copy(), y[indices].copy(), indices

    @property
    def name(self):
        return "SPOT"
