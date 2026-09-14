"""Phase 0 cover.

"""

import numpy as np
from scipy.stats import rankdata

#: Section 4.2: fixed, not adapted.
DEFAULT_GAIN = 0.25


def n_intervals(n_class):
    """``ceil((8 n_c)^(1/5))``, floored at 2 (one slab is not a cover)."""
    return int(max(2, np.ceil((8.0 * int(n_class)) ** 0.2)))


def uniform_intervals(f, n_int, gain=DEFAULT_GAIN):
    """``n_int`` equal-width slabs on ``range(f)``, each widened by ``gain``.

    Slab ``k`` is centred at ``lo + step (k + 1/2)`` and has width
    ``step (1 + gain)``, so consecutive slabs overlap in a band of width
    ``step * gain`` and non-consecutive slabs do not meet at all (for any
    ``gain < 1``).  The union covers ``[lo, hi]`` with a margin at each
    end, so no point of the class falls outside the cover.
    """
    f = np.asarray(f, dtype=float)
    lo, hi = float(f.min()), float(f.max())
    step = (hi - lo) / float(n_int)
    if step <= 0.0:
        # Constant lens: the cover degenerates to a single slab holding
        # everything.  Reported by the caller, never silently patched.
        return [(lo - 0.5, hi + 0.5)]
    half = step * (1.0 + gain) / 2.0
    return [(lo + step * (k + 0.5) - half, lo + step * (k + 0.5) + half)
            for k in range(n_int)]


def rank_reparametrise(f):
    """The lens pushed through its own empirical CDF, onto ``[0, 1]``.

    A strictly monotone reparametrisation of the filter, under which the
    Reeb graph is invariant.  Ties take the ``average`` rank, so points
    sharing a lens value share a reparametrised value and can never be
    placed in different slabs.
    """
    f = np.asarray(f, dtype=float)
    if len(f) < 2:
        return np.zeros(len(f))
    return (rankdata(f, method="average") - 1.0) / (len(f) - 1.0)


def quantile_intervals(f, n_int, gain=DEFAULT_GAIN):
    """Equal-mass slabs: a uniform cover of the rank-reparametrised lens.

    Returned in the REPARAMETRISED coordinate, so callers must test
    membership against :func:`rank_reparametrise` of the same lens rather
    than against the raw values.  :func:`build_cover` returns the
    coordinate to use alongside the cover for exactly this reason.
    """
    return uniform_intervals(rank_reparametrise(f), n_int, gain=gain)


def slab_populations(f, cover):
    """Number of points of ``f`` inside each (closed) slab."""
    f = np.asarray(f, dtype=float)
    return [int(((f >= a) & (f <= b)).sum()) for (a, b) in cover]


def build_cover(f, n_int, min_cluster_size, gain=DEFAULT_GAIN):
    """Uniform cover, with the section-4.2 starvation fallback.

    Returns ``(cover, placement, f_eff)``.  ``f_eff`` is the coordinate
    the cover lives in -- the lens itself under uniform placement, its
    rank reparametrisation under the quantile fallback -- and is what
    slab membership must be tested against.

    The fallback fires when ANY slab of the uniform cover holds fewer
    than ``2 * min_cluster_size`` points: the population below which a
    slab cannot resolve two clusters even in principle, so the cover has
    already degenerated there whatever the space does.
    """
    f = np.asarray(f, dtype=float)
    cover = uniform_intervals(f, n_int, gain=gain)
    if min(slab_populations(f, cover)) < 2 * int(min_cluster_size):
        f_eff = rank_reparametrise(f)
        return uniform_intervals(f_eff, n_int, gain=gain), "quantile", f_eff
    return cover, "uniform", f


def slab_members(f, cover):
    """Index arrays of the points of ``f`` falling in each slab."""
    f = np.asarray(f, dtype=float)
    return [np.where((f >= a) & (f <= b))[0] for (a, b) in cover]
