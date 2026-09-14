"""Correctness of the in-repo SPOT greedy routine.

Two independent references guard it:

* a deliberately naive restatement of SPOTgreedy (recompute the full
  objective for every candidate at every step), which encodes the
  algorithm as written in the paper; and
* ``interpret.utils.SPOT_GreedySubsetSelection``, the NumPy translation
  of the authors' released code that this repository used to call
  directly.

``spot_greedy`` must return the same indices in the same order as both
wherever the greedy choice is well defined.  That is the evidence behind
the README's provenance claim, and it also catches silent drift in the
external package.

Where two candidates tie -- exactly, or to within the floating-point
noise that separates the same quantity computed through two different
BLAS routines -- the reference's ``np.argmax`` returns whichever came
out bitwise larger, which is not a rule.  ``spot_greedy`` snaps its
queue key to ``_KEY_DIGITS`` significant digits and takes the lowest
index instead.  Three tests pin that: the rule is exact on constructed
ties, the result depends neither on ``_BLOCK`` nor on the order of the
target columns, and repeated runs agree.
"""

import numpy as np
import pytest

from baselines.spotgreedy import spot_greedy


def naive_spot_greedy(C, mu, m):
    """O(m * n^2) literal restatement of SPOTgreedy.

    Objective after selecting S: sum_x mu_x * min_{y in S} C[y, x]
    (with the empty-set convention of the authors' code: an initial
    per-target cost of 1e6).  At each step pick the candidate whose
    addition maximally decreases the objective; ties -> lowest index.
    """
    n = C.shape[0]
    mu = mu / mu.sum()
    selected = []
    curr = np.full(C.shape[1], 1e6)
    for _ in range(m):
        best_gain, best_y = -1.0, None
        for yy in range(n):
            if yy in selected:
                continue
            gain = float(np.maximum(curr - C[yy], 0.0) @ mu)
            if gain > best_gain:
                best_gain, best_y = gain, yy
        selected.append(best_y)
        curr = np.minimum(curr, C[best_y])
    return np.asarray(selected)


def _random_cost(rng, n):
    C = rng.random((n, n))
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 0.0)
    return C


def test_matches_naive_reference():
    rng = np.random.default_rng(42)
    for trial in range(5):
        n, m = 40, 8
        C = _random_cost(rng, n)
        mu = np.full(n, 1.0 / n)
        expected = naive_spot_greedy(C.copy(), mu.copy(), m)
        got = spot_greedy(C.copy(), mu.copy(), m)
        assert np.array_equal(got, expected), trial


def test_matches_authors_released_routine():
    from interpret.utils import SPOT_GreedySubsetSelection

    rng = np.random.default_rng(7)
    for trial in range(5):
        n, m = 60, 12
        C = _random_cost(rng, n)
        mu = np.full(n, 1.0 / n)
        expected, _w = SPOT_GreedySubsetSelection(C.copy(), mu.copy(), m)
        got = spot_greedy(C.copy(), mu.copy(), m)
        assert np.array_equal(got, np.asarray(expected)), trial


def test_matches_on_real_euclidean_costs():
    # Duplicate-heavy, tied-distance data is where a tie-break rule
    # actually bites; iris has repeated rows after rounding.
    from interpret.utils import SPOT_GreedySubsetSelection
    from sklearn.datasets import load_iris
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import MinMaxScaler

    X = MinMaxScaler().fit_transform(load_iris().data)
    C = pairwise_distances(X, X, metric="euclidean")
    mu = np.full(len(X), 1.0 / len(X))
    for m in (1, 5, 30):
        expected, _w = SPOT_GreedySubsetSelection(C.copy(), mu.copy(), m)
        got = spot_greedy(C.copy(), mu.copy(), m)
        assert np.array_equal(got, np.asarray(expected)), m


def test_non_uniform_marginal_matches_reference():
    from interpret.utils import SPOT_GreedySubsetSelection

    rng = np.random.default_rng(3)
    n, m = 50, 10
    C = _random_cost(rng, n)
    mu = rng.random(n) + 0.01           # unnormalised on purpose
    expected, _w = SPOT_GreedySubsetSelection(C.copy(), mu.copy(), m)
    got = spot_greedy(C.copy(), mu.copy(), m)
    assert np.array_equal(got, np.asarray(expected))


def _tied_cost(rng, n, n_target):
    """Integer costs: many candidates tie EXACTLY, which is the case
    floating point cannot be trusted to represent."""
    return rng.integers(0, 4, (n, n_target)).astype(float)


def test_exact_ties_resolve_to_the_lowest_index():
    """A genuine tie must go to the lowest index -- and must BE a tie.

    Both halves matter.  The first is the declared rule.  The second is
    what makes the test non-vacuous: unsnapped, two mathematically equal
    gains are separated by floating-point noise rather than equal, and
    which one wins follows the rounding.
    """
    # Rows 1-3 are permutations of row 0's costs under a uniform
    # marginal, so all four have identical gain.
    C = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
        [1.0, 0.0, 3.0, 2.0],
        [2.0, 3.0, 0.0, 1.0],
    ])
    mu = np.full(4, 0.25)
    assert spot_greedy(C, mu, 1)[0] == 0
    order = np.array([2, 3, 0, 1])
    assert spot_greedy(C[order], mu, 1)[0] == 0


def test_snapping_is_load_bearing_not_decorative():
    """Disabling ``_snap`` must break something -- pin exactly what.

    Snapped, the selection is invariant to the block size and to the
    order of the target columns.  Unsnapped, mathematically equal gains
    reaching the key through different summation orders differ in the
    last ulp and the winner follows the rounding, so permuting the
    target columns of ``C`` and ``mu`` together -- which reorders every
    gain's addends without changing the objective (the permuted copy is
    made C-contiguous so the memory layout of the ``gemv`` call is the
    same and only the order changes) -- can change the selection.  This
    test fails if ``_snap`` is removed.

    The discriminator used to be the block-size sweep, which only works
    on a BLAS whose ``gemv`` rounding depends on the call shape: x86
    OpenBLAS does (5-6 of 40 trials flip), Apple Accelerate does not
    (``C @ mu`` was bit-identical across block sizes in 40 of 40 trials,
    so the fixture could not discriminate there, and the retained
    block-size assertions below compare identical computations on that
    platform; the permutation assertion is the one doing the work).
    Reordering the addends changes the rounding on any BLAS that
    accumulates in double precision -- not on one with a wider or exact
    accumulator, where this fixture would report 0 flips -- which is a
    far weaker platform assumption than shape-dependent rounding.
    Measured with this fixture: 8-19 of 40 trials across x86 OpenBLAS
    kernels; on Apple Accelerate the same permutation flipped 12 of 40
    before the contiguous copy was added.
    """
    import baselines.spotgreedy as sg

    rng = np.random.default_rng(17)
    original_block, original_snap = sg._BLOCK, sg._snap
    flips_when_unsnapped = 0
    try:
        for _trial in range(40):
            C = rng.integers(0, 4, (48, 43)).astype(float)
            mu = np.full(43, 1.0 / 43)
            perm = rng.permutation(43)
            C_perm = np.ascontiguousarray(C[:, perm])
            mu_perm = mu[perm]

            sg._BLOCK = original_block
            snapped_ref = spot_greedy(C, mu, 5)
            for block in (3, 7, 4096):
                sg._BLOCK = block
                assert np.array_equal(spot_greedy(C, mu, 5), snapped_ref), (
                    "snapped selection changed with _BLOCK; either snapping "
                    "is broken or a key sits within rounding noise of a "
                    "12-digit boundary (measured margin here: >50 ulps)")
            sg._BLOCK = original_block
            assert np.array_equal(spot_greedy(C_perm, mu_perm, 5),
                                  snapped_ref), (
                "snapped selection changed with the target-column order; "
                "either snapping is broken or a key sits within rounding "
                "noise of a 12-digit boundary (measured margin: >50 ulps)")

            sg._snap = lambda g: np.asarray(g, dtype=float)
            if not np.array_equal(spot_greedy(C_perm, mu_perm, 5),
                                  spot_greedy(C, mu, 5)):
                flips_when_unsnapped += 1
            sg._snap = original_snap
    finally:
        sg._BLOCK, sg._snap = original_block, original_snap
    assert flips_when_unsnapped > 0, (
        "fixture no longer discriminates: disabling _snap must make the "
        "result depend on the summation order of the gains")


def test_repeated_runs_agree():
    rng = np.random.default_rng(5)
    C = _tied_cost(rng, 50, 45)
    mu = np.full(45, 1.0 / 45)
    a = spot_greedy(C, mu, 9)
    b = spot_greedy(C, mu, 9)
    assert np.array_equal(a, b)


def test_snap_puts_ulp_separated_values_on_one_key():
    from baselines.spotgreedy import _snap

    for base in (1e-3, 1.0, 1e6):
        a = base
        b = np.nextafter(base, np.inf)
        assert a != b
        assert float(_snap(a)) == float(_snap(b))
    # ... while a difference that is real at this precision survives
    assert float(_snap(1.0)) != float(_snap(1.0 + 1e-9))


def test_snap_is_finite_and_never_nan():
    from baselines.spotgreedy import _snap

    # Below the representable exponent the scale factor would overflow
    # and the old form returned nan, which a heap mis-orders silently.
    for tiny in (1e-298, 5e-324, -1e-300):
        assert float(_snap(tiny)) == 0.0
    finite = np.array([0.0, 1e-298, 5e-324, 1.0, -2.5, 1e6, 1e300])
    assert np.isfinite(_snap(finite)).all()
    # Non-finite input passes through rather than being invented away.
    assert np.isposinf(float(_snap(np.inf)))
    assert np.isneginf(float(_snap(-np.inf)))
    assert np.isnan(float(_snap(np.nan)))


def test_first_iteration_keeps_full_resolution():
    # Before any selection every target sits at the empty-set cost 1e6,
    # so snapping the ANCHORED gain would leave only ~1e-6 of absolute
    # resolution and call genuinely different candidates tied.  The
    # distance part is snapped instead; a 5e-7 advantage must decide the
    # first pick.
    rng = np.random.default_rng(0)
    n = 40
    C = _random_cost(rng, n)
    mu = np.full(n, 1.0 / n)
    cost = C @ mu
    order = np.argsort(cost)
    i, j = int(order[0]), int(order[1])
    C[j] = C[j] - (cost[j] - cost[i]) - 5.0e-7   # j now wins by 5e-7
    assert (C @ mu)[i] - (C @ mu)[j] == pytest.approx(5.0e-7, rel=1e-6)
    assert int(spot_greedy(C, mu, 1)[0]) == j


def test_negative_marginal_is_rejected():
    # Non-negativity is what makes the objective submodular, and
    # submodularity is the licence for lazy evaluation; without it the
    # queue can return a non-greedy element.
    rng = np.random.default_rng(2)
    C = _random_cost(rng, 20)
    mu = np.full(20, 1.0 / 20)
    mu[3] = -0.01
    with pytest.raises(ValueError, match="non-negative"):
        spot_greedy(C, mu, 4)
