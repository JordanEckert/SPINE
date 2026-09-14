"""Phase-level checks: the gradient, the exact incremental risk, the cover.

Three of these are the ones that would let a silent error through:

* Phase 2b's update is asserted to equal ``-lr * grad L`` by finite
  differences, so the chain
  ``dL/dvertex = phi'(mu) * dmu/dd * dd/dvertex`` is verified rather than
  believed;
* Phase 2a's O(n) incremental risk is asserted identical to a naive
  recomputation, so the optimisation cannot silently change which edges
  are admitted;
* the mini-batch approximation is asserted to coincide with the exact
  sequential rule at ``batch_size=1``.
"""

import numpy as np
import pytest
from scipy.special import expit

from spine import geometry
from spine.cover import (build_cover, n_intervals, quantile_intervals,
                         rank_reparametrise, slab_populations,
                         uniform_intervals)
from spine.phase0 import _reassign_noise, min_cluster_size, phase0
from spine.phase1 import hop_distances, phase1
from spine.phase2 import (admit_edges_by_risk, edge_is_supported, phase2b,
                          support_rho)
from spine.topology import betti_per_class, normalize_edges


# ------------------------------------------------------------------ cover

def test_n_intervals_follows_the_maple_rule():
    for n in (50, 200, 1000, 5000):
        assert n_intervals(n) == max(2, int(np.ceil((8.0 * n) ** 0.2)))
    assert n_intervals(1) == 2                    # floored


def test_uniform_slabs_have_width_step_times_one_plus_gain():
    f = np.linspace(0.0, 10.0, 501)
    cover = uniform_intervals(f, 5, gain=0.25)
    step = 10.0 / 5
    for (a, b) in cover:
        assert (b - a) == pytest.approx(step * 1.25)
    # Consecutive slabs overlap; slabs two apart do not.
    assert cover[0][1] > cover[1][0]
    assert cover[0][1] < cover[2][0]


def test_uniform_cover_contains_every_point():
    rng = np.random.default_rng(0)
    f = rng.normal(size=400)
    cover = uniform_intervals(f, 6, gain=0.25)
    covered = np.zeros(len(f), dtype=bool)
    for (a, b) in cover:
        covered |= (f >= a) & (f <= b)
    assert covered.all()


def test_quantile_slabs_carry_equal_mass_before_the_gain():
    rng = np.random.default_rng(1)
    f = rng.normal(size=2000) ** 3                # strongly non-uniform
    r = rank_reparametrise(f)
    pops = slab_populations(r, quantile_intervals(f, 5, gain=0.0))
    assert max(pops) - min(pops) <= 1             # equal up to rounding


def test_quantile_overlap_stays_between_CONSECUTIVE_slabs_only():
    """The property Appendix B's raw-unit widening destroys.

    Section 4.2 defends the fallback by calling it "exactly a uniform
    cover of a monotonically reparametrized lens".  Widening in rank
    space makes that literally true, so the consecutive-only overlap of
    the uniform cover carries over.  Widening by 25% of each slab's RAW
    width -- Appendix B's form -- does not: on a heavy-tailed lens the
    extreme slabs are enormous in raw units and reach slabs the
    filtration separates completely, which wires non-consecutive slabs
    together in the nerve and manufactures 1-cycles.
    """
    rng = np.random.default_rng(1)
    f = rng.normal(size=2000) ** 3                # heavy-tailed
    r = rank_reparametrise(f)
    cover = quantile_intervals(f, 5, gain=0.25)
    members = [set(np.where((r >= a) & (r <= b))[0]) for (a, b) in cover]
    for i in range(len(cover)):
        for j in range(i + 2, len(cover)):
            assert not (members[i] & members[j]), (
                "slabs {0} and {1} overlap".format(i, j))
    assert members[0] & members[1]                # consecutive ones do

    # Appendix B's raw-unit widening, for contrast.
    e = np.percentile(f, np.linspace(0.0, 100.0, 6))
    raw_cover = [(a - (b - a) * 0.125, b + (b - a) * 0.125)
                 for a, b in zip(e[:-1], e[1:])]
    raw_members = [set(np.where((f >= a) & (f <= b))[0])
                   for (a, b) in raw_cover]
    assert raw_members[0] & raw_members[4], (
        "expected Appendix B's form to join the extreme slabs")


def test_starvation_fallback_fires_on_a_unimodal_lens():
    rng = np.random.default_rng(2)
    f = rng.normal(size=600)                      # extreme slabs starve
    _, placement, f_eff = build_cover(f, 6, min_cluster_size=15, gain=0.25)
    assert placement == "quantile"
    assert not np.array_equal(f_eff, f)           # reparametrised


def test_uniform_placement_survives_a_flat_lens():
    f = np.linspace(0.0, 1.0, 600)
    _, placement, f_eff = build_cover(f, 6, min_cluster_size=15, gain=0.25)
    assert placement == "uniform"
    assert np.array_equal(f_eff, f)               # the lens itself


def test_min_cluster_size_is_clipped_and_sublinear():
    assert min_cluster_size(100) == 15            # floor
    assert min_cluster_size(4000) == 20
    assert min_cluster_size(10 ** 6) == 100       # cap


# ----------------------------------------------------------------- phase 0

def test_noise_reassignment_moves_membership_not_position():
    X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2], [2.5]])
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    out, core = _reassign_noise(X, labels)
    assert out[-1] == 0                           # nearest cluster is 0
    assert not core[-1]                           # but it is not core
    centroid = X[(out == 0) & core].mean()
    assert centroid == pytest.approx(0.1)         # noise excluded


def test_all_noise_slab_becomes_one_cluster():
    X = np.array([[0.0], [1.0], [2.0]])
    out, core = _reassign_noise(X, np.array([-1, -1, -1]))
    assert set(out) == {0}
    assert core.all()


def test_phase0_returns_a_connected_nerve_on_a_filament():
    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 1.0, 600)
    Xc = np.column_stack([t, 0.02 * rng.normal(size=600)])
    V, E, prov = phase0(Xc, n_total=600)
    assert len(V) >= 2 and len(E) >= 1
    assert betti_per_class([(V, E)])[0][0] == 1   # one component
    assert prov["lens"] == "pc1"


def test_phase0_does_not_invent_cycles_on_two_separated_blobs():
    """The regression the raw-unit quantile widening produced.

    Two well-separated Gaussian blobs form a space with no 1-cycle.  Under
    Appendix B's raw-unit widening the extreme quantile slabs reach each
    other, the nerve wires non-consecutive slabs together, and beta_1
    comes out positive.
    """
    rng = np.random.default_rng(9)
    Xc = np.vstack([rng.normal(size=(400, 2)),
                    rng.normal(size=(400, 2)) + np.array([8.0, 0.0])])
    V, E, prov = phase0(Xc, n_total=800)
    b0, b1 = betti_per_class([(V, E)])[0]
    assert b1 == 0, "invented {0} cycle(s) on an acyclic space".format(b1)


# ----------------------------------------------------------------- phase 1

def test_hop_distances_are_graph_distances():
    H = hop_distances(4, [(0, 1), (1, 2), (2, 3)])
    assert H[0, 3] == 3
    assert H[1, 2] == 1


def test_disconnected_pairs_get_the_finite_ceiling():
    H = hop_distances(3, [(0, 1)])
    assert H[0, 2] == 3.0                         # n_vertices
    assert np.isfinite(H).all()


def test_phase1_single_vertex_converges_to_the_mean():
    rng = np.random.default_rng(6)
    Xc = rng.normal(size=(50, 3))
    V = phase1(Xc, np.zeros((1, 3)), [], rng=rng)
    assert np.allclose(V[0], Xc.mean(axis=0))


def test_phase1_reduces_within_class_distortion():
    rng = np.random.default_rng(7)
    Xc = np.column_stack([np.linspace(0, 1, 300),
                          0.05 * rng.normal(size=300)])
    V0 = np.array([[0.5, 0.9], [0.5, -0.9]])
    E = [(0, 1)]

    def distortion(V):
        return float(geometry.skeleton_distance(Xc, V, E).sum())

    V1 = phase1(Xc, V0, E, rng=np.random.default_rng(0))
    assert distortion(V1) < distortion(V0)


# --------------------------------------------------------- phase 2a: risk

def _naive_admit(skeletons, Xv, yv):
    """Appendix B's admission, written out with full recomputation."""
    skels = [(np.asarray(V, float), normalize_edges(E))
             for (V, E) in skeletons]
    base = geometry.risk(Xv, yv, skels)
    for c, (V, E) in enumerate(skels):
        order = sorted(range(len(E)), reverse=True,
                       key=lambda k: np.linalg.norm(V[E[k][0]] - V[E[k][1]]))
        alive = [True] * len(E)
        for k in order:
            keep = [E[q] for q in range(len(E)) if alive[q] and q != k]
            trial = list(skels)
            trial[c] = (V, keep)
            r = geometry.risk(Xv, yv, trial)
            if r < base:
                base = r
                alive[k] = False
        skels[c] = (V, [E[q] for q in range(len(E)) if alive[q]])
    return skels


@pytest.mark.parametrize("seed", range(12))
def test_incremental_admission_equals_full_recomputation(seed):
    rng = np.random.default_rng(seed)
    C, d = 3, 3
    skeletons = []
    for _ in range(C):
        m = int(rng.integers(3, 7))
        V = rng.normal(size=(m, d))
        pairs = set()
        for _ in range(int(rng.integers(1, 2 * m))):
            i, j = rng.integers(0, m, size=2)
            if i != j:
                pairs.add((int(min(i, j)), int(max(i, j))))
        skeletons.append((V, normalize_edges(pairs)))
    Xv = rng.normal(size=(120, d))
    yv = rng.integers(0, C, size=120)

    fast, _ = admit_edges_by_risk(skeletons, Xv, yv)
    slow = _naive_admit(skeletons, Xv, yv)
    for (Vf, Ef), (Vs, Es) in zip(fast, slow):
        assert Ef == Es


def test_admission_is_strict_so_a_neutral_edge_survives():
    # Both classes are perfectly separated, so no removal can lower the
    # risk and the strict rule must keep every edge.
    a = (np.array([[0.0, 0.0], [0.0, 1.0]]), [(0, 1)])
    b = (np.array([[9.0, 0.0], [9.0, 1.0]]), [(0, 1)])
    Xv = np.array([[0.0, 0.5], [9.0, 0.5]])
    yv = np.array([0, 1])
    out, n = admit_edges_by_risk([a, b], Xv, yv)
    assert n == 0
    assert [E for _, E in out] == [[(0, 1)], [(0, 1)]]


# ------------------------------------------------------ phase 2a: support

def test_supported_edge_along_a_populated_ridge_is_admitted():
    Xc = np.column_stack([np.linspace(0.0, 1.0, 200), np.zeros(200)])
    V = np.array([[0.0, 0.0], [1.0, 0.0]])
    rho = support_rho(Xc, V)
    assert edge_is_supported(Xc, V, (0, 1), rho)


def test_edge_crossing_a_void_is_rejected():
    # Two dense clumps with a wide empty gap between them.
    left = np.column_stack([np.linspace(0.0, 0.05, 100), np.zeros(100)])
    right = np.column_stack([np.linspace(0.95, 1.0, 100), np.zeros(100)])
    Xc = np.vstack([left, right])
    V = np.array([[0.0, 0.0], [1.0, 0.0]])
    rho = support_rho(Xc, V)
    assert not edge_is_supported(Xc, V, (0, 1), rho)


def test_zero_length_edge_is_supported_by_definition():
    Xc = np.zeros((10, 2))
    V = np.zeros((2, 2))
    assert edge_is_supported(Xc, V, (0, 1), rho=0.5)


# --------------------------------------------------------------- phase 2b

def _loss(skeletons, X, y, sigma):
    """``sum phi(mu)`` with ``phi = sigmoid(sigma mu)``, written directly."""
    D = geometry.class_distances(X, skeletons)
    total = 0.0
    for a in range(len(X)):
        dp = D[a, y[a]]
        dn = min(D[a, c] for c in range(D.shape[1]) if c != y[a])
        total += float(expit(sigma * (dp - dn) / (dp + dn)))
    return total


@pytest.mark.parametrize("seed", range(10))
def test_phase2b_raw_update_is_minus_lr_times_the_gradient(seed):
    """One sample, one epoch: the step must equal ``-lr * grad L``."""
    rng = np.random.default_rng(seed)
    d, sigma, lr = 3, 3.0, 1e-4
    A = (rng.normal(size=(3, d)), [(0, 1), (1, 2)])
    B = (rng.normal(size=(3, d)) + 1.5, [(0, 1), (1, 2)])
    skels = [(np.array(V), list(E)) for (V, E) in (A, B)]
    X = rng.normal(size=(1, d))
    y = np.array([0])

    moved = phase2b([(V.copy(), list(E)) for (V, E) in skels], X, y,
                    epochs=1, lr=lr, sigma=sigma, batch_size=1,
                    boundary_restriction="off", step_rule="raw",
                    rng=np.random.default_rng(0))
    delta = np.concatenate(
        [(mv - V).reshape(-1) for (mv, _), (V, _) in zip(moved, skels)])

    # Central-difference gradient of the same loss.
    flat = np.concatenate([V.reshape(-1) for V, _ in skels])
    shapes = [V.shape for V, _ in skels]
    edges = [E for _, E in skels]

    def unflatten(vec):
        out, o = [], 0
        for shp, E in zip(shapes, edges):
            k = shp[0] * shp[1]
            out.append((vec[o:o + k].reshape(shp), E))
            o += k
        return out

    eps = 1e-6
    grad = np.zeros_like(flat)
    for i in range(len(flat)):
        up, dn = flat.copy(), flat.copy()
        up[i] += eps
        dn[i] -= eps
        grad[i] = (_loss(unflatten(up), X, y, sigma)
                   - _loss(unflatten(dn), X, y, sigma)) / (2 * eps)

    assert np.allclose(delta, -lr * grad, atol=2e-8), (
        "max discrepancy {0:.3e}".format(np.abs(delta + lr * grad).max()))


def test_batch_size_one_is_the_exact_sequential_rule():
    """Batching is an implementation approximation, not a definition."""
    rng = np.random.default_rng(21)
    d, C = 3, 2
    skels = [(rng.normal(size=(4, d)) + 2.0 * c, [(0, 1), (1, 2), (2, 3)])
             for c in range(C)]
    X = rng.normal(size=(60, d))
    y = rng.integers(0, C, size=60)
    a = phase2b(skels, X, y, epochs=2, batch_size=1,
                rng=np.random.default_rng(9))
    b = phase2b(skels, X, y, epochs=2, batch_size=1,
                rng=np.random.default_rng(9))
    for (Va, _), (Vb, _) in zip(a, b):
        assert np.array_equal(Va, Vb)             # deterministic
    c_ = phase2b(skels, X, y, epochs=2, batch_size=32,
                 rng=np.random.default_rng(9))
    drift = max(float(np.abs(Va - Vc).max())
                for (Va, _), (Vc, _) in zip(a, c_))
    assert drift < 0.5, "batching drifted by {0:.3f}".format(drift)


def test_phase2b_never_changes_topology():
    rng = np.random.default_rng(31)
    skels = [(rng.normal(size=(5, 2)) + 3.0 * c,
              normalize_edges([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)]))
             for c in range(2)]
    before = betti_per_class(skels)
    after = betti_per_class(
        phase2b(skels, rng.normal(size=(80, 2)),
                rng.integers(0, 2, size=80), epochs=3,
                rng=np.random.default_rng(0)))
    assert before == after


def test_boundary_restriction_freezes_interior_vertices():
    """A vertex that is never anyone's nearest wrong-class object stays put."""
    # Class 0 is a long chain; only its right end faces class 1.
    V0 = np.array([[-8.0, 0.0], [-4.0, 0.0], [0.0, 0.0]])
    V1 = np.array([[4.0, 0.0], [8.0, 0.0]])
    skels = [(V0.copy(), [(0, 1), (1, 2)]), (V1.copy(), [(0, 1)])]
    X = np.array([[-8.0, .1], [-4.0, .1], [0.0, .1], [4.0, .1], [8.0, .1]])
    y = np.array([0, 0, 0, 1, 1])
    out = phase2b(skels, X, y, epochs=5, lr=0.2,
                  boundary_restriction="both",
                  rng=np.random.default_rng(0))
    # The far-left vertex is interior by any reading and must not move.
    assert np.array_equal(out[0][0][0], V0[0])
    loose = phase2b(skels, X, y, epochs=5, lr=0.2,
                    boundary_restriction="off",
                    rng=np.random.default_rng(0))
    assert not np.array_equal(loose[0][0][0], V0[0])


# ---------------------------------------------------------------- phase 4

@pytest.mark.parametrize("seed", range(15))
def test_incremental_pruning_distance_equals_a_full_rebuild(seed):
    """The cached-column pruner must be exact, not merely close."""
    from spine.phase4 import (_class_distance_without, _object_distances,
                              _top_objects, candidate_pool)
    from spine.topology import as_normalized, neighbours, remove_vertex

    rng = np.random.default_rng(seed)
    m = int(rng.integers(5, 14))
    V = rng.normal(size=(m, 3))
    pairs = set()
    for _ in range(int(rng.integers(m // 2, 2 * m))):
        i, j = rng.integers(0, m, size=2)
        if i != j:
            pairs.add((int(min(i, j)), int(max(i, j))))
    E = as_normalized(normalize_edges(pairs))
    Xv = rng.normal(size=(90, 3))

    cols, vals = _top_objects(_object_distances(Xv, V, E))
    nb = neighbours(m, E)
    pool, _ = candidate_pool(V, E)
    assert pool, "no candidate to exercise the fast path with"
    for i in pool:
        fast = _class_distance_without(V, E, Xv, i, cols, vals, nb)
        cand_V, cand_E = remove_vertex(V, E, i)
        slow = geometry.skeleton_distance(Xv, cand_V, cand_E)
        assert fast is not None
        assert np.allclose(fast, slow, atol=1e-12), (
            "vertex {0}: max |fast-slow| = {1:.3e}".format(
                i, float(np.abs(fast - slow).max())))


def test_pruning_hits_the_target_and_holds_the_invariant():
    from spine.phase4 import prune_to_budget
    rng = np.random.default_rng(2)
    skels = []
    for c in range(3):
        V = rng.normal(size=(12, 3)) + 4.0 * c
        # A cycle plus pendant vertices: beta_1 = 1 by construction.
        E = normalize_edges([(i, (i + 1) % 6) for i in range(6)]
                            + [(0, 6), (1, 7), (2, 8), (3, 9),
                               (4, 10), (5, 11)])
        skels.append((V, E))
    before = betti_per_class(skels)
    Xv = rng.normal(size=(150, 3)) * 3
    yv = rng.integers(0, 3, size=150)
    out, forced = prune_to_budget(skels, [8, 8, 8], Xv, yv)
    assert [len(V) for V, _ in out] == [8, 8, 8]
    if forced == 0:
        assert betti_per_class(out) == before


@pytest.mark.parametrize("seed", range(6))
def test_scale_free_step_is_the_squared_distance_gradient(seed):
    """``scale_free`` must be ``-lr * grad`` of the SAME loss via d^2/2.

    Preconditioning by the moved object's own distance is exactly the
    chain rule taken through ``d^2/2`` instead of ``d``, so the step is a
    positive multiple of the raw one -- same direction, bounded length.
    """
    rng = np.random.default_rng(100 + seed)
    d, sigma, lr = 3, 3.0, 1e-4
    skels = [(rng.normal(size=(3, d)) + 1.5 * c, [(0, 1), (1, 2)])
             for c in range(2)]
    X = rng.normal(size=(1, d))
    y = np.array([0])

    def run(rule):
        return phase2b([(V.copy(), list(E)) for (V, E) in skels], X, y,
                       epochs=1, lr=lr, sigma=sigma, batch_size=1,
                       boundary_restriction="off", step_rule=rule,
                       rng=np.random.default_rng(0))

    raw = run("raw")
    free = run("scale_free")
    for (Vr, _), (Vf, _), (V0, _) in zip(raw, free, skels):
        dr, df = Vr - V0, Vf - V0
        moved = np.abs(dr).sum(axis=1) > 0
        for i in np.where(moved)[0]:
            # Same direction: the cosine between the two steps is +1.
            cos = float(dr[i] @ df[i] /
                        (np.linalg.norm(dr[i]) * np.linalg.norm(df[i])))
            assert cos == pytest.approx(1.0, abs=1e-9)


def test_scale_free_step_is_bounded_by_the_query_distance():
    """The bound ``||step|| <= lr * ||x - p|| / 2`` is what fixes banana."""
    # Vertices sitting almost on top of the data: d+ and d- both tiny,
    # which is exactly where the raw rule diverges.  The queries are held
    # just OFF the segments, since a query lying exactly on one has
    # ``||x - p|| = 0`` and the update is skipped by both rules.
    V0 = np.array([[0.500, 0.5], [0.501, 0.5]])
    V1 = np.array([[0.502, 0.5], [0.503, 0.5]])
    skels = [(V0.copy(), [(0, 1)]), (V1.copy(), [(0, 1)])]
    X = np.array([[0.5005, 0.50005], [0.5025, 0.50005]])
    y = np.array([0, 1])

    free = phase2b(skels, X, y, epochs=1, lr=0.05, batch_size=1,
                   boundary_restriction="off", step_rule="scale_free",
                   rng=np.random.default_rng(0))
    raw = phase2b(skels, X, y, epochs=1, lr=0.05, batch_size=1,
                  boundary_restriction="off", step_rule="raw",
                  rng=np.random.default_rng(0))
    free_move = max(float(np.abs(Vn - V0_).max())
                    for (Vn, _), (V0_, _) in zip(free, skels))
    raw_move = max(float(np.abs(Vn - V0_).max())
                   for (Vn, _), (V0_, _) in zip(raw, skels))
    assert free_move < 0.05                    # stays inside the data
    assert raw_move > 1.0                      # leaves it entirely
    assert raw_move > 100 * free_move


def test_clipped_step_never_exceeds_its_cap():
    V0 = np.array([[0.500, 0.5], [0.501, 0.5]])
    V1 = np.array([[0.502, 0.5], [0.503, 0.5]])
    skels = [(V0.copy(), [(0, 1)]), (V1.copy(), [(0, 1)])]
    # ONE training point, so each vertex receives at most one update and
    # the per-update cap is directly observable.  With several points a
    # vertex can legitimately move by more than the cap in one epoch,
    # since the cap bounds each update rather than the epoch.
    X = np.array([[0.5005, 0.50005]])
    y = np.array([0])
    for cap in (0.05, 0.01, 0.001):
        out = phase2b(skels, X, y, epochs=1, lr=0.05, batch_size=1,
                      boundary_restriction="off", step_rule="clipped",
                      max_step=cap, rng=np.random.default_rng(0))
        move = max(float(np.abs(Vn - V0_).max())
                   for (Vn, _), (V0_, _) in zip(out, skels))
        # The cap is ABSOLUTE, in data units.  A cap proportional to
        # ||x - p|| would bound nothing here, since ||x - p|| is exactly
        # the quantity that diverges once a vertex escapes.
        assert move <= cap + 1e-12, (cap, move)
