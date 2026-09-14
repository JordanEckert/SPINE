"""SPINE -- Skeletal Prototypes on Iterated Nerve Expansions.

A prototype method whose per-class model is an embedded 1-complex -- a
piecewise-linear curve network -- rather than a finite point set.  The
complex's topology is fixed once, from a class-conditional Mapper
construction, and is then preserved exactly by every subsequent
operation; vertex positions and model size are driven entirely by
classification risk.

Contract in this repository
---------------------------
SPINE is a GENERATION method: its vertices are synthetic points, so
``select`` returns the sentinel index for every prototype and the
harness's ``validate_selection`` checks that.  It is budget-matched:
whatever ``total_count`` the runner injects is the exact number of
VERTICES emitted, apportioned across classes by the harness's own
``allocate_per_class`` -- the same proportional rule the other per-class
comparators use, so the allocation is a property of the dataset rather
than of this method.

Two decision rules, one fitted model
------------------------------------
``select`` returns the vertices, which the harness scores with sklearn's
1-NN exactly like any other prototype set: that is the rung-1 ablation of
section 13, the model with its segments removed at prediction time.  The
fitted skeleton itself is exposed as :attr:`SPINE.estimator_`, whose
``predict`` implements ``argmin_c dist(x, |S_c|)`` -- the actual method.
Scoring the same fitted model both ways is what separates the
hypothesis-class change from everything else in the pipeline.

The validation split
--------------------
Section 1 requires a validation split reserved at the outset, and every
risk-based decision -- edge admission, growth site selection, pruning --
is evaluated on it and never on the data used for fitting.  A consequence
worth stating plainly when results are read: SPINE fits on 75% of the
training fold while the comparators fit on 100% of it.  That is what the
specification asks for, and it is a handicap, not an advantage.  The
split is STRATIFIED here (Appendix B permutes without stratifying), so
that no class can be absent from the validation set and silently escape
every risk-based decision.
"""

import json

import numpy as np

from harness.base import (SYNTHETIC_INDEX, PrototypeSelectorBase,
                          allocate_per_class)
from sklearn.model_selection import StratifiedShuffleSplit

from . import geometry
from .cover import DEFAULT_GAIN
from .phase0 import phase0
from .phase1 import phase1
from .phase2 import phase2a, phase2b
from .phase3 import grow_class
from .phase4 import prune_to_budget
from .topology import betti_per_class, normalize_edges


class SkeletonClassifier:
    """The section-1 decision rule as a fitted, picklable predictor.

    Deliberately not an sklearn estimator: it is never fitted by sklearn,
    never cloned, and never grid-searched.  It holds the skeletons and
    answers ``predict``.
    """

    def __init__(self, skeletons, classes):
        self.skeletons_ = [(np.asarray(V, dtype=float), list(E))
                           for (V, E) in skeletons]
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.classes_[geometry.predict(X, self.skeletons_)]

    def decision_margin(self, X):
        """``mu = (d+ - d-)/(d+ + d-)`` against the predicted class.

        Section 2 notes this is bounded in ``[-1, 1]`` and monotone in
        confidence by construction, so it is the natural soft decision
        for a method whose output is otherwise strictly hard.  Provided
        because it costs nothing; nothing in the pipeline consumes it.
        """
        X = np.asarray(X, dtype=float)
        D = geometry.class_distances(X, self.skeletons_)
        part = np.partition(D, 1, axis=1)
        d1, d2 = part[:, 0], part[:, 1]
        s = d1 + d2
        return np.where(s > 0, (d1 - d2) / np.where(s > 0, s, 1.0), 0.0)

    @property
    def n_vertices(self):
        return int(sum(len(V) for V, _ in self.skeletons_))

    @property
    def n_segments(self):
        return int(sum(len(E) for _, E in self.skeletons_))


class SPINE(PrototypeSelectorBase):
    """Skeletal Prototypes on Iterated Nerve Expansions (generation)."""

    is_generator = True
    deterministic = False
    order_dependent = True

    def __init__(self, lens="pc1", lens_scaling="standardize",
                 gain=DEFAULT_GAIN, n_neighbors=10,
                 val_fraction=0.25, admission="risk",
                 boundary_restriction="both", use_edges=True,
                 phase1_epochs=30, sigma_0=2.0, sigma_f=0.05,
                 eps_0=0.4, eps_f=0.02,
                 fit_epochs=25, refit_epochs=8, final_epochs=10,
                 lr=0.05, sigma=3.0, batch_size=256,
                 step_rule="scale_free", max_step=1.0,
                 overshoot=1.2, grow_chunk=3, grow_rounds=3):
        self.lens = lens
        self.lens_scaling = lens_scaling
        self.gain = gain
        self.n_neighbors = n_neighbors
        self.val_fraction = val_fraction
        self.admission = admission
        self.boundary_restriction = boundary_restriction
        self.use_edges = use_edges
        self.phase1_epochs = phase1_epochs
        self.sigma_0 = sigma_0
        self.sigma_f = sigma_f
        self.eps_0 = eps_0
        self.eps_f = eps_f
        self.fit_epochs = fit_epochs
        self.refit_epochs = refit_epochs
        self.final_epochs = final_epochs
        self.lr = lr
        self.sigma = sigma
        self.batch_size = batch_size
        self.step_rule = step_rule
        self.max_step = max_step
        self.overshoot = overshoot
        self.grow_chunk = grow_chunk
        self.grow_rounds = grow_rounds

    # ------------------------------------------------------------ helpers

    def _split(self, X, y, rng):
        """Stratified train/validation split of the training fold.

        Falls back to the largest stratified split the data admits when a
        class is too small to appear on both sides, and raises when the
        fold cannot support a validation split at all -- that is a
        protocol decision for the caller, not something to absorb.
        """
        _, counts = np.unique(y, return_counts=True)
        n_val = int(round(self.val_fraction * len(X)))
        n_val = max(len(counts), min(n_val, len(X) - len(counts)))
        if counts.min() < 2:
            raise ValueError(
                "SPINE needs at least 2 members in every class to reserve "
                "a stratified validation split; smallest class has {0}"
                .format(int(counts.min())))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=n_val,
            random_state=int(rng.integers(0, 2 ** 31 - 1)))
        train_idx, val_idx = next(splitter.split(X, y))
        return train_idx, val_idx

    def _errors_fn(self, skeletons, c, Xv_c):
        """Callable giving class ``c``'s validation errors for a candidate.

        Only class ``c``'s skeleton changes while it grows, so the other
        classes are reduced once to their pointwise minimum and argmin
        and reused for every candidate -- exact, tie-breaking included.
        """
        if len(Xv_c) == 0:
            return lambda V, E: np.zeros(0, dtype=bool)
        other_min, other_arg = geometry.other_class_extrema(
            Xv_c, skeletons, c)

        def fn(V, E):
            d_c = geometry.skeleton_distance(Xv_c, V, E)
            return ~geometry.predicted_is_class(d_c, other_min, other_arg, c)
        return fn

    # --------------------------------------------------------------- fit

    def select(self, X_train, y_train, **params):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train)
        random_state = params.get("random_state")
        if random_state is None:
            raise ValueError("SPINE requires a random_state in params")
        total_count = params.get("total_count")
        if total_count is None:
            raise ValueError(
                "SPINE is budget-matched and requires total_count")
        total_count = int(total_count)
        rng = np.random.default_rng(random_state)

        self.classes_ = np.unique(y)
        C = len(self.classes_)
        if C < 2:
            raise ValueError(
                "SPINE's margin is defined against a nearest wrong class; "
                "a single-class training fold has none")
        y_pos = np.searchsorted(self.classes_, y)

        # Budget: the harness's own proportional apportionment, computed
        # on the FULL training fold's labels so that it matches what the
        # runner hands the other per-class comparators.
        alloc = allocate_per_class(y, total_count)
        targets = [int(alloc[c]) for c in self.classes_]
        if sum(targets) != total_count:
            raise AssertionError(
                "per-class allocation sums to {0}, not the injected budget "
                "{1}".format(sum(targets), total_count))

        train_idx, val_idx = self._split(X, y_pos, rng)
        Xt, yt = X[train_idx], y_pos[train_idx]
        Xv, yv = X[val_idx], y_pos[val_idx]
        X_by_class = [Xt[yt == c] for c in range(C)]

        # ---- Phase 0: structure --------------------------------------
        provenance = {"phase0": []}
        skeletons = []
        for c in range(C):
            Xc = X_by_class[c]
            if len(Xc) == 0:
                raise ValueError(
                    "class {0!r} has no training points after the "
                    "validation split".format(self.classes_[c]))
            V, E, prov = phase0(Xc, len(Xt), gain=self.gain,
                                lens=self.lens,
                                n_neighbors=self.n_neighbors,
                                lens_scaling=self.lens_scaling)
            provenance["phase0"].append(prov)
            skeletons.append((V, E))
        provenance["betti_nerve"] = betti_per_class(skeletons)

        # ---- Phase 1: annealed representation ------------------------
        skeletons = [
            (phase1(X_by_class[c], V, E, T=self.phase1_epochs,
                    sigma_0=self.sigma_0, sigma_f=self.sigma_f,
                    eps_0=self.eps_0, eps_f=self.eps_f, rng=rng), E)
            for c, (V, E) in enumerate(skeletons)]

        # ---- Phase 2a: admission, then FREEZE ------------------------
        skeletons, admit_info = phase2a(
            skeletons, Xv, yv, X_by_class, criterion=self.admission)
        provenance["admission"] = admit_info
        frozen = betti_per_class(skeletons)
        provenance["betti_frozen"] = frozen

        # ---- Phase 2b: discriminative fitting ------------------------
        skeletons = phase2b(
            skeletons, Xt, yt, epochs=self.fit_epochs, lr=self.lr,
            sigma=self.sigma, batch_size=self.batch_size,
            boundary_restriction=self.boundary_restriction,
            step_rule=self.step_rule, max_step=self.max_step, rng=rng)

        # ---- Phase 3: growth to the overshoot target -----------------
        n_fallback = 0
        overshoot_targets = [
            min(len(X_by_class[c]),
                int(np.ceil(self.overshoot * targets[c])))
            for c in range(C)]
        # Section 7 alternates "fit to convergence, add K vertices, refit"
        # until the overshoot target is reached, and section 11 records K
        # as coarse -- it "affects runtime, not the solution".  Appendix
        # B's driver fixes K = 3 over 3 rounds, i.e. at most 9 vertices per
        # class, which cannot attain a budget of the size this harness
        # injects (RSP3 anchors at 18-26% of the training fold, hundreds
        # of vertices per class).  The number of ALTERNATIONS is therefore
        # what is held fixed here -- at Appendix B's own value of 3 -- and
        # K is what scales, so the fit/grow rhythm matches the reference
        # driver's at every budget instead of the round count growing
        # linearly with M.
        rounds_left = max(1, int(self.grow_rounds))
        while rounds_left > 0:
            grew = False
            for c in range(C):
                need = overshoot_targets[c] - len(skeletons[c][0])
                if need <= 0:
                    continue
                chunk = max(int(self.grow_chunk),
                            int(np.ceil(need / rounds_left)))
                Xv_c = Xv[yv == c]
                V, E, nf = grow_class(
                    skeletons[c][0], skeletons[c][1], Xv_c,
                    self._errors_fn(skeletons, c, Xv_c),
                    X_by_class[c], min(need, chunk))
                n_fallback += nf
                if len(V) == len(skeletons[c][0]):
                    overshoot_targets[c] = len(V)   # no site: stop asking
                    continue
                skeletons[c] = (V, E)
                grew = True
            rounds_left -= 1
            if not grew:
                break
            skeletons = phase2b(
                skeletons, Xt, yt, epochs=self.refit_epochs, lr=self.lr,
                sigma=self.sigma, batch_size=self.batch_size,
                boundary_restriction=self.boundary_restriction,
            step_rule=self.step_rule, max_step=self.max_step, rng=rng)
        provenance["growth_representation_fallbacks"] = int(n_fallback)

        # Growth may have stopped short of the budget only if a class ran
        # out of points; the budget is then unattainable and that is a
        # protocol decision, surfaced rather than absorbed.
        for c in range(C):
            if len(skeletons[c][0]) < targets[c]:
                raise AssertionError(
                    "class {0!r} reached {1} vertices but its budget is "
                    "{2}; the class has {3} training points, so the "
                    "budget cannot be met -- this needs a protocol "
                    "decision, not a silent shortfall".format(
                        self.classes_[c], len(skeletons[c][0]),
                        targets[c], len(X_by_class[c])))

        # ---- Phase 4: prune to the exact budget ----------------------
        skeletons, n_forced = prune_to_budget(skeletons, targets, Xv, yv)
        provenance["forced_removals"] = int(n_forced)
        skeletons = phase2b(
            skeletons, Xt, yt, epochs=self.final_epochs, lr=self.lr,
            sigma=self.sigma, batch_size=self.batch_size,
            boundary_restriction=self.boundary_restriction,
            step_rule=self.step_rule, max_step=self.max_step, rng=rng)

        # ---- Phase 5: finalize ---------------------------------------
        final = betti_per_class(skeletons)
        provenance["betti_final"] = final
        # Section 9.1 -- the implementation CHECK: final == frozen.
        provenance["invariant_held"] = bool(final == frozen)
        # Section 9.2 -- the reportable RESULT: how much of the nerve's
        # topology survived admission.  This is the pair that decides
        # whether the topological framing is load-bearing, so it is a
        # named field rather than something a reader has to derive.
        nerve = provenance["betti_nerve"]
        provenance["topology_retention_nerve_to_frozen"] = {
            "beta0_nerve": sum(b[0] for b in nerve),
            "beta0_frozen": sum(b[0] for b in frozen),
            "beta1_nerve": sum(b[1] for b in nerve),
            "beta1_frozen": sum(b[1] for b in frozen),
        }
        if n_forced == 0 and final != frozen:
            raise AssertionError(
                "section 10 invariant violated with no forced removal: "
                "Betti numbers were {0} at the Phase 2a freeze and {1} at "
                "convergence -- this is a bug in the grammar, not a "
                "result".format(frozen, final))
        # Section 10's SOUNDNESS bound: training may lose topology but can
        # never invent it.  Forced removals are the one operation that can
        # breach it (deleting an isolated vertex lowers beta_0 below the
        # nerve's), so the bound is checked explicitly rather than assumed
        # to follow from the grammar.
        soundness = all(b1f <= b1n and b0f >= b0n
                        for (b0n, b1n), (b0f, b1f) in zip(nerve, final))
        provenance["soundness_bound_holds"] = bool(soundness)
        if not soundness and n_forced == 0:
            raise AssertionError(
                "section 10 soundness bound violated with no forced "
                "removal: nerve {0}, final {1}".format(nerve, final))

        if self.use_edges:
            self.skeletons_ = [(V, normalize_edges(E))
                               for (V, E) in skeletons]
        else:
            # Rung 1 of section 13's ladder: identical vertices, segments
            # removed at prediction time.
            self.skeletons_ = [(V, []) for (V, _) in skeletons]
        self.estimator_ = SkeletonClassifier(self.skeletons_, self.classes_)

        X_proto = np.vstack([V for V, _ in self.skeletons_])
        y_proto = np.concatenate([
            np.full(len(V), self.classes_[c]) for c, (V, _) in
            enumerate(self.skeletons_)]).astype(self.classes_.dtype)
        if len(X_proto) != total_count:
            raise AssertionError(
                "SPINE emitted {0} vertices but the injected budget is {1}"
                .format(len(X_proto), total_count))

        # Exemplars (section 9.3): the nearest actual training point to
        # each vertex.  Recovers the interpretability of prototype
        # SELECTION without ever constraining a prototype to be a data
        # point, and is recorded rather than returned because the
        # harness's index contract reserves the synthetic sentinel for
        # every generated prototype.
        self.exemplars_ = self._exemplars(Xt, yt, train_idx)
        provenance["n_segments"] = int(
            sum(len(E) for _, E in self.skeletons_))
        provenance["step_rule"] = self.step_rule
        provenance["boundary_restriction"] = self.boundary_restriction
        provenance["n_train_fit"] = int(len(Xt))
        provenance["n_val"] = int(len(Xv))
        self.spine_params_ = provenance

        return X_proto, y_proto, np.full(len(X_proto), SYNTHETIC_INDEX)

    def _exemplars(self, Xt, yt, train_idx):
        """Row index (into the ORIGINAL training fold) nearest each vertex. 
        Not used in manuscript results, but recorded in the provenance for interpretability.
        """
        from scipy.spatial.distance import cdist
        out = []
        for c, (V, _) in enumerate(self.skeletons_):
            rows = np.where(yt == c)[0]
            nearest = cdist(V, Xt[rows], "sqeuclidean").argmin(axis=1)
            out.extend(int(train_idx[rows[j]]) for j in nearest)
        return np.asarray(out, dtype=int)

    @property
    def name(self):
        return "SPINE" if self.use_edges else "SPINE-noedges"

    @property
    def params_json(self):
        return json.dumps(self.spine_params_, sort_keys=True, default=str)
