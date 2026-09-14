"""Frozen experimental configuration.

Single source of truth for the evaluation campaign.  Nothing here is
selected on test data; every constant is declared a priori.
"""

#: Base seed for every derived RNG (the date this configuration was frozen).
BASE_SEED = 20260723

N_OUTER_FOLDS = 10
N_INNER_FOLDS = 5

ALPHA = 0.05

#: Scale features (scaler fit on each training fold) before any distance
#: is computed.  True for the frozen protocol; the raw-features ablation
#: flips this to False through the runner's ``normalize`` argument.  WHICH scaler is SCALING.
NORMALIZE = True

#: -----------------------------------------------------------------
#: Feature scaling.  Changed from "minmax" to "standard" on 2026-07-31.
#:
#: The prototype-selection/generation literature min-max scales to
#: [0, 1] -- Triguero et al.'s prototype-generation taxonomy states it
#: outright ("all data set have been normalized between 0 and 1"), and
#: because nearly every PG/PS paper benchmarks on KEEL with KEEL's
#: published partitions, the convention propagated.  That is why this
#: protocol started there.
#:
#: The method under test is not a pure selector, though.  SPINE's
#: Phase 2b is a GLVQ-type gradient update, and
#: that literature standardises, for a reason that bites here: min-max
#: fixes each feature's RANGE, and range is set by extremes, so one
#: outlier compresses the bulk of the data into a sliver of the unit
#: cube.  Small typical distances are exactly where a GLVQ gradient's
#: 1/distance factor explodes.  What that mechanism aggravates,
#: though, is a divergence of the RAW reference step rule
#: (phase2.step_rule = "raw", Appendix B as specified, retained so the
#: defect stays reproducible) rather than a defect of the scaling
#: itself: re-measured on KEEL banana (fold 0, 1167 vertices) on
#: 2026-08-24, the raw rule ends with 99.9% of the vertices outside
#: the scaled training data's own bounding box at 0.553 accuracy under
#: min-max, and it diverges under standardisation too -- 63% of the
#: vertices outside the box, 0.768 by 1-NN while the skeleton readout
#: holds 0.877.  Under the shipped "scale_free" rule the two scalings
#: land within two test points of each other on that fold (0.883 under
#: min-max, 0.887 under standardisation), so what the scaling costs
#: the shipped method across the roster is an open question, measured
#: by scripts/run_scaling.py -- a scaling x step-rule sweep carrying
#: per-epoch trajectory and box-escape readouts -- rather than settled
#: here.
#:
#: Standardisation remains the frozen default: the discriminative
#: phases inherit the preprocessing their own literature uses, and the
#: min-max behaviour is kept reachable (SCALING = "minmax") so that
#: characterisation can be run and reported rather than asserted.
#:
#: "robust" (median/IQR) is available and is the genuinely
#: outlier-resistant option; z-scoring fixes the scale problem but not
#: the outlier problem, since mean and standard deviation are both
#: outlier-sensitive.
#: -----------------------------------------------------------------
SCALING = "standard"

SCALERS = ("standard", "minmax", "robust")


def make_scaler(scaling=None):
    """The protocol's feature scaler, fit by the caller on a train fold."""
    from sklearn.preprocessing import (MinMaxScaler, RobustScaler,
                                       StandardScaler)
    kind = SCALING if scaling is None else scaling
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(
        "unknown scaling {0!r}; expected one of {1}".format(kind, SCALERS))

#: -----------------------------------------------------------------
#: LVQ3 inner-CV tuning grid.  LVQ3 is budget-matched to the anchor's
#: total prototype count (like SPOTGreedy), so the comparison stays at
#: a fixed budget; only its two placement hyperparameters are tuned.
#: Tuned by stratified inner 5-fold CV on the training fold, criterion =
#: highest mean 1-NN accuracy; grid order is the tie-break order (first
#: maximum wins).  The learning rate (alpha = 0.1), the number of epochs
#: (30 passes over the training fold), and the codebook size (the matched
#: budget) are fixed; the window width and the LVQ3 stabilising factor
#: epsilon are searched.
#: -----------------------------------------------------------------
LVQ3_TUNING_GRID = {
    "window": [0.2, 0.3],
    "epsilon": [0.1, 0.3],
}

#: The method whose per-fold prototype count is the total budget of
#: every budget-matched method.  RSP3 is parameter-free and chooses its
#: own count, so anchoring on it puts the method under test at a
#: comparator's own operating point rather than the reverse.  It is
#: required in every run and runs first inside each fold.
BUDGET_ANCHOR = "rsp3"

#: Methods of the main comparison, in table order.  The per-class
#: budgets of 'kmeans' and 'random' are a proportional apportionment of
#: the anchor's total over the training class frequencies -- never read
#: off another method's prototype labels, so no method in the table can
#: hand a comparator its own idea of a good class allocation.
MAIN_METHODS = (
    "full",        # NOP reference (reduction 0)
    "rsp3",        # parameter-free comparator AND the budget anchor
    "spine",       # METHOD UNDER TEST; budget-matched (RSP3 total).
                   # Emits TWO rows from one fit: SPINE+1NN (vertices,
                   # scored like any prototype set) and SPINE+Graph (the
                   # same model under the skeleton decision rule).
    "kmeans",      # budget-matched: proportional per-class counts
    "random",      # budget-matched structureless control (per-class)
    "lvq3",        # budget-matched (RSP3 total); window/epsilon inner-CV tuned
    "spotgreedy",  # budget-matched: RSP3 total count (selection method)
    "glvq",        # budget-matched; the point-prototype counterpart of
                   # SPINE's Phase 2b -- same margin, isolated prototypes
    "gng",         # budget-matched; a graph-building generator whose
                   # edges are training-time only, so it isolates the
                   # hypothesis-class change from having a graph at all
)

#: The result-row label(s) each MAIN_METHODS entry emits, in table
#: order.  'spine' maps to TWO labels because one fit is scored two ways
#: (see harness.stats.SPINE_READOUTS); every other method emits
#: one row.  This mapping is the campaign's DECLARED roster: the
#: statistics validate the collected results against it, so a comparator
#: whose shard failed or was never run cannot silently shrink the field
#: the method under test is ranked against.
METHOD_LABELS = {
    "full": ("Full",),
    "rsp3": ("RSP3",),
    "spine": ("SPINE+1NN", "SPINE+Graph"),
    "kmeans": ("KMeans",),
    "random": ("Random",),
    "lvq3": ("LVQ3",),
    "spotgreedy": ("SPOT",),
    "glvq": ("GLVQ",),
    "gng": ("GNG",),
}

#: Every label a complete campaign emits, in table order.
CAMPAIGN_LABELS = tuple(
    label for name in MAIN_METHODS for label in METHOD_LABELS[name])

#: -----------------------------------------------------------------
#: Budget curve (scripts/run_curve.py).  The main campaign fixes one
#: operating point -- the anchor's own count -- which answers "who wins
#: at RSP3's budget" but not "how does each method degrade as the
#: budget shrinks".  The curve script sweeps the budget instead.
#:
#: The grid is declared as FRACTIONS of the training fold, i.e. as
#: reduction severities, a 1-2-5 grid (approximately log-spaced) from
#: 0.5% to 25% (changed back from prototypes per class 2026-08-23).  A
#: per-class count equalises the per-class CAPACITY across datasets but
#: not the severity: 50 per class is 94% of a wine training fold and
#: 0.6% of a magic one, and the 10% cap that kept the count grid a
#: compression grid then dropped datasets from the upper grid points, so
#: cross-budget comparisons were confounded by roster attrition.  A
#: fraction equalises severity, so the same grid point means the same
#: compression on every dataset and every dataset keeps every point.
#: The realised budget on a fold is
#:
#:     budget = max(round(rho * n_train), n_classes)
#:
#: (round half up), identical for every swept method at that point, and
#: it is split across classes by the same PROPORTIONAL apportionment the
#: main campaign uses (base.allocate_per_class), which never gives a
#: class fewer than one prototype.  The ``max`` is the explicit floor
#: the fraction grid lacked before: below it the per-class comparators
#: cannot honour the budget (wine's 160-row fold at 0.5% is 1 prototype
#: for 3 classes).  Where the floor binds, the row records it
#: (``budget_floored``) and the REALISED fraction ``budget / n_train``
#: is written beside the nominal one, so the x-position the row really
#: sits at is never inferred.  Where two nominal fractions realise the
#: same budget (on this roster only wine and sonar), the construction is
#: reused and the repeated rows carry ``shared_build``; the grid stays
#: complete on every dataset by construction.  The grid's top, 25%,
#: replaces the old cap as the declared upper bound of the compression
#: regime; it is a bound, not a convergence claim (at the campaign's
#: anchor budgets several methods still differ from, and some beat, the
#: Full set).  The RSP3 anchor keeps roughly 10-57% of a fold across
#: the roster (mean ~29%): on ten datasets it lies above the grid's
#: top, on seven below it, so each per-dataset figure marks the
#: anchor's own realised fraction.
#:
#: Frozen a priori like every other constant here.  RSP3 is
#: parameter-free and cannot be moved to a budget, so it appears on the
#: curve at its natural operating point, as a reference level rather
#: than a swept series.  No formal statistics are attached to the curve:
#: it is a graphical instrument.
#: -----------------------------------------------------------------
BUDGET_CURVE_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.25)

#: Classifier sets: 1-NN is primary (all main tables); SVM and MLP at
#: library defaults belong to the "generalization across classifiers"
#: section and are switched on per run (--classifiers svm mlp).
PRIMARY_CLASSIFIER = "1nn"

#: -----------------------------------------------------------------
#: Dataset manifest.  KEEL names as in the repository's classification
#: download URLs (https://sci2s.ugr.es/keel/dataset/data/classification/
#: <keel_name>.zip).  Expected shapes are asserted at load time: a
#: mismatch is a hard error ("figure out why", never silently adapt).
#: n/d/n_classes for entries marked verified=True were checked against
#: real KEEL .dat files; entries marked verified=False carry literature
#: values -- their first load verifies or fails loudly.
#: Two datasets are documented deviations, not in the KEEL repository:
#: EEG Eye State (fetched from the UCI archive) and waveform (fetched
#: from OpenML, data id 60 -- KEEL does not host it).
#: -----------------------------------------------------------------
DATASETS = {
    "banana":            {"source": "keel", "keel_name": "banana",
                          "n": 5300, "d": 2, "n_classes": 2,
                          "verified": True},
    # phoneme: n/d/classes confirmed from a real copy of the data, but
    # the KEEL-format header parse itself could not be exercised in the
    # build sandbox (the available copy was header-stripped CSV).
    "phoneme":           {"source": "keel", "keel_name": "phoneme",
                          "n": 5404, "d": 5, "n_classes": 2,
                          "verified": True},
    # waveform: KEEL does not host it, so it is sourced from OpenML
    # (waveform-5000, data id 60 -- the 40-attribute noisy version) and
    # split by our own seeded stratified k-fold, like EEG Eye State.
    "waveform":          {"source": "openml",
                          "url": ("https://openml.org/data/v1/download/"
                                  "60/waveform-5000.arff"),
                          "n": 5000, "d": 40, "n_classes": 3,
                          "verified": False},
    "texture":           {"source": "keel", "keel_name": "texture",
                          "n": 5500, "d": 40, "n_classes": 11,
                          "verified": True},
    "satimage":          {"source": "keel", "keel_name": "satimage",
                          "n": 6435, "d": 36, "n_classes": 6,
                          "verified": True},
    "ring":              {"source": "keel", "keel_name": "ring",
                          "n": 7400, "d": 20, "n_classes": 2,
                          "verified": True},
    "twonorm":           {"source": "keel", "keel_name": "twonorm",
                          "n": 7400, "d": 20, "n_classes": 2,
                          "verified": True},
    "penbased":          {"source": "keel", "keel_name": "penbased",
                          "n": 10992, "d": 16, "n_classes": 10,
                          "verified": True},
    "eeg-eye-state":     {"source": "uci_eeg",
                          "url": ("https://archive.ics.uci.edu/ml/"
                                  "machine-learning-databases/00264/"
                                  "EEG%20Eye%20State.arff"),
                          "n": 14980, "d": 14, "n_classes": 2,
                          "verified": False},
    "magic":             {"source": "keel", "keel_name": "magic",
                          "n": 19020, "d": 10, "n_classes": 2,
                          "verified": True},
    "letter":            {"source": "keel", "keel_name": "letter",
                          "n": 20000, "d": 16, "n_classes": 26,
                          "verified": True},
    "segment":           {"source": "keel", "keel_name": "segment",
                          "n": 2310, "d": 19, "n_classes": 7,
                          "verified": False},
    # --- datasets shared with the JSM2026 balanced roster (KEEL
    # analogues; verified=False so the first load asserts the shape) ---
    "wine":              {"source": "keel", "keel_name": "wine",
                          "n": 178, "d": 13, "n_classes": 3,
                          "verified": False},
    "sonar":             {"source": "keel", "keel_name": "sonar",
                          "n": 208, "d": 60, "n_classes": 2,
                          "verified": False},
    # ionosphere: KEEL drops UCI's all-zero 2nd attribute -> d=33 not 34.
    "ionosphere":        {"source": "keel", "keel_name": "ionosphere",
                          "n": 351, "d": 33, "n_classes": 2,
                          "verified": False},
    # wdbc = Wisconsin Diagnostic Breast Cancer (JSM2026 "BreastCancer").
    "wdbc":              {"source": "keel", "keel_name": "wdbc",
                          "n": 569, "d": 30, "n_classes": 2,
                          "verified": False},
    # spambase: KEEL ships 4597 rows (UCI/OpenML have 4601).
    "spambase":          {"source": "keel", "keel_name": "spambase",
                          "n": 4597, "d": 57, "n_classes": 2,
                          "verified": False},
}

KEEL_URL = ("https://sci2s.ugr.es/keel/dataset/data/classification/"
            "{keel_name}.zip")

#: KEEL's pre-computed cross-validation partitions live at the same base
#: path (``<keel_name>-10-fold.zip`` / ``<keel_name>-5-fold.zip``); each
#: archive holds ``<keel_name>-<K>-<k>tra.dat`` / ``-<k>tst.dat`` for the
#: k = 1..K train/test pairs.
KEEL_FOLD_URL = ("https://sci2s.ugr.es/keel/dataset/data/classification/"
                 "{keel_name}-{k}-fold.zip")
