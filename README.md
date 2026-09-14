# SPINE

Reference implementation and evaluation harness for **SPINE** — Skeletal
Prototypes on Iterated Nerve Expansions — together with the eight
comparator baselines it is benchmarked against.

## Quickstart

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                              # ~1 min, offline
python scripts/run_experiments.py --datasets wine   # one small dataset
```

`pytest` needs no network and no data. `run_experiments.py` downloads
what it needs into `data/` on first use and writes one CSV per dataset
into `results_run/`. Start with `wine` (seconds) before committing to
`letter` or `magic` (hours). `scripts/download_data.py` prefetches the
whole manifest if you would rather download once and then run offline.

`scripts/run_stats.py` is deliberately *not* in that list: it is a
whole-campaign instrument and refuses a partial one. It hard-errors on
fewer than two datasets, on a duplicated shard, on an incomplete
(dataset, method) block, and on any comparator that
`config.MAIN_METHODS` declares but the results lack. Run it once the
campaign is complete:

```
python scripts/run_experiments.py     # all 17 datasets, all 10 methods
python scripts/run_stats.py           # writes results/stats/
```

`run_stats.py` writes into an existing `results_run/stats/` without clearing
it, so delete that directory before a re-run rather than reading a mix
of old and new tables.

## Layout

One installable distribution, `spine`, carrying three top-level
packages: the method (`spine`) and the two supporting roles
(`baselines`, `harness`) beside it.

```
src/spine/              the method under test
  model.py                the fit, the budget, and SkeletonClassifier —
                          the decision rule as a fitted predictor
  geometry.py             exact distance to an embedded 1-complex
  topology.py             Betti numbers, degrees, the removal guard
  lens.py                 per-class lenses (PC1, eccentricity, Fiedler)
  cover.py                interval count, gain, interval placement
  phase0.py               structure: cover, HDBSCAN, nerve
  phase1.py               annealed representation on the nerve
  phase2.py               edge admission (2a), discriminative fit (2b)
  phase3.py               growth by the homotopy-preserving grammar
  phase4.py               pruning to the exact budget

src/baselines/          the seven comparators
  reference.py            Full (NOP), stratified Random, per-class K-Means
  rsp3.py                 RSP3 (Sanchez 2004), Python port -- also the
                          campaign's budget anchor
  lvq3.py                 LVQ3 (Kohonen), Python port
  spotgreedy.py           SPOTgreedy (Gurumoorthy et al. 2021),
                          lazy-greedy implementation; see provenance
  glvq.py                 GLVQ (Sato & Yamada 1996) via the sklvq
                          package; see provenance
  gng.py                  GNG (Fritzke 1995), run per class over the
                          vendored implementation below
  vendor/
    growing_neural_gas.py   Guille's GNG, MIT-licensed, repaired
                          forward; every edit traced in its docstring

src/harness/            the protocol
  base.py                 method contract + budget allocation + validation
  config.py               frozen protocol constants + dataset manifest
  datasets.py             KEEL/UCI/OpenML loading + KEEL fold-partition
                          loader
  metrics.py              accuracy / reduction-rate evaluation
  runner.py               KEEL-partition outer CV, inner 5-fold tuning
                          (LVQ3's grid), budget matching, timing
  stats.py                Friedman omnibus, then SPINE against each
                          competitor by Wilcoxon and Conover (one shared
                          Holm family), one battery per SPINE readout,
                          plus the paired readout ablation

scripts/                command-line entry points
tests/                  the whole test suite (pytest)
data/, results/         gitignored artifacts (downloaded / produced)
```

Every comparator, and SPINE itself, implements one contract —
`select(X, y, total_count=..., random_state=...)` from
`harness.base` — which is the whole of what the runner needs to
add a method to the campaign.

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                             # offline test suite, ~1 min
```

The suite needs no network and no downloaded data. The `dev` extra adds
`pytest` and `interpret-core`; the latter is only used by three tests in
`tests/test_spotgreedy_algorithm.py`, which check our SPOTgreedy against
the authors' released routine index-for-index.

## Reproducing the study

```
python scripts/run_experiments.py                         # main experiment suite
python scripts/run_stats.py                               # formal statistics
python scripts/run_curve.py                               # accuracy-vs-budget curves
python scripts/run_computational.py                       # timing study + log-log figure
```

There is no separate download step. Each loader fetches what a run needs
into `--data-dir` (default `data/`) on first use and verifies it against
the manifest's declared shape; a download failure, a parse failure or a
shape mismatch is a stop-the-world error, never papered over. Passing
`--no-download` requires the raw files to be present already.

A campaign is optionally parallelised by sharding `run_experiments.py` calls over
`--datasets` and `--folds`, or by running
`scripts/run_experiments_parallel.py`; each shard writes its own CSV and
`run_stats.py` refuses duplicated or incomplete campaigns. Datasets for
which KEEL publishes only a 5-fold partition have outer folds 0–4, so
`--folds` ids 5–9 select nothing for them; every result row records its
`n_outer_folds`.

## Implementation provenance

| Method | Source |
|---|---|
| SPINE | this repository (`src/spine/`) |
| RSP3 | reimplemented in Python from Sanchez (2004) |
| LVQ3 | reimplemented in Python from Kohonen's rule (budget-matched to the anchor's total; window and epsilon inner-CV tuned; alpha, epochs, codebook size fixed) |
| SPOTgreedy | reimplemented in Python from Gurumoorthy, Jawanpuria & Mishra (2021), with lazy (CELF) evaluation of the identical greedy; verified index-for-index against the authors' released routine — see below |
| GLVQ | the `sklvq` package (Sato & Yamada 1996), at its own defaults, under a scikit-learn compatibility shim — see below |
| GNG | vendored from AdrienGuille/GrowingNeuralGas (MIT) at Fritzke's (1995) published parameters, repaired forward — see below |
| K-Means / 1-NN | scikit-learn |

### Why SPOTgreedy is reimplemented

Earlier revisions called `interpret.utils.SPOT_GreedySubsetSelection`
from `interpret-core`, the NumPy translation of the authors' release.
That routine materialises the full `n x n` cost matrix *and two more
`n x n` temporaries on every one of the `m` iterations*, so on magic's
17,118-row training fold it needs roughly 7 GB resident and does
O(m·n²) work — hours per fold at this protocol's budgets, which is why
`magic` and `letter` could not be run.

The SPOT objective is the facility-location form, hence monotone and
submodular, so marginal gains are non-increasing and *lazy* greedy
evaluation (Minoux 1978; CELF, Leskovec et al. 2007) provably returns
the element the exhaustive argmax would. The selection is therefore the
exhaustive greedy's, not an approximation of it:
`tests/test_spotgreedy_algorithm.py` asserts index-for-index equality
against both a naive restatement of the algorithm and interpret-core's
translation of the authors' release — on random costs, on real
Euclidean costs, and under a non-uniform marginal.

Ties get a declared rule rather than an accident. Two candidates can
have mathematically equal gains (duplicate rows, integer-valued or
coarsely quantised features, symmetric configurations), and the same
quantity computed through two different BLAS routines differs in the
last ulp, so a raw comparison would resolve a genuine tie by rounding
noise — and the answer could then depend on the block size. The queue
key is snapped to 12 significant digits before comparison, which puts
mathematically tied gains on one key, and the queue is ordered by
`(-snapped_gain, index)`, so ties go to the lowest row index. Both code
paths that produce a key snap, so the result is invariant to the block
size and reproducible run to run; the test suite pins all three
properties on inputs constructed to tie exactly. The reference does not
do this — its `np.argmax` over a single `matmul` returns the first
*bitwise* maximum — so on an exact tie the two implementations may pick
different members of the tied set. That is the only case in which they
differ, and it is the case in which the reference has no rule at all.

At magic/letter scale this replaces ~5.6e7 gain evaluations with
~1.9e5: about 11 s and 3 GB peak per fold, against an estimated 10 h
and ~7 GB. `interpret-core` is retained as a *test* dependency, purely
as the oracle for that equality check.

### Why GNG is vendored and repaired rather than installed

GNG earns its place in the roster twice over. Its edges are a
training-time device: they decide which units co-adapt, and they are
discarded at prediction time, where the rule is nearest unit. SPINE's
edges are part of the decision rule. GNG is therefore the control that
isolates the hypothesis-class change from the mere presence of a graph
during fitting. Second, its graph is built by competitive Hebbian
learning where SPINE's comes from a Mapper nerve, so a matched-budget
comparison asks whether the Reeb-graph construction buys anything a
cheaper adaptive graph does not.

There is no working Growing Neural Gas on PyPI. `neupy` 0.8.2 imports
`collections.MutableMapping`, which Python removed in 3.10; the MDP
toolkit's version is older still. The most cited standalone
implementation is AdrienGuille/GrowingNeuralGas, but it was written for
networkx 1.x and calls `Graph.node`, which networkx removed in 2.0
(2018), so it raises `AttributeError` on any current install. It is
therefore vendored here under its MIT licence and repaired forward,
with Fritzke's algorithm and Guille's structure — including his numbered
step comments — left legible. Every edit is traced in
`src/baselines/vendor/growing_neural_gas.py`'s docstring, and there are
seven:

1. `Graph.node` → `Graph.nodes` throughout, a pure API rename, without
   which the file does not run at all.
2. Plotting removed. `fit_network` unconditionally wrote PNGs into a
   `visualization/` directory and crashed when it did not exist; a
   baseline inside a campaign must not have file-system side effects.
3. A seeded `numpy.random.Generator` in place of the global
   `np.random`, so the runner can inject the fold's derived seed.
4. A `max_nodes` stop condition: insertion halts once the network
   reaches the injected budget.
5. A `top_up` after fitting. Edge ageing can prune isolated units below
   the target, so the network is topped up to exactly `max_nodes` using
   the algorithm's *own* insertion rule — the result is a network the
   algorithm could itself have produced, and it is what makes the exact
   budget attainable.
6. Seed units initialised at two sampled observations rather than
   `uniform(-2, 2)`, whose hardcoded interval assumes data on roughly
   that scale. Fritzke specifies two units "at random positions", and
   sampling observations is the scale-free reading.
7. A vectorised nearest-unit search, which returns the same two units
   under the same Euclidean metric; only the arithmetic is batched.

Fritzke's parameters (`e_b`, `e_n`, `a_max`, `l`, `a`, `d`) are used at
their published values and are not tuned, which puts GNG on the same
footing as RSP3: a comparator at its author's recommended
settings. The one setting this protocol must choose is `passes`, since
the number of insertions is `steps / l` and the budget has to be
reachable; it is *derived* rather than tuned — one unit is inserted
every `l` presentations from a start of two, so reaching a quota of `k`
needs `(k - 2) * l` presentations — floored at Fritzke's single pass.
GNG is unsupervised, so — exactly as for K-Means here — it is run per
class on that class's points alone, and each class's units are labelled
with that class.

### Why sklvq needs a compatibility shim

GLVQ is delegated to `sklvq` rather than reimplemented, because it is
the direct point-prototype counterpart of SPINE's Phase 2b — same
margin, same loss, isolated prototypes instead of an embedded 1-complex
— and a difference between the two at a matched budget should not be
attributable to a private restatement of the objective.

`sklvq` 0.1.2, the current release, calls `estimator._validate_data`,
which scikit-learn deprecated in 1.6 and *removed* in 1.7, so the
package raises `AttributeError` on any current scikit-learn.
`baselines/glvq.py` restores that method on `sklvq`'s base class in
terms of `sklearn.utils.validation.validate_data`, its documented
replacement, and renames the one keyword that moved with it
(`force_all_finite` → `ensure_all_finite`). The shim is applied on first
use, is a no-op wherever scikit-learn still provides the method, and
changes no numerical behaviour: it restores an input-validation call,
not part of the algorithm. It lives here rather than in a fork because
it is two lines and reverting it is a one-line deletion once `sklvq`
releases a fix. The alternatives were checked and are also broken
against the current stack: `sklearn-lvq` 1.1.1 passes a 2-D `x0` to
`scipy.optimize.minimize`, which SciPy now rejects, and `neupy` 0.8.2
imports `collections.MutableMapping`, removed in Python 3.10.

Everything else is `sklvq`'s own. Only two settings are injected: the
per-class prototype counts (the harness's proportional apportionment of
the anchor's total) and the fold's derived seed, since both the
initialisation and the solver are stochastic. `sklvq` expects those
counts in `numpy.unique(y)` order, which is the order
`allocate_per_class` returns; the mapping is asserted rather than
assumed. Prototypes are initialised from class means and then moved, so
the output is synthetic and carries the generation sentinel, exactly
like LVQ3 and SPINE. A class smaller than its quota is a hard error
rather than a silent cap.

## License

Code in this repository is licensed under the GNU General Public License
v3.0 (see `LICENSE`), matching the license of the KEEL software from
whose repository the datasets are obtained. The vendored Growing Neural
Gas implementation in `src/baselines/vendor/` is MIT-licensed and
carries its own copyright notice in the file.

The datasets themselves are not distributed here (the `data/` directory
is gitignored); they are downloaded from the KEEL dataset repository at
run time and remain under their original terms. KEEL requests that work
using its datasets cite Alcalá-Fdez et al. (2011); see `NOTICE`.
