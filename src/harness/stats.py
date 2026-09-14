"""Formal statistics -- Friedman omnibus, then SPINE against each competitor.

Run once per formally tested metric, alpha = 0.05.

Blocks = datasets (per-dataset scores are the mean over the outer
folds); treatments = methods.  The omnibus is the Friedman test over the
whole roster.  When it rejects, the method under test is compared
against every competitor TWICE, by two procedures that share one
multiplicity correction:

* a pairwise Wilcoxon signed-rank test (:func:`control_wilcoxon`) --
  the primary inference; and
* the Conover post-hoc (:func:`friedman_conover`) -- corroboration.

Both families are SPINE-AS-CONTROL, so each is k-1 comparisons rather
than all k(k-1)/2 pairs, and both are adjusted by the SAME Holm
procedure (:func:`holm`) across those k-1. ``scikit-posthocs``'
``posthoc_conover_friedman`` supplies the Conover matrix; we take it
UNADJUSTED and correct the control row ourselves, because its own
``p_adjust`` spends the correction over all 28 pairs (see
:func:`friedman_conover`).

SPINE's two readouts are tested in SEPARATE batteries
---------------------------------------------------
SPINE is fitted once per fold and scored two ways: ``SPINE+1NN`` takes
its vertices as an ordinary prototype set, and ``SPINE+Graph`` takes the
same fitted model under the skeleton decision rule.  They are two
EVALUATIONS of one fit, not two methods.

The substantive reason: the two readouts are a DETERMINISTIC FUNCTION OF
ONE FIT.  They are not exchangeable treatments, so neither the Friedman
null nor Conover's pooled variance estimate is the right model for them,
and the Conover cell for the pair is meaningless whatever the rank
arithmetic does.  This reason holds in every case, including the ones
where the mechanical effect happens to vanish.

*Construction CPU time.*  The runner records its timing columns for
reference, but the computational comparison is the dedicated
fixed-configuration timing study (``scripts/run_computational.py``),
which isolates construction cost from the inner-CV tuning bill and caps
BLAS threads.  In-campaign timings do neither and are reference values
only.
"""

import glob
import os

import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, wilcoxon

from .config import ALPHA, CAMPAIGN_LABELS

#: The two ways one SPINE fit is scored.  Never two treatments in the
#: same omnibus -- see the module docstring.
SPINE_READOUTS = ("SPINE+1NN", "SPINE+Graph")

#: The SPINE readout the paper reports in its results section.  The
#: other readout's battery still runs -- it is the segments-removed
#: ablation -- but the two are written under names that say which is
#: which, so a table cannot be lifted into the paper from the wrong one.
PAPER_READOUT = "SPINE+Graph"

if PAPER_READOUT not in SPINE_READOUTS:
    raise AssertionError(
        "PAPER_READOUT {0!r} is not one of SPINE_READOUTS {1}; every "
        "battery would be filed as the ablation and the paper's table "
        "would have no file to come from"
        .format(PAPER_READOUT, list(SPINE_READOUTS)))

#: Methods that appear in the raw results but are excluded from EVERY
#: formal test.
#:
#: 'Full' is the NOP reference (1-NN on the whole training fold): its
#: reduction is 0 by construction (so it is trivially last on the
#: reduction metric) and it is the accuracy ceiling rather than a
#: competing selector.  Ranking it against the prototype methods in a
#: Friedman omnibus is therefore degenerate.
#:
#: 'RHC' is here for a different reason, and only for legacy result
#: sets.  RHC is not budget-matched -- it is parameter-free and picks
#: its own prototype count -- so ranking it beside methods held to the
#: anchor's count compares a method against a different operating point.
#: It has since been dropped from ``config.MAIN_METHODS`` entirely, so a
#: campaign run today emits no RHC rows at all and this entry does
#: nothing.  It stays so that result sets collected BEFORE the removal
#: still load: ``comparator_roster`` subtracts this tuple from both the
#: declared roster and the undeclared-method check, so an RHC row in an
#: older CSV is ignored rather than treated as a method the protocol
#: never announced.
#:
#: Both stay in the raw result CSVs as references; neither enters a test.
STATS_EXCLUDED_METHODS = ("Full", "RHC")

#: The formally RANKED metrics: ``(column, higher_is_better)``.  Each is
#: tested once per SPINE readout.
FORMAL_METRICS = (
    ("acc_1nn", True),
)

#: Fewer blocks than this and neither the omnibus nor the paired test
#: means anything -- a one-dataset run produces a chi-square, a Conover
#: matrix of NaNs and a p-value formatted exactly like a real result.
#: The campaign protocol assumes all 17 datasets; this is only the floor
#: below which the output would be actively misleading.
MIN_BLOCKS = 2

#: Metrics written as per-dataset tables but NOT rank-tested.
#: 'reduction' is constant across the budget-matched block by
#: construction (see the module docstring); the timing columns are
#: reference values whose comparison lives in the timing study.
DESCRIPTIVE_METRICS = ("reduction", "n_proto", "cpu_select", "cpu_total")


#: The columns every runner CSV carries.  Used to tell a result shard
#: from some other CSV that happens to share the directory.
RUNNER_KEY_COLUMNS = ("dataset", "fold", "method")


def load_results(results_dir):
    """Concatenate every runner CSV in ``results_dir``.

    Duplicate (dataset, fold, method) rows -- e.g. a shard rerun --
    are a hard error: silently keeping either copy could mix
    inconsistent results.

    A CSV that is not a runner shard is a hard error too, named.  The
    glob takes every ``*.csv`` in the directory, so an analysis table
    saved beside the results used to be concatenated in like a shard:
    lacking a ``method`` column, its every row arrived as method NaN,
    and the failure surfaced as hundreds of "duplicate (dataset, fold,
    method)" records with ``nan`` in each -- a message about the wrong
    problem entirely.  Checking the schema per file says which file and
    what to do about it.  The check is not silent skipping: a stray CSV
    stops the run rather than being quietly excluded, because a shard
    that failed to write its header would otherwise vanish from the
    campaign without a word.
    """
    paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not paths:
        raise IOError("no result CSVs found in {0}".format(results_dir))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = [c for c in RUNNER_KEY_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                "{0} is in the results directory but is not a runner "
                "shard: it has no {1} column(s). Every *.csv here is "
                "loaded as results, so move this file out of {2} (a "
                "subdirectory is enough -- the glob is not recursive)."
                .format(path, ", ".join(repr(c) for c in missing),
                        results_dir))
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(subset=["dataset", "fold", "method"], keep=False)
    if dup.any():
        raise ValueError(
            "duplicate (dataset, fold, method) rows in results: "
            "{0}".format(
                df.loc[dup, ["dataset", "fold", "method"]]
                .drop_duplicates().to_dict("records")))
    return df


def per_dataset_matrix(df, metric, methods=None):
    """Datasets x methods matrix of fold-mean scores (complete blocks).

    An incomplete block (a dataset missing some method) is a hard
    error: the Friedman test requires complete blocks, and a missing
    cell means the campaign is unfinished.

    ``methods``, when given, restricts and ORDERS the columns to that
    roster.  A declared method absent from the results is a hard error
    too -- a metric whose roster silently shrank would be tested on a
    different field than the one declared.
    """
    if metric not in df.columns:
        raise KeyError("metric {0!r} not in results columns {1}".format(
            metric, sorted(df.columns)))
    pivot = (df.groupby(["dataset", "method"])[metric].mean()
             .unstack("method"))
    if methods is not None:
        missing = [m for m in methods if m not in pivot.columns]
        if missing:
            raise ValueError(
                "metric {0!r} declares treatments {1} but the results "
                "contain no rows for {2}".format(
                    metric, list(methods), missing))
        pivot = pivot[list(methods)]
    if pivot.isna().any().any():
        missing = [(d, m) for d in pivot.index for m in pivot.columns
                   if pd.isna(pivot.loc[d, m])]
        raise ValueError(
            "incomplete blocks (dataset, method): {0}".format(missing))
    if pivot.shape[0] < MIN_BLOCKS:
        raise ValueError(
            "metric {0!r} has {1} block(s) ({2}); a rank test over "
            "fewer than {3} datasets is not a result. Run more of the "
            "campaign before calling the statistics."
            .format(metric, pivot.shape[0], list(pivot.index), MIN_BLOCKS))
    return pivot


def comparator_roster(df):
    """Every method in ``df`` that is neither the NOP reference nor SPINE.

    This is the field each SPINE readout is ranked against, and it is
    identical for both readouts -- that is what makes the two batteries
    comparable to each other.

    The roster is ordered by :data:`~harness.config.CAMPAIGN_LABELS`
    rather than by the order rows happen to appear in the concatenated
    CSVs, so the column order of every written table is reproducible
    across shard orderings.

    A DECLARED comparator absent from the results is a hard error.  The
    roster is derived from the results, so without this check a
    comparator whose shard failed would simply vanish and both batteries
    would silently rank SPINE against a smaller field -- an easier field,
    reported in the same format as the declared one.
    """
    excluded = set(STATS_EXCLUDED_METHODS) | set(SPINE_READOUTS)
    declared = [m for m in CAMPAIGN_LABELS if m not in excluded]
    present = set(df["method"])
    missing = [m for m in declared if m not in present]
    if missing:
        raise ValueError(
            "the campaign declares comparators {0} but the results "
            "contain no rows for {1}; run those shards, or amend "
            "config.MAIN_METHODS if the roster really changed"
            .format(declared, missing))
    undeclared = sorted(present - set(declared) - excluded)
    if undeclared:
        raise ValueError(
            "results contain methods {0} that config.MAIN_METHODS does "
            "not declare; the battery would rank a field the protocol "
            "never announced".format(undeclared))
    return declared


def readout_roster(df, readout):
    """One SPINE readout followed by the shared comparator roster."""
    if readout not in SPINE_READOUTS:
        raise ValueError("{0!r} is not a SPINE readout; expected one of "
                         "{1}".format(readout, list(SPINE_READOUTS)))
    return [readout] + comparator_roster(df)


def holm(pvals):
    """Holm step-down adjustment of a family of p-values.

    Returns adjusted p-values in the INPUT order, each capped at 1.0 and
    made monotone in the sorted order (an adjusted value can never fall
    below one that precedes it, which is what makes the step-down
    procedure coherent).  Compare against alpha directly.

    One implementation serves both the Wilcoxon and the Conover family
    on purpose: they are corrected across the same k-1 control
    comparisons, and the claim that their agreement is evidence would
    not survive the two being adjusted by different code.
    """
    pvals = np.asarray(pvals, dtype=float)
    if pvals.ndim != 1 or not len(pvals):
        raise ValueError("holm() needs a non-empty 1-D family of p-values")
    order = np.argsort(pvals)
    out, running = np.empty(len(pvals)), 0.0
    for i, j in enumerate(order):
        running = max(running, (len(pvals) - i) * pvals[j])
        out[j] = min(running, 1.0)
    return out


def friedman_conover(df, metric, control, higher_is_better=True,
                     alpha=ALPHA, methods=None):
    """Friedman omnibus + Conover post-hoc, ``control`` against the rest.

    Returns a dict with the omnibus statistic/df/p-value, average ranks
    (rank 1 = best under ``higher_is_better``), and the control row of
    the Conover matrix both raw and Holm-adjusted.  ``methods`` is the
    battery's treatment roster (default: every method present in
    ``df``); ``control`` is the method under test and must be in it.

    The correction is applied HERE rather than by ``scikit-posthocs``.
    ``posthoc_conover_friedman(..., p_adjust="holm")`` corrects across
    every pair in the matrix -- for a roster of 8 that is 28
    comparisons, and the multiplier on the smallest raw p-value is
    exactly 28.  The protocol compares the method under test against
    each competitor and nothing else, so the family is the k-1 = 7
    control comparisons.  Correcting over 28 would answer a question
    about a set of pairwise contrasts that is never reported, and it
    would break the correspondence with the Wilcoxon family, which is
    corrected over 7 by construction.  So the matrix is taken
    UNADJUSTED and its control row is passed through :func:`holm`.

    The full k x k matrix is returned as ``conover_matrix`` and written
    beside the battery, but it stays UNADJUSTED there too.  Every
    comparator-against-comparator cell is a contrast the protocol does
    not test; correcting the matrix would spend a family-wise budget
    over 28 pairs, of which 21 are never reported, and would leave a
    file of adjusted p-values that answer a question nobody asked.  Raw
    is what a matrix can honestly hold, so the adjusted values live only
    in the k-1 control comparisons that are actually the family.
    """
    import scikit_posthocs as sp

    pivot = per_dataset_matrix(df, metric, methods=methods)
    if control not in pivot.columns:
        raise ValueError(
            "control {0!r} is not in the battery roster {1}".format(
                control, list(pivot.columns)))
    stat, pvalue = friedmanchisquare(
        *[pivot[m].to_numpy() for m in pivot.columns])

    rank_input = pivot if higher_is_better else -pivot
    avg_ranks = rank_input.rank(axis=1, ascending=False).mean(axis=0)

    raw_matrix = sp.posthoc_conover_friedman(pivot.to_numpy(), p_adjust=None)
    raw_matrix.index = raw_matrix.columns = pivot.columns
    control_raw = raw_matrix.loc[control].drop(control)
    if control_raw.isna().any():
        raise ValueError(
            "the Conover matrix has no p-value for {0}; a control "
            "comparison cannot be reported from an incomplete "
            "post-hoc".format(
                list(control_raw.index[control_raw.isna()])))
    _check_family_size(control_raw.index, pivot.columns, control, "Conover")
    control_holm = pd.Series(holm(control_raw.to_numpy()),
                             index=control_raw.index)

    return {
        "metric": metric,
        "control": control,
        "conover_matrix": raw_matrix,
        "higher_is_better": higher_is_better,
        "n_datasets": pivot.shape[0],
        "n_methods": pivot.shape[1],
        "friedman_statistic": float(stat),
        "friedman_df": int(pivot.shape[1] - 1),
        "friedman_pvalue": float(pvalue),
        "significant": bool(pvalue < alpha),
        "alpha": alpha,
        "average_ranks": avg_ranks.sort_values(),
        "scores": pivot,
        "conover_raw": control_raw,
        "conover_holm": control_holm,
        "conover_significant": control_holm < alpha,
        "scipy_version": scipy.__version__,
    }


def _check_family_size(compared, roster, control, family):
    """The correction must be spent over exactly k-1 comparisons.

    The paper names the number of comparisons ("adjusted across the
    seven comparisons by Holm's procedure"), so a roster that grows or
    shrinks without that sentence changing is a silent inconsistency
    between the code and the claim.  A comparison count that is not
    ``len(roster) - 1`` means either the control leaked into its own
    family or a competitor was dropped from it; both change the Holm
    multiplier, and neither is visible in the p-values themselves.
    """
    expected = len(roster) - 1
    if len(compared) != expected:
        raise ValueError(
            "the {0} family compares {1} against {2} method(s) ({3}) but "
            "the battery roster has {4} treatments, so the family should "
            "be {5} comparisons; the Holm multiplier would be wrong"
            .format(family, control, len(compared), list(compared),
                    len(roster), expected))


def control_wilcoxon(df, metric, control, alpha=ALPHA, methods=None):
    """Wilcoxon signed-rank of ``control`` against each other method.

    The primary inference of the comparison: SPINE is the control, so
    this is ``len(roster) - 1`` comparisons rather than all pairs, and
    the family is Holm-adjusted across exactly those.  Returns one row
    dict per competitor, in roster order, each carrying the win / loss /
    tie counts beside its p-value -- the counts and the p-value describe
    the same sample only if they are read together.

    Exact ties are dropped from the signed-rank statistic, the classic
    ``zero_method='wilcox'`` treatment: a dataset where SPINE and a
    competitor score identically contributes no signed rank, and pre-
    filtering the zeros here makes scipy's ``zero_method`` argument
    moot.  ``n_used`` is what the test actually saw.  On this campaign
    it is 17 everywhere except KMeans, where SPINE+Graph and KMeans tie
    exactly on wine (0.955229 on all ten folds), leaving n = 16.

    ``method="exact"`` is PINNED rather than left at ``"auto"``, for the
    reason :func:`readout_ablation` gives: what 'auto' resolves to
    depends on the scipy version as well as the data, and a protocol
    that reports p-values should not have its test chosen for it by a
    dependency's minor version.  The scipy version is recorded in the
    output so the pin can be checked rather than trusted.

    The exact null assumes the retained ``|differences|`` are distinct.
    Where they are not, scipy rounds the statistic conservatively and
    the p-value is conservative rather than exact -- a different claim
    from the one the output makes.  That is a hard error here rather
    than a recorded caveat: on this campaign ``n_abs_ties`` is 0 on all
    seven comparisons, so a nonzero count means the data changed under
    a report that still says "exact".
    """
    scores = per_dataset_matrix(df, metric, methods=methods)
    if control not in scores.columns:
        raise ValueError(
            "control {0!r} is not in the battery roster {1}".format(
                control, list(scores.columns)))
    others = [m for m in scores.columns if m != control]
    _check_family_size(others, scores.columns, control, "Wilcoxon")

    rows, raw = [], []
    for m in others:
        delta = scores[control] - scores[m]
        nonzero = delta != 0
        if not nonzero.any():
            raise ValueError(
                "{0} and {1} tie on every one of the {2} datasets, so the "
                "signed-rank test has no sample; check that both were "
                "actually scored on {3!r}".format(
                    control, m, scores.shape[0], metric))
        retained = delta[nonzero].abs()
        n_abs_ties = int(len(retained) - retained.nunique())
        if n_abs_ties:
            raise ValueError(
                "{0} vs {1} on {2!r}: {3} of the {4} retained differences "
                "share a magnitude, so scipy rounds the statistic and the "
                "exact p-value is conservative rather than exact. Report "
                "it as conservative, or drop the method='exact' pin -- do "
                "not present it as exact.".format(
                    control, m, metric, n_abs_ties, len(retained)))
        statistic, pvalue = wilcoxon(scores[control][nonzero],
                                     scores[m][nonzero], method="exact")
        rows.append({
            "method": m,
            "wins": int((delta > 0).sum()),
            "losses": int((delta < 0).sum()),
            "ties": int((~nonzero).sum()),
            "n_used": int(nonzero.sum()),
            "n_abs_ties": n_abs_ties,
            "statistic": float(statistic),
            "p_raw": float(pvalue),
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
        })
        raw.append(pvalue)

    for row, p_adj in zip(rows, holm(np.array(raw))):
        row["p_holm"] = float(p_adj)
        row["significant"] = bool(p_adj < alpha)
    return rows


def readout_ablation(df, metric, alpha=ALPHA):
    """Paired contrast between the two readouts of one SPINE fit.

    The two readouts share a fit, share a prototype set, share a seed and
    share every fold, so the honest comparison between them is PAIRED,
    not a rank over a field of methods.  The reported difference is
    ``SPINE+Graph - SPINE+1NN`` per dataset: positive means the skeleton
    decision rule beat plain 1-NN on the very same prototypes.  The
    metric is assumed higher-is-better, which
    :func:`run_all_stats` enforces.

    Ties (datasets where the skeleton rule changed no per-dataset score)
    are dropped before the test -- the standard ``zero_method='wilcox'``
    treatment -- and reported as ``n_tied``, so the win/loss counts and
    the p-value never describe different samples.

    The test method is PINNED to ``"exact"`` rather than left at
    ``"auto"``.  Not because 'auto' is wrong here, but because what
    'auto' resolves to depends on both the data and the scipy version:
    in scipy 1.17 it is the exact null when no zero and no tie survives,
    a permutation test when at most 13 differences remain, and the normal
    approximation above that.  A protocol that reports p-values should
    not have its test chosen for it by a dependency's minor version.

    One consequence is stated rather than hidden: the exact null assumes
    the retained ``|differences|`` are distinct.  When they are not,
    scipy rounds the statistic conservatively, so the p-value is
    conservative rather than exact.  ``n_abs_ties`` reports how many
    retained differences share a magnitude, so a reader can tell which
    of the two the number in front of them is.
    """
    scores = per_dataset_matrix(df, metric, methods=list(SPINE_READOUTS))
    points, graph = SPINE_READOUTS
    delta = scores[graph] - scores[points]
    nonzero = delta != 0

    if not nonzero.any():
        # Every dataset tied.  On a real campaign this does not mean
        # "the skeleton rule is exactly neutral"; it is the signature of
        # the graph readout never having been evaluated, or of Phase 2a
        # having stripped every skeleton back to isolated points -- the
        # collapse the strict-improvement rule exists to prevent.  A
        # p-value here would report a finding about a broken run.
        raise ValueError(
            "every dataset ties on {0!r}: {1} and {2} produced identical "
            "per-dataset scores on all {3} datasets. Check that the "
            "skeleton decision rule was actually scored, and check "
            "betti_frozen in the spine_params column -- a skeleton "
            "pruned to isolated vertices predicts exactly like its own "
            "vertices.".format(metric, graph, points, scores.shape[0]))

    retained = delta[nonzero].abs()
    statistic, pvalue = wilcoxon(scores[graph][nonzero],
                                 scores[points][nonzero], method="exact")

    return {
        "metric": metric,
        "contrast": "{0} - {1}".format(graph, points),
        "n_datasets": int(scores.shape[0]),
        "n_tied": int((~nonzero).sum()),
        "n_abs_ties": int(len(retained) - retained.nunique()),
        "n_graph_better": int((delta > 0).sum()),
        "n_points_better": int((delta < 0).sum()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_pvalue": float(pvalue),
        "significant": bool(pvalue < alpha),
        "alpha": alpha,
        "scores": scores,
        "delta": delta,
    }


def write_descriptive_tables(df, out_dir,
                             metrics=DESCRIPTIVE_METRICS):
    """Write per-dataset tables for the recorded-but-unranked metrics.

    Reported, not discarded: EVERY method appears, including 'Full' and
    both SPINE readouts.  STATS_EXCLUDED_METHODS keeps the NOP reference
    out of the *rank* tests, where a treatment with reduction 0 by
    construction is degenerate; it has no bearing on a descriptive
    table, and since this table is the whole substitute for the dropped
    reduction battery, the row that anchors it (reduction 0,
    n_proto = n_train) is exactly the one a reader needs to see.  A
    metric absent from the results is skipped rather than fabricated --
    the runner decides which columns exist.
    """
    written = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        path = os.path.join(
            out_dir, "descriptive_{0}_scores.csv".format(metric))
        per_dataset_matrix(df, metric).to_csv(path)
        written.append(path)
    return written


def readout_slug(readout):
    """File-safe stem for a readout name ('SPINE+1NN' -> 'spine_1nn')."""
    return "".join(c if c.isalnum() else "_" for c in readout).lower()


if len({readout_slug(r) for r in SPINE_READOUTS}) != len(SPINE_READOUTS):
    raise AssertionError(
        "SPINE_READOUTS {0} do not have distinct filename stems; one "
        "battery's tables would overwrite the other's"
        .format(list(SPINE_READOUTS)))


def battery_role(readout):
    """'reported' for the readout the paper's results section carries.

    Both readouts get a full battery, but only one is the confirmatory
    comparison; the other is the segments-removed ablation that belongs
    to the discussion.  The role goes in the FILENAME rather than only
    in a column, because the failure it guards against is a table being
    lifted out of the output directory into the paper from the wrong
    battery -- at which point no column inside the file is being read.
    """
    return "reported" if readout == PAPER_READOUT else "ablation"


def _battery_base(out_dir, metric, readout):
    return os.path.join(out_dir, "{0}_{1}__{2}".format(
        metric, readout_slug(readout), battery_role(readout)))


def _write_battery(res, out_dir, metric, readout):
    """Write the comparison table, the score matrix and the ranks.

    The comparison CSV is the one table the results section needs: one
    row per competitor carrying both families' p-values, the win / loss
    / tie counts and the average rank, under a commented header block
    that records the omnibus, the roster, alpha and the scipy version.
    It is the ONLY file here holding adjusted p-values: the Conover
    matrix is written raw, because Holm covers the control comparisons
    and nothing else.
    The block is written as ``#`` comment lines, so ``pandas.read_csv(...,
    comment='#')`` reads the table back while a human opening the file
    still sees which run produced it.
    """
    base = _battery_base(out_dir, metric, readout)
    res["scores"].to_csv(base + "_scores.csv")
    res["average_ranks"].to_csv(base + "_ranks.csv", header=["avg_rank"])
    # UNADJUSTED, and the filename says so.  Holm here covers only the
    # k-1 control comparisons; a k x k matrix of adjusted values would
    # correct 21 of its 28 cells under a family the paper never uses.
    res["conover_matrix"].to_csv(base + "_conover_raw.csv")

    ranks, conover_raw = res["average_ranks"], res["conover_raw"]
    conover_holm, conover_sig = res["conover_holm"], res["conover_significant"]
    table = pd.DataFrame([
        dict(row,
             conover_raw=float(conover_raw[row["method"]]),
             conover_holm=float(conover_holm[row["method"]]),
             conover_significant=bool(conover_sig[row["method"]]),
             average_rank=float(ranks[row["method"]]))
        for row in res["wilcoxon"]
    ])[["method", "wins", "losses", "ties", "n_used", "n_abs_ties",
        "statistic", "p_raw", "p_holm", "significant",
        "conover_raw", "conover_holm", "conover_significant",
        "mean_delta", "median_delta", "average_rank"]]

    path = base + "_comparison.csv"
    with open(path, "w") as fh:
        for line in [
            "metric = {0}".format(metric),
            "control = {0} ({1} battery)".format(readout,
                                                 battery_role(readout)),
            "control average rank = {0:.4f}".format(float(ranks[readout])),
            "blocks N = {0} datasets, treatments k = {1}".format(
                res["n_datasets"], res["n_methods"]),
            "roster = {0}".format(", ".join(res["scores"].columns)),
            "Friedman chi2 = {0:.4f}, df = {1}, p = {2:.6g}, "
            "significant = {3}".format(
                res["friedman_statistic"], res["friedman_df"],
                res["friedman_pvalue"], res["significant"]),
            "comparisons = {0} (Holm across exactly these, both "
            "families)".format(len(table)),
            "alpha = {0}".format(res["alpha"]),
            "Wilcoxon signed-rank, method=exact; scipy {0}".format(
                res["scipy_version"]),
            "families agree on the significant set = {0}".format(
                res["families_agree"]),
        ]:
            fh.write("# {0}\n".format(line))
        table.to_csv(fh, index=False)
    return path


def _write_ablation(abl, out_dir, metric):
    base = os.path.join(out_dir, "ablation_{0}_spine_readouts".format(metric))
    abl["scores"].assign(delta=abl["delta"]).to_csv(base + "_scores.csv")
    with open(base + ".txt", "w") as fh:
        fh.write(
            "Paired readout ablation on {0}\n"
            "contrast = {1} (positive favours the skeleton rule)\n"
            "{2} datasets: {3} favour the graph, {4} favour the points, "
            "{5} tied (dropped from the test)\n"
            "mean delta   = {6:.6f}\nmedian delta = {7:.6f}\n"
            "Wilcoxon signed-rank, method=exact: "
            "statistic = {8:.6f}, p-value = {9:.6g}\n"
            "{10} retained differences share a magnitude; the p-value is "
            "{11}\n"
            "significant at alpha={12}: {13}\n".format(
                metric, abl["contrast"], abl["n_datasets"],
                abl["n_graph_better"], abl["n_points_better"],
                abl["n_tied"], abl["mean_delta"], abl["median_delta"],
                abl["wilcoxon_statistic"], abl["wilcoxon_pvalue"],
                abl["n_abs_ties"],
                "exact" if not abl["n_abs_ties"] else "conservative",
                abl["alpha"], abl["significant"]))


def run_all_stats(results_dir, out_dir):
    """Run the formal battery; write CSV tables; return the results.

    For every formally ranked metric this runs one battery PER SPINE
    READOUT against the shared comparator roster -- Friedman omnibus,
    then the readout against each competitor by BOTH the Wilcoxon
    signed-rank test and the Conover post-hoc, each Holm-adjusted over
    the same k-1 comparisons -- plus the paired readout ablation.
    Returned shape::

        {metric: {"batteries": {readout: result, ...},
                  "readout_ablation": ablation}}

    Each battery result carries ``families_agree``: whether the two
    procedures reject the same set of competitors.  It is RECORDED, not
    enforced.  Agreement is the campaign's finding and the regression
    tests pin it, but two different tests are entitled to disagree on
    other data, and a run that raised rather than reported would be
    hiding the more interesting outcome of the two.
    """
    df = load_results(results_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Written FIRST, and before anything that needs a comparator roster:
    # the descriptive tables and the paired ablation depend only on the
    # rows they read, so a roster problem must not throw away output
    # that was already computable.
    write_descriptive_tables(df, out_dir)

    out = {}
    for metric, higher in FORMAL_METRICS:
        if not higher:
            raise ValueError(
                "metric {0!r} is declared lower-is-better, but "
                "readout_ablation orients its contrast as "
                "higher-is-better. Orient the metric before adding it "
                "to FORMAL_METRICS.".format(metric))
        # Exclude reference-only methods before this metric's test: the
        # rows stay in the raw result CSVs but are not ranked.
        df_metric = df[~df["method"].isin(STATS_EXCLUDED_METHODS)]

        abl = readout_ablation(df_metric, metric)
        _write_ablation(abl, out_dir, metric)

        batteries = {}
        for readout in SPINE_READOUTS:
            roster = readout_roster(df_metric, readout)
            res = friedman_conover(
                df_metric, metric, control=readout,
                higher_is_better=higher, methods=roster)
            res["wilcoxon"] = control_wilcoxon(
                df_metric, metric, control=readout, methods=roster)
            res["families_agree"] = bool(
                {r["method"] for r in res["wilcoxon"] if r["significant"]}
                == {m for m in res["conover_significant"].index
                    if res["conover_significant"][m]})
            _write_battery(res, out_dir, metric, readout)
            batteries[readout] = res
        out[metric] = {"batteries": batteries, "readout_ablation": abl}
    return out
