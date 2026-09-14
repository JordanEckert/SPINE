#!/usr/bin/env python3
"""Formal statistics over the collected results.

Friedman omnibus on 1-NN accuracy, then the method under test against
each competitor, alpha = 0.05.  Refuses incomplete campaigns (missing
(dataset, method) blocks) and duplicated shards.

SPINE is fitted once per fold and scored two ways -- SPINE+1NN (its
vertices as an ordinary prototype set) and SPINE+Graph (the same model
under the skeleton decision rule).  These are two EVALUATIONS of one
fit -- a deterministic function of the same vertices -- so they are not
exchangeable treatments and no omnibus ever contains both.  The battery
runs once per readout against the identical comparator roster, and the
pair's own contrast, which is paired rather than ranked, is reported
separately as the readout ablation (Wilcoxon signed-rank over the
per-dataset scores).

The inference
-------------
The Friedman test ranks the readout against the seven budget-matched
competitors over the dataset blocks.  Where it rejects, the readout is
compared against each competitor individually, TWICE:

* pairwise Wilcoxon signed-rank, the primary inference -- datasets on
  which the pair ties exactly are dropped from the signed-rank
  statistic, and the win / loss / tie counts are reported beside every
  p-value so the sample the test saw is never left implicit; and
* the Conover post-hoc, as corroboration.

Both families are the method under test against each competitor -- k-1
comparisons, not all k(k-1)/2 pairs -- and both are Holm-adjusted across
exactly those k-1 by one shared implementation.  That is deliberate:
agreement between two procedures is stronger evidence than either alone
only if they were corrected the same way, and ``scikit-posthocs``' own
``p_adjust`` would have spent the Conover correction over all 28 pairs
of an 8-treatment roster.

Bonferroni is not reported (Holm dominates it uniformly at the same
family-wise error rate), and no critical-difference diagram is drawn:
Nemenyi's critical difference is an all-pairs quantity and does not
exist for a control-only family.

Which battery the paper reports
-------------------------------
Both readouts get a full battery, but SPINE+Graph -- the method's actual
decision rule -- is the confirmatory one, and the SPINE+1NN battery is
the segments-removed ablation.  Nothing corrects across the three
families.  The role is in every filename (``__reported`` against
``__ablation``) rather than only in a column, because the mistake it
guards against is a table being lifted into the paper from the wrong
battery, at which point no column inside the file is being read.

Accuracy is the one formally ranked metric.  Reduction rate is not
ranked: the campaign anchors every budget on RSP3's own prototype count,
so no method chooses a count of its own and a rank test over the roster
would report a tautology.  Reduction, prototype counts and the timing
columns are written as per-dataset descriptive tables
(``descriptive_*_scores.csv``) instead.  The accuracy/budget trade-off is
measured by ``scripts/run_curve.py`` and construction cost by
``scripts/run_computational.py``.

Example
-------
::

    python scripts/run_stats.py --results-dir results_run --out-dir results_run/stats
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from harness.stats import battery_role, run_all_stats


def _print_battery(metric, readout, res):
    print("== {0} | SPINE readout: {1} ({2}) ==".format(
        metric, readout, battery_role(readout)))
    print("treatments: {0}".format(", ".join(res["scores"].columns)))
    print("Friedman chi2 = {0:.4f}, df = {1}, p = {2:.4g} "
          "(significant: {3})".format(
              res["friedman_statistic"], res["friedman_df"],
              res["friedman_pvalue"], res["significant"]))
    print("average ranks (best first):")
    print(res["average_ranks"].to_string())
    print()
    print("{0} vs each competitor -- Holm over {1} comparisons, "
          "alpha = {2:g}:".format(readout, len(res["wilcoxon"]),
                                  res["alpha"]))
    print("  {0:<8} {1:>7}  {2:>9} {3:>9}  {4:>9} {5:>9}".format(
        "method", "w/l/t", "wilcoxon", "holm", "conover", "holm"))
    for row in res["wilcoxon"]:
        m = row["method"]
        print("  {0:<8} {1:>7}  {2:9.5f} {3:9.5f}{4} {5:9.3g} {6:9.5f}{7}"
              .format(m,
                      "{0}/{1}/{2}".format(row["wins"], row["losses"],
                                           row["ties"]),
                      row["p_raw"], row["p_holm"],
                      "*" if row["significant"] else " ",
                      float(res["conover_raw"][m]),
                      float(res["conover_holm"][m]),
                      "*" if res["conover_significant"][m] else " "))
    print("  (* significant; n_used = {0})".format(
        ", ".join("{0}:{1}".format(r["method"], r["n_used"])
                  for r in res["wilcoxon"])))
    wil = sorted(r["method"] for r in res["wilcoxon"] if r["significant"])
    con = sorted(m for m in res["conover_significant"].index
                 if res["conover_significant"][m])
    print("  Wilcoxon rejects: {0}".format(", ".join(wil) or "none"))
    print("  Conover  rejects: {0}".format(", ".join(con) or "none"))
    print("  the two families {0}".format(
        "agree" if res["families_agree"]
        else "DISAGREE -- report both sets, not one"))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=os.path.join("results", "stats"))
    args = ap.parse_args()

    out = run_all_stats(args.results_dir, args.out_dir)
    for metric, res in out.items():
        for readout, battery in res["batteries"].items():
            _print_battery(metric, readout, battery)

        abl = res["readout_ablation"]
        print("== {0} | readout ablation ==".format(metric))
        print("contrast: {0} (positive favours the skeleton rule)"
              .format(abl["contrast"]))
        print("{0} datasets: {1} favour the graph, {2} favour the "
              "points, {3} tied (dropped from the test)"
              .format(abl["n_datasets"], abl["n_graph_better"],
                      abl["n_points_better"], abl["n_tied"]))
        print("mean delta = {0:.6f}, median delta = {1:.6f}"
              .format(abl["mean_delta"], abl["median_delta"]))
        print("Wilcoxon signed-rank, method=exact: p = {0:.4g} "
              "(significant: {1}); the p-value is {2} -- {3} retained "
              "difference(s) share a magnitude"
              .format(abl["wilcoxon_pvalue"], abl["significant"],
                      "exact" if not abl["n_abs_ties"] else "conservative",
                      abl["n_abs_ties"]))
        print()

    print("Ranked: {0}, once per SPINE readout. Reduction, prototype counts "
          "and the timing columns are written descriptively (not "
          "rank-tested).".format(", ".join(out)))
    print("Tables written to", args.out_dir)


if __name__ == "__main__":
    main()
