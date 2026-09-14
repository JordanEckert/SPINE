"""The Section 3.4 numbers, pinned against the real campaign.

Every value here was computed from ``results_run/`` (17 datasets x 10
outer folds) and is reproduced by running the SHIPPED pipeline, not a
re-derivation: :func:`harness.stats.run_all_stats` writes the tables and
these tests read what it wrote.  A re-derivation would pass even if the
code that produces the paper's table diverged from it.

A missing ``results_run/`` is a failure rather than a skip.  These are
the numbers in the manuscript; a green suite that quietly tested none of
them is worse than a red one.
"""

import os

import numpy as np
import pandas as pd
import pytest
import scikit_posthocs as sp

from harness.stats import (PAPER_READOUT, holm, load_results,
                           per_dataset_matrix, run_all_stats)

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "results_run")

#: The declared battery: the method under test plus the seven
#: budget-matched competitors.  Full is the accuracy reference and RHC
#: is not budget-matched, so neither is ranked.
ROSTER = ("SPINE+Graph", "RSP3", "KMeans", "Random", "LVQ3", "SPOT",
          "GLVQ", "GNG")

#: Average rank over the 17 datasets, 1 = best.
AVERAGE_RANKS = {
    "SPINE+Graph": 2.4412, "LVQ3": 2.9412, "GNG": 3.1176, "KMeans": 4.6176,
    "RSP3": 4.9412, "SPOT": 5.1765, "GLVQ": 5.9412, "Random": 6.8235,
}

#: method -> (wins, losses, ties, n_used, wilcoxon raw, wilcoxon Holm,
#:            conover raw, conover Holm over the 7 comparisons)
PAIRWISE = {
    "LVQ3":   (12, 5, 0, 17, 0.03052, 0.06104, 4.57e-01, 0.6295),
    "GNG":    (10, 7, 0, 17, 0.07968, 0.07968, 3.15e-01, 0.6295),
    "KMeans": (13, 3, 1, 16, 0.00336, 0.01007, 1.53e-03, 0.0046),
    "RSP3":   (14, 3, 0, 17, 0.00209, 0.00836, 3.00e-04, 0.0012),
    "SPOT":   (14, 3, 0, 17, 0.00066, 0.00394, 8.37e-05, 0.0004),
    "GLVQ":   (14, 3, 0, 17, 0.00066, 0.00394, 8.15e-07, 0.0000),
    "Random": (17, 0, 0, 17, 0.00002, 0.00011, 1.89e-09, 0.0000),
}

SIGNIFICANT = {"KMeans", "RSP3", "SPOT", "GLVQ", "Random"}
NOT_SIGNIFICANT = {"LVQ3", "GNG"}

#: What ``posthoc_conover_friedman(p_adjust="holm")`` returns instead --
#: the correction spent over all 28 pairs.  Recorded so the difference
#: the fix makes is visible as a number rather than only as a diff.
CONOVER_ALL_PAIRS_HOLM = {
    "KMeans": 0.0275, "RSP3": 0.0063, "SPOT": 0.0018,
    "LVQ3": 1.0000, "GNG": 1.0000,
}


@pytest.fixture(scope="module")
def campaign():
    if not os.path.isdir(RESULTS_DIR):
        raise AssertionError(
            "{0} is missing; the Section 3.4 fixture cannot be checked "
            "against anything".format(RESULTS_DIR))
    return load_results(RESULTS_DIR)


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stats")
    out = run_all_stats(RESULTS_DIR, str(out_dir))
    return out["acc_1nn"]["batteries"][PAPER_READOUT], out_dir


def test_the_battery_ranks_the_declared_roster(written):
    battery, _ = written
    assert tuple(battery["scores"].columns) == ROSTER
    assert battery["n_datasets"] == 17
    assert battery["n_methods"] == 8
    assert "Full" not in battery["scores"].columns
    assert "RHC" not in battery["scores"].columns


def test_friedman_omnibus(written):
    battery, _ = written
    assert battery["friedman_statistic"] == pytest.approx(47.5439, abs=1e-4)
    assert battery["friedman_df"] == 7
    assert battery["friedman_pvalue"] == pytest.approx(4.372e-08, rel=1e-3)
    assert battery["significant"]


def test_average_ranks(written):
    battery, _ = written
    ranks = battery["average_ranks"]
    assert list(ranks.index) == sorted(AVERAGE_RANKS,
                                       key=AVERAGE_RANKS.get)
    for method, expected in AVERAGE_RANKS.items():
        assert ranks[method] == pytest.approx(expected, abs=1e-4), method


def test_pairwise_wilcoxon_family(written):
    battery, _ = written
    rows = {r["method"]: r for r in battery["wilcoxon"]}
    assert set(rows) == set(PAIRWISE)
    for method, (w, l, t, n, p_raw, p_holm, _, _) in PAIRWISE.items():
        row = rows[method]
        assert (row["wins"], row["losses"], row["ties"]) == (w, l, t), method
        assert row["n_used"] == n, method
        assert row["p_raw"] == pytest.approx(p_raw, abs=5e-6), method
        assert row["p_holm"] == pytest.approx(p_holm, abs=5e-6), method
        # every comparison's exact null is exact, not conservative
        assert row["n_abs_ties"] == 0, method


def test_the_one_tie_is_kmeans_on_wine(campaign):
    # n = 16 rather than 17 on KMeans is a fact about the data, not a
    # dropped block: SPINE+Graph and KMeans score identically on wine.
    scores = per_dataset_matrix(campaign, "acc_1nn", methods=list(ROSTER))
    tied = scores.index[scores[PAPER_READOUT] == scores["KMeans"]]
    assert list(tied) == ["wine"]
    assert scores.loc["wine", PAPER_READOUT] == pytest.approx(0.955229,
                                                              abs=1e-6)


def test_conover_family_over_the_seven_control_comparisons(written):
    battery, _ = written
    for method, (_, _, _, _, _, _, c_raw, c_holm) in PAIRWISE.items():
        assert battery["conover_raw"][method] == pytest.approx(
            c_raw, rel=1e-2), method
        assert battery["conover_holm"][method] == pytest.approx(
            c_holm, abs=5e-5), method


def test_the_correction_is_over_seven_comparisons_not_twenty_eight(campaign):
    # The live bug this change fixes.  scikit-posthocs' own p_adjust
    # spends the correction over every pair in the matrix; on an
    # 8-treatment roster that is 28, and LVQ3 and GNG saturate at 1.0.
    df = campaign[~campaign["method"].isin(("Full", "RHC"))]
    pivot = per_dataset_matrix(df, "acc_1nn", methods=list(ROSTER))

    all_pairs = sp.posthoc_conover_friedman(pivot.to_numpy(),
                                            p_adjust="holm")
    all_pairs.index = all_pairs.columns = pivot.columns
    for method, expected in CONOVER_ALL_PAIRS_HOLM.items():
        assert all_pairs.loc[PAPER_READOUT, method] == pytest.approx(
            expected, abs=5e-5), method

    raw = sp.posthoc_conover_friedman(pivot.to_numpy(), p_adjust=None)
    raw.index = raw.columns = pivot.columns
    control_raw = raw.loc[PAPER_READOUT].drop(PAPER_READOUT)
    assert len(control_raw) == 7
    # the multiplier the all-pairs adjustment implies is 28, ours is 7
    smallest = control_raw.idxmin()
    implied_all_pairs = (all_pairs.loc[PAPER_READOUT, smallest]
                         / control_raw[smallest])
    assert implied_all_pairs == pytest.approx(28.0, rel=1e-6)
    ours = holm(control_raw.to_numpy())
    assert ours[list(control_raw.index).index(smallest)] / control_raw[
        smallest] == pytest.approx(7.0, rel=1e-6)


def test_the_two_families_reject_the_same_set(written):
    # The corroboration claim itself.  Checking p-values alone would not
    # catch a multiplicity regression that moved both families together;
    # this is the sentence the paper actually makes.
    battery, _ = written
    wilcoxon_sig = {r["method"] for r in battery["wilcoxon"]
                    if r["significant"]}
    conover_sig = {m for m in battery["conover_significant"].index
                   if battery["conover_significant"][m]}
    assert wilcoxon_sig == SIGNIFICANT
    assert conover_sig == SIGNIFICANT
    assert wilcoxon_sig == conover_sig
    assert battery["families_agree"]
    assert set(PAIRWISE) - wilcoxon_sig == NOT_SIGNIFICANT


def test_the_written_table_carries_the_numbers_and_its_provenance(written):
    battery, out_dir = written
    path = os.path.join(str(out_dir),
                        "acc_1nn_spine_graph__reported_comparison.csv")
    assert os.path.exists(path)

    header = [l for l in open(path) if l.startswith("#")]
    blob = "".join(header)
    assert "Friedman chi2 = 47.5439, df = 7" in blob
    assert "blocks N = 17 datasets, treatments k = 8" in blob
    assert "comparisons = 7" in blob
    assert "alpha = 0.05" in blob
    assert "scipy" in blob                       # the version is recorded
    assert "control = SPINE+Graph (reported battery)" in blob

    table = pd.read_csv(path, comment="#")
    assert list(table.columns) == [
        "method", "wins", "losses", "ties", "n_used", "n_abs_ties",
        "statistic", "p_raw", "p_holm", "significant",
        "conover_raw", "conover_holm", "conover_significant",
        "mean_delta", "median_delta", "average_rank"]
    assert set(table["method"]) == set(PAIRWISE)
    assert set(table.loc[table["significant"], "method"]) == SIGNIFICANT
    assert (table["significant"] == table["conover_significant"]).all()
    for _, row in table.iterrows():
        w, l, t, n, p_raw, p_holm, _, _ = PAIRWISE[row["method"]]
        assert row["n_used"] == n
        assert row["p_holm"] == pytest.approx(p_holm, abs=5e-6)
        assert row["average_rank"] == pytest.approx(
            AVERAGE_RANKS[row["method"]], abs=1e-4)


def test_the_ablation_battery_is_written_under_its_own_name(written):
    _, out_dir = written
    assert os.path.exists(os.path.join(
        str(out_dir), "acc_1nn_spine_1nn__ablation_comparison.csv"))
    # and the two batteries never share a file
    assert not os.path.exists(os.path.join(
        str(out_dir), "acc_1nn_spine_1nn__reported_comparison.csv"))


def test_the_conover_matrix_on_disk_is_raw_not_adjusted(written):
    # On the real campaign the two are far apart, so this catches an
    # adjusted matrix being written under the raw name: SPINE+Graph vs
    # LVQ3 is 0.457 raw and 0.6295 under Holm over the seven control
    # comparisons (1.0 under scikit-posthocs' all-pairs adjustment).
    battery, out_dir = written
    path = os.path.join(str(out_dir),
                        "acc_1nn_spine_graph__reported_conover_raw.csv")
    assert os.path.exists(path)
    matrix = pd.read_csv(path, index_col=0)

    assert list(matrix.index) == list(ROSTER)
    assert list(matrix.columns) == list(ROSTER)
    for method, (_, _, _, _, _, _, c_raw, c_holm) in PAIRWISE.items():
        cell = matrix.loc[PAPER_READOUT, method]
        assert cell == pytest.approx(c_raw, rel=1e-2), method
        if c_raw != pytest.approx(c_holm, rel=1e-2):
            assert cell != pytest.approx(c_holm, rel=1e-2), method
    # LVQ3 is the clearest separation of the three possible values
    assert matrix.loc[PAPER_READOUT, "LVQ3"] == pytest.approx(0.457,
                                                              rel=1e-2)

    # no adjusted matrix is written at all, under any name
    names = sorted(os.listdir(str(out_dir)))
    assert [n for n in names if "conover" in n] == [
        "acc_1nn_spine_1nn__ablation_conover_raw.csv",
        "acc_1nn_spine_graph__reported_conover_raw.csv"]
    assert not [n for n in names if "holm" in n or "bonferroni" in n]
