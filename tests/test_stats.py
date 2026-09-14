"""Statistics module: complete-block enforcement and battery shape."""

import numpy as np
import pandas as pd
import pytest

from harness.config import CAMPAIGN_LABELS
from harness.stats import (DESCRIPTIVE_METRICS, FORMAL_METRICS,
                           PAPER_READOUT, SPINE_READOUTS,
                           STATS_EXCLUDED_METHODS, battery_role,
                           comparator_roster, control_wilcoxon,
                           friedman_conover, holm, per_dataset_matrix,
                           readout_ablation, readout_roster,
                           run_all_stats)

#: The campaign's real roster, in the order the runner emits it.  SPINE
#: contributes TWO method names from one fit -- its vertices scored by
#: 1-NN, and the same fitted model under its own decision rule.  They are
#: two EVALUATIONS of one fit, so they are never two treatments in the
#: same omnibus; each gets its own battery against the shared comparator
#: roster, and their own contrast is the paired readout ablation.
CAMPAIGN_METHODS = ("Full", "RSP3", "SPINE+1NN", "SPINE+Graph",
                    "KMeans", "Random", "LVQ3", "SPOT", "GLVQ", "GNG")
#: Budget-matched to the RSP3 anchor: their reduction rate IS RSP3's.
MATCHED_METHODS = ("SPINE+1NN", "SPINE+Graph", "KMeans", "Random", "LVQ3",
                   "SPOT", "GLVQ", "GNG")


def _fake_results(n_datasets=8, methods=("A", "B", "C")):
    rng = np.random.default_rng(7)
    rows = []
    for d in range(n_datasets):
        for m_i, m in enumerate(methods):
            for fold in range(3):
                rows.append({
                    "dataset": "ds{0}".format(d),
                    "fold": fold,
                    "method": m,
                    # the last method is systematically better
                    "acc_1nn": 0.6 + 0.1 * m_i + rng.normal(0, 0.01),
                })
    return pd.DataFrame(rows)


def _campaign_frame(methods=CAMPAIGN_METHODS, seed=11):
    """Fake campaign results with the REAL budget-matching structure.

    'Full' has reduction 0; RSP3 gets its own reduction; every
    budget-matched method copies RSP3's per-fold value exactly, which is
    what makes ranking reduction degenerate.
    """
    df = _fake_results(methods=methods)
    rng = np.random.default_rng(seed)
    order = {m: i for i, m in enumerate(methods)}
    mi = df["method"].map(order).to_numpy()
    df["reduction"] = np.where(
        df.method == "Full", 0.0,
        0.4 + 0.05 * mi + rng.normal(0, 0.01, len(df)))
    df["n_proto"] = (100 - 5 * mi).astype(float)
    df["cpu_select"] = 0.001 * (mi + 1) + rng.normal(0, 1e-4, len(df))
    df["cpu_total"] = df["cpu_select"]
    if "RSP3" in methods:
        anchor = (df[df.method == "RSP3"]
                  .set_index(["dataset", "fold"])["reduction"])
        matched = df.method.isin(MATCHED_METHODS)
        df.loc[matched, "reduction"] = (
            pd.MultiIndex.from_frame(df.loc[matched, ["dataset", "fold"]])
            .map(anchor).to_numpy())
    return df


def _write(df, results_dir):
    results_dir.mkdir()
    for ds, sub in df.groupby("dataset"):
        sub.to_csv(results_dir / "{0}.csv".format(ds), index=False)
    return str(results_dir)


def test_friedman_detects_systematic_difference():
    res = friedman_conover(_fake_results(), "acc_1nn", control="C")
    assert res["significant"]
    ranks = res["average_ranks"]
    assert ranks.index[0] == "C"  # best method ranked first
    assert res["friedman_df"] == res["n_methods"] - 1
    # the Conover family is the control against each other treatment,
    # never the control against itself
    assert list(res["conover_raw"].index) == ["A", "B"]
    assert res["conover_raw"]["A"] < 0.05


def test_incomplete_block_is_hard_error():
    df = _fake_results()
    df = df[~((df.dataset == "ds0") & (df.method == "B"))]
    with pytest.raises(ValueError):
        per_dataset_matrix(df, "acc_1nn")


def test_accuracy_is_the_only_ranked_metric():
    # Reduction left the formal battery when the budget anchor moved to
    # RSP3: every ranked method then emits the anchor's count, so the
    # test would be a tautology.  Guards against it creeping back.
    assert tuple(m for m, _ in FORMAL_METRICS) == ("acc_1nn",)
    assert "reduction" in DESCRIPTIVE_METRICS


def test_run_all_stats_excludes_full_from_formal_tests(tmp_path):
    # 'Full' is the NOP reference (reduction 0 by construction): it stays
    # in the raw result CSVs but must not enter the battery.  Guards
    # against the filter silently no-op'ing (e.g. if the 'Full' label
    # were ever renamed).
    results_dir = tmp_path / "results"
    _write(_campaign_frame(), results_dir)

    out = run_all_stats(str(results_dir), str(tmp_path / "stats"))

    assert set(out) == {"acc_1nn"}
    for battery in out["acc_1nn"]["batteries"].values():
        assert "Full" not in battery["average_ranks"].index
        assert "Full" not in battery["scores"].columns
        # one readout + the 7 comparators; 'Full' and the OTHER readout
        # are both out (10 -> 8)
        assert battery["n_methods"] == len(CAMPAIGN_METHODS) - 2

    # the raw CSVs are untouched: 'Full' is still recorded, just not tested
    raw = pd.concat([pd.read_csv(p) for p in results_dir.glob("*.csv")])
    assert "Full" in set(raw["method"])


def test_reduction_is_written_descriptively_not_ranked(tmp_path):
    # The budget-matched comparators are DEFINED to emit the anchor's
    # prototype count, so their reduction rates are identical to RSP3's.
    # Reduction is reported, not ranked.
    results_dir = tmp_path / "results"
    df = _campaign_frame()
    _write(df, results_dir)

    # the fixture reproduces the degeneracy the change exists for
    wide = df.pivot_table(index=["dataset", "fold"], columns="method",
                          values="reduction")
    for m in MATCHED_METHODS:
        assert np.allclose(wide[m], wide["RSP3"]), m

    stats_dir = tmp_path / "stats"
    out = run_all_stats(str(results_dir), str(stats_dir))

    assert "reduction" not in out
    assert not (stats_dir / "stats_reduction_ranks.csv").exists()

    # ... but every method's reduction is on disk, descriptively --
    # INCLUDING 'Full', which is excluded from the rank tests but is the
    # row that anchors this table (reduction 0, n_proto = n_train).
    table = pd.read_csv(stats_dir / "descriptive_reduction_scores.csv",
                        index_col=0)
    assert set(table.columns) == set(CAMPAIGN_METHODS)
    assert np.allclose(table["Full"], 0.0)
    for m in MATCHED_METHODS:
        assert np.allclose(table[m], table["RSP3"]), m


def test_descriptive_tables_skip_absent_columns(tmp_path):
    # A metric the runner did not record is skipped, never fabricated.
    results_dir = tmp_path / "results"
    df = _campaign_frame(seed=13).drop(columns=["cpu_total"])
    _write(df, results_dir)

    stats_dir = tmp_path / "stats"
    run_all_stats(str(results_dir), str(stats_dir))

    assert (stats_dir / "descriptive_cpu_select_scores.csv").exists()
    assert not (stats_dir / "descriptive_cpu_total_scores.csv").exists()


def test_cpu_columns_recorded_but_not_formally_tested(tmp_path):
    # The timing columns stay in the raw results as reference, but the
    # formal battery ranks accuracy only -- the computational comparison
    # lives in the dedicated timing study.
    results_dir = tmp_path / "results"
    _write(_campaign_frame(seed=13), results_dir)

    out = run_all_stats(str(results_dir), str(tmp_path / "stats"))

    assert set(out) == {"acc_1nn"}
    # Random stays a ranked treatment on accuracy, the metric it can win
    for battery in out["acc_1nn"]["batteries"].values():
        assert "Random" in battery["average_ranks"].index
    # the recorded cpu column survives in the raw results untouched
    raw = pd.concat([pd.read_csv(p) for p in results_dir.glob("*.csv")])
    assert "cpu_select" in raw.columns


# --------------------------------------------------------------------
# SPINE's two readouts are two EVALUATIONS of one fit, not two methods.
# --------------------------------------------------------------------

def test_no_omnibus_contains_both_readouts(tmp_path):
    # The separation guarantee.  One fit ranked twice occupies two of the
    # k rank positions and pushes every genuine comparator down.
    results_dir = tmp_path / "results"
    _write(_campaign_frame(), results_dir)

    out = run_all_stats(str(results_dir), str(tmp_path / "stats"))
    batteries = out["acc_1nn"]["batteries"]

    assert set(batteries) == set(SPINE_READOUTS)
    for readout, battery in batteries.items():
        present = set(battery["scores"].columns) & set(SPINE_READOUTS)
        assert present == {readout}, present


def test_both_readouts_face_the_identical_comparator_roster(tmp_path):
    # The two batteries are only comparable to each other if the field
    # they are ranked against is the same field.
    results_dir = tmp_path / "results"
    _write(_campaign_frame(), results_dir)

    out = run_all_stats(str(results_dir), str(tmp_path / "stats"))
    rosters = [[m for m in b["scores"].columns if m not in SPINE_READOUTS]
               for b in out["acc_1nn"]["batteries"].values()]

    assert rosters[0] == rosters[1]
    assert "Full" not in rosters[0]
    assert set(rosters[0]) == set(CAMPAIGN_METHODS) - {"Full"} - set(
        SPINE_READOUTS)


def test_comparator_roster_drops_reference_and_both_readouts():
    df = _campaign_frame()
    roster = comparator_roster(df)

    assert "Full" not in roster
    assert not set(roster) & set(SPINE_READOUTS)
    # readout_roster puts the readout first, then that same field
    for readout in SPINE_READOUTS:
        assert readout_roster(df, readout) == [readout] + roster


def test_comparator_roster_is_ordered_by_the_declared_campaign():
    # Not by the order rows happen to land in the concatenated CSVs:
    # a re-run shard written with a reordered --methods list would
    # otherwise reshuffle the columns of every written table.
    df = _campaign_frame()
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)

    assert comparator_roster(shuffled) == comparator_roster(df)
    declared = [m for m in CAMPAIGN_LABELS
                if m not in set(STATS_EXCLUDED_METHODS) | set(SPINE_READOUTS)]
    assert comparator_roster(df) == declared


def test_a_missing_comparator_is_a_hard_error_not_a_smaller_field():
    # The roster is derived from the results, so without this guard a
    # comparator whose shard failed would simply vanish and SPINE would
    # be ranked against an easier field, reported in the same format.
    df = _campaign_frame()
    df = df[df.method != "GNG"]

    with pytest.raises(ValueError, match="GNG"):
        comparator_roster(df)


def test_a_fewer_than_two_block_campaign_is_a_hard_error():
    # One dataset yields a chi-square, a Conover matrix of NaNs and a
    # p-value formatted exactly like a real result.
    df = _fake_results(n_datasets=1)
    with pytest.raises(ValueError, match="block"):
        per_dataset_matrix(df, "acc_1nn")


def test_readout_roster_rejects_a_non_readout():
    with pytest.raises(ValueError):
        readout_roster(_campaign_frame(), "KMeans")


def test_ranking_one_fit_twice_shifts_comparator_ranks_by_its_win_rate():
    # The exact law behind the separation, not a slogan.  Adding a
    # second readout to an omnibus raises each comparator's average rank
    # by precisely the readout's per-dataset win rate against it (a tie
    # counting as a half) -- so a comparator the readout never beats
    # does not move at all, and no comparator can improve.
    df = _campaign_frame()
    df = df[~df.method.isin(STATS_EXCLUDED_METHODS)]
    added = "SPINE+Graph"

    one = friedman_conover(df, "acc_1nn", control="SPINE+1NN",
                           methods=readout_roster(df, "SPINE+1NN"))
    both = friedman_conover(df, "acc_1nn", control="SPINE+1NN",
                            methods=list(SPINE_READOUTS)
                            + comparator_roster(df))
    scores = per_dataset_matrix(df, "acc_1nn")

    for m in comparator_roster(df):
        expected = float(((scores[added] > scores[m]).astype(float)
                          + 0.5 * (scores[added] == scores[m])).mean())
        shift = both["average_ranks"][m] - one["average_ranks"][m]
        assert shift == pytest.approx(expected), m
        assert shift >= 0, m


def test_readout_ablation_is_paired_and_directional():
    # In the fixture SPINE+Graph is systematically the better readout,
    # so the contrast (Graph - points) must be positive on every dataset.
    df = _campaign_frame()
    abl = readout_ablation(df, "acc_1nn")

    assert abl["contrast"] == "SPINE+Graph - SPINE+1NN"
    assert abl["n_graph_better"] == abl["n_datasets"]
    assert abl["n_points_better"] == 0
    assert abl["n_tied"] == 0
    assert abl["mean_delta"] > 0
    assert abl["significant"]
    # the ablation is a two-column paired table, never a rank over a field
    assert list(abl["scores"].columns) == list(SPINE_READOUTS)


def test_all_tied_readouts_is_a_hard_error():
    # Identical scores on EVERY dataset is the signature of the graph
    # readout never being scored, or of Phase 2a stripping every
    # skeleton to isolated points -- not of a neutral decision rule.
    # Reporting p = 1.0 would be a finding about a broken run.
    df = _campaign_frame()
    points = df.method == "SPINE+1NN"
    graph = df.method == "SPINE+Graph"
    df.loc[graph, "acc_1nn"] = df.loc[points, "acc_1nn"].to_numpy()

    with pytest.raises(ValueError, match="every dataset ties"):
        readout_ablation(df, "acc_1nn")


def test_readout_ablation_declares_whether_its_p_value_is_exact():
    # The exact null assumes distinct retained |differences|; scipy
    # rounds conservatively otherwise.  The count must be reported so a
    # reader knows which number they have.
    df = _campaign_frame()
    abl = readout_ablation(df, "acc_1nn")
    assert abl["n_abs_ties"] >= 0

    # force two datasets to share a |delta| exactly: identical operands
    # give a bit-identical difference, which float arithmetic on merely
    # equal-looking values would not.
    tied = _campaign_frame()
    for ds in ("ds0", "ds1"):
        rows = tied.dataset == ds
        tied.loc[rows & (tied.method == "SPINE+1NN"), "acc_1nn"] = 0.5
        tied.loc[rows & (tied.method == "SPINE+Graph"), "acc_1nn"] = 0.6
    assert readout_ablation(tied, "acc_1nn")["n_abs_ties"] >= 1


def test_missing_readout_is_a_hard_error():
    # The runner emits both rows from one fit, so a campaign carrying
    # only one of them is broken, not a variant worth adapting to.
    df = _campaign_frame()
    df = df[df.method != "SPINE+Graph"]
    with pytest.raises(ValueError):
        readout_ablation(df, "acc_1nn")


def test_battery_and_ablation_tables_are_written_per_readout(tmp_path):
    results_dir = tmp_path / "results"
    _write(_campaign_frame(), results_dir)
    stats_dir = tmp_path / "stats"

    run_all_stats(str(results_dir), str(stats_dir))

    for slug, role in (("spine_1nn", "ablation"),
                       ("spine_graph", "reported")):
        for suffix in ("_scores.csv", "_ranks.csv", "_comparison.csv",
                       "_conover_raw.csv"):
            path = stats_dir / "acc_1nn_{0}__{1}{2}".format(
                slug, role, suffix)
            assert path.exists(), path
    # the only Conover matrix written is the unadjusted one
    assert (sorted(p.name for p in stats_dir.glob("*conover*"))
            == ["acc_1nn_spine_1nn__ablation_conover_raw.csv",
                "acc_1nn_spine_graph__reported_conover_raw.csv"])
    assert not list(stats_dir.glob("*bonferroni*"))
    assert (stats_dir / "ablation_acc_1nn_spine_readouts.txt").exists()
    assert (stats_dir
            / "ablation_acc_1nn_spine_readouts_scores.csv").exists()
    # the single-battery filenames of the old one-omnibus layout are gone
    assert not (stats_dir / "stats_acc_1nn_ranks.csv").exists()


# --------------------------------------------------------------------
# The control comparison: one Holm family, two procedures.
# --------------------------------------------------------------------

def test_holm_is_step_down_monotone_and_capped():
    # Worked by hand: sorted p = .01, .02, .03, .04 against multipliers
    # 4, 3, 2, 1 gives raw steps .04, .06, .06, .04, and the running
    # maximum carries the .06 forward -- an adjusted p-value can never
    # fall below one that precedes it, which is what makes the step-down
    # procedure coherent (without it the last comparison would be
    # reported as .04, more significant than the .06 above it).
    adj = holm([0.01, 0.02, 0.03, 0.04])
    assert adj == pytest.approx([0.04, 0.06, 0.06, 0.06])
    # order is the INPUT order, not sorted order
    assert holm([0.04, 0.01]) == pytest.approx(holm([0.01, 0.04])[::-1])
    # never above 1, and never below the Bonferroni-of-the-smallest floor
    assert holm([0.5, 0.6, 0.7]).max() == 1.0
    assert holm([0.001, 0.9])[0] == pytest.approx(0.002)
    # a single comparison is uncorrected by definition
    assert holm([0.031]) == pytest.approx([0.031])


def test_the_conover_family_is_holm_over_the_control_comparisons():
    # The correction is applied to the control ROW, over k-1 values, by
    # the same holm() the Wilcoxon family uses.  That the multiplier is
    # k-1 rather than scikit-posthocs' all-pairs k(k-1)/2 needs
    # non-degenerate p-values to show, so it is pinned on the real
    # campaign in tests/test_stats_fixture.py; here the identity is
    # what matters (this fixture's raw p-values underflow to 0.0).
    df = _campaign_frame()
    df = df[~df.method.isin(STATS_EXCLUDED_METHODS)]
    roster = readout_roster(df, PAPER_READOUT)
    res = friedman_conover(df, "acc_1nn", control=PAPER_READOUT,
                           methods=roster)

    assert len(res["conover_raw"]) == len(roster) - 1
    assert list(res["conover_raw"].index) == roster[1:]
    assert list(res["conover_holm"]) == pytest.approx(
        list(holm(res["conover_raw"].to_numpy())))
    assert (res["conover_holm"] >= res["conover_raw"]).all()


def test_control_wilcoxon_is_control_against_each_competitor():
    df = _campaign_frame()
    df = df[~df.method.isin(STATS_EXCLUDED_METHODS)]
    roster = readout_roster(df, PAPER_READOUT)
    rows = control_wilcoxon(df, "acc_1nn", control=PAPER_READOUT,
                            methods=roster)

    # k-1 comparisons, in roster order, control never against itself
    assert [r["method"] for r in rows] == roster[1:]
    assert PAPER_READOUT not in {r["method"] for r in rows}
    scores = per_dataset_matrix(df, "acc_1nn", methods=roster)
    for row in rows:
        # the counts partition the blocks, and n_used is what the
        # signed-rank statistic actually saw
        assert row["wins"] + row["losses"] + row["ties"] == len(scores)
        assert row["n_used"] == row["wins"] + row["losses"]
        delta = scores[PAPER_READOUT] - scores[row["method"]]
        assert row["mean_delta"] == pytest.approx(delta.mean())
        # Holm can only make a p-value larger
        assert row["p_holm"] >= row["p_raw"]
        assert row["significant"] == (row["p_holm"] < 0.05)


def test_exact_ties_are_dropped_from_the_signed_rank_sample():
    # A dataset where the control and a competitor score identically
    # contributes no signed rank; it is counted as a tie and n_used
    # falls, so the count and the p-value describe the same sample.
    df = _campaign_frame()
    df = df[~df.method.isin(STATS_EXCLUDED_METHODS)]
    roster = readout_roster(df, PAPER_READOUT)
    tied_on = roster[1]
    src = df[(df.method == PAPER_READOUT) & (df.dataset == "ds0")]
    df = df.copy()
    for _, row in src.iterrows():
        mask = ((df.method == tied_on) & (df.dataset == "ds0")
                & (df.fold == row["fold"]))
        df.loc[mask, "acc_1nn"] = row["acc_1nn"]

    rows = {r["method"]: r for r in control_wilcoxon(
        df, "acc_1nn", control=PAPER_READOUT, methods=roster)}
    assert rows[tied_on]["ties"] == 1
    assert rows[tied_on]["n_used"] == len(set(df.dataset)) - 1
    assert rows[roster[2]]["ties"] == 0


def test_a_shared_difference_magnitude_is_refused_not_called_exact():
    # scipy's exact null assumes the retained |differences| are
    # distinct; where they are not it rounds the statistic and the
    # p-value is conservative.  Reporting that under a header saying
    # "method=exact" would misdescribe it, so it is a hard error.
    df = _campaign_frame()
    df = df[~df.method.isin(STATS_EXCLUDED_METHODS)]
    roster = readout_roster(df, PAPER_READOUT)
    scores = per_dataset_matrix(df, "acc_1nn", methods=roster)
    victim = roster[1]
    delta = float(scores[PAPER_READOUT]["ds0"] - scores[victim]["ds0"])
    df = df.copy()
    # give ds1 the same signed difference ds0 has
    for fold in sorted(set(df.fold)):
        ctrl = df[(df.method == PAPER_READOUT) & (df.dataset == "ds1")
                  & (df.fold == fold)]["acc_1nn"].iloc[0]
        mask = ((df.method == victim) & (df.dataset == "ds1")
                & (df.fold == fold))
        df.loc[mask, "acc_1nn"] = ctrl - delta

    with pytest.raises(ValueError, match="share a magnitude"):
        control_wilcoxon(df, "acc_1nn", control=PAPER_READOUT,
                         methods=roster)


def test_a_control_outside_its_own_roster_is_a_hard_error():
    df = _campaign_frame()
    df = df[~df.method.isin(STATS_EXCLUDED_METHODS)]
    roster = readout_roster(df, PAPER_READOUT)
    with pytest.raises(ValueError, match="not in the battery roster"):
        friedman_conover(df, "acc_1nn", control="Full", methods=roster)
    with pytest.raises(ValueError, match="not in the battery roster"):
        control_wilcoxon(df, "acc_1nn", control="Full", methods=roster)


def test_both_families_are_recorded_and_their_agreement_reported(tmp_path):
    results_dir = tmp_path / "results"
    _write(_campaign_frame(), results_dir)
    out = run_all_stats(str(results_dir), str(tmp_path / "stats"))

    for readout, battery in out["acc_1nn"]["batteries"].items():
        wil = {r["method"] for r in battery["wilcoxon"] if r["significant"]}
        con = {m for m in battery["conover_significant"].index
               if battery["conover_significant"][m]}
        assert battery["families_agree"] == (wil == con)
        assert battery["scipy_version"]
        # the two families cover exactly the same comparisons
        assert {r["method"] for r in battery["wilcoxon"]} == set(
            battery["conover_raw"].index)


def test_the_paper_battery_is_named_in_its_filename():
    # The output directory holds two batteries; only one is the
    # confirmatory comparison, and the filename is what says so.
    assert battery_role(PAPER_READOUT) == "reported"
    others = [r for r in SPINE_READOUTS if r != PAPER_READOUT]
    assert others and all(battery_role(r) == "ablation" for r in others)


def test_rhc_is_excluded_so_legacy_result_sets_still_load():
    # RHC left config.MAIN_METHODS entirely, so a campaign run today
    # emits no RHC rows.  Result sets collected BEFORE the removal do
    # carry them, and they must be ignored rather than treated as a
    # method the protocol never announced.
    assert "RHC" in STATS_EXCLUDED_METHODS
    assert "RHC" not in CAMPAIGN_LABELS

    df = _campaign_frame()
    legacy = df[df.method == "RSP3"].copy()
    legacy["method"] = "RHC"
    legacy["acc_1nn"] = legacy["acc_1nn"] - 0.05
    df = pd.concat([df, legacy], ignore_index=True)

    roster = comparator_roster(df)
    assert "RHC" not in roster
    assert set(roster) == set(CAMPAIGN_METHODS) - {"Full"} - set(
        SPINE_READOUTS)


def test_the_written_conover_matrix_is_unadjusted(tmp_path):
    # Holm covers the k-1 control comparisons.  A k x k matrix of
    # adjusted values would correct 21 of its 28 cells under a family
    # the protocol never reports, so the matrix is written raw and the
    # adjusted values live only in the control comparison table.
    results_dir = tmp_path / "results"
    _write(_campaign_frame(), results_dir)
    stats_dir = tmp_path / "stats"
    out = run_all_stats(str(results_dir), str(stats_dir))
    battery = out["acc_1nn"]["batteries"][PAPER_READOUT]

    written = pd.read_csv(
        stats_dir / "acc_1nn_spine_graph__reported_conover_raw.csv",
        index_col=0)
    roster = list(battery["scores"].columns)
    assert list(written.index) == roster
    assert list(written.columns) == roster
    assert written.shape == (len(roster), len(roster))

    # every cell is the UNADJUSTED p-value: the control row of the file
    # is the raw control family, not the Holm-adjusted one
    control_row = written.loc[PAPER_READOUT].drop(PAPER_READOUT)
    assert list(control_row) == pytest.approx(
        list(battery["conover_raw"]))
    adjusted = battery["conover_holm"]
    assert not any(
        control_row[m] == pytest.approx(adjusted[m])
        and adjusted[m] != control_row[m] for m in control_row.index)
    # and the matrix is symmetric, which an adjustment applied row-wise
    # would not preserve
    assert written.to_numpy() == pytest.approx(written.to_numpy().T)
