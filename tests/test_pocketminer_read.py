"""The eleventh read: the cryptic-specific baseline the benchmark names.

This is the comparison with the widest margin in the repository, which is exactly
why most of what follows tests the reasons to distrust it: that the baseline was
rebuilt faithfully, that its own data was removed from the fold in a second arm,
that the residue universe is shared, and that the read reports the P2Rank-minus-
PocketMiner difference beside our own so the gap cannot be read as a ranking of
methods when it is mostly a statement about label transfer.
"""
from __future__ import annotations

import json

import pytest

from pocket_bench.paths import ROOT

PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_POCKETMINER.json"
READ = ROOT / "results/official_fold/POCKETMINER_READ.json"
SELFTEST = ROOT / "results/baselines/POCKETMINER_SELFTEST.json"
SCORES = ROOT / "results/baselines/POCKETMINER_SCORES.json"


@pytest.fixture(scope="module")
def plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("plan absent")
    return json.loads(PLAN.read_text())


@pytest.fixture(scope="module")
def read() -> dict:
    if not READ.is_file():
        pytest.skip("read absent")
    return json.loads(READ.read_text())


def test_the_plan_declares_itself_exploratory_and_writes_a_losing_sentence(plan):
    assert plan["status_declared_in_advance"] == "exploratory"
    assert plan["reads_test_fold"] is False
    outcomes = plan["what_will_be_written_under_each_outcome"]
    assert "pocketminer_is_ahead" in outcomes
    assert "the_interval_crosses_zero" in outcomes
    # And a sentence for the case where the rebuild itself is what went wrong,
    # which is the outcome easiest to mistake for a win.
    assert "pocketminer_lands_near_chance" in outcomes


def test_the_plan_concedes_the_label_mismatch_before_seeing_the_result(plan):
    """The caveat that makes this read reportable has to precede it.

    PocketMiner is scored here against a label it was not trained for, and saying
    so afterwards would be indistinguishable from explaining away a result one
    happened to like.
    """
    lab = plan["the_label_it_was_not_trained_for"]
    assert lab["these_are_not_the_same_quantity"] is True
    assert lab["committed_in_advance"]
    assert "state of the art" in lab["committed_in_advance"]


def test_the_rebuild_reproduces_the_published_test_set(read):
    """The load-bearing check is the residue counts, not the AUC.

    An AUC can agree by luck across a mis-aligned label vector; the count of
    positives and negatives cannot.
    """
    rep = read["reproduction_pinned_by_the_plan"]
    assert rep["residue_counts_match_exactly"] is True
    assert abs(rep["ours_on_their_test_set"] - rep["published_roc_auc"]) < 0.01


@pytest.mark.skipif(not SELFTEST.is_file(), reason="self-test absent")
def test_the_read_quotes_the_self_test_it_actually_ran(read):
    st = json.loads(SELFTEST.read_text())["reproduction"]
    rep = read["reproduction_pinned_by_the_plan"]
    assert st["residue_counts_match_exactly"] is True
    assert st["ours"]["n_positive"] == st["published"]["n_positive"]
    assert st["ours"]["n_negative"] == st["published"]["n_negative"]
    assert abs(st["ours"]["roc_auc"] - rep["ours_on_their_test_set"]) < 1e-6
    assert abs(st["published"]["roc_auc"] - rep["published_roc_auc"]) < 1e-9


def test_the_paired_difference_is_over_a_shared_residue_universe(read):
    u = read["residue_universe"]
    assert "intersection" in u["rule"]
    assert u["n_units_compared"] == read["n_units"]
    for row in u["units_where_they_differ"]:
        assert row["n_residues_not_shared"] == abs(row["n_ours"] - row["n_theirs"])


def test_the_gap_is_reported_beside_p2ranks_so_it_cannot_be_read_as_a_ranking(read):
    """Both differences, or the read invites the wrong conclusion.

    If a general-purpose pocket finder also opens a gap on this baseline, the gap
    is mostly about the label definition, and a read that reported only our own
    difference would let that go unnoticed.
    """
    ours = read["primary"]["table_field_minus_pocketminer"]
    theirs = read["primary"]["p2rank_minus_pocketminer"]
    assert ours["n"] == theirs["n"] == read["n_units"]
    assert ours["level_second"] == theirs["level_second"]
    assert ours["n_first_ahead"] + ours["n_second_ahead"] + ours["n_tied"] \
        == ours["n"]


def test_the_contamination_arm_removes_entries_and_can_only_favour_the_baseline(read):
    ca = read["contamination_arm"]
    assert ca["n_units_left"] == read["n_units"] - len(ca["units_removed"])
    assert ca["n_units_left"] < read["n_units"]
    assert "favour PocketMiner" in ca["direction"]
    for unit in ca["units_removed"]:
        assert unit.split("_")[0] in ca["entries_removed"]


def test_the_contamination_arm_does_not_reverse_the_finding(read):
    """A sensitivity that flipped the sign would mean the headline was the overlap.

    Reported rather than asserted: if it ever does flip, this fails and the
    section has to be rewritten instead of regenerated.
    """
    full = read["primary"]["table_field_minus_pocketminer"]["mean"]
    clean = read["contamination_arm"]["table_field_minus_pocketminer"]["mean"]
    assert (full > 0) == (clean > 0)


def test_every_interval_carries_a_corrected_one_and_the_level_is_stated(read):
    m = read["multiplicity"]
    assert m["correction"] == "Bonferroni"
    assert m["corrected_level"] == pytest.approx(0.05 / m["n_paired_tests"],
                                                 abs=1e-5)
    assert len(m["tests"]) == m["n_paired_tests"]
    for key, blk in read["primary"].items():
        if not isinstance(blk, dict):
            continue
        lo, hi = blk["ci"]
        blo, bhi = blk["ci_bonferroni"]
        assert blo <= lo <= blk["mean"] <= hi <= bhi


def test_our_own_column_was_recomputed_through_the_baselines_call(read):
    """Both columns have to come from the same function or they are not paired.

    The baseline's ROC-AUC is produced by driving the harness from a prediction
    dictionary; doing the same for the counting field has to reproduce what the
    harness froze, and the read records the largest disagreement.
    """
    cal = read["calibration_of_our_own_column"]
    assert cal["n_units_recomputed"] >= read["n_units"] - 1
    assert cal["largest_absolute_disagreement"] <= 1e-6


def test_the_thresholded_arms_hold_our_own_rule_fixed(read):
    """Our side must not move between conventions; only the baseline's cut does."""
    ours = {c: blk["positive_class_f1"]["ours"]
            for c, blk in read["thresholded"].items()
            if isinstance(blk, dict) and "positive_class_f1" in blk}
    assert len(ours) >= 2
    assert len(set(round(v, 6) for v in ours.values())) == 1


def test_the_reported_conclusion_is_one_the_plan_wrote(plan, read):
    outcomes = plan["what_will_be_written_under_each_outcome"]
    assert read["outcome_key"] in outcomes
    assert read["conclusion"] == outcomes[read["outcome_key"]]
    assert read["outcome_was_named_in_the_plan"] is True


def test_the_conclusion_admits_it_is_not_a_claim_about_their_task(read):
    assert "not a claim about the label PocketMiner was trained for" \
        in read["conclusion"]


@pytest.mark.skipif(not SCORES.is_file(), reason="scores absent")
def test_the_baseline_scored_every_unit_the_read_compares():
    d = json.loads(SCORES.read_text())
    assert d["n_units"] == len(d["units"])
    for row in d["units"]:
        assert row["n_scored"] <= row["n_residues_in_file"]
        assert 0.0 <= row["max_score"] <= 1.0
        # The comparison against the authors' own featuriser is one-to-one, so it
        # cannot be made on a chain where a residue was dropped. Unverified is
        # allowed there and nowhere else, which is what stops "we did not check"
        # from spreading quietly across the fold.
        if row["agrees_with_official_featurisation"] is None:
            assert row["dropped"], row["unit"]
        else:
            assert row["agrees_with_official_featurisation"] is True
