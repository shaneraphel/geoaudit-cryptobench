"""The external read: computed under the plan, and reporting what it found.

The read is the one place in this repository where a claim can be confirmatory, and
the only thing that makes it so is that nothing about it was decided after the
numbers appeared. So these tests are mostly about the read agreeing with the plan
rather than about the numbers themselves: the statistic, the correction, the
binarisation conventions and the verdict vocabulary all have to come from the plan,
and the sentence the paper will carry has to be one the plan already contained.

Two tests are about the numbers, and they exist because both failures have happened
here before: a paired difference taken over residue universes that were not the
same, and a metric that came from a different function for one method than for
another.
"""
from __future__ import annotations

import json

import pytest

from pocket_bench.paths import ROOT

READ = ROOT / "results/external/EXTERNAL_READ.json"
PLAN = ROOT / "results/external/PREREGISTERED_EXTERNAL.json"
BASELINES = ("p2rank", "plmnn", "pocketminer")


@pytest.fixture(scope="module")
def read() -> dict:
    if not READ.is_file():
        pytest.skip("external read absent")
    return json.loads(READ.read_text())


@pytest.fixture(scope="module")
def plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("plan absent")
    return json.loads(PLAN.read_text())


def test_the_plan_that_governed_the_read_is_the_plan_on_disk(read):
    import hashlib
    assert read["plan_sha256"] == hashlib.sha256(PLAN.read_bytes()).hexdigest()
    assert read["plan"]["is_ancestor_of_head"] is True


def test_the_set_that_was_read_is_the_set_that_was_frozen(read):
    import hashlib
    s = ROOT / "results/external/EXTERNAL_SET.json"
    assert read["set_sha256"] == hashlib.sha256(s.read_bytes()).hexdigest()


def test_it_says_confirmatory_and_did_not_touch_the_old_fold(read):
    assert read["status"] == "confirmatory"
    assert read["reads_cryptobench_test_fold"] is False


def test_all_three_co_primary_comparisons_are_present(read):
    for b in BASELINES:
        assert f"table_field_minus_{b}" in read["co_primary"], b


def test_every_verdict_is_one_the_plan_defined_in_advance(read, plan):
    allowed = set(plan["replication_verdicts_defined_now"])
    for b in BASELINES:
        assert read["co_primary"][f"table_field_minus_{b}"]["verdict"] in allowed


def test_each_comparison_carries_the_number_it_was_predicted_to_be(read, plan):
    """Replication is only checkable if the prediction is beside the result."""
    pinned = plan["what_cryptobench_said_these_numbers_would_be"]
    for b in BASELINES:
        got = read["co_primary"][f"table_field_minus_{b}"]["cryptobench_predicted"]
        assert got == pinned[f"counting_field_minus_{b}_roc_auc"], b


def test_the_verdict_follows_from_the_interval_rather_than_from_taste(read):
    """The plan's table, re-derived here from the numbers in the artifact."""
    for b in BASELINES:
        blk = read["co_primary"][f"table_field_minus_{b}"]
        lo, hi = blk["ci"]
        crosses = lo <= 0.0 <= hi
        assert (blk["verdict"] == "unresolved_externally") == crosses, b
        if not crosses:
            plo, phi = blk["cryptobench_predicted"]["ci"]
            inside = plo <= blk["mean"] <= phi
            same_sign = blk["mean"] * blk["cryptobench_predicted"]["delta"] > 0
            expect = ("replicates" if inside and same_sign else
                      "same_direction_but_outside" if same_sign else
                      "fails_to_replicate")
            assert blk["verdict"] == expect, b


def test_the_sentence_the_paper_will_carry_was_written_before_the_read(read, plan):
    w = read["what_the_paper_must_now_say"]
    sentences = plan["what_will_be_written_under_each_outcome"]
    assert w["headline"] in sentences
    assert w["plmnn"] in sentences
    # The filled sentence must be the template with the placeholders substituted,
    # not a new sentence composed once the numbers were known.
    template = sentences[w["headline"]]
    prefix = template.split("{")[0]
    assert w["headline_sentence"].startswith(prefix)


def test_a_crossing_interval_is_not_reported_as_parity_with_plmnn(read):
    blk = read["co_primary"]["table_field_minus_plmnn"]
    w = read["what_the_paper_must_now_say"]
    if not blk["excludes_zero"]:
        assert w["plmnn"] == "plmnn_deficit_unresolved"
        assert "not read this as parity" in w["plmnn_sentence"] or \
               "do not read this as parity" in w["plmnn_sentence"]


def test_the_pocketminer_result_is_reported_next_to_p2ranks_margin(read):
    """Required by the plan: if P2Rank beats PocketMiner by a similar amount, the
    comparison is measuring the two label definitions, not ranking the methods."""
    assert "p2rank_minus_pocketminer_for_context" in read["co_primary"]


def test_every_comparison_ran_on_one_residue_universe(read):
    """A paired difference over two different universes is not paired. The
    intersection is taken deliberately and what it costs is reported."""
    cov = read["coverage"]
    assert "residues_not_shared_by_all_four" in cov
    for row in cov["residues_not_shared_by_all_four"]:
        assert row["n_shared"] <= min(row["n_table_field"], row["n_p2rank"],
                                      row["n_plmnn"], row["n_pocketminer"])


def test_the_units_that_dropped_out_are_named_with_a_reason(read):
    for row in read["coverage"]["units_skipped"]:
        assert row["why"] and row["unit"]


def test_enough_units_survived_to_be_worth_a_verdict(read):
    assert read["n_units_compared"] >= 30


def test_the_thresholded_arms_use_the_budget_frozen_on_the_training_fold(read):
    t = read["secondary_thresholded_against_p2rank"]
    assert t["q_came_from"]["selected_on"] == "training fold only"
    assert abs(t["q"] - 0.09) < 1e-9


def test_both_binarisation_conventions_are_reported(read):
    """The user-visible question this answers: is the F1 margin a property of the
    scores or of the two methods' different ways of calling a residue positive?"""
    t = read["secondary_thresholded_against_p2rank"]
    for arm in ("as_deployed", "common_budget"):
        for metric in ("precision", "recall", "positive_class_f1", "mcc"):
            blk = t[arm][metric]
            paired = blk["paired_difference_ours_minus_theirs"]
            dropped = max(blk["n_undefined"].values())
            assert paired["n"] == read["n_units_compared"] - dropped, (arm, metric)
            assert paired["ci"][0] <= paired["mean"] <= paired["ci"][1]


def test_a_unit_that_leaves_a_comparison_is_accounted_for(read):
    """A metric averaged over a different number of units than its neighbours is
    how a favourable subset gets chosen without anyone deciding to choose one."""
    t = read["secondary_thresholded_against_p2rank"]
    abst = t["p2rank_predicted_no_pocket_at_all"]
    for arm in ("as_deployed", "common_budget"):
        for metric in ("precision", "recall", "positive_class_f1", "mcc"):
            n_und = t[arm][metric]["n_undefined"]
            # Abstention leaves precision at 0/0 and puts a zero factor in MCC's
            # denominator. Recall and F1 stay defined and equal zero.
            if arm == "as_deployed" and metric in ("precision", "mcc"):
                assert n_und["theirs"] == abst["n"], metric
            else:
                assert sum(n_und.values()) == 0, (arm, metric)
            assert n_und["ours"] == 0, (arm, metric)


def test_the_direction_of_that_bias_is_stated(read):
    """It removes P2Rank's own worst case from two of P2Rank's own averages, so the
    artifact has to say which way it cuts rather than only that it happened."""
    abst = read["secondary_thresholded_against_p2rank"][
        "p2rank_predicted_no_pocket_at_all"]
    cuts = abst["which_way_this_cuts"]
    assert "flatter" in cuts and "MCC" in cuts
    assert abst["n"] == len(abst["units"])


def test_the_bonferroni_interval_is_wider_than_the_plain_one(read):
    for b in BASELINES:
        blk = read["co_primary"][f"table_field_minus_{b}"]
        assert blk["ci_bonferroni"][0] <= blk["ci"][0]
        assert blk["ci_bonferroni"][1] >= blk["ci"][1]


def test_the_trimmed_mean_is_present_but_carries_no_verdict(read):
    """It is exploratory and stays exploratory. Reporting it makes visible that it
    did not quietly become the endpoint when the mean failed to resolve."""
    for b in BASELINES:
        blk = read["co_primary"][f"table_field_minus_{b}"]
        assert "trimmed_mean_exploratory" in blk
        assert "trimmed" not in json.dumps(read["what_the_paper_must_now_say"])


def test_the_four_methods_all_have_a_reported_level(read):
    lv = read["levels"]
    assert set(lv) == {"table_field", "p2rank", "plmnn", "pocketminer"}
    for m, v in lv.items():
        assert 0.0 <= v <= 1.0, m
