"""The external plan: written before the read, and shaped so it cannot be rewritten after.

A preregistration is only worth the constraints it imposes on the person who wrote
it. This one claims the word confirmatory, which no other plan in this repository
can, so the constraints have to be stricter rather than looser: the outcome
sentences are fixed in advance for the bad outcomes as well as the good ones, the
CryptoBench numbers are pinned as predictions so replication is falsifiable, and
the method is forbidden from changing in response to whatever comes back.

These tests check the plan's shape. Whether it was actually committed before the
read is a question about the commit graph, and the gate in the read tool answers
that one.
"""
from __future__ import annotations

import json

import pytest

from pocket_bench.paths import ROOT

PLAN = ROOT / "results/external/PREREGISTERED_EXTERNAL.json"
SET = ROOT / "results/external/EXTERNAL_SET.json"


@pytest.fixture(scope="module")
def plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("plan absent")
    return json.loads(PLAN.read_text())


def test_the_plan_has_not_read_what_it_plans_to_read(plan):
    assert plan["reads_external_set"] is False
    assert plan["reads_test_fold"] is False
    assert plan["written_before_the_read"] is True


def test_it_claims_confirmatory_and_says_what_earns_that(plan):
    """The word is the point of the whole exercise, so it needs its reasons attached."""
    assert plan["status_declared_in_advance"] == "confirmatory"
    why = plan["why_this_one_may_say_confirmatory"]
    assert {"the_set_has_never_been_read", "the_labels_came_first",
            "the_boundary_is_a_date_and_a_cluster_not_a_promise",
            "what_this_does_not_launder"} <= set(why)


def test_it_does_not_claim_to_launder_the_earlier_reads(plan):
    """An external success cannot retrospectively confirm twelve exploratory reads."""
    assert "exploratory" in plan["why_this_one_may_say_confirmatory"][
        "what_this_does_not_launder"]


def test_the_method_may_not_change_after_the_read(plan):
    """The commitment that makes the set single-use. Without it there is no
    external validation, only a slower tuning loop."""
    m = plan["methods_and_what_is_frozen_about_them"]
    for name in ("counting_field", "p2rank", "plmnn", "pocketminer"):
        assert m[name]["may_change_after_this_read"] is False, name
    assert "published bad" in plan["the_commitment_that_matters"]


def test_the_set_is_pinned_by_hash(plan):
    if not SET.is_file():
        pytest.skip("set absent")
    import hashlib
    assert (plan["the_set"]["sha256"]
            == hashlib.sha256(SET.read_bytes()).hexdigest())


def test_the_cryptobench_numbers_are_pinned_as_predictions(plan):
    """Replication is only falsifiable if the prediction is recorded first.

    Each of the three comparisons carries the point estimate and interval
    CryptoBench gave for it, so the external number can be held against something
    rather than described however it lands.
    """
    p = plan["what_cryptobench_said_these_numbers_would_be"]
    for key in ("counting_field_minus_p2rank_roc_auc",
                "counting_field_minus_plmnn_roc_auc",
                "counting_field_minus_pocketminer_roc_auc"):
        assert isinstance(p[key]["delta"], float)
        lo, hi = p[key]["ci"]
        assert lo < p[key]["delta"] < hi, key


def test_the_plmnn_prediction_is_the_one_against_us(plan):
    """The pinned prediction for the strongest baseline is a deficit. A plan that
    only predicted its own wins would not be predicting anything."""
    d = plan["what_cryptobench_said_these_numbers_would_be"][
        "counting_field_minus_plmnn_roc_auc"]
    assert d["delta"] < 0
    assert d["resolved"] is True


def test_all_three_claims_are_on_trial_not_just_the_flattering_one(plan):
    c = plan["co_primary"]
    assert len(c["tests"]) == 3
    assert c["correction"].startswith("Bonferroni")
    assert abs(c["corrected_level"] - 0.05 / 3) < 1e-6


def test_the_primary_functional_is_the_one_already_declared_primary(plan):
    """Switching summaries between the internal and external read would make the
    replication unfalsifiable. This repository has retracted one such switch."""
    s = plan["statistic"]
    assert "mean" in s["primary_functional"]
    assert "exploratory" in s["trimmed_mean_is_not_used_here"]
    assert "unfalsifiable" in s["why_the_same_functional_as_before"]


def test_the_losing_outcomes_have_sentences_written_too(plan):
    w = plan["what_will_be_written_under_each_outcome"]
    assert {"p2rank_parity_replicates", "p2rank_advantage_appears",
            "p2rank_advantage_reverses", "plmnn_deficit_replicates",
            "plmnn_deficit_unresolved", "underpowered"} == set(w)
    assert "withdraw it" in w["p2rank_advantage_reverses"]
    assert "underpowered" in w["underpowered"]


def test_a_failure_to_resolve_is_not_reported_as_parity(plan):
    """The most tempting misreading available here: a smaller set failing to
    resolve the pLM-NN deficit is not evidence the deficit is gone."""
    assert "not evidence" in plan["what_will_be_written_under_each_outcome"][
        "plmnn_deficit_unresolved"]


def test_both_directions_of_bias_in_the_set_are_declared(plan):
    h = plan["how_this_set_is_easier_and_harder_than_cryptobench"]
    assert len(h["easier"]) >= 3 and len(h["harder"]) >= 2


def test_the_read_is_told_what_it_must_verify(plan):
    g = plan["guards_the_read_must_pass"]
    assert len(g) >= 8
    assert any("ancestor of HEAD" in x for x in g)
    assert any("hashes to what is recorded here" in x for x in g)
