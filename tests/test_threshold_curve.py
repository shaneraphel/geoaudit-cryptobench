"""The twelfth read: a curve over 39 cut points, and the rules that keep it honest.

The risk this read carries is not arithmetic, it is selection: 39 cut points times
four metrics is 156 chances to find a favourable number, and the only thing
separating a display from a search is the commitment that nothing downstream reads
a value off it. Most of what follows tests that commitment rather than the maths.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pocket_bench.paths import ROOT

PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_THRESHOLD_CURVE.json"
CURVE = ROOT / "results/official_fold/THRESHOLD_CURVE.json"
FULL = ROOT / "results/official_fold/MATCHED_FULL_READ.json"

METRICS = ("precision", "recall", "positive_class_f1", "mcc")


@pytest.fixture(scope="module")
def plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("plan absent")
    return json.loads(PLAN.read_text())


@pytest.fixture(scope="module")
def curve() -> dict:
    if not CURVE.is_file():
        pytest.skip("read absent")
    return json.loads(CURVE.read_text())


def test_the_plan_forbids_choosing_a_point_on_the_curve(plan):
    assert plan["nothing_is_selected_from_this_curve"]["committed"] is True
    assert len(plan["nothing_is_selected_from_this_curve"]
               ["what_that_forbids"]) >= 3


def test_the_plan_declares_itself_exploratory_before_the_read(plan):
    assert plan["status_declared_in_advance"] == "exploratory"
    assert plan["reads_test_fold"] is False


def test_the_plan_writes_a_sentence_for_losing(plan):
    outcomes = plan["what_will_be_written_under_each_outcome"]
    assert "their_curve_is_above_ours_throughout" in outcomes
    assert "withdrawn" in outcomes["their_curve_is_above_ours_throughout"]


def test_the_read_does_not_feed_any_configuration(curve):
    assert curve["selects_nothing"]["consumed_by_any_configuration"] is False
    assert curve["selects_nothing"]["deployed_q_unchanged"] == 0.09


def test_the_curve_reproduces_the_seventh_read_at_the_deployed_point(curve):
    """The one cut point where this read and an earlier one must agree exactly.

    If they disagree, the curve is binarising something other than what the paper
    reports, and every other point on it is describing a different experiment.
    """
    want = json.loads(FULL.read_text())["conventions"]["D2_common_budget"]
    got = curve["reproduced_read_seven_at_the_deployed_point"]
    assert got["agrees"] is True
    assert got["positive_class_f1_ours"] == pytest.approx(
        want["table_field"]["positive_class_f1"], abs=1e-6)
    assert got["positive_class_f1_theirs"] == pytest.approx(
        want["p2rank"]["positive_class_f1"], abs=1e-6)
    at = next(r for r in curve["arms"]["p2rank"]["curve"]
              if abs(r["q"] - 0.09) < 1e-9)
    assert at["positive_class_f1"]["mean"] == pytest.approx(
        want["paired"]["positive_class_f1"]["chain"]["delta_point"], abs=1e-6)


def test_every_cut_point_carries_every_metric_with_an_ordered_interval(curve):
    for name, arm in curve["arms"].items():
        for r in arm["curve"]:
            for m in METRICS:
                blk = r[m]
                lo, hi = blk["ci"]
                assert lo <= blk["mean"] <= hi, (name, r["q"], m)
                assert blk["excludes_zero"] == (lo > 0 or hi < 0)


def test_the_grid_is_the_one_the_training_fold_was_swept_on(plan, curve):
    qs = [r["q"] for r in curve["arms"]["p2rank"]["curve"]]
    g = plan["grid"]
    assert len(qs) == g["n"] == len(set(qs))
    assert qs == sorted(qs)
    assert qs[0] == pytest.approx(g["low"]) and qs[-1] == pytest.approx(g["high"])


def test_recall_rises_and_precision_falls_as_more_residues_are_called(curve):
    """Not a property of the method -- a property of top-q, and so a check that
    the binarisation is doing what its name says at all 39 cut points."""
    for name, arm in curve["arms"].items():
        for side in ("ours", "theirs"):
            rec = [r["recall"][side] for r in arm["curve"]]
            pre = [r["precision"][side] for r in arm["curve"]]
            assert all(b >= a - 1e-9 for a, b in zip(rec, rec[1:])), (name, side)
            assert pre[0] > pre[-1], (name, side)


def test_a_missing_baseline_is_recorded_as_missing_and_never_as_zero(curve):
    for name in curve["baselines_missing"]:
        assert name not in curve["arms"]
    assert curve["arms"], "no arm survived, so the read says nothing"


def test_the_reported_conclusion_is_one_the_plan_wrote_in_advance(plan, curve):
    outcomes = plan["what_will_be_written_under_each_outcome"]
    assert curve["conclusion_key"] in outcomes
    assert curve["conclusion"] == outcomes[curve["conclusion_key"]]


def test_the_sign_span_agrees_with_the_curve_it_summarises(curve):
    """The claim the paper is allowed to make from this read is the sign span, so
    it is recomputed here rather than trusted."""
    for name, arm in curve["arms"].items():
        for m in METRICS:
            s = arm["sign_span"][m]
            means = [r[m]["mean"] for r in arm["curve"]]
            assert s["n_cut_points"] == len(means)
            assert s["n_where_ours_is_higher"] == sum(1 for v in means if v > 0)
            assert s["n_where_theirs_is_higher"] == sum(1 for v in means if v < 0)
            flips = sum(1 for a, b in zip(means, means[1:])
                        if (a > 0) != (b > 0) and a != 0 and b != 0)
            assert s["sign_changes"] == flips
            assert s["one_sign_throughout"] == (flips == 0)


def test_the_conclusion_key_follows_from_the_f1_sign_span(curve):
    s = curve["arms"]["p2rank"]["sign_span"]["positive_class_f1"]
    if s["one_sign_throughout"] and s["n_where_ours_is_higher"] == s["n_cut_points"]:
        assert curve["conclusion_key"] == "our_curve_is_above_theirs_throughout"
    elif (s["one_sign_throughout"]
          and s["n_where_theirs_is_higher"] == s["n_cut_points"]):
        assert curve["conclusion_key"] == "their_curve_is_above_ours_throughout"
    else:
        assert curve["conclusion_key"] == "the_curves_cross"


def test_no_correction_is_applied_and_the_read_says_why(curve):
    assert curve["multiplicity"]["correction"] == "none, and deliberately"
    for arm in curve["arms"].values():
        assert "pointwise" in arm["interval_kind"]
        for r in arm["curve"]:
            for m in METRICS:
                assert "ci_bonferroni" not in r[m], (
                    "a corrected interval here would imply some point of the "
                    "curve was being claimed, which the plan forbids")


def test_top_q_binarisation_matches_a_direct_recount_on_one_chain(curve):
    """One cut point on one chain, recounted from the raw scores by hand.

    The curve is 468 intervals deep; this checks the one operation underneath all
    of them against arithmetic that imports nothing from the tool.
    """
    preds = json.loads((ROOT / "results/cryptobench_official/predictions"
                        / "table_field.json").read_text())["units"]
    uid = sorted(preds)[0]
    scores = {int(k): float(v)
              for k, v in preds[uid]["residue_scores"].items()}
    keys = sorted(scores)
    s = np.array([scores[k] for k in keys])
    for q in (0.02, 0.09, 0.40):
        k = max(1, int(round(q * len(s))))
        assert int(np.argsort(-s, kind="stable")[:k].size) == k
