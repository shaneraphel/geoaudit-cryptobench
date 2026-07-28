"""The choice of functional has to stay a choice that was made in advance.

A preregistration is worth nothing if the file can drift afterwards into
agreeing with whatever the fold returned. These tests hold the four things that
make it binding: the artifact never touched the fold, the statistic it names is
the one its own selection rule returns from its own recorded numbers, the
forecast it made is still the forecast, and the mean it displaced is still being
reported beside it.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import preregister_statistic as ps  # noqa: E402

ART_PATH = ps.OUT


def _art():
    if not ART_PATH.exists():
        raise unittest.SkipTest("run `make prereg` to produce the artifact")
    return json.loads(ART_PATH.read_text())


class TestTheArtifactAudits(unittest.TestCase):
    def test_its_own_audit_passes(self):
        self.assertEqual(ps.audit(), 0)

    def test_the_training_fold_p2rank_run_declares_the_fold_untouched(self):
        if not ps.P2TRAIN.exists():
            self.skipTest("run tools/run_p2rank_on_train.py")
        doc = json.loads(ps.P2TRAIN.read_text())
        self.assertIs(doc["test_fold_touched"], False)
        self.assertEqual(doc["n_ok"], doc["n_units"])


class TestBothArmsAreOutOfSample(unittest.TestCase):
    """The field's cells are counted on the training fold, so a comparison made
    there is only fair if the field was compiled on the other half."""

    def test_the_comparison_says_so_and_the_halves_are_disjoint(self):
        c = _art()["comparison"]
        self.assertIs(c["both_arms_out_of_sample"], True)
        self.assertEqual(c["n_fit_units"] + c["n_paired_units"],
                         c["n_train_units"])
        self.assertGreater(c["n_paired_units"], 300)


class TestTheSelectionIsNotPostHoc(unittest.TestCase):
    def test_the_named_statistic_is_what_the_rule_returns(self):
        art = _art()
        self.assertEqual(ps._select(art["candidates"])["statistic"],
                         art["preregistered"]["statistic"])

    def test_an_uncalibrated_candidate_could_never_have_been_chosen(self):
        """Guard the rule, not this run: power alone must not be able to win."""
        rows = [
            {"statistic": "mean", "calibrated": False, "claim_strength": 5,
             "mean_power_over_curve": 1.0},
            {"statistic": "median", "calibrated": True, "claim_strength": 3,
             "mean_power_over_curve": 0.4},
        ]
        self.assertEqual(ps._select(rows)["statistic"], "median")

    def test_a_bigger_point_estimate_cannot_win_a_tie(self):
        """A win rate and a ROC-AUC margin are not on one scale, so the larger
        number is not the better candidate."""
        rows = [
            {"statistic": "win_rate", "calibrated": True, "claim_strength": 0,
             "mean_power_over_curve": 0.70, "point_on_pick_half": 0.18},
            {"statistic": "median", "calibrated": True, "claim_strength": 3,
             "mean_power_over_curve": 0.69, "point_on_pick_half": 0.04},
        ]
        self.assertEqual(ps._select(rows)["statistic"], "median")


class TestTheFindingThatMotivatesIt(unittest.TestCase):
    """The mean is a bad functional here. That is the result, and if it stops
    being true the preregistration has lost its reason to exist."""

    def test_the_mean_is_far_weaker_than_what_was_chosen(self):
        art = _art()
        by = {c["statistic"]: c for c in art["candidates"]}
        chosen = art["preregistered"]["statistic"]
        self.assertLess(by["mean"]["mean_power_over_curve"],
                        by[chosen]["mean_power_over_curve"] - 0.2)

    def test_the_mean_all_but_vanishes_at_a_quarter_of_the_effect(self):
        by = {c["statistic"]: c for c in _art()["candidates"]}
        self.assertLess(by["mean"]["power_by_effect_shrink"]["0.25"], 0.15)

    def test_stratifying_on_chain_length_bought_nothing(self):
        """A negative result the manuscript reports: the obvious covariate is
        not where the variance is."""
        by = {c["statistic"]: c for c in _art()["candidates"]}
        if "stratified_by_length" not in by:
            self.skipTest("candidate not present")
        self.assertLess(
            abs(by["stratified_by_length"]["mean_power_over_curve"]
                - by["mean"]["mean_power_over_curve"]), 0.05)


class TestTheForecastIsBinding(unittest.TestCase):
    def test_it_did_not_feed_the_choice(self):
        f = _art()["forecast"]
        self.assertIs(f["used_for_selection"], False)
        self.assertIs(f["conditions_on_a_held_out_number"], True)

    def test_its_prose_matches_its_number(self):
        """So that a read which fails to resolve cannot be reframed later."""
        f = _art()["forecast"]
        likely = f["expected_power"] >= 0.5
        self.assertEqual(likely, "more likely than not to clear zero"
                         in f["reading"])

    def test_the_mean_is_not_being_quietly_dropped(self):
        self.assertTrue(_art()["the_mean_is_still_reported"])


if __name__ == "__main__":
    unittest.main()
