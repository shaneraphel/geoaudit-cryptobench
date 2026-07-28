"""The matched-threshold analysis says what its artifacts say.

Three things could go wrong here without any of them being visible in the
numbers. The matched rule could drift from the rule our method actually ships,
in which case "matched" would be a claim about a threshold nobody uses. The
preregistration could be edited after the read to fit whichever outcome
arrived. And the read's prose conclusion could stop matching its own interval.
Each of those is checked below against the shipped code or the committed JSON,
never against a number retyped into a test.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pocket_bench.methods.table_field import TableField  # noqa: E402

TRAIN_OP = ROOT / "results/architecture_sweep/P2RANK_TRAIN_OPERATING_POINT.json"
PREREG = ROOT / "results/architecture_sweep/PREREGISTERED_MATCHED_OPERATING_POINT.json"
READ = ROOT / "results/official_fold/MATCHED_OPERATING_POINT_READ.json"
TABFIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"

RULE_IDS = ["A_common_q", "B_each_tuned_on_train", "C_p2rank_oracle_q"]


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


class TheMatchedRuleIsOurShippedRule(unittest.TestCase):
    """A matched threshold must be the threshold we actually apply."""

    def test_top_q_call_is_table_fields_positive_call(self):
        import matched_operating_point_read as read

        field = TableField.load(TABFIELD)
        rng = np.random.default_rng(20260729)
        for n in (1, 2, 7, 53, 307, 900):
            for _ in range(4):
                s = rng.normal(size=n)
                # Ties are the interesting case: a rule that broke them
                # differently for the two methods would not be matched.
                s[: n // 3] = 0.5
                mine = read.top_q_call(s, field.q)
                theirs = field.positive_call(s)
                self.assertTrue(
                    np.array_equal(mine, theirs),
                    f"the matched rule differs from TableField.positive_call "
                    f"at n={n}")

    def test_the_read_uses_the_shipped_q_for_our_side(self):
        d = _load(READ)
        if d is None:
            self.skipTest("read not present")
        q = float(json.loads(TABFIELD.read_text())["operating_point"]["q"])
        for rule in ("A_common_q", "B_each_tuned_on_train"):
            self.assertAlmostEqual(d["matched"][rule]["q_ours"], q, places=9)

    def test_our_f1_is_identical_under_both_rules(self):
        """Only P2Rank's binarisation moves between rule A and rule B."""
        d = _load(READ)
        if d is None:
            self.skipTest("read not present")
        self.assertEqual(d["matched"]["A_common_q"]["table_field_f1"],
                         d["matched"]["B_each_tuned_on_train"]["table_field_f1"])

    def test_recomputing_our_own_call_did_not_move_our_own_arm(self):
        """The matched delta must be about the threshold, not about us."""
        d = _load(READ)
        if d is None:
            self.skipTest("read not present")
        rec = d["our_rule_recovers_our_own_call"]
        self.assertLessEqual(rec["f1_drift_from_recomputing_it"], 1e-6)
        self.assertAlmostEqual(d["matched"]["A_common_q"]["table_field_f1"],
                              d["native_call_reference"]["table_field_f1"],
                              places=6,
                              msg="our arm under the matched rule should be "
                                  "our published arm, because our published "
                                  "call already is the matched rule")


class TheTrainingSelectionIsWhatItClaims(unittest.TestCase):
    def setUp(self):
        self.d = _load(TRAIN_OP)
        if self.d is None:
            self.skipTest(f"{TRAIN_OP.name} not present")

    def test_the_selected_q_is_the_argmax_of_the_committed_curve(self):
        best = max(self.d["f1_against_q"], key=lambda r: r["pooled_train_f1"])
        self.assertAlmostEqual(best["q"], self.d["p2rank_selected_q"], places=9)

    def test_the_curve_covers_the_grid_table_field_searched(self):
        qs = [r["q"] for r in self.d["f1_against_q"]]
        self.assertEqual(qs[0], 0.02)
        self.assertEqual(qs[-1], 0.40)
        self.assertEqual(len(qs), 39)

    def test_it_did_not_touch_the_test_fold(self):
        self.assertFalse(self.d["test_fold_touched"])

    def test_the_rerun_agreed_with_the_committed_training_summary(self):
        a = self.d["audit_against_committed_summary"]
        self.assertTrue(a["checked"])
        self.assertTrue(a["agrees"], f"worst disagreement "
                                     f"{a['largest_disagreement']}")

    def test_the_value_of_tuning_is_the_difference_it_claims(self):
        self.assertAlmostEqual(
            self.d["tuning_is_worth_to_p2rank"],
            self.d["p2rank_pooled_train_f1_at_selected_q"]
            - self.d["p2rank_pooled_train_f1_at_native_call"],
            places=6)


class ThePlanWasFixedFirst(unittest.TestCase):
    def setUp(self):
        self.d = _load(PREREG)
        if self.d is None:
            self.skipTest(f"{PREREG.name} not present")

    def test_the_three_rules_are_the_ones_that_were_planned(self):
        self.assertEqual([r["id"] for r in self.d["rules_to_be_read"]],
                         RULE_IDS)

    def test_a_sentence_is_committed_for_every_outcome(self):
        said = self.d["what_will_be_written_under_each_outcome"]
        for k in ("if_it_survives", "if_it_does_not_survive",
                  "if_the_two_matched_rules_disagree"):
            self.assertTrue(said.get(k), f"no sentence for {k}")
        self.assertIn("depends on the operating-point convention",
                      said["if_it_does_not_survive"],
                      "the losing branch must commit to weakening the claim")

    def test_the_plan_does_not_read_the_fold(self):
        self.assertFalse(self.d["reads_test_fold"])

    def test_the_forecast_is_the_subtraction_it_describes(self):
        f = self.d["forecast"]
        self.assertAlmostEqual(
            f["expected_matched_delta_if_the_gain_transfers"],
            f["published_delta"] - f["p2rank_gains_from_tuning_on_train"],
            places=6)

    def test_the_plan_commit_exists_in_this_history(self):
        r = subprocess.run(["git", "cat-file", "-e", self.d["commit"]],
                           cwd=ROOT, capture_output=True)
        if subprocess.run(["git", "rev-parse", "--is-shallow-repository"],
                          cwd=ROOT, capture_output=True,
                          text=True).stdout.strip() == "true":
            self.skipTest("shallow clone")
        self.assertEqual(r.returncode, 0,
                         "the preregistration names a commit this repository "
                         "does not contain")


class TheReadIsGovernedByThePlan(unittest.TestCase):
    def setUp(self):
        self.d = _load(READ)
        if self.d is None:
            self.skipTest(f"{READ.name} not present")
        self.plan = _load(PREREG)

    def test_it_is_the_sixth_indexed_read_and_rescored_nothing(self):
        self.assertEqual(self.d["test_fold_read_index"], 6)
        self.assertFalse(self.d["rescored_anything"])
        self.assertTrue(self.d["why_it_is_indexed_anyway"])

    def test_it_reproduced_the_published_numbers_before_reporting_new_ones(self):
        for m, r in self.d["reproduces_the_published_numbers"].items():
            self.assertTrue(r["agrees"], f"{m} did not reproduce")

    def test_the_plan_precedes_the_read_in_git(self):
        o = self.d["ordering"]
        self.assertTrue(o["preregistration_is_an_ancestor_of_the_read"])
        if subprocess.run(["git", "rev-parse", "--is-shallow-repository"],
                          cwd=ROOT, capture_output=True,
                          text=True).stdout.strip() == "true":
            self.skipTest("shallow clone")
        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor",
             o["preregistration_commit"], "HEAD"], cwd=ROOT,
            capture_output=True)
        self.assertEqual(r.returncode, 0,
                         "the recorded ordering does not hold in this history")

    def test_the_conclusion_is_the_preregistered_sentence(self):
        survived = self.d["survives_under"][self.d["governing_rule"]]
        key = "if_it_survives" if survived else "if_it_does_not_survive"
        self.assertEqual(
            self.d["conclusion"],
            self.plan["what_will_be_written_under_each_outcome"][key],
            "the conclusion was not the sentence written for this outcome")

    def test_every_interval_is_labelled_consistently(self):
        for k, v in self.d["matched"].items():
            for which in ("primary", "secondary_trimmed_mean"):
                p = v[which]
                self.assertEqual(
                    p["crosses_zero"],
                    p["delta_ci_low"] <= 0.0 <= p["delta_ci_high"],
                    f"{k}/{which} mislabels its interval")

    def test_the_oracle_is_the_argmax_of_the_published_curve(self):
        curve = self.d["f1_against_q"]
        best = max(curve, key=lambda r: r["p2rank_f1"])
        self.assertAlmostEqual(
            best["q"], self.d["oracle"]["p2rank_best_q_on_the_held_out_fold"],
            places=9)
        self.assertAlmostEqual(
            best["p2rank_f1"], self.d["oracle"]["p2rank_f1_at_its_oracle_q"],
            places=9)

    def test_the_oracle_is_labelled_as_an_oracle(self):
        self.assertTrue(self.d["oracle"]["is_an_oracle"]
                        if "is_an_oracle" in self.d["oracle"] else True)
        self.assertIn("upper bound", self.d["oracle"]["what_this_is"])

    def test_the_governing_rule_is_the_one_the_plan_named(self):
        decisive = [r["id"] for r in self.plan["rules_to_be_read"]
                    if r.get("this_is_the_decisive_one")]
        self.assertEqual([self.d["governing_rule"]], decisive)


if __name__ == "__main__":
    unittest.main()
