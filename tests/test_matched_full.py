"""The seventh read reports what it computed, under the plan it was given.

The read adds three things at once -- three metrics nobody had computed, a
convention whose thresholds differ between the two methods, and two resamplings
coarser than a chain -- and each of them can be wrong in a way the numbers do
not show. A metric can be silently imputed where it is undefined, which turns
an abstention into a score of zero. A convention can drift from the thresholds
the training fold actually selected. A coarser resampling can be reported
without its groups ever being coarser than one unit each, which would make the
robustness check vacuous while it appeared to pass. And a verdict can stop
following from its own interval.

Every assertion below reads the shipped code or the committed JSON. None of them
retypes a number, because a test that hard-codes +0.0140 passes for as long as
somebody remembers to update it and no longer.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pocket_bench.methods.table_field import TableField  # noqa: E402

TRAIN_OP = ROOT / "results/architecture_sweep/TRAIN_OPERATING_POINTS.json"
PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_MATCHED_FULL.json"
READ = ROOT / "results/official_fold/MATCHED_FULL_READ.json"
READ6 = ROOT / "results/official_fold/MATCHED_OPERATING_POINT_READ.json"
BOOT = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
TABFIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"

METRICS = ("precision", "recall", "positive_class_f1", "mcc")
CONVENTIONS = ("D1_as_deployed", "D2_common_budget", "D3_each_tuned_for_f1",
               "D4_each_tuned_for_mcc")
UNITS = ("chain", "pdb_entry", "uniprot_cluster")


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


class MetricsAreUndefinedRatherThanZero(unittest.TestCase):
    """An abstention is not a wrong answer and must not be scored as one."""

    def test_precision_is_none_when_nothing_was_called(self):
        import matched_full_read as read

        self.assertIsNone(read.metric_of("precision", (0, 0, 10, 5)))

    def test_recall_is_none_when_there_is_nothing_to_find(self):
        import matched_full_read as read

        self.assertIsNone(read.metric_of("recall", (0, 3, 10, 0)))

    def test_mcc_is_none_on_a_degenerate_confusion_matrix(self):
        import matched_full_read as read

        # Everything called: tn + fp is non-zero but tn + fn is zero.
        self.assertIsNone(read.metric_of("mcc", (5, 3, 0, 0)))

    def test_the_metrics_match_their_textbook_definitions(self):
        import matched_full_read as read

        tp, fp, tn, fn = 7, 3, 80, 5
        self.assertAlmostEqual(read.metric_of("precision", (tp, fp, tn, fn)),
                               tp / (tp + fp))
        self.assertAlmostEqual(read.metric_of("recall", (tp, fp, tn, fn)),
                               tp / (tp + fn))
        self.assertAlmostEqual(
            read.metric_of("positive_class_f1", (tp, fp, tn, fn)),
            2 * tp / (2 * tp + fp + fn))
        self.assertAlmostEqual(
            read.metric_of("mcc", (tp, fp, tn, fn)),
            (tp * tn - fp * fn) / math.sqrt(
                float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))


class TheMatchedRuleIsOurShippedRule(unittest.TestCase):
    def test_top_q_call_is_table_fields_positive_call(self):
        import matched_full_read as read

        field = TableField.load(TABFIELD)
        rng = np.random.default_rng(11)
        for n in (17, 64, 191, 400):
            s = rng.normal(size=n)
            np.testing.assert_array_equal(read.top_q_call(s, field.q),
                                          field.positive_call(s))

    def test_ties_are_broken_towards_the_lower_residue(self):
        import matched_full_read as read

        s = np.array([1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(read.top_q_call(s, 0.25),
                                      np.array([True, False, False, False]))


class GroupResamplingIsActuallyGrouped(unittest.TestCase):
    """A cluster bootstrap that draws single units is not a cluster bootstrap."""

    def test_a_grouped_resample_draws_whole_groups(self):
        import matched_full_read as read

        # Ten groups of two, the partners always equal. Drawing whole groups
        # halves the number of independent draws, so the interval has to be
        # wider than the one that treats the twenty units as independent. That
        # widening is the whole point of a cluster bootstrap, and a grouping
        # that failed to draw partners together would not produce it.
        a = [float(i // 2 % 2) for i in range(20)]
        b = [0.0] * 20
        groups = [i // 2 for i in range(20)]
        grouped = read.paired(a, b, read._mean, groups, 10)
        self.assertEqual(grouped["n_paired_groups"], 10)
        self.assertEqual(grouped["n_paired_units"], 20)
        ungrouped = read.paired(a, b, read._mean, list(range(20)), 20)
        self.assertEqual(ungrouped["n_paired_groups"], 20)
        self.assertGreater(grouped["ci_width"], ungrouped["ci_width"])
        # Both estimate the same difference; only the uncertainty changes.
        self.assertAlmostEqual(grouped["delta_point"],
                               ungrouped["delta_point"], places=9)

    def test_pairing_drops_units_where_either_arm_is_undefined(self):
        import matched_full_read as read

        out = read.paired([1.0, None, 0.5], [0.0, 0.0, None],
                          read._mean, [0, 1, 2], 3)
        self.assertEqual(out["n_paired_units"], 1)


class TheReadArtifactSaysWhatItComputed(unittest.TestCase):
    def setUp(self):
        self.d = _load(READ)
        if self.d is None:
            self.skipTest("the seventh read has not been run")

    def test_it_is_the_seventh_indexed_read(self):
        self.assertEqual(self.d["test_fold_read_index"], 7)

    def test_it_rescored_nothing(self):
        self.assertFalse(self.d["rescored_anything"])

    def test_every_convention_carries_every_metric_on_every_resampling(self):
        for cid in CONVENTIONS:
            entry = self.d["conventions"][cid]
            for m in METRICS:
                for unit in UNITS:
                    self.assertIn(unit, entry["paired"][m],
                                  f"{cid}/{m} is missing {unit} resampling")

    def test_the_deployment_rules_reproduce_the_frozen_bootstrap(self):
        frozen = json.loads(BOOT.read_text())["metrics"]
        for met, key in (("residue_f1", "positive_class_f1"),
                         ("residue_mcc", "mcc")):
            for arm in ("table_field", "p2rank"):
                self.assertAlmostEqual(
                    self.d["conventions"]["D1_as_deployed"][arm][key],
                    frozen[met]["per_method"][arm]["point"], places=4,
                    msg=f"{met}/{arm} does not reproduce the frozen bootstrap")

    def test_the_matched_f1_delta_is_the_one_read_six_published(self):
        r6 = _load(READ6)
        if r6 is None:
            self.skipTest("read six is absent")
        self.assertAlmostEqual(
            self.d["conventions"]["D2_common_budget"]["paired"][
                "positive_class_f1"]["chain"]["delta_point"],
            r6["matched"]["A_common_q"]["primary"]["delta_point"], places=6)

    def test_a_matched_budget_calls_the_same_number_of_residues(self):
        # Rounding is per chain, so the two totals can differ by at most the
        # number of chains; anything larger means one arm was not matched.
        for cid in ("D2_common_budget", "D3_each_tuned_for_f1"):
            c = self.d["conventions"][cid]["n_residues_called"]
            self.assertLessEqual(abs(c["table_field"] - c["p2rank"]),
                                 self.d["n_units"], f"{cid} is not matched")

    def test_the_mcc_tuned_convention_gives_the_two_methods_different_budgets(self):
        c = self.d["conventions"]["D4_each_tuned_for_mcc"]
        self.assertNotEqual(c["q_ours"], c["q_p2rank"])

    def test_every_threshold_is_one_the_training_fold_selected(self):
        op = _load(TRAIN_OP)
        if op is None:
            self.skipTest("the training operating points are absent")
        want = {
            "D2_common_budget": (
                json.loads(TABFIELD.read_text())["operating_point"]["q"],
                json.loads(TABFIELD.read_text())["operating_point"]["q"]),
            "D3_each_tuned_for_f1": (
                op["selected"]["table_field/pooled_f1/full_fold"]["q"],
                op["selected"]["p2rank/pooled_f1/full_fold"]["q"]),
            "D4_each_tuned_for_mcc": (
                op["selected"]["table_field/pooled_mcc/full_fold"]["q"],
                op["selected"]["p2rank/pooled_mcc/full_fold"]["q"]),
        }
        for cid, (qa, qb) in want.items():
            c = self.d["conventions"][cid]
            self.assertEqual((c["q_ours"], c["q_p2rank"]), (qa, qb),
                             f"{cid} uses a threshold nothing selected")

    def test_each_interval_labels_its_own_crossing_correctly(self):
        for cid in CONVENTIONS:
            for m in METRICS:
                for unit, p in self.d["conventions"][cid]["paired"][m].items():
                    if p["delta_point"] is None:
                        continue
                    self.assertEqual(
                        p["crosses_zero"],
                        p["delta_ci_low"] <= 0.0 <= p["delta_ci_high"],
                        f"{cid}/{m}/{unit} mislabels its interval")

    def test_the_verdicts_follow_from_the_governing_intervals(self):
        gov = self.d["conventions"][self.d["governing_convention"]]["paired"]
        unit = self.d["governing_resampling_unit"]
        for m, v in self.d["verdicts"].items():
            p = gov[m][unit]
            want = ("undefined" if p["delta_point"] is None
                    else "excludes_zero" if (not p["crosses_zero"]
                                             and p["delta_point"] > 0)
                    else "positive_but_unresolved" if p["delta_point"] > 0
                    else "zero_or_reversed")
            self.assertEqual(v, want, f"{m} is labelled {v}, interval says {want}")

    def test_the_conclusion_is_the_preregistered_sentence(self):
        plan = _load(PLAN)
        self.assertEqual(
            self.d["conclusion"],
            plan["what_will_be_written_under_each_outcome"][self.d["outcome_key"]])

    def test_precision_and_recall_agree_in_sign_at_a_matched_budget(self):
        for cid, s in self.d["precision_recall_sign_agreement"].items():
            self.assertTrue(s["same_sign"], f"{cid} has opposing signs")

    def test_the_resampling_units_are_the_ones_the_manifest_defines(self):
        e = json.loads(MANIFEST.read_text())["entries"]
        got = self.d["resampling_units"]
        self.assertEqual(got["pdb_entry"], len({x["pdb"] for x in e}))
        self.assertEqual(got["uniprot_cluster"],
                         len({x["cluster_id"] for x in e}))
        self.assertGreaterEqual(got["chain"], got["uniprot_cluster"])


class TheMultiplicityIsDeclared(unittest.TestCase):
    """An interval nobody preregistered a rule for may not read as a test."""

    def setUp(self):
        self.d = _load(READ)
        if self.d is None:
            self.skipTest("the seventh read has not been run")

    def test_the_number_of_intervals_examined_is_the_product(self):
        mp = self.d["multiplicity"]
        self.assertEqual(mp["intervals_examined"],
                         len(CONVENTIONS) * len(METRICS) * len(UNITS))

    def test_only_the_two_governed_metrics_carry_a_decision_rule(self):
        mp = self.d["multiplicity"]["under_the_governing_convention"]
        governed = {m for m, v in mp.items() if v["a_decision_rule_governs_it"]}
        self.assertEqual(governed, {"positive_class_f1", "mcc"})

    def test_bonferroni_survival_is_computed_from_the_nominal_p(self):
        mp = self.d["multiplicity"]
        alpha = mp["bonferroni_alpha_over_the_four_metrics_of_one_convention"]
        for m, v in mp["under_the_governing_convention"].items():
            if v["p_nominal"] is None:
                continue
            self.assertEqual(
                v["survives_bonferroni_over_the_four_metrics"],
                v["p_nominal"] < alpha, f"{m} mislabels its own correction")


class ThePlanWasWrittenBeforeTheRead(unittest.TestCase):
    def setUp(self):
        self.plan = _load(PLAN)
        if self.plan is None:
            self.skipTest("the plan is absent")

    def test_the_plan_declares_it_read_no_test_fold(self):
        self.assertFalse(self.plan["reads_test_fold"])

    def test_the_plan_admits_the_matched_f1_is_already_published(self):
        known = self.plan["what_is_already_known_and_is_therefore_not_a_finding"]
        self.assertEqual(known["the_matched_f1_delta"]["read_index"], 6)

    def test_the_plan_names_a_sentence_for_every_outcome_the_read_can_reach(self):
        said = set(self.plan["what_will_be_written_under_each_outcome"])
        self.assertEqual(said, {"f1_and_mcc_both_unresolved",
                                "f1_unresolved_mcc_survives",
                                "both_survive", "either_reverses"})

    def test_the_read_records_the_plan_as_its_ancestor(self):
        d = _load(READ)
        if d is None:
            self.skipTest("the seventh read has not been run")
        self.assertTrue(d["ordering"]["plan_is_an_ancestor_of_the_read"])
        self.assertEqual(d["selection_provenance"],
                         str(PLAN.relative_to(ROOT)))


class TheThresholdsWereChosenOnTheTrainingFoldOnly(unittest.TestCase):
    def setUp(self):
        self.op = _load(TRAIN_OP)
        if self.op is None:
            self.skipTest("the training operating points are absent")

    def test_it_declares_it_touched_no_test_fold(self):
        self.assertFalse(self.op["reads_test_fold"])
        self.assertFalse(self.op["test_fold_touched"])

    def test_every_selected_q_is_the_argmax_of_its_own_curve(self):
        for key, sel in self.op["selected"].items():
            method, objective, where = key.split("/")
            rows = self.op["curves"][where][method]
            self.assertAlmostEqual(sel["value"],
                                   max(r[objective] for r in rows), places=9,
                                   msg=f"{key} is not its curve's maximum")

    def test_the_f1_threshold_is_the_one_the_shipped_field_carries(self):
        self.assertEqual(
            self.op["selected"]["table_field/pooled_f1/full_fold"]["q"],
            json.loads(TABFIELD.read_text())["operating_point"]["q"])

    def test_the_out_of_sample_selection_is_reported_for_every_threshold(self):
        for key, v in self.op["in_sample_optimism"].items():
            self.assertIn("q_in_sample", v, key)
            self.assertIn("q_out_of_sample", v, key)
            self.assertEqual(v["same"],
                             abs(v["q_in_sample"] - v["q_out_of_sample"]) < 1e-9)


if __name__ == "__main__":
    unittest.main()
