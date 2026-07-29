#!/usr/bin/env python3
"""A cost measurement that flatters the measurer is the default outcome.

This artifact exists because the speed factor it replaces was not a measurement:
it compared a warm Python process against a cold JVM and reported the difference
as an architectural property. The replacement can fail in the same direction in
quieter ways -- a verdict sentence left over from a run that said something else,
a thread budget requested and not granted, a steady state read one way in the
prose and another way in the numbers. These tests check the artifact against
itself, so that a rerun which reverses the finding cannot leave the old sentence
standing next to the new ratio.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "results/architecture_sweep/RUNTIME_COST.json"

HAVE = ART.is_file()
REASON = f"{ART.relative_to(ROOT)} not built"


def art() -> dict:
    return json.loads(ART.read_text())


@unittest.skipUnless(HAVE, REASON)
class TheVerdictFollowsFromTheNumbers(unittest.TestCase):
    def test_the_wall_clock_verdict_names_the_side_the_ratio_names(self):
        w = art()["warm"]
        ours_is_cheaper = w["ratio_p2rank_over_table_field"] > 1.0
        said_ours = "the counting field is the cheaper" in w["verdict"]
        self.assertEqual(ours_is_cheaper, said_ours, w["verdict"])

    def test_the_cpu_verdict_names_the_side_the_cpu_ratio_names(self):
        w = art()["warm"]
        if "ratio_p2rank_over_table_field_cpu" not in w:
            self.skipTest("artifact predates the CPU-second reading")
        ours_is_cheaper = w["ratio_p2rank_over_table_field_cpu"] > 1.0
        said_ours = ("the counting field is the cheaper"
                     in w["verdict_on_cpu_seconds"])
        self.assertEqual(ours_is_cheaper, said_ours, w["verdict_on_cpu_seconds"])

    def test_the_two_readings_of_the_steady_state_agree(self):
        """Wall clock and CPU seconds can disagree, and then the chapter has to
        say which one it is reporting. The manuscript currently says they agree,
        so this fails rather than letting the prose drift from the artifact."""
        w = art()["warm"]
        if "ratio_p2rank_over_table_field_cpu" not in w:
            self.skipTest("artifact predates the CPU-second reading")
        self.assertEqual(w["ratio_p2rank_over_table_field"] > 1.0,
                         w["ratio_p2rank_over_table_field_cpu"] > 1.0)

    def test_the_warm_ratio_is_the_two_warm_numbers_divided(self):
        w = art()["warm"]
        self.assertAlmostEqual(
            w["ratio_p2rank_over_table_field"],
            w["p2rank"]["amortised_per_chain_s"] / w["table_field"]["median_s"],
            places=2)

    def test_the_cold_ratio_is_the_two_cold_medians_divided(self):
        c = art()["cold"]
        self.assertAlmostEqual(
            c["ratio_of_medians"],
            c["p2rank"]["median_s"] / c["table_field"]["median_s"], places=2)


@unittest.skipUnless(HAVE, REASON)
class TheThreadBudgetIsMeasuredNotAssumed(unittest.TestCase):
    def test_an_uneven_grant_does_not_favour_the_declared_winner(self):
        """One thread was asked for and, on this platform, not always given.
        That is tolerable only while the side that got extra cores is the side
        the wall-clock comparison went against."""
        d = art()
        par = d["did_either_side_get_more_than_one_thread"]
        if par["both_within_tolerance"]:
            return
        wall_favours = ("table_field"
                        if d["warm"]["ratio_p2rank_over_table_field"] > 1.0
                        else "p2rank")
        self.assertNotEqual(
            par["who_got_more"], wall_favours,
            "the faster side is also the side that was granted more cores, so "
            "the comparison is partly a measurement of core count")

    def test_both_sides_report_a_parallelism_at_least_one(self):
        """A process cannot spend less CPU than wall time while it is running,
        so a figure below one means the accounting is wrong, not that the tool
        was frugal."""
        par = art()["did_either_side_get_more_than_one_thread"]
        for side in ("table_field", "p2rank_batch"):
            with self.subTest(side=side):
                self.assertGreaterEqual(par[side], 0.9)


@unittest.skipUnless(HAVE, REASON)
class TheMeasurementIsWhatItSaysItIs(unittest.TestCase):
    def test_it_reads_no_test_fold(self):
        d = art()
        self.assertIsNone(d["test_fold_read_index"])
        self.assertIn("train", d["population"]["which"])

    def test_our_cost_splits_into_features_and_scoring(self):
        """The two parts have to account for the whole, near enough.

        Not exactly: each figure is a median over chains, and the chain with the
        median total need not be the chain with the median feature time, so the
        parts are free to miss the whole by a little. A gap larger than a couple
        of per cent would mean something else is being timed.
        """
        tf = art()["warm"]["table_field"]
        parts = tf["features_median_s"] + tf["scoring_median_s"]
        self.assertLess(abs(parts - tf["median_s"]) / tf["median_s"], 0.02)

    def test_the_feature_fraction_is_that_split_and_not_a_separate_claim(self):
        tf = art()["warm"]["table_field"]
        self.assertAlmostEqual(
            tf["fraction_of_the_median_spent_on_features"],
            tf["features_median_s"] / tf["median_s"], places=3)

    def test_the_quartiles_bracket_the_median(self):
        for regime in ("warm", "cold"):
            tf = art()[regime]["table_field"]
            with self.subTest(regime=regime):
                self.assertLessEqual(tf["q1_s"], tf["median_s"])
                self.assertLessEqual(tf["median_s"], tf["q3_s"])
                self.assertAlmostEqual(tf["iqr_s"], tf["q3_s"] - tf["q1_s"],
                                       places=3)

    def test_both_regimes_cover_the_chains_they_claim(self):
        d = art()
        self.assertEqual(d["warm"]["table_field"]["n"], d["population"]["chains"])
        self.assertEqual(d["warm"]["p2rank"]["n"], d["population"]["chains"])
        self.assertEqual(d["cold"]["table_field"]["n"], d["cold"]["n_chains"])
        self.assertEqual(d["cold"]["p2rank"]["n"], d["cold"]["n_chains"])

    def test_the_jvm_fraction_is_the_jvm_over_the_cold_median(self):
        c = art()["cold"]
        self.assertAlmostEqual(
            c["jvm_start_as_fraction_of_p2ranks_cold_median"],
            c["jvm_start_median_s"] / c["p2rank"]["median_s"], places=3)

    def test_the_compiled_detector_is_smaller_than_the_baselines_model(self):
        """The one cost claim that survived, so it is the one worth pinning."""
        m = art()["model_size"]
        self.assertLess(m["table_field_json_bytes"],
                        m["p2rank_default_model_bytes"])


if __name__ == "__main__":
    unittest.main()
