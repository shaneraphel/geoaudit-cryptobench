#!/usr/bin/env python3
"""A ladder of readouts is only informative if every rung is the same climb.

The comparison exists to answer whether the counting field's accuracy is in its
architecture or in the wires it reads, and the ways it could quietly fail all
look like success. An arm scored on a different half, a published row that is
not in fact the published configuration, a summary difference typed in rather
than subtracted, an interval attached to the wrong arm: each leaves a table that
reads well and means nothing. These tests check the rungs against each other and
against the sweep that measured the same configuration first.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "results/architecture_sweep/INTERPRETABLE_BASELINES.json"
SENS = ROOT / "results/architecture_sweep/SENSITIVITY_SWEEP.json"

HAVE = ART.is_file()
REASON = f"{ART.relative_to(ROOT)} not built"
PUBLISHED = "pairs, sixteen rounds"


def art() -> dict:
    return json.loads(ART.read_text())


def by(name: str) -> dict:
    return next(r for r in art()["rows"] if r["arm"] == name)


@unittest.skipUnless(HAVE, REASON)
class EveryArmIsTheSameClimb(unittest.TestCase):
    def test_the_ladder_has_every_rung(self):
        want = {"ridge direction", "logistic regression", "additive over bins",
                "pairs, one round", PUBLISHED,
                "pairs, sixteen rounds, unrounded"}
        self.assertEqual(want, {r["arm"] for r in art()["rows"]})

    def test_the_published_arm_is_the_published_configuration(self):
        r = by(PUBLISHED)
        f = art()["frozen_configuration"]
        self.assertEqual(r["width"], f["width"])
        self.assertEqual(r["rounds"], f["rounds"])
        self.assertTrue(r["quantised"])
        self.assertTrue(r["integer_fan_out"])

    def test_the_published_arm_scores_what_the_sweep_says_it_scores(self):
        """Two tools, one configuration, one half. If they disagree, one of them
        is not running the shipped method and the ladder has no fixed point."""
        if not SENS.is_file():
            self.skipTest("sensitivity sweep not built")
        sweep = json.loads(SENS.read_text())["published_pick_half_roc_auc"]
        self.assertAlmostEqual(by(PUBLISHED)["pick_half_roc_auc"], sweep,
                               places=9)

    def test_the_split_is_the_one_the_field_was_selected_under(self):
        sweep_seed = (json.loads(SENS.read_text())["split"]["seed"]
                      if SENS.is_file() else None)
        if sweep_seed is None:
            self.skipTest("sensitivity sweep not built")
        self.assertEqual(art()["split"]["seed"], sweep_seed)

    def test_no_arm_reads_the_held_out_fold(self):
        self.assertFalse(art()["reads_test_fold"])


@unittest.skipUnless(HAVE, REASON)
class TheNumbersAreTheNumbersBeneathThem(unittest.TestCase):
    def test_every_mean_is_the_mean_of_its_own_per_chain_vector(self):
        for r in art()["rows"]:
            v = [x for x in r["per_unit_gated_auc"] if x is not None]
            with self.subTest(arm=r["arm"]):
                self.assertTrue(v)
                self.assertAlmostEqual(sum(v) / len(v), r["pick_half_roc_auc"],
                                       places=5)

    def test_every_arm_scored_the_same_chains(self):
        """A difference between two arms is paired only if the pairing exists."""
        lens = {len(r["per_unit_gated_auc"]) for r in art()["rows"]}
        self.assertEqual(1, len(lens))
        missing = {tuple(i for i, x in enumerate(r["per_unit_gated_auc"])
                         if x is None) for r in art()["rows"]}
        self.assertEqual(1, len(missing),
                         "the arms disagree about which chains are scorable")

    def test_the_smoothing_gain_is_gated_minus_raw(self):
        for r in art()["rows"]:
            with self.subTest(arm=r["arm"]):
                self.assertAlmostEqual(
                    r["spatial_smoothing_gain"],
                    r["pick_half_roc_auc"] - r["pick_half_roc_auc_raw"],
                    places=5)

    def test_each_step_is_worth_the_subtraction_it_claims(self):
        w = art()["what_each_step_is_worth"]
        pub = by(PUBLISHED)["pick_half_roc_auc"]
        for key, got in (
                ("repeating_the_pairing_sixteen_times",
                 pub - by("pairs, one round")["pick_half_roc_auc"]),
                ("rounding_the_fusion_to_integers",
                 pub - by("pairs, sixteen rounds, unrounded")[
                     "pick_half_roc_auc"]),
                ("pairing_the_wires_over_an_additive_model",
                 by("pairs, one round")["pick_half_roc_auc"]
                 - by("additive over bins")["pick_half_roc_auc"])):
            with self.subTest(step=key):
                self.assertAlmostEqual(w[key], got, places=5)

    def test_the_linear_comparison_uses_the_stronger_of_the_two_linear_arms(self):
        """Quoting the weaker linear readout would flatter the architecture, and
        both are in the artifact precisely so that it cannot."""
        w = art()["what_each_step_is_worth"]
        lin = max(by("ridge direction")["pick_half_roc_auc"],
                  by("logistic regression")["pick_half_roc_auc"])
        self.assertAlmostEqual(
            w["the_bins_and_the_tables_over_a_linear_readout"],
            by("additive over bins")["pick_half_roc_auc"] - lin, places=5)


@unittest.skipUnless(HAVE, REASON)
class TheIntervalsSayWhatCanBeSeparated(unittest.TestCase):
    def test_every_other_arm_has_a_paired_interval(self):
        d = art()
        self.assertEqual(
            set(d["published_readout_against_each_other_arm"]),
            {r["arm"] for r in d["rows"]} - {PUBLISHED})

    def test_each_interval_brackets_its_own_point_estimate(self):
        for arm, ci in art()[
                "published_readout_against_each_other_arm"].items():
            with self.subTest(arm=arm):
                self.assertLessEqual(ci["ci95"][0], ci["delta"])
                self.assertLessEqual(ci["delta"], ci["ci95"][1])

    def test_each_delta_is_the_difference_of_the_two_reported_means(self):
        """The interval is over paired chains and the means are over the same
        chains, so the point estimate has to be their difference."""
        pub = by(PUBLISHED)["pick_half_roc_auc"]
        for arm, ci in art()[
                "published_readout_against_each_other_arm"].items():
            with self.subTest(arm=arm):
                self.assertAlmostEqual(ci["delta"],
                                       pub - by(arm)["pick_half_roc_auc"],
                                       places=5)

    def test_the_unresolved_list_follows_from_the_intervals(self):
        d = art()
        self.assertEqual(
            sorted(k for k, v in
                   d["published_readout_against_each_other_arm"].items()
                   if not v["excludes_zero"]),
            d["arms_it_cannot_be_separated_from"])

    def test_the_resampling_unit_is_a_chain_and_not_a_residue(self):
        """A residue-level resample would treat neighbouring residues on one
        protein as independent draws and shrink every interval."""
        r = art()["resampling"]
        self.assertIn("chain", r["unit"])
        self.assertGreaterEqual(r["draws"], 1000)


if __name__ == "__main__":
    unittest.main()
