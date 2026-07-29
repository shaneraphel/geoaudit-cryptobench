"""The pocket stage was built for this read, so the read has to be checkable.

Everything else in this repository scores a prediction that already existed. This
read is different: the counting field emits no candidate sites, so the read
constructs them, and a construction invented to be measured is exactly the kind of
thing that can be tuned until it measures well. The plan fixed the construction
before the fold was opened; these tests check that the artifact still describes
that construction and that its favourable headline is bounded by its own numbers.

The specific hazards:

Hit rates at larger K are not independent when the median chain yields three
candidates, because a top-5 rate is then mostly a top-3 rate. The read has to say
so rather than let the reader read three numbers as three findings.

A hit rate can be raised by offering more candidates. The read compares at matched
K and reports the candidate counts, so the tests check that the field's advantage
is not bought with a larger candidate budget -- here it is not, because the field
offers fewer.

Per-unit detail has to reconstruct the summary, or the summary is unfalsifiable.
"""
from __future__ import annotations

import json
import unittest

from pocket_bench.paths import ROOT

import pocket_read as PR

READ = ROOT / "results/official_fold/POCKET_READ.json"
PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_POCKETS.json"


def _json(p):
    return json.loads(p.read_text())


class TheGatePasses(unittest.TestCase):
    def test_check(self):
        self.assertEqual(PR.check(), 0)


class TheReadIsIndexedAndExploratory(unittest.TestCase):
    def test_it_claims_read_nine(self):
        self.assertEqual(_json(READ)["test_fold_read_index"], 9)

    def test_it_is_exploratory(self):
        self.assertIn("exploratory", _json(READ)["status"].lower())

    def test_it_says_it_rescored_nothing_yet_is_indexed_anyway(self):
        # Building a pocket stage from frozen residue scores adds no new scoring,
        # but scoring that stage against the labels is a new inferential use of
        # the fold, which is the ledger's own definition of a read.
        d = _json(READ)
        self.assertFalse(d["rescored_anything"])
        self.assertTrue(d["why_it_is_indexed_anyway"].strip())

    def test_the_plan_precedes_it_in_the_history(self):
        self.assertTrue(
            _json(READ)["provenance_of_the_plan"]["is_ancestor_of_head"])


class TheConstructionIsTheOneThePlanFixed(unittest.TestCase):
    def test_the_clustering_cutoff_is_an_existing_constant(self):
        # The cutoff is the pinch radius the descriptors already use, not a
        # number chosen for this read. A cutoff tuned here would be the whole
        # finding.
        self.assertEqual(_json(READ)["primary"]["clustering_cutoff_angstrom"],
                         7.0)

    def test_the_read_declares_that_it_built_the_stage(self):
        d = _json(READ)["what_was_built_for_this_read"]
        self.assertIn("placeholder", d)

    def test_a_looser_cutoff_is_reported_as_a_sensitivity(self):
        # One cutoff with a favourable result and no second cutoff would leave
        # the reader unable to tell whether the choice carried the finding.
        self.assertIn("sensitivity_at_the_looser_cutoff", _json(READ))


class TheAdvantageIsNotBoughtWithCandidates(unittest.TestCase):
    def test_the_field_does_not_offer_more_candidates_than_p2rank(self):
        c = _json(READ)["primary"]["candidates_per_chain"]
        self.assertLessEqual(c["ours_mean"], c["p2rank_mean"])

    def test_offering_fewer_candidates_is_reported_against_the_field(self):
        keys = {o["key"] for o
                in _json(READ)["additional_preregistered_outcomes_that_apply"]}
        self.assertIn("the_field_offers_too_few_candidates", keys)

    def test_the_larger_k_are_declared_not_independent(self):
        extra = next(o for o
                     in _json(READ)["additional_preregistered_outcomes_that_apply"]
                     if o["key"] == "the_field_offers_too_few_candidates")
        self.assertIn("not independent", extra["why_it_applies"])

    def test_p2rank_is_also_scored_at_its_own_residue_centroid(self):
        # P2Rank's centre is a cavity centre and ours is a residue centroid, so a
        # distance comparison between them partly measures that difference. The
        # corrected arm removes it, and the headline has to survive there too.
        arm = _json(READ)[
            "fairness_correction_p2rank_scored_at_its_own_residue_centroid"]
        self.assertTrue(arm)


class ThePerUnitDetailReconstructsTheSummary(unittest.TestCase):
    def test_the_top1_hit_rate_is_the_mean_of_the_per_unit_hits(self):
        d = _json(READ)["primary"]
        rows = [r for r in d["per_unit"]
                if r["ours"]["n_candidates"] and r["theirs"]["n_candidates"]]
        for side, key in (("ours", "ours"), ("theirs", "p2rank")):
            hits = [r[side]["hit"]["4.0"]["1"] for r in rows]
            self.assertAlmostEqual(sum(hits) / len(hits),
                                   d["hit_rates"]["4.0A/top1"][key], places=5)

    def test_every_paired_test_uses_the_same_paired_count(self):
        d = _json(READ)["primary"]
        n = d["n_units_both_offer_a_candidate"]
        for name, block in d["hit_rates"].items():
            self.assertEqual(block["paired_95"]["n_paired"], n, name)

    def test_a_hit_at_a_smaller_radius_is_a_hit_at_a_larger_one(self):
        for r in _json(READ)["primary"]["per_unit"]:
            for side in ("ours", "theirs"):
                h = r[side]["hit"]
                if not r[side]["n_candidates"]:
                    continue
                for k in ("1", "3", "5"):
                    self.assertLessEqual(h["4.0"][k], h["6.0"][k])
                    self.assertLessEqual(h["6.0"][k], h["8.0"][k])

    def test_a_hit_at_top1_is_a_hit_at_top3(self):
        for r in _json(READ)["primary"]["per_unit"]:
            for side in ("ours", "theirs"):
                if not r[side]["n_candidates"]:
                    continue
                for radius in ("4.0", "6.0", "8.0"):
                    h = r[side]["hit"][radius]
                    self.assertLessEqual(h["1"], h["3"])
                    self.assertLessEqual(h["3"], h["5"])


class TheCorrectedIntervalIsWider(unittest.TestCase):
    def test_bonferroni_never_narrows_a_corrected_test(self):
        for name, block in _json(READ)["primary"]["hit_rates"].items():
            if not block.get("is_a_corrected_test"):
                continue
            a, b = block["paired_95"]["ci"], block["paired_bonferroni"]["ci"]
            self.assertLessEqual(b[0], a[0] + 1e-9, name)
            self.assertGreaterEqual(b[1], a[1] - 1e-9, name)

    def test_the_reported_outcome_is_one_the_plan_named(self):
        d = _json(READ)
        self.assertIn(d["outcome_key"],
                      set(_json(PLAN)["what_will_be_written_under_each_outcome"]))


if __name__ == "__main__":
    unittest.main()
