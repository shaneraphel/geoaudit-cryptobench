"""The endpoint demotion has to keep following from the commit graph.

The manuscript leads with an unresolved mean and reports the preregistered
trimmed mean as exploratory. That ordering is not a stylistic choice: it is what
the commit graph licenses, given how many times the fold had been read before the
statistic was fixed. These tests check the artifact still says so, that it says
so consistently with the two artifacts it summarises, and that it cannot silently
turn back into a paper that leads with the trimmed mean.
"""
from __future__ import annotations

import json
import unittest

from pocket_bench.paths import ROOT

import endpoint_status as ES

ART = ROOT / "results/official_fold/ENDPOINT_STATUS.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
READ = ROOT / "results/official_fold/PREREGISTERED_READ.json"


def _load(p):
    return json.loads(p.read_text())


class TheArtifactExists(unittest.TestCase):
    def test_the_gate_passes(self):
        self.assertEqual(ES.check(), 0)


class TheDemotionFollowsFromTheGraph(unittest.TestCase):
    def test_some_read_precedes_the_preregistration(self):
        # If nothing preceded it, the preregistration would be worth taking at
        # face value and this whole artifact would have no subject.
        d = _load(ART)
        self.assertGreater(d["n_reads_before_the_preregistration"], 0)

    def test_the_recorded_count_matches_a_live_recount(self):
        d = _load(ART)
        live = ES.build()
        self.assertEqual(d["n_reads_before_the_preregistration"],
                         live["n_reads_before_the_preregistration"])
        self.assertEqual(d["preregistration_commit"],
                         live["preregistration_commit"])

    def test_reads_partition_the_ledger(self):
        # Every indexed read is on exactly one side of the preregistration; a
        # read that fell out of both lists would be a read whose ordering was
        # never actually decided.
        d = _load(ART)
        idx = [r["read_index"] for r in d["reads_before_the_preregistration"]]
        idx += [r["read_index"] for r in d["reads_after_the_preregistration"]]
        want = [e["read_index"]
                for e in _load(LEDGER)["indexed_read_sequence"]]
        self.assertEqual(sorted(idx), sorted(want))
        self.assertEqual(len(idx), len(set(idx)))

    def test_the_preceding_reads_are_named_with_their_commits(self):
        for r in _load(ART)["reads_before_the_preregistration"]:
            self.assertIsNotNone(r["first_committed_in"])
            self.assertTrue(r["artifact"])

    def test_the_preregistration_commit_is_the_one_the_read_records(self):
        self.assertEqual(
            _load(ART)["preregistration_commit"],
            _load(READ)["provenance_of_the_choice"]["committed_in"])


class TheMeanIsPrimary(unittest.TestCase):
    def test_the_primary_endpoint_is_the_mean_and_is_unresolved(self):
        p = _load(ART)["primary_endpoint"]
        self.assertEqual(p["statistic"], "mean")
        self.assertFalse(p["resolves"])
        self.assertLessEqual(p["ci95"][0], 0.0)
        self.assertGreaterEqual(p["ci95"][1], 0.0)

    def test_the_primary_endpoint_matches_the_preregistered_read(self):
        p = _load(ART)["primary_endpoint"]
        m = _load(READ)["mean_reported_beside_it"]
        self.assertEqual(p["delta"], m["point"])
        self.assertEqual(p["ci95"], [m["ci_low"], m["ci_high"]])

    def test_the_trimmed_mean_is_exploratory(self):
        d = _load(ART)
        pre = _load(READ)["preregistered_result"]["statistic"]
        names = {e["statistic"] for e in d["exploratory_endpoints"]}
        self.assertIn(pre, names)
        self.assertNotEqual(d["primary_endpoint"]["statistic"], pre)

    def test_every_exploratory_endpoint_says_it_is_exploratory(self):
        for e in _load(ART)["exploratory_endpoints"]:
            self.assertEqual(e["status"], "exploratory")

    def test_no_statistic_is_both_primary_and_exploratory(self):
        d = _load(ART)
        self.assertNotIn(d["primary_endpoint"]["statistic"],
                         {e["statistic"] for e in d["exploratory_endpoints"]})

    def test_a_resolving_exploratory_endpoint_exists(self):
        # The demotion is only interesting because something did clear zero; if
        # nothing had, there would be no temptation to lead with it.
        self.assertTrue(any(e["resolves"]
                            for e in _load(ART)["exploratory_endpoints"]))


class TheDistributionIsReported(unittest.TestCase):
    def test_wins_and_losses_account_for_every_unit(self):
        c = _load(ART)["per_chain_outcome"]
        self.assertEqual(c["n_field_ahead"] + c["n_baseline_ahead"],
                         c["n_units"])

    def test_the_shape_matches_the_preregistered_read(self):
        c = _load(ART)["per_chain_outcome"]
        s = _load(READ)["shape_of_the_differences"]
        self.assertEqual(c["n_units"], s["n"])
        self.assertEqual(c["n_field_ahead"], s["n_field_ahead"])
        self.assertEqual(c["median_difference"], s["quantiles"]["0.5"])

    def test_it_wins_more_often_and_loses_harder(self):
        # This is the asymmetry the manuscript uses to explain why a trimmed
        # mean and a mean disagree, so it should be an assertion and not a
        # remark.
        c = _load(ART)["per_chain_outcome"]
        self.assertGreater(c["n_field_ahead"], c["n_baseline_ahead"])
        self.assertGreater(abs(c["worst_losses_mean"]), abs(c["best_wins_mean"]))


class ItIsNotItselfARead(unittest.TestCase):
    def test_it_claims_no_read_index(self):
        d = _load(ART)
        self.assertIsNone(d["test_fold_read_index"])
        self.assertTrue(d["why_this_is_not_an_indexed_read"])

    def test_every_number_it_reports_appears_in_an_earlier_artifact(self):
        # The artifact must be a restatement, not a computation on the fold.
        # Any endpoint whose point estimate is absent from the committed read
        # would mean something was scored here.
        read = _load(READ)
        known = {c["statistic"]: c for c in read["candidates"]}
        known[read["mean_reported_beside_it"]["statistic"]] = \
            read["mean_reported_beside_it"]
        d = _load(ART)
        for e in [d["primary_endpoint"], *d["exploratory_endpoints"]]:
            self.assertIn(e["statistic"], known)
            self.assertEqual(e["delta"], known[e["statistic"]]["point"])


class TheGateRejectsAPromotedTrimmedMean(unittest.TestCase):
    def _with(self, mutate):
        d = _load(ART)
        mutate(d)
        original = ART.read_text()
        try:
            ART.write_text(json.dumps(d, indent=2) + "\n")
            return ES.check()
        finally:
            ART.write_text(original)

    def test_a_resolving_primary_endpoint_fails(self):
        def m(d):
            d["primary_endpoint"]["resolves"] = True
        self.assertEqual(self._with(m), 1)

    def test_renaming_the_primary_endpoint_fails(self):
        def m(d):
            d["primary_endpoint"]["statistic"] = "trimmed20"
        self.assertEqual(self._with(m), 1)

    def test_an_exploratory_endpoint_marked_primary_fails(self):
        def m(d):
            d["exploratory_endpoints"][0]["status"] = "primary"
        self.assertEqual(self._with(m), 1)

    def test_a_stale_read_count_fails(self):
        def m(d):
            d["n_reads_before_the_preregistration"] += 1
        self.assertEqual(self._with(m), 1)

    def test_claiming_a_read_index_fails(self):
        def m(d):
            d["test_fold_read_index"] = 99
        self.assertEqual(self._with(m), 1)

    def test_the_gate_restored_the_file(self):
        self.assertEqual(ES.check(), 0)


if __name__ == "__main__":
    unittest.main()
