#!/usr/bin/env python3
"""The decomposition has to be the score, not a story about the score.

Most of what could go wrong here is silent. A family split that drops a table
still produces plausible-looking numbers; a decomposition of the cached float32
wires still ranks residues almost the same way; a case list quietly reselected to
whichever chains read well still validates. So these tests check the arithmetic
closes, that the cases came from the committed selection, and that the residues
were chosen by the stated rules rather than by outcome.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "results/official_fold/AUDIT_DECOMPOSITION.json"
CASES = ROOT / "results/official_fold/CASE_STUDIES.json"
FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"

HAVE = ART.is_file()
REASON = f"{ART.relative_to(ROOT)} not built"


def art() -> dict:
    return json.loads(ART.read_text())


@unittest.skipUnless(HAVE, REASON)
class TheArithmeticCloses(unittest.TestCase):
    def test_family_terms_sum_to_the_stated_deviation(self):
        """The point of an exact decomposition is that it is exact."""
        for c in art()["cases"]:
            for r in c["residues"]:
                with self.subTest(unit=c["unit_id"], res=r["resnum"]):
                    self.assertAlmostEqual(
                        sum(r["by_family"].values()),
                        r["deviation_from_chain_mean"], places=2)

    def test_the_coarse_rollup_loses_nothing(self):
        for c in art()["cases"]:
            for r in c["residues"]:
                with self.subTest(unit=c["unit_id"], res=r["resnum"]):
                    self.assertAlmostEqual(sum(r["by_coarse_family"].values()),
                                           sum(r["by_family"].values()),
                                           places=2)

    def test_the_score_reconstruction_is_exact_not_merely_close(self):
        """A 1e-3 agreement would mean the cached float32 wires were used.

        That cache round-trips the features through float32, and because the
        quantisation is a within-chain rank, near-ties flip digits and the score
        moves in the third significant figure. The called set survives it, so a
        loose tolerance here would hide the substitution.
        """
        d = art()
        self.assertTrue(d["reconstruction"]["agrees"])
        self.assertLess(d["reconstruction"]["worst_relative_error"], 1e-12)

    def test_the_deviation_is_a_deviation(self):
        """Score minus chain mean, as claimed, and not some other centring."""
        for c in art()["cases"]:
            for r in c["residues"]:
                with self.subTest(unit=c["unit_id"], res=r["resnum"]):
                    self.assertAlmostEqual(
                        r["score"] - r["chain_mean_score"],
                        r["deviation_from_chain_mean"], places=2)


@unittest.skipUnless(HAVE, REASON)
class EveryTableIsAccountedFor(unittest.TestCase):
    def test_the_family_split_covers_the_whole_bank(self):
        fa = art()["family_assignment"]
        n_field = len(json.loads(FIELD.read_text())["tables"])
        self.assertEqual(fa["n_tables"], n_field)
        self.assertLessEqual(fa["n_tables_spanning_two_families"], fa["n_tables"])

    def test_every_named_family_has_a_coarse_home(self):
        """A family with no rollup would vanish from the reported summary."""
        fa = art()["family_assignment"]
        for fam in fa["families"]:
            with self.subTest(family=fam):
                self.assertIn(fam, fa["coarse_rollup"])

    def test_the_mixed_table_convention_is_declared(self):
        fa = art()["family_assignment"]
        self.assertGreater(fa["n_tables_spanning_two_families"], 0)
        self.assertIn("split", fa["mixed_table_convention"])


@unittest.skipUnless(HAVE and CASES.is_file(), REASON)
class TheCasesWereNotChosenHere(unittest.TestCase):
    def test_the_cases_are_the_committed_ones_in_order(self):
        committed = [c["unit_id"] for c in json.loads(CASES.read_text())["cases"]]
        d = art()
        self.assertEqual(d["cases_are_not_chosen_here"]["case_ids"], committed)
        self.assertEqual([c["unit_id"] for c in d["cases"]], committed)

    def test_all_four_comparison_outcomes_are_represented(self):
        """Reporting only the chains we win on would be a demonstration."""
        got = {c["case"] for c in art()["cases"]}
        self.assertEqual(got, {"both_locate", "table_field_only",
                               "p2rank_only", "neither_locates"})

    def test_the_labels_and_calls_match_the_committed_case_study(self):
        cases = {c["unit_id"]: c for c in json.loads(CASES.read_text())["cases"]}
        for c in art()["cases"]:
            src = cases[c["unit_id"]]
            with self.subTest(unit=c["unit_id"]):
                self.assertEqual(c["n_labelled_cryptic"],
                                 len(src["cryptic_residues"]))
                self.assertEqual(c["n_called_by_us"],
                                 len(src["called_residues"]["table_field"]))
                self.assertEqual(c["n_called_by_p2rank"],
                                 len(src["called_residues"]["p2rank"]))

    def test_the_roles_agree_with_the_committed_membership(self):
        """A residue labelled 'false positive' must really be one."""
        cases = {c["unit_id"]: c for c in json.loads(CASES.read_text())["cases"]}
        for c in art()["cases"]:
            src = cases[c["unit_id"]]
            truth = {int(x) for x in src["cryptic_residues"]}
            ours = {int(x) for x in src["called_residues"]["table_field"]}
            theirs = {int(x) for x in src["called_residues"]["p2rank"]}
            for r in c["residues"]:
                n, role = r["resnum"], r["role"]
                with self.subTest(unit=c["unit_id"], res=n, role=role):
                    self.assertEqual(r["is_labelled_cryptic"], n in truth)
                    self.assertEqual(r["we_called_it"], n in ours)
                    self.assertEqual(r["p2rank_called_it"], n in theirs)
                    if role == "true positive":
                        self.assertTrue(n in ours and n in truth)
                    if role == "false positive":
                        self.assertTrue(n in ours and n not in truth)
                    if role in ("near miss", "deepest miss"):
                        self.assertTrue(n not in ours and n in truth)
                    if role == "we called it, P2Rank did not":
                        self.assertTrue(n in ours and n not in theirs)
                    if role == "P2Rank called it, we did not":
                        self.assertTrue(n in theirs and n not in ours)


@unittest.skipUnless(HAVE, REASON)
class TheResiduesWereChosenByRule(unittest.TestCase):
    def test_the_near_miss_outranks_the_deepest_miss(self):
        """Otherwise the two miss rules have been swapped or collapsed."""
        for c in art()["cases"]:
            by = {r["role"]: r for r in c["residues"]}
            if "near miss" in by and "deepest miss" in by:
                with self.subTest(unit=c["unit_id"]):
                    if by["near miss"]["resnum"] != by["deepest miss"]["resnum"]:
                        self.assertLess(by["near miss"]["score_rank_in_chain"],
                                        by["deepest miss"]["score_rank_in_chain"])

    def test_the_near_miss_ranks_below_the_calling_budget(self):
        """A missed residue cannot be inside the top-9% it was cut from."""
        q = json.loads(FIELD.read_text())["operating_point"]["q"]
        for c in art()["cases"]:
            k = max(1, round(q * c["n_residues"]))
            for r in c["residues"]:
                if r["role"] in ("near miss", "deepest miss"):
                    with self.subTest(unit=c["unit_id"], res=r["resnum"]):
                        self.assertGreater(r["score_rank_in_chain"], k)

    def test_a_true_positive_is_inside_the_budget(self):
        q = json.loads(FIELD.read_text())["operating_point"]["q"]
        for c in art()["cases"]:
            k = max(1, round(q * c["n_residues"]))
            for r in c["residues"]:
                if r["we_called_it"]:
                    with self.subTest(unit=c["unit_id"], res=r["resnum"]):
                        self.assertLessEqual(r["score_rank_in_chain"], k)

    def test_every_case_reports_a_hit_and_a_miss_and_a_disagreement(self):
        for c in art()["cases"]:
            roles = {r["role"] for r in c["residues"]}
            with self.subTest(unit=c["unit_id"]):
                self.assertIn("true positive", roles)
                self.assertTrue({"near miss", "deepest miss"} & roles)
                self.assertTrue({"we called it, P2Rank did not",
                                 "P2Rank called it, we did not"} & roles)


@unittest.skipUnless(HAVE, REASON)
class TheTableLevelDetailIsReadable(unittest.TestCase):
    def test_each_cited_table_states_a_complete_lookup(self):
        n_levels = 4
        for c in art()["cases"]:
            for r in c["residues"]:
                for t in r["largest_single_tables"]:
                    with self.subTest(unit=c["unit_id"], res=r["resnum"],
                                      q=t["quantities"]):
                        self.assertEqual(len(t["quantities"]), 2)
                        self.assertEqual(len(t["quartile_of_this_residue"]), 2)
                        for level in t["quartile_of_this_residue"]:
                            self.assertGreaterEqual(level, 1)
                            self.assertLessEqual(level, n_levels)
                        self.assertGreaterEqual(
                            t["cell_binding_rate_in_training"], 0.0)
                        self.assertLessEqual(
                            t["cell_binding_rate_in_training"], 1.0)
                        self.assertNotEqual(t["multiplicity"], 0)

    def test_the_cited_tables_are_the_largest_movers(self):
        """They are sorted by absolute effect, so the list is not arbitrary."""
        for c in art()["cases"]:
            for r in c["residues"]:
                mags = [abs(t["contribution_above_chain_mean"])
                        for t in r["largest_single_tables"]]
                with self.subTest(unit=c["unit_id"], res=r["resnum"]):
                    self.assertEqual(mags, sorted(mags, reverse=True))

    def test_the_base_rate_multiple_is_consistent(self):
        for c in art()["cases"]:
            for r in c["residues"]:
                for t in r["largest_single_tables"]:
                    with self.subTest(unit=c["unit_id"], res=r["resnum"]):
                        self.assertAlmostEqual(
                            t["cell_binding_rate_in_training"]
                            / t["fold_base_rate"],
                            t["cell_is_this_many_times_the_base_rate"],
                            places=1)


@unittest.skipUnless(HAVE, REASON)
class ThePatternClaimsMatchTheNumbers(unittest.TestCase):
    def test_the_hit_versus_miss_margins_are_the_stated_subtraction(self):
        sep = art()["what_separates_a_hit_from_a_miss"]
        p = sep["mean_contribution_by_residue_class"]
        self.assertAlmostEqual(
            p["called and labelled"]["geometric"]
            - p["labelled and missed"]["geometric"],
            sep["geometric_margin_of_hits_over_misses"], places=2)
        self.assertAlmostEqual(
            p["called and labelled"]["spatial smoothing"]
            - p["labelled and missed"]["spatial smoothing"],
            sep["spatial_smoothing_margin_of_hits_over_misses"], places=2)

    def test_the_gate_claim_is_not_asserted_against_the_numbers(self):
        sep = art()["what_separates_a_hit_from_a_miss"]
        mis = sep["mean_contribution_by_residue_class"]["labelled and missed"]
        self.assertEqual(
            sep["on_missed_positives_the_gate_outweighs_local_geometry"],
            mis["spatial smoothing"] > mis["geometric"] > 0)

    def test_the_false_positive_gap_is_smaller_than_the_miss_gap(self):
        """The reported reading depends on this ordering, so it is checked."""
        fp = art()["what_the_false_positives_are_made_of"]
        self.assertLess(fp["largest_family_gap_to_false_positives"],
                        fp["largest_family_gap_to_missed_positives"])
        self.assertGreater(fp["ratio"], 1.0)

    def test_every_residue_of_every_case_chain_is_classified(self):
        """The classes partition the chain, so nothing is quietly excluded."""
        for c in art()["cases"]:
            counts = c["mean_contribution_by_residue_class"]
            with self.subTest(unit=c["unit_id"]):
                self.assertEqual(sum(v["n"] for v in counts.values()),
                                 c["n_residues"])
                labelled = sum(v["n"] for k, v in counts.items()
                               if "labelled" in k and "not labelled" not in k)
                self.assertEqual(labelled, c["n_labelled_cryptic"])
                called = sum(v["n"] for k, v in counts.items()
                             if k.startswith("called"))
                self.assertEqual(called, c["n_called_by_us"])

    def test_the_pooling_is_by_chain_not_by_residue(self):
        """Residue-weighting would let the 297-residue chain decide the pattern."""
        d = art()
        sep = d["what_separates_a_hit_from_a_miss"]
        self.assertIn("equal weight per chain", sep["pooling"])
        for name, row in sep["mean_contribution_by_residue_class"].items():
            per_chain = [c["mean_contribution_by_residue_class"][name]
                         for c in d["cases"]
                         if name in c["mean_contribution_by_residue_class"]]
            with self.subTest(cls=name):
                self.assertEqual(row["n_chains"], len(per_chain))
                self.assertEqual(row["n_residues"],
                                 sum(r["n"] for r in per_chain))
                for key in row:
                    if key in ("n_chains", "n_residues"):
                        continue
                    self.assertAlmostEqual(
                        row[key],
                        float(np.mean([r[key] for r in per_chain])), places=2)


@unittest.skipUnless(HAVE, REASON)
class ItDoesNotSpendATestFoldRead(unittest.TestCase):
    def test_it_declares_no_read_index(self):
        """It re-explains frozen calls; it produces no comparable statistic."""
        d = art()
        self.assertIsNone(d["test_fold_read_index"])
        self.assertIn("no statistic", d["why_this_is_not_an_indexed_read"])

    def test_it_cannot_be_used_to_choose_anything(self):
        """No metric here is reported for a configuration other than the one
        shipped, which is what would make it a selection."""
        text = ART.read_text()
        for forbidden in ("\"roc_auc\"", "\"pr_auc\"", "candidate",
                          "alternative_configuration"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
