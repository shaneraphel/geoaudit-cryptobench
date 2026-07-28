"""The counterattack's two artifacts have to keep telling the same story.

The selection says a construction beat the dense bank on every training split.
The probe says it did not beat it on the held-out fold. That pair is the whole
point of the section, and each half is easy to invalidate by accident: rerun the
selection with a different candidate pool and the training claim changes without
the paper noticing; regenerate the probe from a pipeline that no longer
reproduces the control and the negative claim stops meaning anything.
"""
from __future__ import annotations

import json
import unittest

from pocket_bench.methods.quotient_tables import (
    n_cells_dense, n_orbits, widest_admissible,
)
from pocket_bench.paths import ROOT

SEL = ROOT / "results/architecture_sweep/COUNTERATTACK_QUOTIENT.json"
PROBE = ROOT / "results/official_fold/COUNTERATTACK_QUOTIENT_PROBE.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
FROZEN_DENSE_AUC = 0.7668089153970957


class TestTheSelectionNeverReadTheFold(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(SEL.read_text())

    def test_it_says_so_and_carries_no_official_units(self):
        self.assertFalse(self.doc["reads_test_fold"])
        text = SEL.read_text()
        self.assertNotIn("official_fold", text)
        self.assertNotIn("test_fold_read_index", text)

    def test_every_split_ranked_the_same_pool(self):
        pools = {tuple(sorted(r["architecture"] for r in s["ranking"]))
                 for s in self.doc["splits"]}
        self.assertEqual(len(pools), 1)
        self.assertEqual(len(next(iter(pools))), self.doc["n_candidates"])

    def test_both_kinds_of_disjointness_are_present(self):
        kinds = {s["split"].split(":")[0] for s in self.doc["splits"]}
        self.assertEqual(kinds, {"cv", "half"})

    def test_the_failed_constructions_were_not_dropped(self):
        names = [r["architecture"] for r in self.doc["summary"]]
        self.assertTrue(any("S_35" in n for n in names),
                        "the widest quotient is the one that failed; a candidate "
                        "list without it is a selection over an unstated pool")
        losers = [r for r in self.doc["summary"]
                  if r["mean_delta_vs_control"] < 0]
        self.assertGreaterEqual(len(losers), 3)

    def test_the_capacity_block_recomputes(self):
        c = self.doc["capacity"]
        rN = c["n_train_positives"]
        self.assertEqual(c["dense_widest_admissible_L4"],
                         widest_admissible(4, rN, symmetric=False))
        self.assertEqual(c["quotient_widest_admissible_L4"],
                         widest_admissible(4, rN, symmetric=True))
        self.assertEqual(c["dense_cells_d6_L4"], n_cells_dense(6, 4))
        self.assertEqual(c["quotient_cells_d6_L4"], n_orbits(6, 4))
        self.assertEqual(c["quotient_cells_d35_L4"], n_orbits(35, 4))

    def test_the_one_split_number_is_larger_than_the_honest_one(self):
        """Not a law, but it is what happened, and dropping it would be the
        omission the field the number lives in exists to prevent."""
        h = self.doc["selection_honesty"]
        self.assertGreater(h["delta_on_the_split_it_was_found_on"],
                           h["delta_over_all_splits"])


class TestTheProbeIsOneReadAndReportsALoss(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(PROBE.read_text())

    def test_it_is_the_fourth_read_and_the_ledger_agrees(self):
        idx = self.doc["test_fold_read_index"]
        led = json.loads(LEDGER.read_text())
        indexed = {p["read_index"] for p in led["indexed_read_sequence"]}
        self.assertIn(idx, indexed)
        self.assertEqual(sorted(indexed), list(range(1, len(indexed) + 1)),
                         "read indices must be contiguous from one")

    def test_the_control_reproduces_the_frozen_detector(self):
        r = self.doc["reproduction_check"]
        self.assertFalse(r["counts_as_a_fold_read"])
        self.assertAlmostEqual(r["residue_auc_mean"], FROZEN_DENSE_AUC, places=3)
        self.assertLess(abs(r["difference"]), 5e-4)

    def test_the_verdict_matches_the_paired_numbers(self):
        v = self.doc["verdict"]
        af = self.doc["paired_vs"]["algebraic_field"]["residue_auc"]
        p2 = self.doc["paired_vs"]["p2rank"]["residue_auc"]
        self.assertEqual(v["delta_vs_dense_counting_field"],
                         af["paired_difference"])
        self.assertEqual(v["beats_the_dense_counting_field"],
                         af["paired_difference"] > 0)
        self.assertEqual(v["delta_vs_p2rank"], p2["paired_difference"])
        self.assertEqual(v["separable_from_p2rank"], p2["excludes_zero"])

    def test_the_gain_did_not_transfer(self):
        """The section's claim, asserted so that a rerun that quietly starts
        winning cannot slip past the prose that says it did not."""
        af = self.doc["paired_vs"]["algebraic_field"]["residue_auc"]
        prov = self.doc["selection_provenance"]
        self.assertFalse(prov["reads_test_fold"])
        self.assertGreater(prov["mean_delta_vs_dense_bank_on_train"], 0.0)
        self.assertGreater(prov["worst_delta_vs_dense_bank_on_train"], 0.0)
        self.assertFalse(af["excludes_zero"])
        self.assertLess(af["paired_difference"],
                        prov["mean_delta_vs_dense_bank_on_train"])

    def test_it_scored_every_official_unit(self):
        self.assertEqual(self.doc["n_test_units"], 192)
        self.assertEqual(self.doc["n_scored_units"], 192)
        self.assertEqual(len(self.doc["per_structure"]), 192)

    def test_no_parameter_was_fitted(self):
        self.assertEqual(self.doc["architecture"]["fitted_parameters"],
                         "none; cells are counts and fan-out is a rank")


if __name__ == "__main__":
    unittest.main()
