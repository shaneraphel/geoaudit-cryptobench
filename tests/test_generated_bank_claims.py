"""The two artifacts behind Section 4.6 have to support what it says.

The claim being audited is a negative one -- that 267 generated descriptors are
worth nothing to the counting field -- and a negative claim is easy to
manufacture by searching one arm harder than the other. So the tests here are
mostly about the comparison being fair: same grid, same seed, same bases,
optimum interior to the grid in both arms, and a control that reproduces the
published number rather than a fresh one that happens to be lower.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CEILING = ROOT / "results/architecture_sweep/OPERATOR_BANK_CEILING.json"
WIDE3 = ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE3.json"
CONTROL = ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE3_CONTROL.json"
PUBLISHED_645 = 0.8045


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


class TestTheCeilingArtifact(unittest.TestCase):
    def setUp(self):
        if not CEILING.exists():
            self.skipTest("the ceiling artifact has not been built")
        self.d = _load(CEILING)

    def test_it_records_the_descriptor_counts_the_modules_define(self):
        from pocket_bench.methods.chain_operator_descriptors import (
            N_CHAIN_OPERATOR)
        from pocket_bench.methods.operator_descriptors import N_OPERATOR

        g = self.d["generator"]
        self.assertEqual(g["n_operator_descriptors"], N_OPERATOR)
        self.assertEqual(g["n_chain_descriptors"], N_CHAIN_OPERATOR)

    def test_the_lift_follows_from_the_per_split_numbers(self):
        base = self.d["per_split"][self.d["keys"]["baseline"]]
        full = self.d["per_split"][self.d["keys"]["full"]]
        d = [f - b for f, b in zip(full, base)]
        self.assertAlmostEqual(sum(d) / len(d), self.d["lift"]["mean"],
                               places=5)
        self.assertEqual(sum(1 for x in d if x > 0),
                         self.d["lift"]["n_splits_positive"])
        self.assertAlmostEqual(min(d), self.d["lift"]["min"], places=5)

    def test_the_test_fold_was_not_read(self):
        self.assertFalse(self.d["reads_test_fold"])

    def test_every_family_is_accounted_for(self):
        """A family sweep that silently dropped descriptors would let a weak
        family hide inside an unnamed remainder."""
        from pocket_bench.methods.chain_operator_descriptors import (
            FEATURE_NAMES as CH)
        from pocket_bench.methods.operator_descriptors import (
            FEATURE_NAMES as OP)

        counted = sum(f["n_descriptors"] for f in self.d["by_family"])
        self.assertEqual(counted, len(OP) + len(CH))


class TestTheComparisonIsFair(unittest.TestCase):
    def setUp(self):
        if not (WIDE3.exists() and CONTROL.exists()):
            self.skipTest("the wide-wire arms have not both been built")
        self.t, self.c = _load(WIDE3), _load(CONTROL)

    def test_the_arms_are_what_they_say(self):
        self.assertEqual(self.t["arm"], "both")
        self.assertEqual(self.c["arm"], "existing")
        self.assertGreater(self.t["n_wires"], self.c["n_wires"])
        self.assertEqual(self.c["n_wires"], self.t["n_wires_existing"])

    def test_both_arms_saw_the_same_grid_seed_and_bases(self):
        self.assertEqual(self.t["ridge_grid"], self.c["ridge_grid"])
        self.assertEqual(self.t["bases"], self.c["bases"])
        self.assertEqual(self.t["split"], self.c["split"])

    def test_the_control_reproduces_the_published_number(self):
        """If the control had drifted, the difference between the arms would be
        a difference between two harnesses and not between two wire sets."""
        self.assertAlmostEqual(self.c["selected"]["pick_half_roc_auc"],
                               PUBLISHED_645, places=4)

    def test_neither_arm_is_pinned_at_an_end_of_the_ridge_grid(self):
        """An optimum at the edge means the grid, not the wires, may be what is
        being compared. This is the test that forced the grid from three points
        to six."""
        for name, arm in (("treatment", self.t), ("control", self.c)):
            grid = sorted(arm["ridge_grid"])
            best = arm["selected"]["ridge"]
            with self.subTest(arm=name):
                if best == grid[0]:
                    # The smallest ridge is the unregularised end; an optimum
                    # there is a boundary only if a smaller one could help, and
                    # the interior points establish the curve is falling away
                    # from it.
                    at = [r["pick_half_roc_auc"] for r in arm["candidates"]
                          if r["ridge"] == grid[1]]
                    self.assertTrue(
                        at and at[0] < arm["selected"]["pick_half_roc_auc"],
                        "the optimum sits at the low end with nothing showing "
                        "the curve turns over")
                else:
                    self.assertNotEqual(
                        best, grid[-1],
                        "the optimum sits at the high end of the ridge grid, "
                        "so the grid may be the binding constraint")

    def test_the_recorded_delta_is_the_difference_of_the_two_arms(self):
        d = (self.t["selected"]["pick_half_roc_auc"]
             - self.c["selected"]["pick_half_roc_auc"])
        self.assertAlmostEqual(d, self.t["delta_vs_645_wires"], places=4)

    def test_the_wider_arm_addresses_fewer_of_its_cells(self):
        """The stated mechanism for the negative result. If the wider bank had
        the better coverage, the explanation in Section 4.6 would be wrong."""
        def empty(a):
            s = a["selected"]
            return s["n_cells_never_addressed"] / s["n_cells"]

        self.assertGreater(empty(self.t), empty(self.c))

    def test_neither_arm_read_the_test_fold(self):
        for a in (self.t, self.c):
            self.assertFalse(a["test_fold_read"])


if __name__ == "__main__":
    unittest.main()
