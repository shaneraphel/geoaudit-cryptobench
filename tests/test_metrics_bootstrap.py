"""Paired bootstrap CI + MCC/F1 gates."""
from __future__ import annotations

import unittest

from pocket_bench.metrics_bootstrap import (
    f1,
    f1_from_counts,
    mcc,
    mcc_from_counts,
    paired_bootstrap,
    per_structure_values,
)


class TestMccF1(unittest.TestCase):
    def test_mcc_perfect_and_inverse(self) -> None:
        s, y = [1, 1, 0, 0], [1, 1, 0, 0]
        self.assertAlmostEqual(mcc(s, y, 0.5), 1.0)
        self.assertAlmostEqual(mcc(s, [0, 0, 1, 1], 0.5), -1.0)

    def test_mcc_undefined_when_degenerate(self) -> None:
        self.assertIsNone(mcc_from_counts(0, 0, 4, 0))  # no positives predicted/true

    def test_f1(self) -> None:
        self.assertAlmostEqual(f1([1, 1, 0, 0], [1, 1, 0, 0], 0.5), 1.0)
        self.assertAlmostEqual(f1_from_counts(2, 1, 1), 2 * 2 / (2 * 2 + 1 + 1))


class TestPairedBootstrap(unittest.TestCase):
    def test_separates_clearly_better_method(self) -> None:
        # method A strictly dominates baseline on every structure -> Δ CI excludes 0
        vals = {
            "A": [0.9, 0.8, 0.85, 0.95, 0.7],
            "base": [0.5, 0.45, 0.55, 0.5, 0.4],
        }
        res = paired_bootstrap(vals, baseline="base", n_boot=3000, seed=1)
        d = res["paired_vs_baseline"]["A"]
        self.assertGreater(d["delta_point"], 0)
        self.assertFalse(d["crosses_zero"])

    def test_tie_crosses_zero(self) -> None:
        vals = {
            "A": [0.6, 0.4, 0.55, 0.45, 0.5],
            "base": [0.55, 0.45, 0.5, 0.5, 0.5],
        }
        res = paired_bootstrap(vals, baseline="base", n_boot=3000, seed=1)
        self.assertTrue(res["paired_vs_baseline"]["A"]["crosses_zero"])

    def test_deterministic(self) -> None:
        vals = {"A": [0.6, 0.7, 0.5], "base": [0.5, 0.5, 0.5]}
        r1 = paired_bootstrap(vals, baseline="base", n_boot=2000, seed=7)
        r2 = paired_bootstrap(vals, baseline="base", n_boot=2000, seed=7)
        self.assertEqual(r1["paired_vs_baseline"], r2["paired_vs_baseline"])

    def test_none_values_ignored_not_imputed(self) -> None:
        vals = {"A": [0.6, None, 0.5], "base": [0.5, 0.5, 0.5]}
        res = paired_bootstrap(vals, baseline="base", n_boot=1000, seed=3)
        self.assertEqual(res["per_method"]["A"]["n_structures_scored"], 2)

    def test_per_structure_alignment(self) -> None:
        rows = [
            {"pdb": "b", "method": "m", "residue_auc": 0.7},
            {"pdb": "a", "method": "m", "residue_auc": 0.6},
        ]
        out = per_structure_values(rows, "residue_auc")
        self.assertEqual(out["m"], [0.6, 0.7])  # sorted by pdb (a,b)

    def test_two_chains_of_one_entry_are_separate_units(self) -> None:
        """The official fold has entries contributing two chains (3lnz C/O).

        Keyed on pdb alone these overwrote each other and vanished from the
        resample; both must survive as independent units.
        """
        rows = [
            {"pdb": "3lnz", "chain": "C", "unit_id": "3lnz_C",
             "method": "m", "residue_auc": 0.6},
            {"pdb": "3lnz", "chain": "O", "unit_id": "3lnz_O",
             "method": "m", "residue_auc": 0.8},
        ]
        out = per_structure_values(rows, "residue_auc")
        self.assertEqual(out["m"], [0.6, 0.8])

    def test_colliding_unit_keys_raise_instead_of_dropping(self) -> None:
        rows = [
            {"pdb": "3lnz", "method": "m", "residue_auc": 0.6},
            {"pdb": "3lnz", "method": "m", "residue_auc": 0.8},
        ]
        with self.assertRaises(ValueError):
            per_structure_values(rows, "residue_auc")


if __name__ == "__main__":
    unittest.main()
