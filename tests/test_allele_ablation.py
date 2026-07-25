"""GF(4) wrong-allele / allele-shuffle ablation gate.

Proves the allele-conditioning is structurally exclusionary (algebraic, not
chemical): a candidate that satisfies the correct KRAS G12D syndrome fails closed
against every wrong allele, and the rejection is emergent from distinct syndromes
+ a nonsingular operator (verified), not asserted. `clinical_grade=false`.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "gf4_allele_shuffle_ablation", ROOT / "tools" / "gf4_allele_shuffle_ablation.py"
)
gf4 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gf4)  # type: ignore[union-attr]


class TestGF4Field(unittest.TestCase):
    def test_multiplication_table(self) -> None:
        A, A2, ONE = gf4.GF4_A, gf4.GF4_A2, gf4.GF4_ONE
        self.assertEqual(gf4.gf4_mul(A, A), A2)     # a * a = a^2
        self.assertEqual(gf4.gf4_mul(A, A2), ONE)   # a * a^2 = 1
        self.assertEqual(gf4.gf4_mul(A2, A2), A)    # a^2 * a^2 = a
        self.assertEqual(gf4.gf4_mul(ONE, A), A)    # identity

    def test_addition_is_characteristic_two(self) -> None:
        A, A2, ONE = gf4.GF4_A, gf4.GF4_A2, gf4.GF4_ONE
        self.assertEqual(gf4.gf4_add(ONE, A), A2)   # 1 + a = a^2
        self.assertEqual(gf4.gf4_add(A, A2), ONE)   # a + a^2 = 1
        self.assertEqual(gf4.gf4_add(A, A), 0)      # x + x = 0

    def test_operators_nonsingular_forward_solve_exact(self) -> None:
        H, B = gf4.build_operators()
        delta = gf4.allele_syndrome("G12D")
        rhs = gf4.matvec(B, delta)
        x = gf4.solve_unit_lower_triangular(H, rhs)
        self.assertEqual(gf4.matvec(H, x), rhs)  # exact solution over GF(4)


class TestAlleleAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = gf4.run_ablation(n_shuffles=2000, seed=20260725)

    def test_syndromes_are_allele_distinct(self) -> None:
        self.assertTrue(self.report["all_syndromes_distinct"])
        # each allele's residual is non-zero only inside codon 12 (indices 21..23)
        for a, idx in self.report["syndrome_nonzero_index"].items():
            self.assertTrue(all(21 <= i <= 23 for i in idx),
                            f"{a} residual leaked outside codon 12: {idx}")

    def test_correct_allele_admissible(self) -> None:
        self.assertTrue(self.report["correct_allele_G12D_admissible"])
        self.assertTrue(self.report["correct_residual_zero"])

    def test_wrong_alleles_fail_closed(self) -> None:
        wrong = self.report["wrong_allele_fail_closed"]
        for a in ("G12C", "G12V", "WT"):
            self.assertTrue(wrong[a]["rejected"],
                            f"wrong allele {a} was NOT rejected (leak)")
            self.assertGreater(wrong[a]["residual_weight"], 0)
        self.assertTrue(self.report["all_wrong_alleles_rejected"])

    def test_shuffle_rejection_high_but_not_rigged(self) -> None:
        sh = self.report["allele_shuffle"]
        # emergent, not tautological: a shuffle that restores the residual locus
        # legitimately passes, so the rate is high yet strictly below 1.0.
        self.assertGreaterEqual(sh["rejection_rate"], 0.90)
        self.assertLess(sh["rejection_rate"], 1.0)

    def test_deterministic(self) -> None:
        r2 = gf4.run_ablation(n_shuffles=2000, seed=20260725)
        self.assertEqual(self.report["syndromes"], r2["syndromes"])
        self.assertEqual(self.report["allele_shuffle"]["rejected"],
                         r2["allele_shuffle"]["rejected"])


if __name__ == "__main__":
    unittest.main()
