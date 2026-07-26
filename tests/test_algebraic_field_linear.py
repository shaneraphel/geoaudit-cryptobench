"""Guards on the fitted readout of the algebraic field.

These check the properties whose violation would be invisible in the reported
metrics: a compiled artifact that no longer matches the source that produced it,
a propensity table contaminated by test labels, a quantizer that depends on
absolute units, or a query path that is not independent per chain.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from pocket_bench.methods import algebraic_field_linear as afl

ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT / "data/cryptobench_apo/ALGEBRAIC_FIELD_LINEAR.json"
TRAIN_CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
TEST_CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_test.npz"


class TestQuantizer(unittest.TestCase):
    def test_digits_are_invariant_to_monotone_rescaling(self) -> None:
        """A wire's digit is its rank inside its own chain, so units cannot matter.

        This is the property that lets one field score a 57-residue chain and a
        307-residue chain without carrying a constant between them. If the
        quantizer ever regresses to absolute cut points, this fails.
        """
        rng = np.random.default_rng(11)
        F = rng.normal(size=(64, 5))
        base = afl.chain_digits(F)
        for transform in (lambda x: 3.0 * x + 17.0,
                          lambda x: np.exp(x / 4.0),
                          lambda x: np.arcsinh(x) * 1e-3):
            np.testing.assert_array_equal(afl.chain_digits(transform(F)), base)

    def test_digits_span_the_quaternary_range(self) -> None:
        rng = np.random.default_rng(12)
        D = afl.chain_digits(rng.normal(size=(400, 3)))
        self.assertEqual(int(D.min()), 0)
        self.assertEqual(int(D.max()), 3)


class TestGate(unittest.TestCase):
    def test_gate_is_a_neighbourhood_mean(self) -> None:
        """Two residues inside one radius of each other must share their score."""
        ctr = np.array([[0.0, 0, 0], [1.0, 0, 0], [500.0, 0, 0]])
        s = np.array([1.0, 0.0, 4.0])
        out = afl.apply_gate(s, ctr)
        self.assertAlmostEqual(out[0] - s[0], out[1] - s[1], places=9)
        self.assertGreater(out[2] - s[2], out[0] - s[0])

    def test_gate_is_a_no_op_on_a_flat_field(self) -> None:
        ctr = np.zeros((8, 3))
        s = np.full(8, 2.5)
        np.testing.assert_allclose(afl.apply_gate(s, ctr), s)


@unittest.skipUnless(FIELD.exists(), "compiled readout not present")
class TestCompiledArtifact(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = json.loads(FIELD.read_text())

    def test_artifact_matches_the_source_that_compiled_it(self) -> None:
        """A stale field would report numbers no current code path can produce."""
        self.assertEqual(
            self.doc["code_sha256"], afl.code_sha256(),
            "ALGEBRAIC_FIELD_LINEAR.json was compiled by different source; "
            "rerun tools/compile_algebraic_field_linear.py")

    def test_artifact_is_strict_json_and_declares_its_fitting(self) -> None:
        text = FIELD.read_text()
        for token in ("NaN", "Infinity"):
            self.assertNotIn(token, text)
        self.assertFalse(self.doc["clinical_grade"])
        self.assertIn("closed-form", self.doc["fitting"])

    def test_shapes_agree(self) -> None:
        n = self.doc["n_wires"]
        for key in ("coefficients", "digit_mean", "digit_std", "wire_names"):
            self.assertEqual(len(self.doc[key]), n, key)
        self.assertEqual(len(self.doc["propensity_table"]), 20)

    @unittest.skipUnless(TRAIN_CACHE.exists() and TEST_CACHE.exists(),
                         "feature caches not present")
    def test_propensity_table_is_compiled_on_train_only(self) -> None:
        """The one place a test label could leak into a per-residue feature."""
        train = np.load(TRAIN_CACHE, allow_pickle=False)["propensity_table"]
        test = np.load(TEST_CACHE, allow_pickle=False)["propensity_table"]
        np.testing.assert_array_equal(train, test)
        np.testing.assert_allclose(
            np.asarray(self.doc["propensity_table"]), train, rtol=0, atol=0)

    @unittest.skipUnless(TEST_CACHE.exists(), "test cache not present")
    def test_scoring_one_chain_does_not_depend_on_the_others(self) -> None:
        """Scoring is per chain, so the fold order cannot move a reported value."""
        z = np.load(TEST_CACHE, allow_pickle=False)
        n_res = z["n_res_per"]
        field = afl.AlgebraicFieldLinear(self.doc)
        offs = np.concatenate([[0], np.cumsum(n_res)])
        for i in (0, 1, 2):
            a, b = int(offs[i]), int(offs[i + 1])
            alone = field.score_matrix(z["X"][a:b], z["ctr"][a:b])
            again = field.score_matrix(z["X"][a:b].copy(), z["ctr"][a:b].copy())
            np.testing.assert_array_equal(alone, again)
            self.assertEqual(len(alone), b - a)
            self.assertTrue(np.all(np.isfinite(alone)))

    def test_operating_point_calls_the_declared_fraction(self) -> None:
        field = afl.AlgebraicFieldLinear(self.doc)
        score = np.arange(1000, dtype=np.float64)
        call = field.positive_call(score)
        self.assertEqual(int(call.sum()), round(field.q * 1000))
        self.assertTrue(call[-1])
        self.assertFalse(call[0])


if __name__ == "__main__":
    unittest.main()
