"""Paired bootstrap CI + MCC/F1 gates."""
from __future__ import annotations

import json
import math
import unittest

from pocket_bench.metrics_bootstrap import (
    f1,
    f1_from_counts,
    json_safe,
    mcc,
    mcc_from_counts,
    paired_bootstrap,
    per_structure_values,
)


class TestJsonStrictness(unittest.TestCase):
    """Non-finite floats must serialize as JSON null, never as a bare NaN token."""

    def test_non_finite_becomes_none(self) -> None:
        self.assertIsNone(json_safe(float("nan")))
        self.assertIsNone(json_safe(float("inf")))
        self.assertIsNone(json_safe(float("-inf")))
        self.assertEqual(json_safe(0.25), 0.25)

    def test_nested_structures_sanitized(self) -> None:
        raw = {"per_method": {"m": {"ci_low": float("nan"), "point": 0.5}},
               "xs": [1.0, float("inf"), {"y": float("nan")}]}
        safe = json_safe(raw)
        self.assertIsNone(safe["per_method"]["m"]["ci_low"])
        self.assertEqual(safe["per_method"]["m"]["point"], 0.5)
        self.assertEqual(safe["xs"], [1.0, None, {"y": None}])

    def test_emits_strict_json(self) -> None:
        raw = {"ci_low": float("nan")}
        # allow_nan=False is what a strict consumer effectively enforces.
        with self.assertRaises(ValueError):
            json.dumps(raw, allow_nan=False)
        text = json.dumps(json_safe(raw), allow_nan=False)
        self.assertIn("null", text)
        self.assertNotIn("NaN", text)
        self.assertIsNone(json.loads(text)["ci_low"])

    def test_bootstrap_output_is_strict_without_json_safe(self) -> None:
        """The producer, not only the sanitizer, must be strict.

        Relying on every caller to remember ``json_safe`` is how a bare NaN
        reached a frozen report: a method that scored on no structure has an
        empty bootstrap distribution, and an empty percentile used to be NaN.
        """
        vals = {
            "scored": [0.7, 0.6, 0.8],
            "never_scored": [None, None, None],
        }
        out = paired_bootstrap(vals, baseline="scored", n_boot=64, seed=1)
        text = json.dumps(out, allow_nan=False)   # raises if any NaN survives
        self.assertNotIn("NaN", text)
        empty = out["per_method"]["never_scored"]
        self.assertIsNone(empty["point"])
        self.assertIsNone(empty["ci_low"])
        self.assertIsNone(empty["ci_high"])
        self.assertEqual(empty["n_structures_scored"], 0)
        delta = out["paired_vs_baseline"]["never_scored"]
        self.assertIsNone(delta["delta_point"])
        self.assertIsNone(delta["crosses_zero"],
                          "an unestimated difference must not read as a verdict")

    def test_bootstrap_report_is_strict_json_without_sanitizing(self) -> None:
        """A method scored on nothing has no CI, and "no CI" is null, not NaN.

        json_safe exists as a belt, but a report that is only strict because
        every caller remembered to route it through a helper is one forgotten
        call away from emitting a bare NaN token. paired_bootstrap must not be
        able to produce one in the first place.
        """
        res = paired_bootstrap(
            {"base": [0.5, 0.6], "empty": [None, None]},
            baseline="base", n_boot=32,
        )
        empty = res["per_method"]["empty"]
        self.assertIsNone(empty["ci_low"])
        self.assertIsNone(empty["ci_high"])
        self.assertEqual(empty["n_structures_scored"], 0)

        text = json.dumps(res, allow_nan=False)   # no json_safe on this path
        self.assertNotIn("NaN", text)
        self.assertIsNone(json.loads(text)["per_method"]["empty"]["ci_low"])

    def test_undefined_delta_is_not_reported_as_significant(self) -> None:
        res = paired_bootstrap(
            {"base": [0.5, 0.6], "empty": [None, None]},
            baseline="base", n_boot=32,
        )
        d = res["paired_vs_baseline"]["empty"]
        self.assertIsNone(d["delta_point"])
        self.assertIsNone(d["crosses_zero"])
        self.assertIsNone(d["p_two_sided_bootstrap"])

    def test_json_safe_still_available_for_upstream_nans(self) -> None:
        self.assertIsNone(json_safe(float("nan")))
        self.assertTrue(math.isnan(float("nan")))


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
