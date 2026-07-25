"""0-masking telemetry gates: faithful metrics + denominator discipline."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from pocket_bench.metrics import average_precision, residue_auc_pr, roc_auc
from pocket_bench.telemetry import (
    aggregate,
    assert_denominator_discipline,
    declared_available_tools,
    telemetry_row,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ENV = ROOT / "data/manifests/BASELINE_ENV.json"


class TestFaithfulMetrics(unittest.TestCase):
    def test_roc_auc_known_value(self) -> None:
        # perfect separation -> 1.0 ; inverted -> 0.0 ; tie handling
        self.assertEqual(roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertEqual(roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)
        self.assertAlmostEqual(roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5)
        self.assertIsNone(roc_auc([0.5, 0.6], [1, 1]))  # one class

    def test_average_precision(self) -> None:
        self.assertAlmostEqual(average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertIsNone(average_precision([0.1, 0.2], [0, 0]))

    def test_residue_auc_pr_available(self) -> None:
        pockets = [
            {"rank": 1, "residues": ["A:ALA10", "A:GLY11", "A:SER12"]},
            {"rank": 2, "residues": ["A:VAL40"]},
        ]
        universe = list(range(1, 51))
        truth = [10, 11, 12]  # exactly the rank-1 pocket
        out = residue_auc_pr(pockets, truth, universe)
        self.assertTrue(out["available"])
        self.assertEqual(out["residue_auc"], 1.0)


class TestTelemetryRow(unittest.TestCase):
    def test_row_has_mandated_fields(self) -> None:
        row = telemetry_row(
            method="geometric_foundation", pdb="1a4u", split="test", status="OK",
            scored={"top1": {"success": True, "best_dca": 2.78}, "dcc_top1": 3.9},
            label={"cryptic_residues": [10, 11], "binding_residues": ["A:ALA10"]},
            prediction={"pockets": [{"rank": 1, "residues": ["A:ALA10"], "center_xyz": [0, 0, 0]}]},
            universe_residues=list(range(1, 30)),
            tool_version=None, env_sha="abc", seed=0, runtime_s=1.2,
        )
        for f in ("residue_auc", "residue_pr_auc", "residue_f1", "top1_dca",
                  "top1_success", "n_pockets", "residue_metrics_available"):
            self.assertIn(f, row)
        self.assertIs(row["clinical_grade"], False)
        self.assertTrue(row["top1_success"])


class TestDenominatorDiscipline(unittest.TestCase):
    def _rows(self, method, statuses):
        return [
            telemetry_row(method=method, pdb=f"p{i}", split="test", status=st,
                          scored={"top1": {"success": st == "OK", "best_dca": 1.0}},
                          label=None, prediction=None)
            for i, st in enumerate(statuses)
        ]

    def test_declared_available_from_baseline(self) -> None:
        env = json.loads(BASELINE_ENV.read_text())
        avail = declared_available_tools(env)
        self.assertIn("p2rank", avail)
        self.assertIn("fpocket", avail)
        self.assertNotIn("deeppocket", avail)  # declared TOOL_UNAVAILABLE

    def test_valid_run_passes(self) -> None:
        rows = self._rows("geometric_foundation", ["OK", "OK", "EMPTY"])
        summ = aggregate(rows, {"geometric_foundation": 3})
        assert_denominator_discipline(summ, {"p2rank", "fpocket"})
        s = summ["per_method"]["geometric_foundation"]
        self.assertEqual(s["intention_to_evaluate_denominator"], 3)
        self.assertEqual(s["available_denominator"], 3)
        self.assertEqual(s["hits_over_intention"], "2/3")

    def test_empty_counts_as_miss_not_mask(self) -> None:
        # p2rank returns EMPTY (declared available) -> must count in denominator, no mask
        rows = self._rows("p2rank", ["OK", "EMPTY", "CRASH"])
        summ = aggregate(rows, {"p2rank": 3})
        assert_denominator_discipline(summ, {"p2rank", "fpocket"})
        s = summ["per_method"]["p2rank"]
        self.assertEqual(s["available_denominator"], 3)  # nothing masked away
        self.assertEqual(s["top1_hits"], 1)

    def test_false_tool_unavailable_for_available_tool_raises(self) -> None:
        rows = self._rows("p2rank", ["OK", "TOOL_UNAVAILABLE"])
        summ = aggregate(rows, {"p2rank": 2})
        with self.assertRaises(AssertionError):
            assert_denominator_discipline(summ, {"p2rank", "fpocket"})

    def test_internal_method_cannot_be_unavailable(self) -> None:
        rows = self._rows("geometric_foundation", ["TOOL_UNAVAILABLE"])
        summ = aggregate(rows, {"geometric_foundation": 1})
        with self.assertRaises(AssertionError):
            assert_denominator_discipline(summ, {"p2rank", "fpocket"})

    def test_intention_denominator_mismatch_raises(self) -> None:
        rows = self._rows("fpocket", ["OK", "OK"])
        summ = aggregate(rows, {"fpocket": 5})  # lies: only 2 attempted
        with self.assertRaises(AssertionError):
            assert_denominator_discipline(summ, {"p2rank", "fpocket"})

    def test_declared_unavailable_tool_allowed_with_dual_denominators(self) -> None:
        rows = self._rows("deeppocket", ["TOOL_UNAVAILABLE", "TOOL_UNAVAILABLE"])
        summ = aggregate(rows, {"deeppocket": 2})
        assert_denominator_discipline(summ, {"p2rank", "fpocket"})  # deeppocket not declared avail
        s = summ["per_method"]["deeppocket"]
        self.assertEqual(s["available_denominator"], 0)
        self.assertEqual(s["intention_to_evaluate_denominator"], 2)


if __name__ == "__main__":
    unittest.main()
