"""P2Rank must be scored on its OWN residue-level output.

P2Rank is natively a residue predictor. Reconstructing a residue signal from
pocket centres (top-5 pockets, 6 A ball, 1/rank weight) scores a different
prediction than the one it made and caps the achievable AUROC, because 1/rank is
a 5-valued staircase with every remaining residue tied at 0.
"""
from __future__ import annotations

import unittest

from pocket_bench.methods.p2rank_wrap import parse_residues_csv
from pocket_bench.metrics import native_residue_scores, residue_auc_pr

_CSV = """chain, residue_label, residue_name, score, zscore, probability, pocket
A,    1, LYS,  0.0179,  -0.3695,   0.0005, 0
A,    2, GLU,  0.9106,   3.1735,   0.9104, 1
A,    3, LEU,  0.7479,   2.3528,   0.7010, 1
A,    4, VAL,  0.0023,  -0.3782,   0.0003, 0
B,    5, ALA,  0.5000,   1.0000,   0.5000, 2
"""


def _write(tmpdir, text: str):
    from pathlib import Path
    p = Path(tmpdir) / "rec.pdb_residues.csv"
    p.write_text(text)
    return p


class TestResidueCsvParsing(unittest.TestCase):
    def test_parses_probability_and_native_positive_call(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            scores, positive = parse_residues_csv(_write(td, _CSV))
        self.assertEqual(scores[2], 0.9104)
        self.assertEqual(scores[1], 0.0005)
        # `pocket > 0` is P2Rank's own binary call, not a distance cutoff.
        self.assertEqual(positive, {2, 3, 5})

    def test_chain_filter(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            scores, positive = parse_residues_csv(_write(td, _CSV), chain="A")
        self.assertNotIn(5, scores)
        self.assertEqual(positive, {2, 3})

    def test_scores_are_continuous_not_a_rank_staircase(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            scores, _ = parse_residues_csv(_write(td, _CSV), chain="A")
        # Four residues, four distinct probabilities. The old 1/rank scheme could
        # only ever emit values from {1, 1/2, 1/3, 1/4, 1/5, 0}.
        self.assertEqual(len(set(scores.values())), 4)


class TestNativeScoresDriveMetrics(unittest.TestCase):
    def test_native_scores_preferred_over_pocket_derived(self) -> None:
        universe = [1, 2, 3, 4]
        prediction = {
            "residue_scores": {"1": 0.01, "2": 0.91, "3": 0.70, "4": 0.00},
            "residue_positive": [2, 3],
        }
        got = native_residue_scores(prediction, universe)
        self.assertIsNotNone(got)
        scores, positive = got
        self.assertEqual(scores[2], 0.91)
        self.assertEqual(positive, {2, 3})

    def test_absent_native_table_falls_back(self) -> None:
        self.assertIsNone(native_residue_scores({"pockets": []}, [1, 2]))
        self.assertIsNone(native_residue_scores(None, [1, 2]))

    def test_auc_uses_native_scores_and_native_operating_point(self) -> None:
        universe = [1, 2, 3, 4]
        prediction = {
            "residue_scores": {"1": 0.01, "2": 0.91, "3": 0.70, "4": 0.00},
            "residue_positive": [2, 3],
        }
        out = residue_auc_pr([], true_residues=[2, 3], universe=universe,
                             prediction=prediction)
        self.assertTrue(out["available"])
        self.assertEqual(out["operating_point"], "predictor_native_binary_call")
        self.assertEqual(out["residue_auc"], 1.0)
        self.assertEqual(out["residue_f1_universe"], 1.0)

    def test_pocket_only_detector_keeps_pocket_operating_point(self) -> None:
        pockets = [{"rank": 1, "residues": [2, 3]}]
        out = residue_auc_pr(pockets, true_residues=[2, 3], universe=[1, 2, 3, 4],
                             prediction={"pockets": pockets})
        self.assertEqual(out["operating_point"], "residue_in_any_predicted_pocket")


if __name__ == "__main__":
    unittest.main()
