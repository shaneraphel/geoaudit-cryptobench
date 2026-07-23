"""Unit tests: leak guard, DCA, TOOL_UNAVAILABLE accounting, sample-size semantics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pocket_bench.metrics import dca, score_prediction, topk_dca_success
from pocket_bench.dataset.clustering import (
    build_structure_cluster_ledger,
)
from pocket_bench.methods import prediction
from pocket_bench.methods.foliation_pocket import predict as fol_predict
from pocket_bench.pdb_io import assert_no_hetatm, write_receptor_only_pdb, parse_pdb_atoms
from pocket_bench.stats import aggregate_primary


class TestLeakGuard(unittest.TestCase):
    def test_hetatm_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.pdb"
            p.write_text(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
                "HETATM    2  C1  LIG A   2       1.000   1.000   1.000  1.00 20.00           C\n"
                "END\n"
            )
            with self.assertRaises(AssertionError):
                assert_no_hetatm(p)


class TestMetrics(unittest.TestCase):
    def test_dca_and_topk(self):
        lig = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        self.assertLessEqual(dca([0.5, 0.0, 0.0], lig), 0.5 + 1e-9)
        pockets = [{"rank": 1, "center_xyz": [10.0, 0, 0]}, {"rank": 2, "center_xyz": [0.1, 0, 0]}]
        t1 = topk_dca_success(pockets, lig, k=1, threshold_a=4.0)
        self.assertFalse(t1["success"])
        t3 = topk_dca_success(pockets, lig, k=3, threshold_a=4.0)
        self.assertTrue(t3["success"])

    def test_tool_unavailable_not_in_miss_denom(self):
        rows = [
            score_prediction(
                prediction(method="x", pdb_id="A", status="OK", pockets=[{"rank": 1, "center_xyz": [0, 0, 0]}]),
                {
                    "ligand_heavy_coords": [[0.1, 0, 0]],
                    "ligand_centroid": [0.1, 0, 0],
                    "binding_residues": [],
                },
            ),
            score_prediction(
                prediction(method="x", pdb_id="B", status="TOOL_UNAVAILABLE"),
                {
                    "ligand_heavy_coords": [[0.1, 0, 0]],
                    "ligand_centroid": [0.1, 0, 0],
                    "binding_residues": [],
                },
            ),
        ]
        agg = aggregate_primary(rows)
        self.assertEqual(agg["n_eligible_primary"], 1)
        self.assertEqual(agg["n_tool_failures"], 1)
        self.assertAlmostEqual(agg["top1_success_rate"], 1.0)
        self.assertAlmostEqual(agg["failure_rate"], 0.5)
        self.assertIn("mean_runtime_s", agg)
        self.assertIn("mean_residue_f1", agg)
        self.assertEqual(agg["n_tool_unavailable"], 1)
        self.assertEqual(agg["top1_intention_to_evaluate"]["n"], 1)
        self.assertGreater(agg["top1_wilson_95ci"]["ci_low"], 0.0)

    def test_crash_counts_as_intention_to_evaluate_failure(self):
        rows = [
            score_prediction(
                prediction(method="x", pdb_id="A", status="CRASH"),
                {
                    "ligand_heavy_coords": [[0.1, 0, 0]],
                    "ligand_centroid": [0.1, 0, 0],
                    "binding_residues": [],
                },
            )
        ]
        aggregate = aggregate_primary(rows)
        self.assertEqual(aggregate["top1_intention_to_evaluate"]["n"], 1)
        self.assertEqual(aggregate["top1_intention_to_evaluate"]["successes"], 0)

    def test_residue_f1_when_pockets_emit_residues(self):
        rows = [
            score_prediction(
                prediction(
                    method="x",
                    pdb_id="A",
                    status="OK",
                    pockets=[{"rank": 1, "center_xyz": [0, 0, 0], "residues": ["A:ALA1", "A:LEU2"]}],
                    runtime_s=1.5,
                ),
                {
                    "ligand_heavy_coords": [[0.1, 0, 0]],
                    "ligand_centroid": [0.1, 0, 0],
                    "binding_residues": ["A:ALA1", "A:GLY9"],
                },
            ),
        ]
        self.assertTrue(rows[0]["residue_f1"]["available"])
        agg = aggregate_primary(rows)
        self.assertEqual(agg["n_residue_f1"], 1)
        self.assertGreater(agg["mean_residue_f1"], 0.0)
        self.assertAlmostEqual(agg["mean_runtime_s"], 1.5)


class TestFoliationNoLigand(unittest.TestCase):
    def test_synthetic_receptor(self):
        # Build a small hollow box of atoms
        lines = ["REMARK synthetic"]
        n = 0
        for x in range(0, 12, 2):
            for y in range(0, 12, 2):
                for z in (0, 10):
                    n += 1
                    lines.append(
                        f"ATOM  {n:5d}  CA  ALA A{n:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
                    )
        for z in range(2, 10, 2):
            for x, y in ((0, 0), (0, 10), (10, 0), (10, 10)):
                n += 1
                lines.append(
                    f"ATOM  {n:5d}  CA  ALA A{n:4d}    {float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00 20.00           C"
                )
        # pad atoms
        while n < 60:
            n += 1
            lines.append(
                f"ATOM  {n:5d}  CA  ALA A{n:4d}    {float(n % 10):8.3f}{float((n * 3) % 10):8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
        lines.append("END")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rec.pdb"
            p.write_text("\n".join(lines) + "\n")
            before = p.read_bytes()
            out = fol_predict(p, pdb_id="SYN", grid_step=2.0, top_k=3)
            after = p.read_bytes()
            self.assertEqual(before, after, "predictor must not mutate receptor PDB")
            self.assertEqual(out["status"], "OK")
            self.assertGreaterEqual(len(out["pockets"]), 1)
            self.assertNotIn("HETATM", before.decode())
            self.assertEqual(out.get("input_contract"), "receptor_only_pdb_no_ligand_hetatm")


class TestStructureClusters(unittest.TestCase):
    @staticmethod
    def _receptor(path: Path, offset: float) -> None:
        lines = []
        for index in range(1, 61):
            x = float(index % 10) + offset
            y = float((index // 10) % 6)
            z = float(index % 3)
            lines.append(
                f"ATOM  {index:5d}  CA  ALA A{index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
        path.write_text("\n".join(lines) + "\nEND\n")

    def test_related_receptors_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pdb_id, offset in (("AAAA", 0.0), ("BBBB", 0.2)):
                self._receptor(root / f"{pdb_id}.pdb", offset)
                (root / f"{pdb_id}_raw.pdb").write_text("HEADER TEST\n")
            entries = [
                {
                    "pdb_id": "AAAA",
                    "genotype": "WT",
                    "split": "validation",
                    "receptor_pdb": "AAAA.pdb",
                    "raw_pdb": "AAAA_raw.pdb",
                },
                {
                    "pdb_id": "BBBB",
                    "genotype": "WT",
                    "split": "locked_test",
                    "receptor_pdb": "BBBB.pdb",
                    "raw_pdb": "BBBB_raw.pdb",
                },
            ]
            ledger = build_structure_cluster_ledger(entries, root=root)
            self.assertFalse(ledger["split_integrity_passed"])
            self.assertEqual(
                ledger["recommended_evidence_level"], "retrospective_pilot_only"
            )


if __name__ == "__main__":
    unittest.main()
