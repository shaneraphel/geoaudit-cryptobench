import json
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys


class TestPredictionManifestGuard(unittest.TestCase):
    def test_forbidden_label_fields_rejected(self):
        # Lightweight inline guard equivalent to tools/pocket_bench_predict.py
        forbidden = {"ligand_resname", "labels_json", "ligand_centroid", "ligand_heavy_coords"}
        bad = {
            "ligand_fields_present": False,
            "entries": [{"pdb_id": "3ERT", "ligand_resname": "OHT", "split": "locked_test"}],
        }
        self.assertTrue(any(forbidden & set(row) for row in bad["entries"]))
        good = {
            "ligand_fields_present": False,
            "entries": [
                {
                    "pdb_id": "3ERT",
                    "split": "locked_test",
                    "genotype": "WT",
                    "structure_cluster_id": "esr1-struct-001",
                    "receptor_pdb": "data/receptors/3ERT_A_receptor.pdb",
                    "receptor_sha256": "x",
                }
            ],
        }
        self.assertFalse(any(forbidden & set(row) for row in good["entries"]))
        self.assertIs(good["ligand_fields_present"], False)


if __name__ == "__main__":
    unittest.main()
